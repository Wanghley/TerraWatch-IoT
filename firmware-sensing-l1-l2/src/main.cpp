// ====== IMPORTS ======
#include <Arduino.h>
#include "ArduinoJson.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <esp_task_wdt.h>
#include <esp_system.h>
#include <string.h>

// --- KEY IMPORTS ---
#include "shared_types.h"
#include "predictor.h"

#include "sleep_manager.h"
#include "led_manager.h"
#include "thermal_array_manager.h"
#include "mmWave_array_manager.h"
#include "mic_manager.h"
#include "deterrent_manager.h"

// ====== USER CONFIG ======
// PIR pins
#define LPIR 12
#define CPIR 13
#define RPIR 14

// LED brightness (passed to LedManager, but your current implementation ignores it)
#define BRIGHTNESS 10

// Watchdog timeout
#define WDT_TIMEOUT_SECONDS 120

// I2C & PIN DEFS
#define T0_SDA 48
#define T0_SCL 47
#define T1_SDA 8
#define T1_SCL 9
#define DETERRENT_PIN 11

// mmWave UART pins
#define RADAR1_RX 10
#define RADAR1_TX 16
#define RADAR2_RX 18
#define RADAR2_TX 17

// =========================================================
// 🧠 AI THRESHOLDS (UPDATED FOR YOUR ACTUAL MODEL)
// =========================================================
// Your actual optimal threshold is 0.49, not 0.73
// Adjust all thresholds accordingly

// >40%: Likely detection (below optimal to catch edge cases)
static constexpr float PROB_THRESHOLD_WEAK   = 0.40f; 

// >65%: Confirmed detection (high confidence zone)
static constexpr float PROB_THRESHOLD_STRONG = 0.65f; 

// HYSTERESIS & STABILITY
// Trigger at 0.45, reset at 0.35 to prevent flickering
static constexpr float THRESHOLD_TRIGGER = 0.45f; 
static constexpr float THRESHOLD_RESET   = 0.35f; 

// Require 3 consecutive hits for more stability
static constexpr int   N_CONSECUTIVE_HITS = 3;
static constexpr unsigned long AI_MIN_RETRIGGER_MS = 2000; 

// Robustness Configs
#define PIR_DEBOUNCE_MS     150   // PIR must be HIGH for 150ms
#define KEEP_ALIVE_MS       5000  // Stay awake 5s after last motion
#define AI_SAMPLE_RATE_MS   333   // ~3 samples/sec

// Cooldowns
#define DETER_COOLDOWN_MS   10000 // Ignore PIR for 10s after any deterrent

// =========================================================
// LOGGING CONFIG
// =========================================================
// Levels: 0=OFF, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG
#ifndef LOG_LEVEL
  #define LOG_LEVEL 4
#endif

#if LOG_LEVEL >= 1
  #define LOG_ERROR(...)  Serial.printf("[ERROR] " __VA_ARGS__)
#else
  #define LOG_ERROR(...)
#endif

#if LOG_LEVEL >= 2
  #define LOG_WARN(...)   Serial.printf("[WARN] " __VA_ARGS__)
#else
  #define LOG_WARN(...)
#endif

#if LOG_LEVEL >= 3
  #define LOG_INFO(...)   Serial.printf("[INFO] " __VA_ARGS__)
#else
  #define LOG_INFO(...)
#endif

#if LOG_LEVEL >= 4
  #define LOG_DEBUG(...)  Serial.printf("[DEBUG] " __VA_ARGS__)
#else
  #define LOG_DEBUG(...)
#endif

// --- GLOBALS ---
QueueHandle_t packetQueue      = nullptr;
TaskHandle_t  sensorTaskHandle = nullptr;
TaskHandle_t  uplinkTaskHandle = nullptr;

// Predictor mutex
SemaphoreHandle_t predictorMutex = nullptr;

// State mutex to protect RTC/shared state
SemaphoreHandle_t stateMutex = nullptr;

// [RTC] State Tracking (survives deep sleep reset)
RTC_DATA_ATTR volatile unsigned long g_lastDeterrentTime   = 0; // last time deterrent fired
RTC_DATA_ATTR volatile bool          g_alarmActive         = false;
RTC_DATA_ATTR volatile int           g_consecutiveHits     = 0;
RTC_DATA_ATTR volatile unsigned long g_lastAlarmChangeMs   = 0;

// Managers
SleepManager        sleepManager(LPIR, CPIR, RPIR, true);
LedManager          ledManager(RGB_BUILTIN, BRIGHTNESS);
ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, false);
mmWaveArrayManager  mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, false);
MicManager          micManager(0.2, false);
DeterrentManager    deterrentManager(DETERRENT_PIN, (LOG_LEVEL >= 4));

// The AI Engine
Predictor predictor;

// Task Prototypes
void sensorCoreTask(void *p);
void uplinkCoreTask(void *p);

// [HELPER] Flush UART Buffers to prevent 'atof' crashes on resume
static void flushUARTs() {
    if (Serial1) {
        while (Serial1.available() > 0) { Serial1.read(); }
    }
    if (Serial2) {
        while (Serial2.available() > 0) { Serial2.read(); }
    }
}

// Safe time-diff helper (millis wraparound safe)
static inline bool timeSinceLessThan(unsigned long prev, unsigned long duration_ms) {
    unsigned long now = millis();
    return (now - prev) < duration_ms;
}

// =========================================================
// SETUP
// =========================================================
void setup() {
    Serial.begin(115200);
    delay(500);

    // Basic pin directions
    pinMode(LPIR, INPUT);
    pinMode(CPIR, INPUT);
    pinMode(RPIR, INPUT);
    pinMode(DETERRENT_PIN, OUTPUT);

    // Init Watchdog
    esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true);
    esp_task_wdt_add(NULL);

    LOG_INFO("\n--- Booting TerraWatch AI 2.0 ---\n");
    LOG_INFO("Log level: %d\n", LOG_LEVEL);

    // predictor mutex
    predictorMutex = xSemaphoreCreateMutex();
    if (!predictorMutex) {
        LOG_WARN("predictor mutex allocation failed.\n");
    }

    // state mutex
    stateMutex = xSemaphoreCreateMutex();
    if (!stateMutex) {
        LOG_ERROR("state mutex creation failed. Halting.\n");
        while (1) {
            delay(1000);
        }
    }

    // Initialize AI first
    LOG_INFO("[System] Allocating AI Memory...\n");
    if (!predictor.begin()) {
        LOG_ERROR("AI Failed to Initialize. Halting.\n");
        while (1) {
            ledManager.setColor(255, 0, 0);
            delay(100);
            ledManager.setColor(0, 0, 0);
            delay(100);
        }
    }
    LOG_INFO("AI Neural Network Ready.\n");

    // Initialize hardware
    sleepManager.configure();
    ledManager.begin();

    if (LOG_LEVEL >= 4) {
        LOG_DEBUG("[LED] Running LED Test...\n");
        ledManager.setColor(255, 0, 0);
        LOG_DEBUG("  -> Red\n");
        delay(500);
        ledManager.setColor(0, 255, 0);
        LOG_DEBUG("  -> Green\n");
        delay(500);
        ledManager.setColor(0, 0, 255);
        LOG_DEBUG("  -> Blue\n");
        delay(500);
        ledManager.setOff();
        LOG_DEBUG("  -> Off\n");
    }

    bool thermalOk = thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL);
    if (!thermalOk) {
        LOG_WARN("[thermal] Thermal sensors unavailable. Using zeros.\n");
    }

    mmWaveManager.begin();
    micManager.begin();
    deterrentManager.begin();
    deterrentManager.enablePersistent(true); // latch behavior

    // Create Queue and Tasks
    packetQueue = xQueueCreate(1, sizeof(SensorPacket));

    xTaskCreatePinnedToCore(sensorCoreTask, "sensors",
                            10240, nullptr, 2, &sensorTaskHandle, 0);
    xTaskCreatePinnedToCore(uplinkCoreTask, "ai_logic",
                            8192, nullptr, 1, &uplinkTaskHandle, 1);

    LOG_INFO("System Ready.\n");
    ledManager.setColor(0, 255, 0); // Green idle
    
    // LED test sequence
    delay(500);
    ledManager.setOff();
    delay(500);
    ledManager.setColor(0, 255, 0); // Green idle
    delay(500);
    ledManager.setOff();
    delay(500);
    ledManager.setColor(0, 255, 0); // Green idle - SETUP MARKER
    delay(500);
    ledManager.setOff();
    
    LOG_INFO("[System] 🚀 Hardware initialization complete. Starting 45s warmup...\n");
    
    esp_task_wdt_reset();
}

// =========================================================
// SENSOR TASK (CORE 0)
// =========================================================
void sensorCoreTask(void *p) {
    SensorPacket pkt;
    double micL_temp, micR_temp;

    esp_task_wdt_add(NULL);

    for (;;) {
        // service radar regularly
        uint32_t notificationCount = ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(10));
        mmWaveManager.update();

        if (notificationCount > 0) {
            LOG_DEBUG("[Sensors] Reading Data...\n");

            // 1. Thermal
            thermalManager.readRotated();
            ThermalReadings thermalData = thermalManager.getObject();
            memcpy(pkt.thermal_left,   thermalData.left,   sizeof(float) * 64);
            memcpy(pkt.thermal_center, thermalData.center, sizeof(float) * 64);
            memcpy(pkt.thermal_right,  thermalData.right,  sizeof(float) * 64);

            // 2. Radar
            RadarData r1 = mmWaveManager.getRadar1();
            pkt.r1.range_cm   = r1.range_cm;
            pkt.r1.speed_ms   = r1.speed_ms;
            pkt.r1.energy     = r1.energy;
            pkt.r1.numTargets = r1.numTargets;

            RadarData r2 = mmWaveManager.getRadar2();
            pkt.r2.range_cm   = r2.range_cm;
            pkt.r2.speed_ms   = r2.speed_ms;
            pkt.r2.energy     = r2.energy;
            pkt.r2.numTargets = r2.numTargets;

            // 3. Mic
            micManager.read(micL_temp, micR_temp);
            pkt.micL = (float)micL_temp;
            pkt.micR = (float)micR_temp;

            pkt.timestamp = millis();

            xQueueOverwrite(packetQueue, &pkt);
        }

        esp_task_wdt_reset();
    }

    esp_task_wdt_delete(NULL);
    vTaskDelete(NULL);
}

// =========================================================
// UPLINK / AI TASK (CORE 1)
// =========================================================
void uplinkCoreTask(void *p) {
    SensorPacket pkt;
    esp_task_wdt_add(NULL);

    float emaProbability = 0.0f;

    for (;;) {
        if (xQueueReceive(packetQueue, &pkt, portMAX_DELAY) != pdTRUE)
            continue;

        esp_task_wdt_reset();

        unsigned long now = millis();
        bool inDeterCooldown = false;

        if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            if (g_lastDeterrentTime > 0) {
                inDeterCooldown = (now - g_lastDeterrentTime < DETER_COOLDOWN_MS);
            }
            xSemaphoreGive(stateMutex);
        }

        if (inDeterCooldown) {
            LOG_DEBUG("Packet dropped (Cooldown)\n");
            continue;
        }

        // --- Run Predictor ---
        unsigned long startAI = millis();
        float probability = 0.0f;
        bool predictor_ok = true;

        if (predictorMutex) {
            if (xSemaphoreTake(predictorMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                probability = predictor.update(pkt);
                xSemaphoreGive(predictorMutex);
            } else {
                predictor_ok = false;
                LOG_WARN("Predictor busy, skipping sample.\n");
            }
        } else {
            probability = predictor.update(pkt);
        }

        if (!predictor_ok) continue;

        unsigned long durAI = millis() - startAI;

        // EMA smoothing
        // TUNED: Changed alpha from 0.25 to 0.60.
        // 60% new data, 40% history. This makes it much more responsive.
        emaProbability = (0.60f * probability) + (0.40f * emaProbability);

        // Logging (rate limited)
        static unsigned long lastLog = 0;
        if ((millis() - lastLog) > 500 && LOG_LEVEL >= 3) {
            Serial.printf("🔮 AI Prob: %.1f%% (EMA %.1f%%) (Took %lums)\n",
                          probability * 100.0f, emaProbability * 100.0f, durAI);
            if (probability > 0.05f && probability < PROB_THRESHOLD_WEAK) {
                Serial.printf("   [NOISE] Background spike: %.1f%%\n",
                              probability * 100.0f);
            }
            lastLog = millis();
        }

        // =====================================================
        //  A. HYSTERESIS + CONSECUTIVE HITS
        // =====================================================
        now = millis();

        if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            if (probability >= THRESHOLD_TRIGGER) {
                g_consecutiveHits++;
            } else if (probability <= THRESHOLD_RESET) {
                g_consecutiveHits = 0;
                if (g_alarmActive) {
                    g_alarmActive = false;
                    g_lastAlarmChangeMs = now;
                    LOG_INFO("Alarm reset (probability below reset threshold).\n");
                }
            }
            xSemaphoreGive(stateMutex);
        } else {
            LOG_WARN("stateMutex busy; skipping state update.\n");
            continue;
        }

        // Evaluate triggering conditions
        bool canTriggerNewAlarm = false;
        if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            if (!g_alarmActive &&
                (g_consecutiveHits >= N_CONSECUTIVE_HITS) &&
                (now - g_lastAlarmChangeMs > AI_MIN_RETRIGGER_MS)) {
                canTriggerNewAlarm = true;
            }
            xSemaphoreGive(stateMutex);
        }

        // =====================================================
        //  B. HIGH-LEVEL DECISION
        // =====================================================
        // Using emaProbability helps filter single-frame glitches
        if (canTriggerNewAlarm && emaProbability > PROB_THRESHOLD_WEAK) {
            if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
                g_alarmActive       = true;
                g_lastAlarmChangeMs = now;
                g_lastDeterrentTime = now;
                xSemaphoreGive(stateMutex);
            } else {
                LOG_WARN("stateMutex busy; skipped alarm trigger.\n");
                continue;
            }

            if (emaProbability > PROB_THRESHOLD_STRONG) {
                LOG_INFO("🚨 DETECTED: ANIMAL/HUMAN (STRONG)!\n");
                ledManager.setColor(255, 0, 0); // Red
                deterrentManager.signalSureDetection();
            } else {
                LOG_INFO("⚠️ Suspicious activity... (UNSURE)\n");
                ledManager.setColor(255, 255, 0); // Yellow/Orange
                deterrentManager.signalUnsureDetection();
            }
        } else {
            bool alarmCopy = false;
            if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
                alarmCopy = g_alarmActive;
                xSemaphoreGive(stateMutex);
            }
            if (!alarmCopy) {
                ledManager.setColor(0, 255, 0); // Green idle
                deterrentManager.deactivate();
            }
        }
    }

    esp_task_wdt_delete(NULL);
    vTaskDelete(NULL);
}

// =========================================================
// MAIN LOOP
// =========================================================
void loop() {
    esp_task_wdt_reset();
    deterrentManager.update();

    static unsigned long lastStats = 0;
    if ((LOG_LEVEL >= 3) && (millis() - lastStats > 10000)) {
        lastStats = millis();
        LOG_INFO("[SYS] Free Heap: %u bytes\n", esp_get_free_heap_size());
    }

    // --- PIR + COOLDOWN ---
    bool rawMotion = (digitalRead(LPIR) == HIGH) ||
                     (digitalRead(CPIR) == HIGH) ||
                     (digitalRead(RPIR) == HIGH);

    unsigned long now = millis();
    bool inDeterCooldown = false;
    unsigned long lastDetCopy = 0;

    if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        lastDetCopy = g_lastDeterrentTime;
        if (g_lastDeterrentTime > 0) {
            inDeterCooldown = (now - g_lastDeterrentTime < DETER_COOLDOWN_MS);
        }
        xSemaphoreGive(stateMutex);
    }

    if (inDeterCooldown) {
        rawMotion = false;
        static unsigned long lastCoolLog = 0;
        if ((LOG_LEVEL >= 3) && (now - lastCoolLog > 2000)) {
            unsigned long remain = (DETER_COOLDOWN_MS - (now - lastDetCopy)) / 1000;
            LOG_INFO("❄️ COOLDOWN: %lus remaining. Ignoring PIR.\n", remain);
            lastCoolLog = now;
        }
    }

    // --- PIR DEBOUNCE ---
    static unsigned long pirHighStartTime = 0;
    static bool motionConfirmed = false;

    if (rawMotion) {
        if (pirHighStartTime == 0) {
            pirHighStartTime = millis();
            LOG_DEBUG("[PIR] Rising Edge Detected\n");
        } else if ((millis() - pirHighStartTime) > PIR_DEBOUNCE_MS) {
            if (!motionConfirmed) {
                LOG_INFO("[PIR] Motion Confirmed (Stable)\n");
                motionConfirmed = true;
            }
        }
    } else {
        pirHighStartTime = 0;
        motionConfirmed  = false;
    }

    // --- AWAKE/SLEEP STATE MACHINE ---
    static unsigned long lastMotionTime  = millis();
    static unsigned long lastSampleTime  = 0;
    static bool sensorsSuspended         = false;
    static bool setupCompleteFlag        = false;
    static unsigned long setupStartTime  = millis();
    static unsigned long lastAlarmResetTime = millis();

    // Force stay awake until setup is complete (after green LED blink + 45 seconds)
    if (!setupCompleteFlag) {
        lastMotionTime = millis(); // Keep timer fresh during warmup
        unsigned long setupElapsed = millis() - setupStartTime;
        
        // Setup is complete after 45 seconds (45000 ms)
        if (setupElapsed > 45000) {
            setupCompleteFlag = true;
            LOG_INFO("[System] ✅ Setup complete (%.1fs). Entering normal operation.\n", 
                     setupElapsed / 1000.0f);
        } else {
            // Log progress every 10 seconds during setup
            static unsigned long lastSetupLog = 0;
            if ((LOG_LEVEL >= 3) && (millis() - lastSetupLog > 10000)) {
                LOG_INFO("[System] ⏳ Setup in progress... %lus remaining\n", 
                         (45000 - setupElapsed) / 1000);
                lastSetupLog = millis();
            }
        }
    }

    if (motionConfirmed) {
        lastMotionTime = millis();
    }

    bool stayAwake = false;
    if (inDeterCooldown) {
        stayAwake = deterrentManager.isSignaling();
    } else {
        stayAwake = !setupCompleteFlag ||
                    (millis() - lastMotionTime < KEEP_ALIVE_MS) ||
                    deterrentManager.isSignaling();
    }

    if (stayAwake) {
        if (!inDeterCooldown) {
            if (sensorsSuspended && sensorTaskHandle != NULL) {
                flushUARTs();
                vTaskResume(sensorTaskHandle);
                sensorsSuspended = false;
                LOG_INFO("⚡ Sensors Resumed\n");
            }

            if (millis() - lastSampleTime > AI_SAMPLE_RATE_MS) {
                if (sensorTaskHandle) {
                    xTaskNotifyGive(sensorTaskHandle);
                }
                lastSampleTime = millis();
                LOG_DEBUG(".");
            }
        }
    } else {
        // Check if enough time has passed since last alarm reset (minimum 2 minutes)
        unsigned long timeSinceLastReset = millis() - lastAlarmResetTime;
        const unsigned long MIN_RESET_INTERVAL = 120000; // 2 minutes in milliseconds
        
        if (timeSinceLastReset < MIN_RESET_INTERVAL) {
            unsigned long remainingTime = (MIN_RESET_INTERVAL - timeSinceLastReset) / 1000;
            LOG_INFO("⏱️ Cannot sleep yet. %lus remaining before next sleep allowed.\n", 
                     remainingTime);
            delay(1000);
            return; // Skip sleep, stay awake
        }
        
        // Prepare for deep sleep
        ledManager.setColor(0, 0, 255); // Blue idle
        LOG_INFO("\n💤 No motion. Entering Deep Sleep...\n");
        Serial.flush();

        esp_task_wdt_delete(NULL);

        unsigned long sleepStart = millis();
        while (deterrentManager.isSignaling() && (millis() - sleepStart < 5000)) {
            deterrentManager.update();
            delay(10);
        }

        if (!sensorsSuspended && sensorTaskHandle != NULL) {
            vTaskSuspend(sensorTaskHandle);
            sensorsSuspended = true;
        }

        sleepManager.goToSleep(0);

        // On wake
        esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true);
        esp_task_wdt_add(NULL);
        ledManager.setOff();
        LOG_INFO("⏰ Woke up!\n");
        
        // Update last reset time after waking
        lastAlarmResetTime = millis();
        
        // Sensors will be resumed on next loop() when conditions are met
    }
}