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

// =========================================================
// OPERATIONAL MODES
// =========================================================
enum OperationMode {
    MODE_AI_SURE_ONLY,        // AI enabled, fire only on SURE (EMA >= 0.75)
    MODE_AI_UNSURE_ENABLED,   // AI enabled, fire on both SURE and UNSURE
    MODE_PIR_UNSURE_ONLY,     // PIR-triggered, always send UNSURE signal
    MODE_PIR_SURE_ONLY,       // PIR-triggered, always send SURE signal
    MODE_RAW_DATA_COLLECTION  // Raw sensor data collection (NO AI, NO sleep, continuous streaming)
};

// =========== SELECT OPERATING MODE HERE ===========
static constexpr OperationMode OPERATING_MODE = MODE_PIR_UNSURE_ONLY;
// ==================================================

// ====== USER CONFIG ======
// PIR pins
#define LPIR 12
#define CPIR 13
#define RPIR 14

#define BRIGHTNESS 10
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

// Raw data collection settings
#define RAW_DATA_SAMPLE_RATE_MS 20  // 50Hz sampling
#define RAW_DATA_SERIAL_BAUD 115200  // High-speed for continuous data

// =========================================================
// AI THRESHOLDS
// =========================================================
static constexpr float PROB_THRESHOLD_TRIGGER = 0.40f;   // Hit counter increment
static constexpr float PROB_THRESHOLD_RESET   = 0.25f;   // Reset counter
static constexpr float PROB_THRESHOLD_WEAK    = 0.60f;   // UNSURE signal threshold
static constexpr float PROB_THRESHOLD_STRONG  = 0.75f;   // SURE signal threshold

// Hit counter & timing
static constexpr int   N_CONSECUTIVE_HITS = 1;
static constexpr unsigned long AI_MIN_RETRIGGER_MS = 1500;

// Timing configs
#define PIR_DEBOUNCE_MS     200
#define KEEP_ALIVE_MS       5000
#define AI_SAMPLE_RATE_MS   333
#define DETER_COOLDOWN_MS   15000
#define PIR_ONLY_COOLDOWN_MS 8000
#define SLEEP_TIMEOUT_MS    30000

// =========================================================
// LOGGING CONFIG
// =========================================================
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

// =========================================================
// GLOBALS
// =========================================================
QueueHandle_t packetQueue = nullptr;
TaskHandle_t sensorTaskHandle = nullptr;
TaskHandle_t uplinkTaskHandle = nullptr;

SemaphoreHandle_t predictorMutex = nullptr;
SemaphoreHandle_t stateMutex = nullptr;

// [RTC] State Tracking
RTC_DATA_ATTR volatile unsigned long g_lastDeterrentTime = 0;
RTC_DATA_ATTR volatile bool g_alarmActive = false;
RTC_DATA_ATTR volatile int g_consecutiveHits = 0;
RTC_DATA_ATTR volatile unsigned long g_lastAlarmChangeMs = 0;
RTC_DATA_ATTR volatile unsigned long g_bootTime = 0;
RTC_DATA_ATTR volatile unsigned int g_wakeCount = 0;

// Managers
SleepManager sleepManager(LPIR, CPIR, RPIR, true);
LedManager ledManager(RGB_BUILTIN, BRIGHTNESS);
ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, false);
mmWaveArrayManager mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, false);
MicManager micManager(0.2, false);
DeterrentManager deterrentManager(DETERRENT_PIN, (LOG_LEVEL >= 4));
Predictor predictor;

// Task Prototypes
void sensorCoreTask(void *p);
void uplinkCoreTask(void *p);
void rawDataCollectionLoop();

// =========================================================
// HELPER: Get operation mode name
// =========================================================
const char* getModeString() {
    switch (OPERATING_MODE) {
        case MODE_AI_SURE_ONLY:        return "AI (SURE only)";
        case MODE_AI_UNSURE_ENABLED:   return "AI (SURE+UNSURE)";
        case MODE_PIR_UNSURE_ONLY:     return "PIR (UNSURE only)";
        case MODE_PIR_SURE_ONLY:       return "PIR (SURE only)";
        case MODE_RAW_DATA_COLLECTION: return "RAW DATA COLLECTION";
        default:                        return "UNKNOWN";
    }
}

// =========================================================
// HELPER: Enhanced UART Flush with Timeout
// =========================================================
static void fullSystemFlush() {
    unsigned long start = millis();
    const unsigned long FLUSH_TIMEOUT = 500;
    
    while (Serial1.available() > 0 && (millis() - start < FLUSH_TIMEOUT)) {
        Serial1.read();
    }
    
    start = millis();
    while (Serial2.available() > 0 && (millis() - start < FLUSH_TIMEOUT)) {
        Serial2.read();
    }
    
    delay(50);
    
    start = millis();
    while ((Serial1.available() > 0 || Serial2.available() > 0) && 
           (millis() - start < FLUSH_TIMEOUT)) {
        if (Serial1.available()) Serial1.read();
        if (Serial2.available()) Serial2.read();
    }
    
    LOG_DEBUG("[System] UART buffers flushed\n");
}

// =========================================================
// HELPER: Graceful Task Suspension
// =========================================================
static void suspendAllTasks() {
    if (sensorTaskHandle != NULL) {
        LOG_DEBUG("[Sleep] Suspending sensor task...\n");
        vTaskSuspend(sensorTaskHandle);
    }
    if (uplinkTaskHandle != NULL && (OPERATING_MODE == MODE_AI_SURE_ONLY || 
                                      OPERATING_MODE == MODE_AI_UNSURE_ENABLED)) {
        LOG_DEBUG("[Sleep] Suspending uplink task...\n");
        vTaskSuspend(uplinkTaskHandle);
    }
}

// =========================================================
// HELPER: Resume All Tasks
// =========================================================
static void resumeAllTasks() {
    fullSystemFlush();
    
    if (sensorTaskHandle != NULL) {
        LOG_DEBUG("[Sleep] Resuming sensor task...\n");
        vTaskResume(sensorTaskHandle);
    }
    if (uplinkTaskHandle != NULL && (OPERATING_MODE == MODE_AI_SURE_ONLY || 
                                      OPERATING_MODE == MODE_AI_UNSURE_ENABLED)) {
        LOG_DEBUG("[Sleep] Resuming uplink task...\n");
        vTaskResume(uplinkTaskHandle);
    }
}

// =========================================================
// RAW DATA COLLECTION MODE
// =========================================================
void rawDataCollectionLoop() {
    LOG_INFO("\n=== RAW DATA COLLECTION MODE ===\n");
    LOG_INFO("Sampling at %.1f Hz\n", 1000.0f / RAW_DATA_SAMPLE_RATE_MS);
    LOG_INFO("Streaming raw JSON to Serial at %d baud\n", RAW_DATA_SERIAL_BAUD);
    LOG_INFO("Commands: 'STOP' to halt, 'RESET' to restart\n\n");

    ledManager.setColor(0, 255, 255);  // Cyan = data collection

    unsigned long lastSampleTime = millis();
    bool isRunning = true;

    // Disable WDT for raw collection mode
    esp_task_wdt_deinit();

    while (isRunning) {
        // Check for serial commands
        if (Serial.available()) {
            String cmd = Serial.readStringUntil('\n');
            cmd.trim();
            cmd.toUpperCase();

            if (cmd == "STOP") {
                LOG_INFO("\n📋 Data collection stopped by user\n");
                isRunning = false;
                break;
            } else if (cmd == "RESET") {
                LOG_INFO("🔄 Resetting and continuing...\n");
                lastSampleTime = millis();
            } else if (cmd.startsWith("RATE")) {
                // RATE <ms> to change sample rate
                int spaceIdx = cmd.indexOf(' ');
                if (spaceIdx != -1) {
                    String rateStr = cmd.substring(spaceIdx + 1);
                    int newRate = rateStr.toInt();
                    if (newRate > 0 && newRate < 10000) {
                        LOG_INFO("Sample rate changed to %d ms (%.1f Hz)\n", 
                                 newRate, 1000.0f / newRate);
                    }
                }
            }
        }

        // Sample at configured rate
        if (millis() - lastSampleTime >= RAW_DATA_SAMPLE_RATE_MS) {
            // Read all sensors
            thermalManager.readRotated();
            ThermalReadings thermal = thermalManager.getObject();
            
            mmWaveManager.update();
            RadarData r1 = mmWaveManager.getRadar1();
            RadarData r2 = mmWaveManager.getRadar2();
            
            double leftMic = 0, rightMic = 0;
            micManager.read(leftMic, rightMic);

            // Build JSON with raw data
            StaticJsonDocument<2048> doc;
            doc["timestamp"] = millis();

            // Thermal array data (all 3 sensors, 64 pixels each)
            JsonObject thermalObj = doc.createNestedObject("thermal");
            JsonArray leftArr = thermalObj.createNestedArray("left");
            JsonArray centerArr = thermalObj.createNestedArray("center");
            JsonArray rightArr = thermalObj.createNestedArray("right");
            
            for (int i = 0; i < 64; i++) {
                leftArr.add(serialized(String(thermal.left[i], 2)));
                centerArr.add(serialized(String(thermal.center[i], 2)));
                rightArr.add(serialized(String(thermal.right[i], 2)));
            }

            // mmWave radar data
            JsonObject radarObj = doc.createNestedObject("mmWave");
            
            JsonObject r1Obj = radarObj.createNestedObject("R1");
            r1Obj["numTargets"] = r1.numTargets;
            r1Obj["range_cm"] = r1.range_cm;
            r1Obj["speed_ms"] = serialized(String(r1.speed_ms, 2));
            r1Obj["energy"] = r1.energy;
            r1Obj["valid"] = r1.isValid;

            JsonObject r2Obj = radarObj.createNestedObject("R2");
            r2Obj["numTargets"] = r2.numTargets;
            r2Obj["range_cm"] = r2.range_cm;
            r2Obj["speed_ms"] = serialized(String(r2.speed_ms, 2));
            r2Obj["energy"] = r2.energy;
            r2Obj["valid"] = r2.isValid;

            // Microphone data
            JsonObject micObj = doc.createNestedObject("mic");
            micObj["left"] = serialized(String(leftMic, 4));
            micObj["right"] = serialized(String(rightMic, 4));

            // Serialize and send
            serializeJson(doc, Serial);
            Serial.println();

            lastSampleTime = millis();
        }

        // Prevent watchdog
        delay(1);
    }

    // Cleanup on exit
    LOG_INFO("🛑 Exiting raw data collection mode\n");
    ledManager.setColor(255, 0, 0);
    delay(1000);
    ledManager.setOff();

    // Reinitialize WDT
    esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true);
    esp_task_wdt_add(NULL);
}

// =========================================================
// SETUP
// =========================================================
void setup() {
    // For raw data mode, use high baud rate
    if (OPERATING_MODE == MODE_RAW_DATA_COLLECTION) {
        Serial.begin(RAW_DATA_SERIAL_BAUD);
    } else {
        Serial.begin(115200);
    }
    delay(500);

    if (g_bootTime == 0) {
        g_bootTime = millis();
        g_wakeCount = 0;
    }
    g_wakeCount++;

    pinMode(LPIR, INPUT);
    pinMode(CPIR, INPUT);
    pinMode(RPIR, INPUT);
    pinMode(DETERRENT_PIN, OUTPUT);

    esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true);
    esp_task_wdt_add(NULL);

    LOG_INFO("\n--- TerraWatch AI 3.0 ---\n");
    LOG_INFO("Wake #%u | Mode: %s | Heap: %u bytes\n", g_wakeCount, getModeString(), esp_get_free_heap_size());
    LOG_INFO("Log level: %d\n", LOG_LEVEL);

    // ===== IMMEDIATE ENTRY FOR RAW DATA MODE =====
    if (OPERATING_MODE == MODE_RAW_DATA_COLLECTION) {
        ledManager.begin();
        thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL);
        mmWaveManager.begin();
        micManager.begin();

        LOG_INFO("Thermal init complete\n");
        LOG_INFO("mmWave init complete\n");
        LOG_INFO("Mic init complete\n");
        delay(500);

        // Enter raw data collection immediately
        rawDataCollectionLoop();

        // If we exit, loop infinitely with error state
        while (1) {
            ledManager.setColor(255, 0, 0);
            delay(500);
            ledManager.setOff();
            delay(500);
        }
    }

    // ===== NORMAL INITIALIZATION FOR OTHER MODES =====
    esp_sleep_wakeup_cause_t wakeup_reason = esp_sleep_get_wakeup_cause();
    switch (wakeup_reason) {
        case ESP_SLEEP_WAKEUP_EXT1:
            LOG_INFO("💤→ Woke from PIR interrupt (EXT1)\n");
            break;
        case ESP_SLEEP_WAKEUP_TIMER:
            LOG_INFO("💤→ Woke from timer\n");
            break;
        default:
            LOG_INFO("💤→ Cold boot or unknown wake source (%d)\n", wakeup_reason);
            break;
    }

    predictorMutex = xSemaphoreCreateMutex();
    if (!predictorMutex) LOG_WARN("Predictor mutex creation failed.\n");

    stateMutex = xSemaphoreCreateMutex();
    if (!stateMutex) {
        LOG_ERROR("State mutex creation failed. Halting.\n");
        while (1) delay(1000);
    }

    if (OPERATING_MODE == MODE_AI_SURE_ONLY || OPERATING_MODE == MODE_AI_UNSURE_ENABLED) {
        LOG_INFO("[System] Allocating AI Memory...\n");
        if (!predictor.begin()) {
            LOG_ERROR("AI Failed to Initialize. Halting.\n");
            while (1) {
                ledManager.setColor(255, 0, 0);
                delay(100);
                ledManager.setOff();
                delay(100);
            }
        }
        LOG_INFO("AI Neural Network Ready.\n");
    } else {
        LOG_INFO("AI disabled (PIR-only mode).\n");
    }

    sleepManager.configure();
    ledManager.begin();

    bool thermalOk = thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL);
    if (!thermalOk && (OPERATING_MODE == MODE_AI_SURE_ONLY || OPERATING_MODE == MODE_AI_UNSURE_ENABLED)) {
        LOG_WARN("[thermal] Thermal sensors unavailable.\n");
    }

    mmWaveManager.begin();
    micManager.begin();
    deterrentManager.begin();
    deterrentManager.enablePersistent(false);

    packetQueue = xQueueCreate(1, sizeof(SensorPacket));

    xTaskCreatePinnedToCore(sensorCoreTask, "sensors", 10240, nullptr, 2, &sensorTaskHandle, 0);
    xTaskCreatePinnedToCore(uplinkCoreTask, "ai_logic", 8192, nullptr, 1, &uplinkTaskHandle, 1);

    LOG_INFO("System Ready.\n");
    ledManager.setColor(0, 255, 0);
    delay(200);
    ledManager.setOff();
    delay(200);
    ledManager.setColor(0, 255, 0);
    
    LOG_INFO("[System] 🚀 Initialization complete.\n");
    esp_task_wdt_reset();
}

// =========================================================
// SENSOR TASK (CORE 0)
// =========================================================
void sensorCoreTask(void *p) {
    SensorPacket pkt;
    double micL_temp, micR_temp;
    esp_task_wdt_add(NULL);

    LOG_INFO("[Sensor Task] Started on Core 0\n");

    for (;;) {
        uint32_t notificationCount = ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
        
        mmWaveManager.update();

        if (notificationCount > 0 && (OPERATING_MODE == MODE_AI_SURE_ONLY || OPERATING_MODE == MODE_AI_UNSURE_ENABLED)) {
            LOG_DEBUG("[Sensors] Reading Data...\n");

            thermalManager.readRotated();
            ThermalReadings thermalData = thermalManager.getObject();
            memcpy(pkt.thermal_left, thermalData.left, sizeof(float) * 64);
            memcpy(pkt.thermal_center, thermalData.center, sizeof(float) * 64);
            memcpy(pkt.thermal_right, thermalData.right, sizeof(float) * 64);

            RadarData r1 = mmWaveManager.getRadar1();
            pkt.r1.range_cm = r1.range_cm;
            pkt.r1.speed_ms = r1.speed_ms;
            pkt.r1.energy = r1.energy;
            pkt.r1.numTargets = r1.numTargets;

            RadarData r2 = mmWaveManager.getRadar2();
            pkt.r2.range_cm = r2.range_cm;
            pkt.r2.speed_ms = r2.speed_ms;
            pkt.r2.energy = r2.energy;
            pkt.r2.numTargets = r2.numTargets;

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
    unsigned long lastValidPacketTime = millis();

    LOG_INFO("[AI Task] Started on Core 1\n");

    if (OPERATING_MODE != MODE_AI_SURE_ONLY && OPERATING_MODE != MODE_AI_UNSURE_ENABLED) {
        LOG_INFO("[AI Task] Disabled in PIR-only mode. Suspending.\n");
        vTaskSuspend(NULL);
    }

    for (;;) {
        if (xQueueReceive(packetQueue, &pkt, pdMS_TO_TICKS(5000)) != pdTRUE) {
            LOG_WARN("[AI] No sensor data for 5s.\n");
            continue;
        }

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

        unsigned long startAI = millis();
        float probability = 0.0f;
        bool predictor_ok = true;

        if (predictorMutex) {
            if (xSemaphoreTake(predictorMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                probability = predictor.update(pkt);
                xSemaphoreGive(predictorMutex);
                lastValidPacketTime = millis();
            } else {
                predictor_ok = false;
                LOG_WARN("Predictor busy, skipping sample.\n");
            }
        } else {
            probability = predictor.update(pkt);
            lastValidPacketTime = millis();
        }

        if (!predictor_ok) continue;

        unsigned long durAI = millis() - startAI;
        emaProbability = (0.60f * probability) + (0.40f * emaProbability);

        LOG_DEBUG("Raw: %.1f%% -> EMA: %.1f%%\n", probability * 100.0f, emaProbability * 100.0f);

        static unsigned long lastLog = 0;
        if ((millis() - lastLog) > 500 && LOG_LEVEL >= 3) {
            Serial.printf("🔮 AI: Raw=%.1f%% EMA=%.1f%% (Took %lums)\n",
                          probability * 100.0f, emaProbability * 100.0f, durAI);
            lastLog = millis();
        }

        now = millis();

        if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            if (emaProbability >= PROB_THRESHOLD_TRIGGER) {
                g_consecutiveHits++;
                LOG_DEBUG("Hit %d/%d (EMA: %.1f%%)\n", g_consecutiveHits, N_CONSECUTIVE_HITS, emaProbability * 100.0f);
            } else if (emaProbability <= PROB_THRESHOLD_RESET) {
                if (g_consecutiveHits > 0) {
                    LOG_DEBUG("Hit counter reset (EMA: %.1f%% < %.1f%%)\n", emaProbability * 100.0f, PROB_THRESHOLD_RESET * 100.0f);
                    g_consecutiveHits = 0;
                }
                if (g_alarmActive) {
                    g_alarmActive = false;
                    g_lastAlarmChangeMs = now;
                    LOG_INFO("🔓 Alarm reset.\n");
                }
            } else {
                LOG_DEBUG("In hysteresis zone (%.1f%%), maintaining %d hits\n", emaProbability * 100.0f, g_consecutiveHits);
            }
            xSemaphoreGive(stateMutex);
        } else {
            LOG_WARN("stateMutex busy.\n");
            continue;
        }

        bool canTriggerNewAlarm = false;
        if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            if (!g_alarmActive &&
                (g_consecutiveHits >= N_CONSECUTIVE_HITS) &&
                (now - g_lastAlarmChangeMs > AI_MIN_RETRIGGER_MS)) {
                canTriggerNewAlarm = true;
                LOG_INFO("✅ Trigger conditions met (hits=%d)\n", g_consecutiveHits);
            }
            xSemaphoreGive(stateMutex);
        }

        if (canTriggerNewAlarm) {
            bool shouldFireSure = false;
            bool shouldFireUnsure = false;

            if (OPERATING_MODE == MODE_AI_SURE_ONLY) {
                shouldFireSure = (emaProbability >= PROB_THRESHOLD_STRONG);
            } else if (OPERATING_MODE == MODE_AI_UNSURE_ENABLED) {
                shouldFireSure = (emaProbability >= PROB_THRESHOLD_STRONG);
                shouldFireUnsure = (emaProbability >= PROB_THRESHOLD_WEAK && emaProbability < PROB_THRESHOLD_STRONG);
            }

            if (shouldFireSure || shouldFireUnsure) {
                LOG_INFO("🚨 Firing deterrent (EMA: %.1f%%)\n", emaProbability * 100.0f);

                if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
                    g_alarmActive = true;
                    g_lastAlarmChangeMs = now;
                    g_lastDeterrentTime = now;
                    g_consecutiveHits = 0;
                    xSemaphoreGive(stateMutex);
                } else {
                    LOG_WARN("stateMutex busy; skipped trigger.\n");
                    continue;
                }

                if (shouldFireSure) {
                    LOG_INFO("💥 SURE detection (%.1f%%)\n", emaProbability * 100.0f);
                    ledManager.setColor(255, 0, 0);
                    deterrentManager.signalSureDetection();
                } else {
                    LOG_INFO("⚠️ UNSURE detection (%.1f%%)\n", emaProbability * 100.0f);
                    ledManager.setColor(255, 165, 0);
                    deterrentManager.signalUnsureDetection();
                }
            }
        } else {
            bool alarmCopy = false;
            if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
                alarmCopy = g_alarmActive;
                xSemaphoreGive(stateMutex);
            }
            if (!alarmCopy) {
                ledManager.setColor(0, 255, 0);
                deterrentManager.deactivate();
            }
        }
    }

    esp_task_wdt_delete(NULL);
    vTaskDelete(NULL);
}

// =========================================================
// MAIN LOOP - WITH IMPROVED SLEEP LOGIC
// =========================================================
void loop() {
    esp_task_wdt_reset();
    deterrentManager.update();

    static unsigned long lastStats = 0;
    if ((LOG_LEVEL >= 3) && (millis() - lastStats > 10000)) {
        lastStats = millis();
        LOG_INFO("[SYS] Heap: %u bytes | Uptime: %lus\n", 
                 esp_get_free_heap_size(), millis() / 1000);
    }

    bool rawMotion = (digitalRead(LPIR) == HIGH) ||
                     (digitalRead(CPIR) == HIGH) ||
                     (digitalRead(RPIR) == HIGH);

    unsigned long now = millis();
    bool inDeterCooldown = false;
    unsigned long lastDetCopy = 0;
    unsigned long effectiveCooldown = (OPERATING_MODE == MODE_PIR_UNSURE_ONLY || OPERATING_MODE == MODE_PIR_SURE_ONLY) 
        ? PIR_ONLY_COOLDOWN_MS : DETER_COOLDOWN_MS;

    if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        lastDetCopy = g_lastDeterrentTime;
        if (g_lastDeterrentTime > 0) {
            inDeterCooldown = (now - g_lastDeterrentTime < effectiveCooldown);
        }
        xSemaphoreGive(stateMutex);
    }

    if (inDeterCooldown) {
        rawMotion = false;
        static unsigned long lastCoolLog = 0;
        if ((LOG_LEVEL >= 3) && (now - lastCoolLog > 2000)) {
            unsigned long remain = (effectiveCooldown - (now - lastDetCopy)) / 1000;
            LOG_INFO("❄️ Cooldown: %lus remaining\n", remain);
            lastCoolLog = now;
        }
    }

    static unsigned long pirHighStartTime = 0;
    static bool motionConfirmed = false;

    if (rawMotion) {
        if (pirHighStartTime == 0) {
            pirHighStartTime = millis();
            LOG_DEBUG("[PIR] Rising edge\n");
        } else if ((millis() - pirHighStartTime) > PIR_DEBOUNCE_MS) {
            if (!motionConfirmed) {
                LOG_INFO("[PIR] Motion confirmed\n");
                motionConfirmed = true;

                if (OPERATING_MODE == MODE_PIR_UNSURE_ONLY || OPERATING_MODE == MODE_PIR_SURE_ONLY) {
                    if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
                        if (!inDeterCooldown) {
                            LOG_INFO("🚨 PIR triggered deterrent\n");

                            if (OPERATING_MODE == MODE_PIR_SURE_ONLY) {
                                LOG_INFO("💥 Sending SURE signal\n");
                                ledManager.setColor(255, 0, 0);
                                deterrentManager.signalSureDetection();
                            } else {
                                LOG_INFO("⚠️ Sending UNSURE signal\n");
                                ledManager.setColor(255, 165, 0);
                                deterrentManager.signalUnsureDetection();
                            }

                            g_lastDeterrentTime = now;
                            g_alarmActive = true;
                        }
                        xSemaphoreGive(stateMutex);
                    }
                }
            }
        }
    } else {
        pirHighStartTime = 0;
        motionConfirmed = false;
    }

    static unsigned long lastMotionTime = millis();
    static unsigned long lastSampleTime = 0;
    static bool sensorsSuspended = false;

    if (motionConfirmed) {
        lastMotionTime = millis();
    }

    bool deterrentSignaling = deterrentManager.isSignaling();
    bool recentCooldown = (inDeterCooldown && (now - lastDetCopy) < 5000);
    
    bool stayAwake = (millis() - lastMotionTime < KEEP_ALIVE_MS) || 
                     deterrentSignaling || 
                     recentCooldown;

    if (stayAwake) {
        if (sensorsSuspended) {
            resumeAllTasks();
            sensorsSuspended = false;
            LOG_INFO("⚡ System awakened\n");
        }

        if (OPERATING_MODE == MODE_AI_SURE_ONLY || OPERATING_MODE == MODE_AI_UNSURE_ENABLED) {
            if (millis() - lastSampleTime > AI_SAMPLE_RATE_MS) {
                if (sensorTaskHandle) xTaskNotifyGive(sensorTaskHandle);
                lastSampleTime = millis();
                LOG_DEBUG(".");
            }
        }
    } else {
        LOG_INFO("\n💤 SLEEP SEQUENCE INITIATED\n");
        LOG_INFO("  Idle time: %lums | Cooldown: %s | Signaling: %s\n",
                 now - lastMotionTime, 
                 inDeterCooldown ? "YES" : "NO",
                 deterrentSignaling ? "YES" : "NO");

        ledManager.setColor(0, 0, 100);
        Serial.flush();

        esp_task_wdt_delete(NULL);

        if (deterrentSignaling) {
            LOG_INFO("  Waiting for deterrent to finish...\n");
            unsigned long sleepStart = millis();
            while (deterrentManager.isSignaling() && (millis() - sleepStart < 5000)) {
                deterrentManager.update();
                delay(10);
            }
        }

        suspendAllTasks();
        sensorsSuspended = true;

        fullSystemFlush();
        delay(100);

        LOG_INFO("  Entering light sleep (PIR active)...\n");
        Serial.flush();
        
        sleepManager.goToSleep(0);

        LOG_INFO("⏰ WOKE FROM SLEEP\n");

        esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true);
        esp_task_wdt_add(NULL);

        resumeAllTasks();
        sensorsSuspended = false;

        lastMotionTime = millis();
        lastSampleTime = millis();
        
        ledManager.setColor(0, 255, 0);
        LOG_INFO("  System resumed and ready\n");
    }

    delay(10);
}