// ====== IMPORTS ======
#include <Arduino.h> 
#include "ArduinoJson.h" 
#include "freertos/FreeRTOS.h" 
#include "freertos/task.h" 
#include "freertos/semphr.h" 
#include <esp_task_wdt.h> 
#include <string.h> 

// --- KEY IMPORTS ---
#include "shared_types.h"       // <--- STEP 1: The struct definition
#include "predictor.h"          // <--- STEP 2: The AI Engine

#include "sleep_manager.h" 
#include "led_manager.h" 
#include "thermal_array_manager.h" 
#include "mmWave_array_manager.h" 
#include "mic_manager.h" 
#include "deterrent_manager.h" 

// ====== USER CONFIG ======
#define LPIR 12 
#define CPIR 13
#define RPIR 14
#define DEBUG true
#define BRIGHTNESS 50
#define WDT_TIMEOUT_SECONDS 120 

// I2C & PIN DEFS
#define T0_SDA 48
#define T0_SCL 47
#define T1_SDA 8
#define T1_SCL 9
#define DETERRENT_PIN 36

#define RADAR1_RX 10
#define RADAR1_TX 16
#define RADAR2_RX 18
#define RADAR2_TX 17

// ====== LOGGING MACROS ======
#if DEBUG
    #define LOG_PRINT(x) Serial.print(x)
    #define LOG_PRINTLN(x) Serial.println(x)
    #define LOG_PRINTF(...) Serial.printf(__VA_ARGS__)
#else
    #define LOG_PRINT(x)
    #define LOG_PRINTLN(x)
    #define LOG_PRINTF(...)
#endif

// --- GLOBALS ---
QueueHandle_t packetQueue = nullptr;
TaskHandle_t sensorTaskHandle = nullptr;
TaskHandle_t uplinkTaskHandle = nullptr;

// Managers
SleepManager sleepManager(LPIR, CPIR, RPIR, DEBUG);
LedManager ledManager(LED_BUILTIN, BRIGHTNESS);
ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, DEBUG);
mmWaveArrayManager mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, DEBUG);
MicManager micManager(0.2, DEBUG);
DeterrentManager deterrentManager(DETERRENT_PIN, DEBUG);

// CRITICAL: This MUST be a global variable. 
// If placed inside setup(), the 80KB Tensor Arena will overflow the 8KB stack and crash.
Predictor predictor; 

// Task Prototypes
void sensorCoreTask(void *p);
void uplinkCoreTask(void *p);

void setup()
{
    // 1. Initialize Serial
    Serial.begin(115200);
    delay(1000); 

    // 2. Initialize Watchdog
    esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true); 
    esp_task_wdt_add(NULL); 

    if(DEBUG){
        LOG_PRINTLN("\n--- TerraWatch Agronauts AI L1-L2 Firmware ---");
        LOG_PRINTLN("Firmware Version: 1.0.0");
        LOG_PRINTLN("Build Date: " __DATE__ " " __TIME__);
        LOG_PRINTLN("Author: @wanghley");
        LOG_PRINTLN("-----------------------------------------------\n"); 
    }

    // 3. Initialize Sleep Manager (BEFORE creating tasks)
    sleepManager.configure();
    
    // 4. Initialize Hardware
    ledManager.begin();
    ledManager.setColor(100, 100, 0); // Yellow = Booting

    thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL);
    mmWaveManager.begin();
    micManager.begin();
    deterrentManager.begin();

    // 5. Initialize AI
    // This loads the model and allocates the Tensor Arena
    LOG_PRINTLN("[AI] Initializing TensorFlow Lite Micro...");
    if (!predictor.begin())
    {
        LOG_PRINTLN("❌ AI Model Load Failed! Restarting...");
        ledManager.setColor(255, 0, 0); 
        delay(5000);
        ESP.restart(); 
    }
    LOG_PRINTLN("✅ AI Model Loaded.");

    // 6. Create Tasks
    packetQueue = xQueueCreate(1, sizeof(SensorPacket));
    
    // Core 0: Sensor Reading (High Priority)
    xTaskCreatePinnedToCore(sensorCoreTask, "sensors", 8192, nullptr, 2, &sensorTaskHandle, 0);
    
    // Core 1: AI Inference (Lower Priority, Heavy Computation)
    // NOTE: Stack size increased to 12000 for TFLite
    xTaskCreatePinnedToCore(uplinkCoreTask, "ai_logic", 12000, nullptr, 1, &uplinkTaskHandle, 1);
    
    LOG_PRINTLN("✅ System Ready.");
    ledManager.setColor(0, 100, 0); // Green = Ready
}

void sensorCoreTask(void *p)
{
    SensorPacket pkt;
    double micL_temp, micR_temp; 

    for (;;)
    {
        // Wait for notification from loop()
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // --- 1. Thermal Read ---
        thermalManager.readRotated();
        ThermalReadings thermalData = thermalManager.getObject();
        memcpy(pkt.thermal_left, thermalData.left, sizeof(float) * 64);
        memcpy(pkt.thermal_center, thermalData.center, sizeof(float) * 64);
        memcpy(pkt.thermal_right, thermalData.right, sizeof(float) * 64);

        // --- 2. Radar Read ---
        mmWaveManager.update();
        RadarData r1 = mmWaveManager.getRadar1();
        pkt.r1.range_cm = r1.range_cm;
        pkt.r1.speed_ms = r1.speed_ms;
        pkt.r1.energy = r1.energy;
        pkt.r1.numTargets = r1.numTargets; 
        pkt.r1.isValid = r1.isValid;

        RadarData r2 = mmWaveManager.getRadar2();
        pkt.r2.range_cm = r2.range_cm;
        pkt.r2.speed_ms = r2.speed_ms;
        pkt.r2.energy = r2.energy;
        pkt.r2.numTargets = r2.numTargets;
        pkt.r2.isValid = r2.isValid;

        // --- 3. Mic Read ---
        micManager.read(micL_temp, micR_temp);
        pkt.micL = (float)micL_temp;
        pkt.micR = (float)micR_temp;

        pkt.timestamp = millis();

        // Send to Queue (Overwrite if full to keep latest data)
        xQueueOverwrite(packetQueue, &pkt);
    }
}

void uplinkCoreTask(void *p)
{
    SensorPacket pkt;
    for (;;)
    {
        // Wait for data from Sensor Task
        if (xQueueReceive(packetQueue, &pkt, portMAX_DELAY) != pdTRUE)
            continue;

        // --- PREDICTION PIPELINE ---
        // update() adds data to history and runs TFLite inference
        float probability = predictor.update(pkt);

        // Check if Warmup is complete (Predictor returns -1.0 during warmup)
        if (probability >= 0.0f)
        {
            // Optional: Only print every N ms or on change to reduce serial spam
            LOG_PRINTF("Prob: %.2f\n", probability);

            if (probability > 0.5f) // Threshold
            {
                LOG_PRINTF("🚨 THREAT DETECTED (%.1f%%)\n", probability * 100.0);
                ledManager.setColor(255, 0, 0); // RED
                if(probability > 0.8f) {
                    deterrentManager.signalSureDetection();
                } else {
                    deterrentManager.signalUnsureDetection();
                }
            }
            else
            {
                // LOG_PRINTLN("... All Clear ...");
                ledManager.setColor(0, 255, 0); // GREEN
                deterrentManager.deactivate(); // Ensure deterrent is off
            }
        }
        else
        {
            // Still filling the 198-frame buffer
            static unsigned long lastPrint = 0;
            if(millis() - lastPrint > 1000) {
                LOG_PRINTLN("⏳ AI Warming Up (Buffer Filling)...");
                lastPrint = millis();
            }
        }
    }
}

void loop()
{
    // 1. Reset Watchdog
    esp_task_wdt_reset();

    // 2. Update Deterrent State Machine (Non-blocking)
    deterrentManager.update();

    // 3. Debug Stats (Every 10s)
    static unsigned long lastStats = 0;
    if (DEBUG && millis() - lastStats > 10000) {
        lastStats = millis();
        LOG_PRINTF("[SYS] Heap: %u | Min Heap: %u\n", esp_get_free_heap_size(), esp_get_minimum_free_heap_size());
        if (uplinkTaskHandle) LOG_PRINTF("[AI TASK] Stack High Mark: %u\n", uxTaskGetStackHighWaterMark(uplinkTaskHandle));
    }

    // 4. SLEEP LOGIC: Only trigger sensors if PIR detected motion
    // Check if any PIR is HIGH (motion detected)
    bool motionDetected = (digitalRead(LPIR) == HIGH) || 
                          (digitalRead(CPIR) == HIGH) || 
                          (digitalRead(RPIR) == HIGH);

    if (motionDetected)
    {
        // Motion detected - trigger sensor reading
        if (sensorTaskHandle)
        {
            xTaskNotifyGive(sensorTaskHandle);
        }
        delay(20);  // ~50Hz sampling
    }
    else
    {
        // No motion - prepare for sleep
        ledManager.setColor(0, 0, 100);  // Blue = sleeping
        
        if (DEBUG) {
            LOG_PRINTLN("💤 No motion detected. Going to sleep...");
            Serial.flush();
        }
        
        // CRITICAL: Disable watchdog before sleeping, as light sleep blocks loop()
        esp_task_wdt_delete(NULL);
        
        sleepManager.goToSleep();  // This will block until PIR wakes us
        
        // WAKE UP - PIR triggered
        // Re-enable watchdog after waking
        esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true);
        esp_task_wdt_add(NULL);
        
        ledManager.setColor(0, 100, 0);  // Green = awake
        if (DEBUG) {
            LOG_PRINTLN("⏰ Woke up! PIR triggered.");
            Serial.flush();
        }
    }
}