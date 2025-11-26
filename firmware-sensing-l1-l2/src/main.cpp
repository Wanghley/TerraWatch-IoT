// ====== IMPORTS ======
#include <Arduino.h> 
#include "ArduinoJson.h" 
#include "freertos/FreeRTOS.h" 
#include "freertos/task.h" 
#include "freertos/semphr.h" 
#include <esp_task_wdt.h> 
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
#define LPIR 12 
#define CPIR 13
#define RPIR 14
#define DEBUG true
#define SENSOR_DEBUG true 
#define BRIGHTNESS 10
#define WDT_TIMEOUT_SECONDS 120 

// I2C & PIN DEFS
#define T0_SDA 48
#define T0_SCL 47
#define T1_SDA 8
#define T1_SCL 9
#define DETERRENT_PIN 11

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
ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, SENSOR_DEBUG);
mmWaveArrayManager mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, SENSOR_DEBUG);
MicManager micManager(0.2, SENSOR_DEBUG);
DeterrentManager deterrentManager(DETERRENT_PIN, DEBUG);

// The AI Engine
Predictor predictor; 

// Task Prototypes
void sensorCoreTask(void *p);
void uplinkCoreTask(void *p);

void setup() {
    Serial.begin(115200);
    delay(1000); // Give serial monitor time to catch up

    // 1. Init Watchdog
    esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true); 
    esp_task_wdt_add(NULL); 

    if(DEBUG) LOG_PRINTLN("\n--- Booting TerraWatch AI 2.0 ---");

    // 2. INITIALIZE AI FIRST (Priority Allocation)
    // We do this before ANY other sensors to guarantee we get the RAM we need.
    LOG_PRINTLN("[System] Allocating AI Memory...");
    if (!predictor.begin()) {
        LOG_PRINTLN("❌ AI Failed. Halting.");
        while(1) { ledManager.setColor(255, 0, 0); delay(100); ledManager.setColor(0,0,0); delay(100); }
    }
    LOG_PRINTLN("✅ AI Ready.");

    // 3. Initialize Sleep & Hardware
    sleepManager.configure();
    ledManager.begin();
    ledManager.setColor(100, 100, 0); // Yellow

    // Thermal
    bool thermalOk = thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL);
    if (!thermalOk) {
        LOG_PRINTLN("[thermal] WARNING: thermal sensors unavailable. Using zeros.");
    }

    mmWaveManager.begin();
    micManager.begin();
    deterrentManager.begin();
    deterrentManager.enablePersistent(true); // enable latch
    // deterrentManager.signalSureDetection(); // stays ON until deactivate()

    // 4. Create Tasks
    packetQueue = xQueueCreate(1, sizeof(SensorPacket));
    
    // Use slightly less stack for AI now that the Arena is on the Heap
    xTaskCreatePinnedToCore(sensorCoreTask, "sensors", 8192, nullptr, 2, &sensorTaskHandle, 0);
    xTaskCreatePinnedToCore(uplinkCoreTask, "ai_logic", 8192, nullptr, 1, &uplinkTaskHandle, 1);
    
    LOG_PRINTLN("✅ System Ready.");
    ledManager.setColor(0, 100, 0);
}

void sensorCoreTask(void *p)
{
    SensorPacket pkt;
    double micL_temp, micR_temp; 

    for (;;)
    {
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
        
        RadarData r2 = mmWaveManager.getRadar2();
        pkt.r2.range_cm = r2.range_cm;
        pkt.r2.speed_ms = r2.speed_ms;
        pkt.r2.energy = r2.energy;
        pkt.r2.numTargets = r2.numTargets;

        // --- 3. Mic Read ---
        micManager.read(micL_temp, micR_temp);
        // NOTE: Ensure these names match shared_types.h
        pkt.micL = (float)micL_temp; 
        pkt.micR = (float)micR_temp;

        pkt.timestamp = millis();

        // Send to Queue
        xQueueOverwrite(packetQueue, &pkt);
    }
}

void uplinkCoreTask(void *p)
{
    SensorPacket pkt;
    // REMOVED: int warmupCount = 0; 

    for (;;)
    {
        // Wait for data
        if (xQueueReceive(packetQueue, &pkt, portMAX_DELAY) != pdTRUE)
            continue;

        // --- INSTANT PREDICTION ---
        // update() now returns valid probability immediately (no -1.0 return)
        float probability = predictor.update(pkt);

        // --- LOGIC ---
        // Only print every 500ms to avoid flooding Serial Monitor
        static unsigned long lastLog = 0;
        if(millis() - lastLog > 500) {
            LOG_PRINTF("Prob: %.2f\n", probability);
            lastLog = millis();
        }

        if (probability > 0.6f) // Slightly higher threshold for safety
        {
            ledManager.setColor(255, 0, 0); // RED
            
            // Strong detection
            if(probability > 0.85f) {
                LOG_PRINTLN("🚨 CONFIRMED THREAT");
                deterrentManager.signalSureDetection();
            } 
            // Weak detection
            else {
                deterrentManager.signalUnsureDetection();
            }
        }
        else
        {
            ledManager.setColor(0, 255, 0); // GREEN
            // LOG_PRINTLN("✅ No Threat Detected, sending unsure signal");
            // deterrentManager.signalUnsureDetection();
            deterrentManager.deactivate();
        }
    }
}

void loop()
{
    esp_task_wdt_reset();
    deterrentManager.update();

    static unsigned long lastStats = 0;
    if (DEBUG && millis() - lastStats > 10000) {
        lastStats = millis();
        LOG_PRINTF("[SYS] Heap: %u\n", esp_get_free_heap_size());
    }

    // PIR Logic
    bool motionDetected = (digitalRead(LPIR) == HIGH) || 
                          (digitalRead(CPIR) == HIGH) || 
                          (digitalRead(RPIR) == HIGH);

    if (motionDetected)
    {
        if (sensorTaskHandle) xTaskNotifyGive(sensorTaskHandle);
        delay(20);  // ~50Hz sampling
    }
    else
    {
        ledManager.setColor(0, 0, 100); 
        if (DEBUG) {
            LOG_PRINTLN("💤 No motion. Sleeping...");
            Serial.flush();
        }
        
        esp_task_wdt_delete(NULL);
        while(deterrentManager.isSignaling()) {
            deterrentManager.update();
        }
        sleepManager.goToSleep(); 
        
        // WAKE UP
        esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true);
        esp_task_wdt_add(NULL);
        ledManager.setColor(0, 100, 0); 
        if (DEBUG) LOG_PRINTLN("⏰ Woke up!");
    }
}