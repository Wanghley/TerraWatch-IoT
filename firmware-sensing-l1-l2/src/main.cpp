// ====== IMPORTS ======
#include <Arduino.h> // Core Arduino library
#include "ArduinoJson.h" // JSON handling library
#include "freertos/FreeRTOS.h" // FreeRTOS core
#include "freertos/task.h" // FreeRTOS task management
#include "freertos/semphr.h" // FreeRTOS semaphores
#include <esp_task_wdt.h> // ESP32 Watchdog for system reliability
#include <string.h> // For memcpy

#include "shared_types.h" // Shared types between tasks
#include "predictor.h" // AI Predictor

#include "sleep_manager.h" // power management and awake/sleep cycles
#include "led_manager.h" // LED Manager for status indication
#include "thermal_array_manager.h" // Thermal Array Manager
#include "mmWave_array_manager.h" // mmWave Array Manager
#include "mic_manager.h" // Microphone Manager
#include "deterrent_manager.h" // Deterrent Manager

// ====== USER CONFIG ======
#define LPIR 12 // left Passive Infrared Sensor (PIR)
#define CPIR 13
#define RPIR 14
#define DEBUG true
#define BRIGHTNESS 50
#define kPlaceholderSignalWindowMs 60
#define WDT_TIMEOUT_SECONDS 15 

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

#define T0_SDA 48
#define T0_SCL 47
#define T1_SDA 8
#define T1_SCL 9
#define DETERRENT_PIN 36

#define RADAR1_RX 16
#define RADAR1_TX 10
#define RADAR2_RX 17
#define RADAR2_TX 18

// --- GLOBALS ---
QueueHandle_t packetQueue = nullptr;
TaskHandle_t sensorTaskHandle = nullptr;
TaskHandle_t uplinkTaskHandle = nullptr;

SleepManager sleepManager(LPIR, CPIR, RPIR, DEBUG);
LedManager ledManager(LED_BUILTIN, BRIGHTNESS);
ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, DEBUG);
mmWaveArrayManager mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, DEBUG);
MicManager micManager(0.2, DEBUG);
DeterrentManager deterrentManager(DETERRENT_PIN, DEBUG);

Predictor predictor;

void sensorCoreTask(void *);
void uplinkCoreTask(void *);

void setup()
{
    // 1. Initialize Serial
    Serial.begin(115200);
    unsigned long start = millis();
    while (!Serial && (millis() - start < 3000)) delay(100);

    // 2. Initialize Watchdog
    esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true); // Panic (reset) if WDT triggers
    esp_task_wdt_add(NULL); // Add current thread (loopTask) to WDT

    if(DEBUG){
        LOG_PRINTLN("\n--- TerraWatch Agronauts AI L1-L2 Firmware ---");
        LOG_PRINTLN("Firmware Version: 1.0.0");
        LOG_PRINTLN("Build Date: " __DATE__ " " __TIME__);
        LOG_PRINTLN("Author: @wanghley");
        LOG_PRINTLN("-----------------------------------------------\n");
    }

    // 3. Initialize Hardware
    ledManager.begin();
    ledManager.setColor(100, 100, 0); // Yellow = Booting
    sleepManager.configure();

    thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL);
    delay(250);
    mmWaveManager.begin();
    micManager.begin();
    deterrentManager.begin();

    // 4. Initialize AI
    if (!predictor.begin())
    {
        LOG_PRINTLN("❌ AI Model Load Failed! System will restart in 5 seconds...");
        ledManager.setColor(255, 0, 0); // Red = Error
        delay(5000);
        ESP.restart(); // Restart instead of hanging
    }
    LOG_PRINTLN("✅ AI Model Loaded.");

    // 5. Create Tasks
    packetQueue = xQueueCreate(1, sizeof(SensorPacket));
    
    // Note: Stack sizes increased for safety. Monitor with uxTaskGetStackHighWaterMark
    xTaskCreatePinnedToCore(sensorCoreTask, "sensors", 8192, nullptr, 2, &sensorTaskHandle, 0);
    xTaskCreatePinnedToCore(uplinkCoreTask, "uplink", 8192, nullptr, 1, &uplinkTaskHandle, 1);
    
    LOG_PRINTLN("✅ System Ready.");
    ledManager.setColor(0, 100, 0); // Green = Ready
}

void sensorCoreTask(void *p)
{
    SensorPacket pkt;
    double micL_temp, micR_temp; // Temp variables for double& reference

    for (;;)
    {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // 1. Thermal
        thermalManager.readRotated();
        ThermalReadings thermalData = thermalManager.getObject();
        memcpy(pkt.thermal_left, thermalData.left, sizeof(float) * 64);
        memcpy(pkt.thermal_center, thermalData.center, sizeof(float) * 64);
        memcpy(pkt.thermal_right, thermalData.right, sizeof(float) * 64);

        // 2. Radar (Map RadarData to RadarReading)
        mmWaveManager.update();
        RadarData r1 = mmWaveManager.getRadar1();
        pkt.r1.range_cm = r1.range_cm;
        pkt.r1.speed_ms = r1.speed_ms;
        pkt.r1.energy = r1.energy;
        pkt.r1.lastDetection = r1.lastDetection;
        pkt.r1.isValid = r1.isValid;

        RadarData r2 = mmWaveManager.getRadar2();
        pkt.r2.range_cm = r2.range_cm;
        pkt.r2.speed_ms = r2.speed_ms;
        pkt.r2.energy = r2.energy;
        pkt.r2.lastDetection = r2.lastDetection;
        pkt.r2.isValid = r2.isValid;

        // 3. Mic (Handle double to float conversion)
        micManager.read(micL_temp, micR_temp);
        pkt.micL = (float)micL_temp;
        pkt.micR = (float)micR_temp;

        xQueueOverwrite(packetQueue, &pkt);
    }
}

void uplinkCoreTask(void *p)
{
    SensorPacket pkt;
    for (;;)
    {
        if (xQueueReceive(packetQueue, &pkt, portMAX_DELAY) != pdTRUE)
            continue;

        // --- PREDICTION PIPELINE ---
        float probability = predictor.update(pkt);

        if (probability >= 0.0)
        {
            LOG_PRINTF("\n>>> PREDICTION: %.1f%% <<<\n", probability * 100.0);

            if (probability > 0.5)
            {
                LOG_PRINTLN("🚨 EVENT DETECTED 🚨");
                ledManager.setColor(255, 0, 0); // RED
                // deterrentManager.activate();
            }
            else
            {
                LOG_PRINTLN("... All Clear ...");
                ledManager.setColor(0, 255, 0); // GREEN
            }
        }
        else
        {
            // Optional: Print dot to show buffering, or comment out to reduce noise
            // LOG_PRINT(".");
        }
    }
}

void loop()
{
    // 1. Reset Watchdog
    esp_task_wdt_reset();

    // 2. Trigger sensor reading
    if (sensorTaskHandle)
    {
        xTaskNotifyGive(sensorTaskHandle);
    }

    // 3. Update Deterrent
    deterrentManager.update();

    // 4. System Health Check (Every 10 seconds)
    static unsigned long lastStats = 0;
    if (DEBUG && millis() - lastStats > 10000) {
        lastStats = millis();
        LOG_PRINTF("[SYS] Free Heap: %u bytes (Min: %u)\n", esp_get_free_heap_size(), esp_get_minimum_free_heap_size());
        if (sensorTaskHandle) LOG_PRINTF("[TASK] Sensor Stack High Water Mark: %u\n", uxTaskGetStackHighWaterMark(sensorTaskHandle));
        if (uplinkTaskHandle) LOG_PRINTF("[TASK] Uplink Stack High Water Mark: %u\n", uxTaskGetStackHighWaterMark(uplinkTaskHandle));
    }

    // Run at ~4Hz
    delay(250);
}