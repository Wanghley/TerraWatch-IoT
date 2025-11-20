#include <Arduino.h>
#include "ArduinoJson.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h> // For memcpy

// Note: sleep_manager.h is NOT included
#include "led_manager.h"
#include "thermal_array_manager.h"
#include "mmWave_array_manager.h"
#include "mic_manager.h"
#include "esp_wifi.h"
#include "esp_bt.h"
#include "esp_bt_main.h"

// ====== USER CONFIG ======
// --- Sleep/PIR pins removed ---

#define DEBUG false // Keep debug off for max speed
#define BRIGHTNESS 50

// Thermal I2C pins
#define T0_SDA 48
#define T0_SCL 47
#define T1_SDA 8
#define T1_SCL 9

// mmWave pins
#define RADAR1_RX 16 // LEFT
#define RADAR1_TX 10 // LEFT
#define RADAR2_RX 17 // RIGHT
#define RADAR2_TX 18 // RIGHT
// ==========================

struct SensorPacket {
    float thermal_left[64];
    float thermal_center[64];
    float thermal_right[64];
    RadarData r1;
    RadarData r2;
    double micL;
    double micR;
};

void sensorCoreTask(void*);
void uplinkCoreTask(void*);

QueueHandle_t packetQueue = nullptr;
TaskHandle_t sensorTaskHandle = nullptr;
TaskHandle_t uplinkTaskHandle = nullptr;

// Semaphore for sync
SemaphoreHandle_t uplinkDoneSemaphore = nullptr;

// Manager objects
// Note: SleepManager is NOT created
LedManager ledManager(LED_BUILTIN, BRIGHTNESS);
ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, DEBUG);
mmWaveArrayManager mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, DEBUG);
MicManager micManager(0.2, DEBUG);

void disableRadios() {
    esp_bt_controller_mem_release(ESP_BT_MODE_BTDM);
    esp_bt_controller_disable();
    esp_wifi_stop();
    esp_wifi_deinit();
}

void setup() {
    disableRadios();
    delay(100);
    Serial.begin(921600); // High-speed baud rate

    // Wait for Serial
    unsigned long start = millis();
    while (!Serial && (millis() - start < 3000)) {
        delay(100);
    }
    
    ledManager.begin();
    ledManager.setColor(100, 0, 100); // Purple = Booting data collector

    // --- Stagger sensor init ---
    if (!thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL)) {
         ledManager.setColor(100, 0, 0); // Red = Error
         while(1) delay(100);
    }
    delay(250); 
    if (!mmWaveManager.begin()) {
         ledManager.setColor(100, 0, 0); // Red = Error
         while(1) delay(100);
    }
    delay(250); 
    micManager.begin();
    
    // Create the queue and tasks
    packetQueue = xQueueCreate(1, sizeof(SensorPacket));
    uplinkDoneSemaphore = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(sensorCoreTask, "sensors", 8192, nullptr, 2, &sensorTaskHandle, 0); // Core 0, Prio 2
    xTaskCreatePinnedToCore(uplinkCoreTask, "uplink", 6144, nullptr, 1, &uplinkTaskHandle, 1); // Core 1, Prio 1
    
    ledManager.setColor(0, 0, 100); // Blue = Ready to sample
}

void sensorCoreTask(void* p) {
    SensorPacket pkt;
    for (;;) {
        // Wait for the notification from loop()
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        // --- Sensor Read Pipeline (Core 0) ---
        thermalManager.readRotated();
        ThermalReadings thermalData = thermalManager.getObject(); 
        memcpy(pkt.thermal_left, thermalData.left, sizeof(float) * 64);
        memcpy(pkt.thermal_center, thermalData.center, sizeof(float) * 64);
        memcpy(pkt.thermal_right, thermalData.right, sizeof(float) * 64);
        
        mmWaveManager.update();
        pkt.r1 = mmWaveManager.getRadar1();
        pkt.r2 = mmWaveManager.getRadar2();
        
        micManager.read(pkt.micL, pkt.micR);
        
        // Send to Core 1
        xQueueOverwrite(packetQueue, &pkt);
    }
}

void uplinkCoreTask(void* p) {
    SensorPacket pkt;
    JsonDocument doc; 
    for (;;) {
        // Wait for a packet from Core 0
        if (xQueueReceive(packetQueue, &pkt, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        
        // --- JSON & Uplink Pipeline (Core 1) ---
        doc.clear();

        JsonArray thermal_left = doc["thermal"].to<JsonObject>()["left"].to<JsonArray>();
        for (int i = 0; i < 64; i++) thermal_left.add(pkt.thermal_left[i]);
        
        JsonArray thermal_center = doc["thermal"]["center"].to<JsonArray>();
        for (int i = 0; i < 64; i++) thermal_center.add(pkt.thermal_center[i]);

        JsonArray thermal_right = doc["thermal"]["right"].to<JsonArray>();
        for (int i = 0; i < 64; i++) thermal_right.add(pkt.thermal_right[i]);
        
        doc["radar"]["left"]["range"] = pkt.r1.range_cm;
        doc["radar"]["left"]["speed"] = pkt.r1.speed_ms;
        doc["radar"]["left"]["energy"] = pkt.r1.energy;
        doc["radar"]["left"]["lastDetection"] = pkt.r1.lastDetection;
        doc["radar"]["left"]["isValid"] = pkt.r1.isValid;
        doc["radar"]["right"]["range"] = pkt.r2.range_cm;
        doc["radar"]["right"]["speed"] = pkt.r2.speed_ms;
        doc["radar"]["right"]["energy"] = pkt.r2.energy;
        doc["radar"]["right"]["lastDetection"] = pkt.r2.lastDetection;
        doc["radar"]["right"]["isValid"] = pkt.r2.isValid;
        doc["mic"]["left"] = pkt.micL;
        doc["mic"]["right"] = pkt.micR;
        
        serializeJson(doc, Serial);
        Serial.println();
        Serial.flush(); // Force send the data

        // Signal Core 0 that printing is done
        xSemaphoreGive(uplinkDoneSemaphore);
    }
}

void loop() {
    // --- Continuous Sampling Loop (Core 0) ---
    
    // Notify the sensor task to start reading
    if (sensorTaskHandle) {
        xTaskNotifyGive(sensorTaskHandle);
    }
    
    // Wait for Core 1 to signal that it has finished printing.
    xSemaphoreTake(uplinkDoneSemaphore, portMAX_DELAY);

    // loop() will now finish and immediately run again
}