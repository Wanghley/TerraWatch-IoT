#include <Arduino.h>

// --- Added Missing Headers ---
#include "ArduinoJson.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h> // <-- ADDED for memcpy
// -----------------------------

#include "sleep_manager.h"
#include "led_manager.h"
#include "thermal_array_manager.h"
#include "mmWave_array_manager.h"
#include "mic_manager.h"
#include "esp_wifi.h"
#include "esp_bt.h"
#include "esp_bt_main.h"

// ====== USER CONFIG ======
#define LPIR 12
#define CPIR 13
#define RPIR 14

#define DEBUG false

#define BRIGHTNESS 50  // RGB LED brightness (0-255)

// --- Removed unused WiFi/UDP vars ---
// const char* WIFI_SSID     = "ECE449deco";
// const char* WIFI_PASSWORD = "ece449$$";
// const char* TARGET_ID = "GROUP2_DETER_ESP";
// unsigned int UDP_PORT = 4210;

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

// Manager objects
SleepManager sleepManager(LPIR, CPIR, RPIR, DEBUG);
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
    Serial.begin(115200);

    // --- FIX: Wait for Serial on S3 native USB ---
    // Give 3 seconds to connect the Serial monitor
    unsigned long start = millis();
    while (!Serial && (millis() - start < 3000)) {
        delay(100);
    }
    // --------------------------------------------

    delay(100);
    if (DEBUG) {
        Serial.println("\n--- TerraWatch Agronauts L1/L2 Sensing Firmware ---");
        Serial.flush(); // Force print before next step
    }

    // Init LED
    ledManager.begin();
    ledManager.setColor(100, 100, 0); // Yellow = setup start

    // delay for stability
    delay(100);

    // Configure sleep
    if (DEBUG) {
        Serial.println("Configuring sleep...");
        Serial.flush();
    }
    sleepManager.configure();
    if (DEBUG) {
        Serial.println("Sleep configured.");
        Serial.flush();
    }

    // Init sensors
    if (DEBUG) {
        Serial.println("Initializing thermal sensors...");
        Serial.flush();
    }
    if (!thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL)) {
        if (DEBUG) {
            Serial.println("⚠️ Thermal sensors failed!");
            Serial.flush();
        }
    } else {
        if (DEBUG) {
            Serial.println("Thermal sensors initialized.");
            Serial.flush();
        }
    }

    delay(250); // Stagger init

    if (DEBUG) {
        Serial.println("Initializing mmWave radars...");
        Serial.flush();
    }
    if (!mmWaveManager.begin()) {
        if (DEBUG) {
            Serial.println("⚠️ mmWave radars failed!");
            Serial.flush();
        }
    } else {
        if (DEBUG) {
            Serial.println("mmWave radars initialized.");
            Serial.flush();
        }
    }

    delay(250); // Stagger init

    if (DEBUG) {
        Serial.println("Initializing mic manager...");
        Serial.flush();
    }
    micManager.begin();
    if (DEBUG) {
        Serial.println("Mic manager initialized.");
        Serial.flush();
    }
    
    // Create the queue and tasks
    if (DEBUG) {
        Serial.println("Creating FreeRTOS tasks...");
        Serial.flush();
    }
    packetQueue = xQueueCreate(1, sizeof(SensorPacket));
    xTaskCreatePinnedToCore(sensorCoreTask, "sensors", 8192, nullptr, 2, &sensorTaskHandle, 0); // Core 0, Prio 2
    xTaskCreatePinnedToCore(uplinkCoreTask, "uplink", 6144, nullptr, 1, &uplinkTaskHandle, 1); // Core 1, Prio 1
    if (DEBUG) {
        Serial.println("Setup complete. Handing over to tasks.");
        Serial.flush();
    }
}

void sensorCoreTask(void* p) {
    SensorPacket pkt;
    for (;;) {
        // Wait for the notification from loop()
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        // --- Sensor Read Pipeline (Core 0) ---
        thermalManager.readRotated();
        ThermalReadings thermalData = thermalManager.getObject(); // Get object with pointers

        // Assumes thermal arrays are 64 elements (8x8)
        memcpy(pkt.thermal_left, thermalData.left, sizeof(float) * 64);
        memcpy(pkt.thermal_center, thermalData.center, sizeof(float) * 64);
        memcpy(pkt.thermal_right, thermalData.right, sizeof(float) * 64);
        
        mmWaveManager.update();
        pkt.r1 = mmWaveManager.getRadar1();
        pkt.r2 = mmWaveManager.getRadar2();
        
        micManager.read(pkt.micL, pkt.micR);
        
        // Send to Core 1 (overwrites if Core 1 is still busy)
        xQueueOverwrite(packetQueue, &pkt);
    }
}

void uplinkCoreTask(void* p) {
    SensorPacket pkt;
    JsonDocument doc;
    for (;;) {
        // Wait for a packet from Core 0
        if (xQueueReceive(packetQueue, &pkt, portMAX_DELAY) != pdTRUE) {
            continue; // Should never happen with portMAX_DELAY
        }
        
        // --- JSON & Uplink Pipeline (Core 1)
        doc.clear();

        JsonArray thermal_left = doc["thermal"].to<JsonObject>()["left"].to<JsonArray>();
        for (int i = 0; i < 64; i++) {
            thermal_left.add(pkt.thermal_left[i]);
        }
        
        JsonArray thermal_center = doc["thermal"]["center"].to<JsonArray>();
        for (int i = 0; i < 64; i++) {
            thermal_center.add(pkt.thermal_center[i]);
        }

        JsonArray thermal_right = doc["thermal"]["right"].to<JsonArray>();
        for (int i = 0; i < 64; i++) {
            thermal_right.add(pkt.thermal_right[i]);
        }

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
    }
}

void loop() {
    // Sleep Management (Core 0)
    ledManager.setColor(0, 0, 100); // Blue = sleeping
    sleepManager.goToSleep();       // Block here until PIR interrupt
    
    // --- WAKE UP ---
    ledManager.setColor(0, 100, 0); // Green = awake
    
    // Notify the sensor task (also on Core 0) to start reading
    if (sensorTaskHandle) {
        xTaskNotifyGive(sensorTaskHandle);
    }
}