#include <Arduino.h>
#include "ArduinoJson.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include <string.h>

#include "sleep_manager.h"
#include "led_manager.h"
#include "thermal_array_manager.h"
#include "mmWave_array_manager.h"
#include "mic_manager.h"
#include "esp_wifi.h"
#include "esp_bt.h"
#include "esp_bt_main.h"
#include "deterrent_manager.h"

// ====== USER CONFIG ======
#define LPIR 12
#define CPIR 13
#define RPIR 14
#define DETERRENT_PIN 36

#define DEBUG true
#define BRIGHTNESS 50
#define kPlaceholderSignalWindowMs 60

// ====== THERMAL SENSOR PINS ======
#define T0_SDA 48
#define T0_SCL 47
#define T1_SDA 8
#define T1_SCL 9

// ====== MMWAVE RADAR PINS ======
#define RADAR1_RX 10
#define RADAR1_TX 16
#define RADAR2_RX 18
#define RADAR2_TX 17

// ====== DATA STRUCTURES ======
struct RadarReading {
    float range_cm;
    float speed_ms;
    float energy;
    double lastDetection;
    bool isValid;
};

struct SensorPacket {
    float thermal_left[64];
    float thermal_center[64];
    float thermal_right[64];
    RadarReading r1;
    RadarReading r2;
    double micL;
    double micR;
};

// ====== GLOBAL VARIABLES ======
QueueHandle_t packetQueue = nullptr;
TaskHandle_t sensorTaskHandle = nullptr;
TaskHandle_t uplinkTaskHandle = nullptr;

// Manager objects
SleepManager sleepManager(LPIR, CPIR, RPIR, DEBUG);
LedManager ledManager(LED_BUILTIN, BRIGHTNESS);
ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, DEBUG);
mmWaveArrayManager mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, DEBUG);
MicManager micManager(0.2, DEBUG);
DeterrentManager deterrentManager(DETERRENT_PIN, DEBUG);

// ====== FORWARD DECLARATIONS ======
void sensorCoreTask(void* p);
void uplinkCoreTask(void* p);

void disableRadios() {
    esp_bt_controller_mem_release(ESP_BT_MODE_BTDM);
    esp_wifi_stop();
    esp_wifi_deinit();
}

void setup() {
    disableRadios();
    delay(100);
    Serial.begin(115200);

    // Wait for Serial on native USB
    unsigned long start = millis();
    while (!Serial && (millis() - start < 3000)) {
        delay(100);
    }

    delay(100);
    if (DEBUG) {
        Serial.println("\n--- TerraWatch Agronauts L1/L2 Sensing Firmware ---");
        Serial.flush();
    }

    ledManager.begin();
    ledManager.setColor(100, 100, 0); // Yellow = setup start
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

    // Initialize thermal sensors
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
            Serial.println("✓ Thermal sensors initialized.");
            Serial.flush();
        }
    }

    delay(250);

    // Initialize mmWave radars
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
            Serial.println("✓ mmWave radars initialized.");
            Serial.flush();
        }
    }

    delay(250);

    // Initialize mic and deterrent
    if (DEBUG) {
        Serial.println("Initializing mic and deterrent managers...");
        Serial.flush();
    }
    micManager.begin();
    deterrentManager.begin();
    if (DEBUG) {
        Serial.println("✓ Mic and deterrent managers initialized.");
        Serial.flush();
    }

    // Create FreeRTOS components
    if (DEBUG) {
        Serial.println("Creating FreeRTOS tasks...");
        Serial.flush();
    }
    packetQueue = xQueueCreate(1, sizeof(SensorPacket));
    
    xTaskCreatePinnedToCore(
        sensorCoreTask,
        "sensors",
        8192,
        nullptr,
        2,
        &sensorTaskHandle,
        0  // Core 0
    );

    xTaskCreatePinnedToCore(
        uplinkCoreTask,
        "uplink",
        6144,
        nullptr,
        1,
        &uplinkTaskHandle,
        1  // Core 1
    );

    if (DEBUG) {
        Serial.println("✓ Setup complete. Handing over to tasks.");
        Serial.flush();
    }

    ledManager.setColor(0, 100, 0); // Green = ready
}

void sensorCoreTask(void* p) {
    SensorPacket pkt;
    
    for (;;) {
        // Wait for notification from loop()
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // Read thermal sensors
        thermalManager.readRotated();
        ThermalReadings thermalData = thermalManager.getObject();

        memcpy(pkt.thermal_left, thermalData.left, sizeof(float) * 64);
        memcpy(pkt.thermal_center, thermalData.center, sizeof(float) * 64);
        memcpy(pkt.thermal_right, thermalData.right, sizeof(float) * 64);

        // Read radar sensors
        mmWaveManager.update();
        RadarData r1_data = mmWaveManager.getRadar1();
        pkt.r1.range_cm = r1_data.range_cm;
        pkt.r1.speed_ms = r1_data.speed_ms;
        pkt.r1.energy = r1_data.energy;
        pkt.r1.lastDetection = r1_data.lastDetection;
        pkt.r1.isValid = r1_data.isValid;
        
        RadarData r2_data = mmWaveManager.getRadar2();
        pkt.r2.range_cm = r2_data.range_cm;
        pkt.r2.speed_ms = r2_data.speed_ms;
        pkt.r2.energy = r2_data.energy;
        pkt.r2.lastDetection = r2_data.lastDetection;
        pkt.r2.isValid = r2_data.isValid;

        // Read microphones
        micManager.read(pkt.micL, pkt.micR);

        // Send to Core 1
        xQueueOverwrite(packetQueue, &pkt);
    }
}

void uplinkCoreTask(void* p) {
    SensorPacket pkt;
    JsonDocument doc;

    for (;;) {
        // Wait for packet from Core 0
        if (xQueueReceive(packetQueue, &pkt, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        doc.clear();

        // Thermal arrays
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

        // Radar data
        doc["radar"]["left"]["range"] = pkt.r1.range_cm;
        doc["radar"]["left"]["speed"] = pkt.r1.speed_ms;
        doc["radar"]["left"]["energy"] = pkt.r1.energy;
        doc["radar"]["left"]["isValid"] = pkt.r1.isValid;

        doc["radar"]["right"]["range"] = pkt.r2.range_cm;
        doc["radar"]["right"]["speed"] = pkt.r2.speed_ms;
        doc["radar"]["right"]["energy"] = pkt.r2.energy;
        doc["radar"]["right"]["isValid"] = pkt.r2.isValid;

        // Microphone data
        doc["mic"]["left"] = pkt.micL;
        doc["mic"]["right"] = pkt.micR;

        // Send JSON to Serial
        serializeJson(doc, Serial);
        Serial.println();
    }
}

void loop() {
    // Update deterrent
    deterrentManager.update();

    // Sleep Management
    ledManager.setColor(0, 0, 100); // Blue = sleeping
    sleepManager.goToSleep();

    // WAKE UP
    ledManager.setColor(0, 100, 0); // Green = awake

    // Signal sensor task
    if (sensorTaskHandle) {
        xTaskNotifyGive(sensorTaskHandle);
    }

    // Signal deterrent
    deterrentManager.signalUnsureDetection();
    unsigned long signalDeadline = millis() + kPlaceholderSignalWindowMs;
    while (deterrentManager.isSignaling() && millis() < signalDeadline) {
        deterrentManager.update();
        delay(1);
    }
}