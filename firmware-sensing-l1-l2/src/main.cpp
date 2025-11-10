#include <Arduino.h>
#include "wifi_manager.h"
#include "sleep_manager.h"
#include "led_manager.h"
#include "thermal_array_manager.h"
#include "mmWave_array_manager.h"
#include "mic_manager.h"

// ====== USER CONFIG ======
#define LPIR 3
#define CPIR 13
#define RPIR 2

#define DEBUG true

#define BRIGHTNESS 15  // RGB LED brightness (0-255)
const char* WIFI_SSID     = "ECE449deco";
const char* WIFI_PASSWORD = "ece449$$";

// Thermal I2C pins
#define T0_SDA 48
#define T0_SCL 47
#define T1_SDA 45
#define T1_SCL 20

// mmWave pins
#define RADAR1_RX 16 // LEFT
#define RADAR1_TX 15 // LEFT
#define RADAR2_RX 17 // RIGHT
#define RADAR2_TX 18 // RIGHT
// ==========================

// Manager objects
WifiManager wifiManager(WIFI_SSID, WIFI_PASSWORD, DEBUG);
SleepManager sleepManager(LPIR, CPIR, RPIR);
LedManager ledManager(LED_BUILTIN, BRIGHTNESS);

ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, DEBUG);
mmWaveArrayManager mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, DEBUG);
MicManager micManager(0.2, DEBUG);

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n--- Farm Defense System Booting Up ---");

    // Init LED
    ledManager.begin();
    ledManager.setColor(255, 255, 0); // Yellow = setup start

    // Init Wi-Fi
    wifiManager.connect();

    // Configure sleep
    sleepManager.configure();

    // Init sensors
    if (!thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL)) {
        Serial.println("⚠️ Thermal sensors failed!");
    } else {
        Serial.println("Thermal sensors initialized.");
    }

    if (!mmWaveManager.begin()) {
        Serial.println("⚠️ mmWave radars failed!");
    } else {
        Serial.println("mmWave radars initialized.");
    }

    micManager.begin();
}

void loop() {
    // --- GO TO SLEEP ---
    Serial.println("Entering light sleep, waiting for animal...");
    ledManager.setColor(0, 0, 255); // Blue = sleep

    sleepManager.goToSleep();  // Blocking, wake on PIR

    // --- WOKE UP ---
    ledManager.setColor(0, 255, 0); // Green = awake
    Serial.println("\n--- WOKE UP! Animal detected! ---");

    // --- CHECK WIFI ---
    if (!wifiManager.isConnected()) {
        Serial.println("⚠️ Wi-Fi disconnected. Reconnecting...");
        wifiManager.connect();
    }

    // --- 1. READ THERMAL ---
    thermalManager.readRotated();
    ThermalReadings thermal = thermalManager.getObject();

    // --- 2. READ mmWave ---
    mmWaveManager.update();
    RadarData r1 = mmWaveManager.getRadar1();
    RadarData r2 = mmWaveManager.getRadar2();

    // --- 3. READ MIC ---
    double leftMic = 0, rightMic = 0;
    micManager.read(leftMic, rightMic);

    // --- 4. BUILD COMBINED JSON ---
    StaticJsonDocument<2048> doc;

    // Thermal
    JsonObject thermalObj = doc.createNestedObject("thermal");
    JsonArray leftArr = thermalObj.createNestedArray("left");
    JsonArray centerArr = thermalObj.createNestedArray("center");
    JsonArray rightArr = thermalObj.createNestedArray("right");
    for (int i = 0; i < 64; i++) {
        leftArr.add(thermal.left[i]);
        centerArr.add(thermal.center[i]);
        rightArr.add(thermal.right[i]);
    }

    // mmWave
    JsonObject radarObj = doc.createNestedObject("mmWave");
    JsonObject r1Obj = radarObj.createNestedObject("R1");
    r1Obj["numTargets"] = r1.numTargets;
    r1Obj["range"] = r1.range_cm;
    r1Obj["speed"] = r1.speed_ms;
    r1Obj["energy"] = r1.energy;
    r1Obj["valid"] = r1.isValid;

    JsonObject r2Obj = radarObj.createNestedObject("R2");
    r2Obj["numTargets"] = r2.numTargets;
    r2Obj["range"] = r2.range_cm;
    r2Obj["speed"] = r2.speed_ms;
    r2Obj["energy"] = r2.energy;
    r2Obj["valid"] = r2.isValid;

    // Mic
    JsonObject micObj = doc.createNestedObject("mic");
    micObj["left"] = leftMic;
    micObj["right"] = rightMic;

    // --- 5. SEND JSON ---
    String output;
    serializeJson(doc, output);
    Serial.println(output);

    Serial.println("Alert sent. Going back to sleep.\n");
}
