#include <Arduino.h>
#include "wifi_manager.h"
#include "sleep_manager.h"
#include "led_manager.h"
#include "thermal_array_manager.h"
#include "mmWave_array_manager.h"
#include "mic_manager.h"
#include "ping_wire.h"

// ====== USER CONFIG ======
#define LPIR 12
#define CPIR 13
#define RPIR 14

#define DEBUG true

#define BRIGHTNESS 5  // RGB LED brightness (0-255)
const char* WIFI_SSID     = "ECE449deco";
const char* WIFI_PASSWORD = "ece449$$";
const char* TARGET_ID = "GROUP2_DETER_ESP";  // the one we want to stop
unsigned int UDP_PORT = 4210;

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

// Manager objects
WifiManager wifiManager(WIFI_SSID, WIFI_PASSWORD, DEBUG, IPAddress(0,0,0,0), UDP_PORT, TARGET_ID);
SleepManager sleepManager(LPIR, CPIR, RPIR);
LedManager ledManager(LED_BUILTIN, BRIGHTNESS);

ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, DEBUG);
mmWaveArrayManager mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, DEBUG);
MicManager micManager(0.2, DEBUG);

void setup() {
    Serial.begin(115200);
    delay(500);
    if (DEBUG) {
        Serial.println("\n--- Farm Defense System Booting Up ---");
    }

    // Init LED
    ledManager.begin();
    ledManager.setColor(100, 100, 0); // Yellow = setup start

    // Init Wi-Fi
    wifiManager.connect();

    // Configure sleep
    sleepManager.configure();

    // Init sensors
    if (!thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL)) {
        if (DEBUG) {
            Serial.println("⚠️ Thermal sensors failed!");
        }
    } else {
        if (DEBUG) {
            Serial.println("Thermal sensors initialized.");
        }
    }

    if (!mmWaveManager.begin()) {
        if (DEBUG) {
            Serial.println("⚠️ mmWave radars failed!");
        }
    } else {
        if (DEBUG) {
            Serial.println("mmWave radars initialized.");
        }
    }

    micManager.begin();
    if (DEBUG) {
        Serial.println("Mic manager initialized.");
    }
}

void loop() {
    // --- GO TO SLEEP ---
    if (DEBUG) {
        Serial.println("\n--- GOING TO SLEEP ---");
    }
    ledManager.setColor(0, 0, 0); // Blue = sleep

    sleepManager.goToSleep();  // Blocking, wake on PIR

    // --- WOKE UP ---
    ledManager.setColor(0, 100, 0); // Green = awake
    if (DEBUG) {
        Serial.println("\n--- WOKE UP ---");
    }

    // --- CHECK WIFI ---
    if (!wifiManager.isConnected()) {
        if (DEBUG) {
           Serial.println("⚠️ Wi-Fi disconnected. Reconnecting...");
        }
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

    wifiManager.triggerDeterrenceSystem(10,10,"V1.0","WAN");

    if (DEBUG) {
        Serial.println("\n--- DATA SENT ---");
    }
}
