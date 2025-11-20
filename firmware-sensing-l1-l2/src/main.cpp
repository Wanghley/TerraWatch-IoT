#include <Arduino.h>
#include <ArduinoJson.h>
#include <WiFi.h>        // <-- ADDED (for IPAddress)

#include "wifi_manager.h"
#include "sleep_manager.h"
#include "led_manager.h"
#include "thermal_array_manager.h"
#include "mmWave_array_manager.h"
#include "mic_manager.h"
#include "ping_wire.h"
#include "inference_manager.h"  // <--- This is correct

// ====== USER CONFIG ======
#define LPIR 12
#define CPIR 13
#define RPIR 14

#define DEBUG true

#define BRIGHTNESS 5
const char* WIFI_SSID     = "ECE449deco";
const char* WIFI_PASSWORD = "ece449$$";
const char* TARGET_ID = "GROUP2_DETER_ESP";
unsigned int UDP_PORT = 4210;

#define T0_SDA 48
#define T0_SCL 47
#define T1_SDA 8
#define T1_SCL 9

#define RADAR1_RX 10
#define RADAR1_TX 16
#define RADAR2_RX 18
#define RADAR2_TX 17
// ==========================

// Manager objects
WifiManager wifiManager(WIFI_SSID, WIFI_PASSWORD, DEBUG, IPAddress(0,0,0,0), UDP_PORT, TARGET_ID);
SleepManager sleepManager(LPIR, CPIR, RPIR);
LedManager ledManager(LED_BUILTIN, BRIGHTNESS);
ThermalArrayManager thermalManager(0x68, 0x69, 0x69, Wire, Wire1, DEBUG);
mmWaveArrayManager mmWaveManager(RADAR1_RX, RADAR1_TX, RADAR2_RX, RADAR2_TX, DEBUG);
MicManager micManager(0.2, DEBUG);

// ADD ML INFERENCE MANAGER
InferenceManager mlInference(DEBUG);

void setup() {
    Serial.begin(115200);
    delay(500);
    if (DEBUG) {
        Serial.println("\n--- Farm Defense System Booting Up ---");
    }

    ledManager.begin();

    // bool wifiConnected = wifiManager.connect();
    // if (wifiConnected) {
    //     ledManager.setColor(0, 100, 0); // Green when online
    // } else {
    //     if (DEBUG) Serial.println("⚠️ Wi-Fi connection unavailable, continuing without Wi-Fi.");
    //     ledManager.setColor(100, 0, 0); // Red when offline
    // }

    sleepManager.configure();

    if (!thermalManager.begin(T0_SDA, T0_SCL, T1_SDA, T1_SCL)) {
        if (DEBUG) Serial.println("⚠️ Thermal sensors failed!");
    } else {
        if (DEBUG) Serial.println("Thermal sensors initialized.");
    }

    if (!mmWaveManager.begin()) {
        if (DEBUG) Serial.println("⚠️ mmWave radars failed!");
    } else {
        if (DEBUG) Serial.println("mmWave radars initialized.");
    }

    micManager.begin();
    if (DEBUG) Serial.println("Mic manager initialized.");

    // INITIALIZE ML MODEL
    if (!mlInference.begin()) {
        if (DEBUG) Serial.println("⚠️ ML inference failed to initialize!");
        ledManager.setColor(255, 0, 0); // Red = error
        delay(2000);
    } else {
        if (DEBUG) Serial.println("✓ ML inference ready!");
    }
}

void loop() {
    // --- GO TO SLEEP ---
    if (DEBUG) Serial.println("\n--- GOING TO SLEEP ---");
    ledManager.setColor(0, 0, 0);
    sleepManager.goToSleep();

    // --- WOKE UP ---
    // if (wifiManager.isConnected()) {
    //     ledManager.setColor(0, 100, 0);
    // } else {
    //     ledManager.setColor(100, 0, 0);
    // }
    if (DEBUG) Serial.println("\n--- WOKE UP ---");

    // --- CHECK WIFI ---
    // if (!wifiManager.isConnected()) {
    //     if (DEBUG) Serial.println("⚠️ Wi-Fi disconnected. Reconnecting...");
    //     ledManager.setColor(0, 0, 0);
    //     bool reconnected = wifiManager.connect();
    //     if (reconnected) {
    //         ledManager.setColor(0, 100, 0);
    //     } else {
    //         ledManager.setColor(100, 0, 0);
    //     }
    // }

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

    // --- 4. ADD TO ML BUFFER ---
    mlInference.addFrame(thermal, r1, r2, leftMic, rightMic);

    // --- 5. RUN INFERENCE IF READY ---
    if (mlInference.isReady()) {
        int prediction = mlInference.predict();
        
        if (prediction == 0) {
            // HUMAN DETECTED
            if (DEBUG) {
                Serial.println("\n>>> 👤 HUMAN DETECTED <<<");
                Serial.printf("Confidence: %.1f%%\n", 
                             mlInference.getHumanConfidence() * 100);
            }
            ledManager.setColor(0, 255, 0); // Green for human
            
            // DO NOT TRIGGER DETERRENCE
            
        } else if (prediction == 1) {
            // ANIMAL DETECTED
            if (DEBUG) {
                Serial.println("\n>>> 🐾 ANIMAL DETECTED <<<");
                Serial.printf("Confidence: %.1f%%\n", 
                             mlInference.getAnimalConfidence() * 100);
            }
            ledManager.setColor(255, 0, 0); // Red for animal
            
            // TRIGGER DETERRENCE SYSTEM
            // wifiManager.triggerDeterrenceSystem(10, 10, "V1.0", "ML");
            
        } else if (prediction == 2) {
            if (DEBUG) {
                Serial.println("Buffering data for ML...");
            }
        }
    } else {
        if (DEBUG) {
            Serial.println("Building ML sequence buffer...");
        }
    }

    // --- 6. BUILD JSON (for logging/debugging) ---
    StaticJsonDocument<2048> doc;
    
    JsonObject thermalObj = doc.createNestedObject("thermal");
    JsonArray leftArr = thermalObj.createNestedArray("left");
    JsonArray centerArr = thermalObj.createNestedArray("center");
    JsonArray rightArr = thermalObj.createNestedArray("right");
    for (int i = 0; i < 64; i++) {
        leftArr.add(thermal.left[i]);
        centerArr.add(thermal.center[i]);
        rightArr.add(thermal.right[i]);
    }

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

    JsonObject micObj = doc.createNestedObject("mic");
    micObj["left"] = leftMic;
    micObj["right"] = rightMic;

    // Add ML prediction to JSON
    if (mlInference.isReady()) {
        JsonObject mlObj = doc.createNestedObject("ml");
        mlObj["human_conf"] = mlInference.getHumanConfidence();
        mlObj["animal_conf"] = mlInference.getAnimalConfidence();
        mlObj["inference_ms"] = mlInference.getInferenceTimeMs();
    }

    String output;
    serializeJson(doc, output);
    Serial.println(output);

    if (DEBUG) {
        Serial.println("\n--- DATA SENT ---");
    }
}