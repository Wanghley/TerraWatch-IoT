#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <WiFi.h>
#include <HTTPClient.h>

#include "wake_manager.h"
#include "doppler.h"
#include "thermal.h"
#include "mic.h"
#include "feature_extractor.h"
#include "inference.h"

// ===== WiFi Configuration =====
const char* WIFI_SSID = "ECE449deco";
const char* WIFI_PASSWORD = "ece449$$";

// Server endpoint
const char* SERVER_URL = "http://192.168.68.104:80/";

// ==============================
SleepManager sleepManager(12, 11, 10);

// Robust WiFi connection function for mesh networks
bool connectWiFi() {
    Serial.println("\nStarting WiFi connection...");

    // Force disconnect any old connections
    WiFi.disconnect(true);
    delay(500);
    
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);

    // Optional: set static IP if DHCP is unreliable
    /*
    IPAddress localIP(192,168,68,200);
    IPAddress gateway(192,168,68,1);
    IPAddress subnet(255,255,255,0);
    if (!WiFi.config(localIP, gateway, subnet)) {
        Serial.println("[WiFi] Static IP configuration failed");
    }
    */

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int maxRetries = 60;  // ~30 seconds
    int retryCount = 0;

    while (WiFi.status() != WL_CONNECTED && retryCount < maxRetries) {
        delay(500);
        Serial.print(".");
        retryCount++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Connected!");
        Serial.print("[WiFi] IP address: ");
        Serial.println(WiFi.localIP());
        return true;
    } else {
        Serial.println("\n[WiFi] Failed to connect!");
        return false;
    }
}

// HTTP GET request function
void sendGET() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        if (http.begin(SERVER_URL)) {
            int httpCode = http.GET();
            if (httpCode > 0) {
                Serial.printf("[HTTP] GET response code: %d\n", httpCode);
                String payload = http.getString();
                Serial.println("[HTTP] Response: " + payload);
            } else {
                Serial.printf("[HTTP] GET failed, error: %s\n", http.errorToString(httpCode).c_str());
            }
            http.end();
        } else {
            Serial.println("[HTTP] begin() failed");
        }
    } else {
        Serial.println("[HTTP] WiFi not connected. Cannot send data.");
    }
}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    Serial.println("\n=== System Starting ===");

    // Initialize wake manager
    sleepManager.begin();

    // Connect to WiFi
    bool wifiConnected = connectWiFi();
    if (!wifiConnected) {
        Serial.println("[WiFi] Continuing without connection...");
    }

    // Initialize microphone
    mic_begin();
    Serial.printf("Wake count: %d\n", sleepManager.getWakeCount());

    // Initialize sensors
    setupDoppler(true, 5, 2400, 20);
    if (!setupThermalSensor()) {
        Serial.println("Thermal sensor failed. Halting.");
        while (1);
    }

    // Initialize TFLite Micro
    if (!Inference::begin()) {
        Serial.println("Inference setup failed. Halting.");
        while (1);
    }

    unsigned long startTime = millis();
    const unsigned long durationMs = 3UL * 1000UL; // 3 seconds
    const unsigned long sampleIntervalMs = 100;    // ~100 Hz → 10 ms per sample
    int sampleCount = 0;
    float sum_prob = 0.0f;
    int prob_count = 0;

    // --- Sampling loop ---
    while (millis() - startTime < durationMs)
    {
        DopplerData doppler = readDoppler();
        ThermalFrame thermal = readThermalFrame();
        double micRMS = mic_readRMS();
        int lastPeak = mic_getPeak();

        // Output JSON-like line
        Serial.print("{");
        Serial.printf("\"sample\":%d,", ++sampleCount);
        Serial.printf("\"doppler\":{\"speed\":%.2f,\"range\":%.2f,\"energy\":%.2f},",
                      doppler.speed, doppler.range, doppler.energy);
        Serial.print("\"thermal\":[");
        for (int j = 0; j < AMG88xx_PIXEL_ARRAY_SIZE; j++) {
            Serial.print(thermal.pixels[j], 2);
            if (j < AMG88xx_PIXEL_ARRAY_SIZE - 1) Serial.print(",");
        }
        Serial.print("],");
        Serial.printf("\"mic\":{\"rms\":%.4f,\"peak\":%d}", micRMS, lastPeak);
        Serial.println("}");

        // Feature extraction
        unsigned long featureStart = millis();
        float micRMSFloat = static_cast<float>(micRMS);
        Features features = FeatureExtractor::extractFeatures(
            0, doppler, thermal, &micRMSFloat, 1, micRMS, lastPeak);
        unsigned long featureExtractionTime = millis() - featureStart;
        Serial.printf("Feature extraction time: %lu ms\n", featureExtractionTime);
        FeatureExtractor::printFeatures(features);

        // Inference
        float movementProb = Inference::predict(features);
        Serial.printf("Movement probability: %.3f\n", movementProb);
        sum_prob += movementProb;
        prob_count++;

        // Delay to maintain sample interval
        unsigned long elapsed = millis() - featureStart;
        unsigned long nextSampleTime = sampleCount * sampleIntervalMs;
        if (nextSampleTime > elapsed) {
            delay(nextSampleTime - elapsed);
        }
    }

    float avg_prob = (prob_count > 0) ? (sum_prob / static_cast<float>(prob_count)) : 0.0f;
    Serial.printf("\nAverage movement probability over %d samples: %.3f\n", prob_count, avg_prob);

    // --- Classification and WiFi sending ---
    const float CLASSIFICATION_THRESHOLD = 0.05f;

    if (avg_prob > CLASSIFICATION_THRESHOLD) {
        // Class 1 (Animal)
        Serial.println("\n=== Classification: Animal (1) ===");
        sendGET(); // Send GET request if WiFi connected
    } else {
        // Class 0 (Human)
        Serial.println("\n=== Classification: Human (0) ===");
        Serial.println("No data sent (human detected).");
    }

    Serial.println("\nDisplaying result for 5 seconds before sleeping...");
    delay(5000);

    // Go to deep sleep
    Serial.println("Entering deep sleep...");
    sleepManager.sleepNow();
}

void loop() {
    // Not used
}
