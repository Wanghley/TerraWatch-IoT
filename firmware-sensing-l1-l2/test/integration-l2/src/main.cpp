#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#include "wake_manager.h"
#include "doppler.h"
#include "thermal.h"
#include "mic.h"
#include "feature_extractor.h"
#include "inference.h"

// PIR pins: LPIR, CPIR, RPIR
SleepManager sleepManager(12, 11, 10);

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ;
    // Initialize wake manager
    sleepManager.begin();

    // Initialize microphone
    mic_begin();
    Serial.printf("Wake count: %d\n", sleepManager.getWakeCount());

    // Initialize sensors
    setupDoppler(true, 5, 2400, 20);
    if (!setupThermalSensor())
    {
        Serial.println("Thermal sensor failed. Halting.");
        while (1)
            ;
    }

    // Initialize TFLite Micro
    if (!Inference::begin())
    {
        Serial.println("Inference setup failed. Halting.");
        while (1)
            ;
    }

    unsigned long startTime = millis();
    const unsigned long durationMs = 3UL * 1000UL; // 3 seconds
    const unsigned long sampleIntervalMs = 100;    // ~100 Hz → 10 ms per sample
    int sampleCount = 0;

    float sum_prob = 0.0f;
    int prob_count = 0;

    // --- Start of sampling loop ---
    while (millis() - startTime < durationMs)
    {
        // [ ... Your existing sensor reading and printing code ... ]
        // (Doppler, Thermal, Mic data printing)

        DopplerData doppler = readDoppler();
        ThermalFrame thermal = readThermalFrame();
        double micRMS = mic_readRMS();
        int lastPeak = mic_getPeak();

        // Output as single-line JSON-like format for Octave
        Serial.print("{");
        Serial.printf("\"sample\":%d,", ++sampleCount);
        Serial.printf("\"doppler\":{\"speed\":%.2f,\"range\":%.2f,\"energy\":%.2f},",
                      doppler.speed, doppler.range, doppler.energy);
        Serial.print("\"thermal\":[");
        for (int j = 0; j < AMG88xx_PIXEL_ARRAY_SIZE; j++)
        {
            Serial.print(thermal.pixels[j], 2);
            if (j < AMG88xx_PIXEL_ARRAY_SIZE - 1)
                Serial.print(",");
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

        // --- Run inference ---
        float movementProb = Inference::predict(features);
        Serial.printf("Movement probability: %.3f\n", movementProb);
        sum_prob += movementProb;
        prob_count++;

        // [ ... Your existing delay logic ... ]
        unsigned long elapsed = millis() - featureStart;
        unsigned long nextSampleTime = sampleCount * sampleIntervalMs;
        if (nextSampleTime > elapsed)
        {
             // delay(nextSampleTime - elapsed); // Your original logic
        }
    }
    // --- End of sampling loop ---

    float avg_prob = (prob_count > 0) ? (sum_prob / static_cast<float>(prob_count)) : 0.0f;
    Serial.printf("Average movement probability over %d samples: %.3f\n", prob_count, avg_prob);

    // --- UPDATED: Classification and LED logic ---
    const float CLASSIFICATION_THRESHOLD = 0.05f;

    if (avg_prob > CLASSIFICATION_THRESHOLD)
    {
        // Class 1 (Animal) -> Set LED to Red
        Serial.println("Classification: Animal (1)");
    }
    else
    {
        // Class 0 (Human) -> Set LED to Green
        Serial.println("Classification: Human (0)");
    }

    // Keep the LED on for a few seconds to see the result
    Serial.println("Displaying result for 5 seconds before sleeping...");
    delay(5000);

    // Go back to deep sleep
    sleepManager.sleepNow();
}

void loop()
{
    // Not used
}