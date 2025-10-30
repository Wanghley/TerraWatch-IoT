#include <Arduino.h>
#include "doppler.h"
#include "thermal.h"
#include "mic.h"

const int TRIGGER_PIN = 21;
const int LABEL_PIN = 1;

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ;

    pinMode(TRIGGER_PIN, INPUT_PULLUP);
    pinMode(LABEL_PIN, INPUT);

    mic_begin();
    setupDoppler(true, 5, 1200, 20);

    if (!setupThermalSensor())
    {
        Serial.println("Thermal sensor failed. Halting.");
        while (1)
            ;
    }

    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.println("Ready. Waiting for trigger...");

    while (digitalRead(TRIGGER_PIN) == HIGH)
    {
        delay(10);
    }

    int label = digitalRead(LABEL_PIN); 
    // 0 - human
    // 1 - animal
    Serial.printf("Triggered! Label: %d\n", label);
    delay(500);

    Serial.println("=== BEGIN DATA STREAM ===");

    unsigned long startTime = millis();
    const unsigned long durationMs = 10000; // 10 seconds
    const unsigned long sampleIntervalMs = 10;
    int sampleCount = 0;

    const int micSamplesPerRead = 10;
    static float micSamples[micSamplesPerRead]; // static keeps it off the stack

    while (millis() - startTime < durationMs)
    {
        DopplerData doppler = readDoppler();
        ThermalFrame thermal = readThermalFrame();

        double micSum = 0;
        int peakSum = 0;
        for (int i = 0; i < micSamplesPerRead; i++)
        {
            micSamples[i] = mic_readRMS();
            micSum += micSamples[i];
            peakSum += mic_getPeak();
            delay(5);
        }
        double micRMSmean = micSum / micSamplesPerRead;
        int micPeakMean = peakSum / micSamplesPerRead;

        // Print JSON piece by piece to avoid big local buffer
        Serial.print("{\"sample\":");
        Serial.print(++sampleCount);
        Serial.print(",\"timestamp\":");
        Serial.print(millis());
        Serial.print(",\"label\":");
        Serial.print(label);

        Serial.print(",\"doppler\":{\"speed\":");
        Serial.print(doppler.speed, 2);
        Serial.print(",\"range\":");
        Serial.print(doppler.range, 2);
        Serial.print(",\"energy\":");
        Serial.print(doppler.energy, 2);
        Serial.print("}");

        Serial.print(",\"thermal\":[");
        for (int j = 0; j < AMG88xx_PIXEL_ARRAY_SIZE; j++)
        {
            Serial.print(thermal.pixels[j], 2);
            if (j < AMG88xx_PIXEL_ARRAY_SIZE - 1)
                Serial.print(",");
        }
        Serial.print("]");

        Serial.print(",\"mic\":{\"rms_mean\":");
        Serial.print(micRMSmean, 4);
        Serial.print(",\"peak_mean\":");
        Serial.print(micPeakMean);
        Serial.print(",\"rms_samples\":[");
        for (int i = 0; i < micSamplesPerRead; i++)
        {
            Serial.print(micSamples[i], 4);
            if (i < micSamplesPerRead - 1)
                Serial.print(",");
        }
        Serial.println("]}}");

        yield();
        unsigned long elapsed = millis() - startTime;
        unsigned long nextSampleTime = sampleCount * sampleIntervalMs;
        if (nextSampleTime > elapsed)
            delay(nextSampleTime - elapsed);
    }

    Serial.println("=== END DATA STREAM ===");
    Serial.printf("Final free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.println("Done. Reset to collect again.");
}

void loop()
{
    delay(1000);
}

// #include <Arduino.h>
// #include "wake_manager.h"
// #include "doppler.h"
// #include "thermal.h"
// #include "mic.h"

// // PIR pins: LPIR, CPIR, RPIR
// SleepManager sleepManager(12, 11, 10);

// void setup()
// {
//     Serial.begin(115200);
//     while (!Serial)
//         ;

//     // Initialize wake manager
//     sleepManager.begin();

//     // Initialize microphone
//     mic_begin();
//     Serial.printf("Wake count: %d\n", sleepManager.getWakeCount());

//     // Initialize sensors
//     setupDoppler(true, 5, 2400, 20);
//     if (!setupThermalSensor())
//     {
//         Serial.println("Thermal sensor failed. Halting.");
//         while (1)
//             ;
//     }

//     unsigned long startTime = millis();
//     const unsigned long durationMs = 3UL * 1000UL; // 3 seconds
//     const unsigned long sampleIntervalMs = 100;      // ~100 Hz → 10 ms per sample
//     int sampleCount = 0;

//     while (millis() - startTime < durationMs)
//     {
//         DopplerData doppler = readDoppler();
//         ThermalFrame thermal = readThermalFrame();
//         double micRMS = mic_readRMS();
//         int lastPeak = mic_getPeak();

//         // Output as single-line JSON-like format for Octave
//         Serial.print("{");
//         Serial.printf("\"sample\":%d,", ++sampleCount);

//         // Doppler
//         Serial.printf("\"doppler\":{\"speed\":%.2f,\"range\":%.2f,\"energy\":%.2f},",
//                       doppler.speed, doppler.range, doppler.energy);

//         // Thermal
//         Serial.print("\"thermal\":[");
//         for (int j = 0; j < AMG88xx_PIXEL_ARRAY_SIZE; j++)
//         {
//             Serial.print(thermal.pixels[j], 2);
//             if (j < AMG88xx_PIXEL_ARRAY_SIZE - 1)
//                 Serial.print(",");
//         }
//         Serial.print("],"); // close thermal array

//         // Microphone
//         Serial.printf("\"mic\":{\"rms\":%.4f,\"peak\":%d}", micRMS, lastPeak);

//         Serial.println("}"); // close sample object

//         // Wait for next sample
//         unsigned long elapsed = millis() - startTime;
//         unsigned long nextSampleTime = sampleCount * sampleIntervalMs;
//         if (nextSampleTime > elapsed)
//         {
//             delay(nextSampleTime - elapsed);
//         }
//     }

//     // Go back to deep sleep
//     sleepManager.sleepNow();
// }

// void loop()
// {
//     // not used
// }
