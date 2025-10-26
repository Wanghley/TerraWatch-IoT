#include <Arduino.h>
#include "wake_manager.h"
#include "doppler.h"
#include "thermal.h"
#include "mic.h"

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

    unsigned long startTime = millis();
    const unsigned long durationMs = 3UL * 1000UL; // 3 seconds
    const unsigned long sampleIntervalMs = 100;      // ~100 Hz → 10 ms per sample
    int sampleCount = 0;

    while (millis() - startTime < durationMs)
    {
        DopplerData doppler = readDoppler();
        ThermalFrame thermal = readThermalFrame();
        double micRMS = mic_readRMS();
        int lastPeak = mic_getPeak();

        // Output as single-line JSON-like format for Octave
        Serial.print("{");
        Serial.printf("\"sample\":%d,", ++sampleCount);

        // Doppler
        Serial.printf("\"doppler\":{\"speed\":%.2f,\"range\":%.2f,\"energy\":%.2f},",
                      doppler.speed, doppler.range, doppler.energy);

        // Thermal
        Serial.print("\"thermal\":[");
        for (int j = 0; j < AMG88xx_PIXEL_ARRAY_SIZE; j++)
        {
            Serial.print(thermal.pixels[j], 2);
            if (j < AMG88xx_PIXEL_ARRAY_SIZE - 1)
                Serial.print(",");
        }
        Serial.print("],"); // close thermal array

        // Microphone
        Serial.printf("\"mic\":{\"rms\":%.4f,\"peak\":%d}", micRMS, lastPeak);

        Serial.println("}"); // close sample object

        // Wait for next sample
        unsigned long elapsed = millis() - startTime;
        unsigned long nextSampleTime = sampleCount * sampleIntervalMs;
        if (nextSampleTime > elapsed)
        {
            delay(nextSampleTime - elapsed);
        }
    }

    // Go back to deep sleep
    sleepManager.sleepNow();
}

void loop()
{
    // not used
}
