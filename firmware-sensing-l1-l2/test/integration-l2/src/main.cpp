#include <Arduino.h>
#include "doppler.h"
#include "thermal.h"

void setup() {
    Serial.begin(115200);
    while (!Serial);

    Serial.println("=== Initializing Sensors ===");

    // --- Initialize Doppler ---
    setupDoppler(true, 5, 2400, 20);

    // --- Initialize Thermal ---
    if (!setupThermalSensor()) {
        Serial.println("Thermal sensor failed to initialize. Halting.");
        while (1);
    }

    Serial.println("=== Sensors Initialized Successfully ===");
}

void loop() {
    // --- Read Doppler ---
    DopplerData doppler = readDoppler();
    Serial.println("=== Doppler Data ===");
    Serial.print("Speed: "); Serial.print(doppler.speed); Serial.println(" m/s");
    Serial.print("Range: "); Serial.print(doppler.range); Serial.println(" m");
    Serial.print("Energy: "); Serial.println(doppler.energy);

    // --- Read Thermal ---
    ThermalFrame thermal = readThermalFrame();
    Serial.println("=== Thermal Data ===");
    printThermalFrame(thermal);

    delay(100);  // ~10 Hz loop
}
