#include "doppler.h"

// UART object (UART2)
HardwareSerial DopplerSerial(2);

// Doppler sensor object
DFRobot_C4001_UART Doppler(&DopplerSerial, 9600, DOPPLER_RX_PIN, DOPPLER_TX_PIN);

void setupDoppler(boolean frettingOn, int min, int max, int thres) {
    Serial.println("Initializing Doppler sensor...");

    // Start UART
    DopplerSerial.begin(9600, SERIAL_8N1, DOPPLER_RX_PIN, DOPPLER_TX_PIN);
    delay(100);

    // Initialize sensor with watchdog-safe retry
    unsigned long start = millis();
    while (!Doppler.begin()) {
        Serial.println("Radar not found! Check wiring & power.");
        if (millis() - start > 10000) {  // 10-second timeout
            Serial.println("Initialization timeout. Continuing anyway.");
            break;
        }
        delay(500);
    }
    Serial.println("Radar Initialized Successfully!");

    // MAX SENSITIVITY CONFIGURATION
    Doppler.setSensorMode(eSpeedMode);
    Doppler.setFrettingDetection(frettingOn ? eON : eOFF);
    Doppler.setDetectThres(min, max, thres);

    // Print current configuration
    sSensorStatus_t data = Doppler.getStatus();
    Serial.print("Work status: "); Serial.println(data.workStatus);
    Serial.print("Work mode  : "); Serial.println(data.workMode);
    Serial.print("Init status: "); Serial.println(data.initStatus);
    Serial.print("Min range  : "); Serial.println(Doppler.getTMinRange());
    Serial.print("Max range  : "); Serial.println(Doppler.getTMaxRange());
    Serial.print("Threshold  : "); Serial.println(Doppler.getThresRange());
    Serial.print("Fretting   : "); Serial.println(Doppler.getFrettingDetection());
    Serial.println("--------------------------------------");
}

DopplerData readDoppler() {
    DopplerData d;
    if (Doppler.getTargetNumber() > 0) {
        d.speed = Doppler.getTargetSpeed();
        d.range = Doppler.getTargetRange();
        d.energy = Doppler.getTargetEnergy();
    } else {
        d.speed = 0;
        d.range = 0;
        d.energy = 0;
    }
    return d;
}
