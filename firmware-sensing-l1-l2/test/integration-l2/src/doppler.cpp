#include "doppler.h"

RTC_DATA_ATTR bool dopplerConfigured = false; // persist across deep sleep

// UART object (UART2)
HardwareSerial DopplerSerial(2);

// Doppler sensor object
DFRobot_C4001_UART Doppler(&DopplerSerial, 9600, DOPPLER_RX_PIN, DOPPLER_TX_PIN);

void setupDoppler(boolean frettingOn, int min, int max, int thres) {
    DopplerSerial.begin(9600, SERIAL_8N1, DOPPLER_RX_PIN, DOPPLER_TX_PIN);
    delay(50); // minimal settle time

    if (!dopplerConfigured) {
        Serial.println("Initializing Doppler sensor for the first time...");

        // Try to begin with short retry loop
        unsigned long start = millis();
        while (!Doppler.begin()) {
            if (millis() - start > 3000) { // 3s timeout
                Serial.println("Initialization timeout. Continuing anyway.");
                break;
            }
            delay(20); // short retry
        }

        Doppler.setSensorMode(eSpeedMode);
        Doppler.setFrettingDetection(frettingOn ? eON : eOFF);
        Doppler.setDetectThres(min, max, thres);

        dopplerConfigured = true; // mark as configured
        Serial.println("Doppler initialized and configured.");
    }
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
