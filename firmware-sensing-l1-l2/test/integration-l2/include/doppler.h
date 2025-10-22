#pragma once
#include <Arduino.h>
#include "DFRobot_C4001.h"

// UART pins for Doppler sensor
#define DOPPLER_RX_PIN 16
#define DOPPLER_TX_PIN 17

// UART and sensor objects
extern HardwareSerial DopplerSerial;
extern DFRobot_C4001_UART Doppler;

// Struct for sensor readings
struct DopplerData {
    float speed;
    float range;
    float energy;
};

// Functions
void setupDoppler(boolean frettingOn = true, int min = 5, int max = 2400, int thres = 20);
DopplerData readDoppler();
