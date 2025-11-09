#pragma once // Prevents file from being included multiple times

#include "esp_sleep.h"

class SleepManager {
public:
    // Constructor: Takes the PIR sensor pins
    SleepManager(int lpir, int cpir, int rpir);

    // Sets up the GPIOs and wake-up sources
    void configure();

    // Puts the ESP32 into light sleep
    void goToSleep();

private:
    int _lpir, _cpir, _rpir;
    uint64_t _wakeMask;
};