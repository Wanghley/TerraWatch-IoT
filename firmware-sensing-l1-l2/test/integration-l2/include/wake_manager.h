#pragma once
#include <Arduino.h>
#include "esp_sleep.h"

class SleepManager {
private:
    int LPIR, CPIR, RPIR;
    int LED = LED_BUILTIN;
    RTC_DATA_ATTR static int wakeCount;

public:
    SleepManager(int l, int c, int r) : LPIR(l), CPIR(c), RPIR(r) {}

    void begin();           // Initialize pins, increment wake count
    int getWakeCount();     // Return wake count
    void sleepNow();        // Go to deep sleep
};
