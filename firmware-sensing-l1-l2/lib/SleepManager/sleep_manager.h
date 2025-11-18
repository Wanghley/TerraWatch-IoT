#pragma once
#include "esp_sleep.h"

class SleepManager {
public:
    SleepManager(int lpir, int cpir, int rpir, bool debug = false);
    void configure();
    void goToSleep();

private:
    int _lpir;
    int _cpir;
    int _rpir;
    uint64_t _wakeMask;
    bool _debug;
};