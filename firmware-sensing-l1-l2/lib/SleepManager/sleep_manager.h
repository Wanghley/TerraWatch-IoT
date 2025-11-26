#pragma once
#include "esp_sleep.h"

class SleepManager {
public:
    // debug flag enables Serial logging
    SleepManager(int lpir, int cpir, int rpir, bool debug = false);

    // Call once at boot. Handles wake-cause diagnostics.
    void configure();

    // Enter light sleep with EXT1 (PIRs) and optional timer wakeup.
    // If sleepMs == 0 => no timer, only EXT1.
    void goToSleep(uint32_t sleepMs = 0);

    // Last EXT1 mask (which PIRs woke us)
    uint64_t getWakeMask() const { return _wakeMask; }

private:
    int _lpir;
    int _cpir;
    int _rpir;
    uint64_t _wakeMask;
    bool _debug;
};