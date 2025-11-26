#include "sleep_manager.h"
#include <Arduino.h>

SleepManager::SleepManager(int lpir, int cpir, int rpir, bool debug)
    : _lpir(lpir), _cpir(cpir), _rpir(rpir), _wakeMask(0), _debug(debug) {}

void SleepManager::configure() {
    // initialize serial for debug if requested
    if (_debug && !Serial) {
        Serial.begin(115200);
        delay(200);
    }

    pinMode(_lpir, INPUT);
    pinMode(_cpir, INPUT);
    pinMode(_rpir, INPUT);

    esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
    if (_debug) Serial.printf("[Sleep] wake cause code: %d\n", cause);

    if (cause == ESP_SLEEP_WAKEUP_EXT1) {
        delay(50); // settle
        uint64_t mask = esp_sleep_get_ext1_wakeup_status();
        _wakeMask = mask;
        bool l = (mask & (1ULL << _lpir)) != 0;
        bool c = (mask & (1ULL << _cpir)) != 0;
        bool r = (mask & (1ULL << _rpir)) != 0;
        if (_debug) {
            Serial.printf("[Sleep] EXT1 wake: L=%d C=%d R=%d (mask=0x%llX)\n",
                          l, c, r, (unsigned long long)mask);
        }
    } else if (_debug) {
        Serial.println("[Sleep] Woke from non-EXT1 cause.");
    }
}

void SleepManager::goToSleep(uint32_t sleepMs) {
    if (_debug) {
        Serial.println("[Sleep] Arming EXT1 wakeup (PIRs)...");
        Serial.flush();
    }

    // Wait up to 1s for all PIRs to go LOW to avoid immediate wake
    unsigned long start = millis();
    while (millis() - start < 1000) {
        int l = digitalRead(_lpir);
        int c = digitalRead(_cpir);
        int r = digitalRead(_rpir);
        if (l == LOW && c == LOW && r == LOW) break;
        if (_debug) Serial.printf("[Sleep] Waiting for PIR idle: L=%d C=%d R=%d\n", l, c, r);
        delay(50);
    }

    uint64_t wakeMask = (1ULL << _lpir) | (1ULL << _cpir) | (1ULL << _rpir);
    esp_sleep_enable_ext1_wakeup(wakeMask, ESP_EXT1_WAKEUP_ANY_HIGH);

    if (sleepMs > 0) {
        // Convert ms to µs
        esp_sleep_enable_timer_wakeup((uint64_t)sleepMs * 1000ULL);
        if (_debug) {
            Serial.printf("[Sleep] Timer wake enabled: %lu ms\n", (unsigned long)sleepMs);
        }
    }

    delay(10); // settle after arming

    if (_debug) {
        Serial.println("[Sleep] Entering LIGHT sleep (EXT1 enabled)...");
        Serial.flush();
    }

    // Use light sleep to preserve RAM and running tasks.
    esp_light_sleep_start();

    // On return from light sleep, tasks continue.
    if (_debug) {
        Serial.println("[Sleep] Woke from light sleep.");
    }
}