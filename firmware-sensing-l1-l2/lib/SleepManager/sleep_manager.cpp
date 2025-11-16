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
    if (_debug) Serial.printf("wake cause code: %d\n", cause);

    switch (cause) {
        case ESP_SLEEP_WAKEUP_EXT1: {
            // small settle time and read EXT1 mask
            delay(50);
            uint64_t mask = esp_sleep_get_ext1_wakeup_status();
            _wakeMask = mask;
            bool l = (mask & (1ULL << _lpir)) != 0;
            bool c = (mask & (1ULL << _cpir)) != 0;
            bool r = (mask & (1ULL << _rpir)) != 0;
            if (_debug) {
                Serial.printf("Wake reason: EXT1 (mask=0x%llx)\n", mask);
                Serial.printf("  -> LPIR triggered: %d\n", l);
                Serial.printf("  -> CPIR triggered: %d\n", c);
                Serial.printf("  -> RPIR triggered: %d\n", r);
                Serial.printf("Raw reads: L=%d C=%d R=%d\n",
                              digitalRead(_lpir), digitalRead(_cpir), digitalRead(_rpir));
            }
            break;
        }
        case ESP_SLEEP_WAKEUP_EXT0:
            if (_debug) {
                Serial.println("Wake reason: EXT0");
                Serial.printf("Raw reads: L=%d C=%d R=%d\n",
                              digitalRead(_lpir), digitalRead(_cpir), digitalRead(_rpir));
            }
            break;
        case ESP_SLEEP_WAKEUP_TIMER:
            if (_debug) Serial.println("Wake reason: TIMER");
            break;
        case ESP_SLEEP_WAKEUP_TOUCHPAD:
            if (_debug) Serial.println("Wake reason: TOUCHPAD");
            break;
        case ESP_SLEEP_WAKEUP_UNDEFINED:
        default:
            if (_debug) Serial.println("Wake reason: UNDEFINED/OTHER");
            break;
    }
}

void SleepManager::goToSleep() {
    if (_debug) {
        Serial.println("SleepManager: going to light sleep (EXT1 enabled)...");
        Serial.flush();
    }
    uint64_t wakeMask = (1ULL << _lpir) | (1ULL << _cpir) | (1ULL << _rpir);
    esp_sleep_enable_ext1_wakeup(wakeMask, ESP_EXT1_WAKEUP_ANY_HIGH);
    delay(100); // allow pins to settle
    // light sleep
    esp_light_sleep_start();
}