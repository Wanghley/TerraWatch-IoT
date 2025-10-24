#include "wake_manager.h"

RTC_DATA_ATTR int SleepManager::wakeCount = 0;

void SleepManager::begin() {
    pinMode(LPIR, INPUT);
    pinMode(CPIR, INPUT);
    pinMode(RPIR, INPUT);
    pinMode(LED, OUTPUT);
    digitalWrite(LED, HIGH); // LED on when awake

    ++wakeCount;
    Serial.printf("\n--- Wake #%d ---\n", wakeCount);

    esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
    Serial.printf("Wake cause code: %d\n", cause);

    if (cause == ESP_SLEEP_WAKEUP_EXT1) {
        delay(50);
        uint64_t mask = esp_sleep_get_ext1_wakeup_status();
        bool l = (mask & (1ULL << LPIR)) != 0;
        bool c = (mask & (1ULL << CPIR)) != 0;
        bool r = (mask & (1ULL << RPIR)) != 0;
        Serial.printf("EXT1 mask bits: L=%d C=%d R=%d\n", l, c, r);
    }

    Serial.printf("Raw reads: L=%d C=%d R=%d\n",
                  digitalRead(LPIR), digitalRead(CPIR), digitalRead(RPIR));
}

int SleepManager::getWakeCount() {
    return wakeCount;
}

void SleepManager::sleepNow() {
    Serial.println("Going to deep sleep...");

    uint64_t wakeMask = (1ULL << LPIR) | (1ULL << CPIR) | (1ULL << RPIR);
    esp_sleep_enable_ext1_wakeup(wakeMask, ESP_EXT1_WAKEUP_ANY_HIGH);

    delay(100); // settle
    digitalWrite(LED, LOW);
    Serial.flush();
    esp_deep_sleep_start();
}
