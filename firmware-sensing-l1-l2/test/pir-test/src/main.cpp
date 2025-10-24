#include <Arduino.h>
#include "esp_sleep.h"

#define LPIR 12  // A0
#define CPIR 11  // A1
#define RPIR 10  // A2 (input-only)

RTC_DATA_ATTR int wakeCount = 0;

void enterDeepSleep();

void setup() {
  Serial.begin(115200);
  delay(300);
  pinMode(LPIR, INPUT);
  pinMode(CPIR, INPUT);
  pinMode(RPIR, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH); // on when awake
  ++wakeCount;

  Serial.printf("\n--- Wake #%d ---\n", wakeCount);
  esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
  Serial.printf("wake cause code: %d\n", cause);

  if (cause == ESP_SLEEP_WAKEUP_EXT1) {
    // small settle time
    delay(50);
    uint64_t mask = esp_sleep_get_ext1_wakeup_status();
    bool l = (mask & (1ULL << LPIR)) != 0;
    bool c = (mask & (1ULL << CPIR)) != 0;
    bool r = (mask & (1ULL << RPIR)) != 0;
    Serial.printf("EXT1 mask bits: L=%d C=%d R=%d\n", l, c, r);
  }

  // ALSO print raw GPIO states for diagnostics
  Serial.printf("Raw reads: L=%d C=%d R=%d\n",
                digitalRead(LPIR), digitalRead(CPIR), digitalRead(RPIR));

  // short demo processing
  for (int i=0;i<10;i++) { Serial.println("processing..."); delay(1000); }

  enterDeepSleep();
}

void loop() {
  // not used
}

void enterDeepSleep() {
  Serial.println("Configuring EXT1 wakeup...");
  uint64_t wakeMask = (1ULL << LPIR) | (1ULL << CPIR) | (1ULL << RPIR);
  esp_sleep_enable_ext1_wakeup(wakeMask, ESP_EXT1_WAKEUP_ANY_HIGH);

  delay(500);

  Serial.println("Going to deep sleep...");
  digitalWrite(LED_BUILTIN, LOW);
  Serial.println("... zzz ...");
  Serial.flush();
  esp_deep_sleep_start();
}

