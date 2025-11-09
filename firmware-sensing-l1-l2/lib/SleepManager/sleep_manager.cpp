#include "sleep_manager.h"
#include <Arduino.h>

SleepManager::SleepManager(int lpir, int cpir, int rpir)
  : _lpir(lpir), _cpir(cpir), _rpir(rpir) {
  // Calculate the bitmask for the pins
  _wakeMask = (1ULL << _lpir) | (1ULL << _cpir) | (1ULL << _rpir);
}

void SleepManager::configure() {
  // Set pins as inputs
  pinMode(_lpir, INPUT);
  pinMode(_cpir, INPUT);
  pinMode(_rpir, INPUT);

  // Configure the wake-up source
  // We will wake up if ANY of the pins in the mask go HIGH
  esp_sleep_enable_ext1_wakeup(_wakeMask, ESP_EXT1_WAKEUP_ANY_HIGH);
}

void SleepManager::goToSleep() {
  Serial.flush(); // Finish printing before sleeping
  esp_light_sleep_start();
}