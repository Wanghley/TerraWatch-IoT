/**
 * @file src/main.cpp
 * @brief Test code for deep sleep and wakeup handling with RTC GPIO pins
 * 
 * This file contains the implementation for handling deep sleep and wakeup events
 * using RTC GPIO pins connected to PIR sensors. The system wakes up on signals from
 * these sensors, processes the event, and goes back to deep sleep when idle.
 * 
 * @author Wanghley Soares Martins <me@wanghley.com>
 * @date 2025-10-05
 */
#include <Arduino.h>
#include "esp_sleep.h"

// RTC GPIO pins for PIR sensors
#define LPIRpin 0
#define CPIRpin 2
#define RPIRpin 4

RTC_DATA_ATTR int wakeupCounter = 0; // Counter to track wakeup events
RTC_DATA_ATTR bool isIdle = false; // Flag to indicate if the system is idle

void setup()
{
  pinMode(LPIRpin, INPUT);
  pinMode(CPIRpin, INPUT);
  pinMode(RPIRpin, INPUT);
  ++wakeupCounter;

  Serial.begin(115200);
  delay(1000); // Give time for Serial to initialize

  handleWakeup();
}

void loop()
{
  // TODO
}

/**
 * @brief Handle wakeup events and process accordingly
 */
void handleWakeup()
{
  // activate onboard LED to indicate wakeup
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100); // LED on for 100ms
  if (isIdle)
  {
    Serial.println("System is idle. No further processing.");
    enterDeepSleep();
    return;
  }

  // Determine wakeup reason
  esp_sleep_wakeup_cause_t wakeupReason = esp_sleep_get_wakeup_cause();

  uint64_t wakeupStatus = esp_sleep_get_ext1_wakeup_status();
  bool lpirState = (wakeupStatus & (1ULL << LPIRpin)) != 0;
  bool cpirState = (wakeupStatus & (1ULL << CPIRpin)) != 0;
  bool rpirState = (wakeupStatus & (1ULL << RPIRpin)) != 0;

  Serial.printf("Wake up caused by movement detection. LPIR: %s, CPIR: %s, RPIR: %s\n",
                lpirState ? "TRUE" : "FALSE",
                cpirState ? "TRUE" : "FALSE",
                rpirState ? "TRUE" : "FALSE");

  switch (wakeupReason)
  {
  case ESP_SLEEP_WAKEUP_EXT0:
    Serial.println("Wakeup caused by external signal using RTC_IO");
    break;
  case ESP_SLEEP_WAKEUP_EXT1:
    Serial.println("Wakeup caused by external signal using RTC_CNTL");
    break;
  case ESP_SLEEP_WAKEUP_TIMER:
    Serial.println("Wakeup caused by timer");
    break;
  default:
    Serial.println("Wakeup caused by unknown reason");
    break;
  }
  Serial.printf("Wakeup count: %d\n", wakeupCounter);

  // TODO: implement processing logic here
  // For demonstration, we will have the system busy for a few cycles before going idle
  for (int i = 0; i < 5; i++)
  {
    Serial.println("Processing...");
    delay(1000); // Simulate processing delay
  }
  isIdle = true;
  enterDeepSleep();
}
void enterDeepSleep()
{
  Serial.println("Entering deep sleep mode...");

  digitalWrite(LED_BUILTIN, LOW); // Turn off LED before sleeping
  delay(100);                     // Ensure LED state is set

  // Configure wakeup sources using EXT1 (supports multiple pins)
  uint64_t wakeupMask = (1ULL << LPIRpin) | (1ULL << CPIRpin) | (1ULL << RPIRpin);
  esp_sleep_enable_ext1_wakeup(wakeupMask, ESP_EXT1_WAKEUP_ANY_HIGH);

  // Enter deep sleep
  esp_deep_sleep_start();
}