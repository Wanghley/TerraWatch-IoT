#include <Arduino.h>
#include "DFRobot_C4001.h"

// UART pins for ESP32-S3
#define RADAR_RX_PIN 16
#define RADAR_TX_PIN 17

HardwareSerial RadarSerial(2); // UART2
DFRobot_C4001_UART radar(&RadarSerial, 9600, RADAR_RX_PIN, RADAR_TX_PIN);

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("DFRobot mmWave Radar C4001 - MAX Sensitivity");

  RadarSerial.begin(9600, SERIAL_8N1, RADAR_RX_PIN, RADAR_TX_PIN);
  delay(100);

  Serial.println("Initializing radar...");
  while (!radar.begin()) {
    Serial.println("Radar not found! Check wiring & power.");
    delay(1000);
  }
  Serial.println("Radar Initialized Successfully!");

  // --- MAX SENSITIVITY CONFIGURATION ---
  radar.setSensorMode(eSpeedMode);
  radar.setFrettingDetection(eON);      // detect weak movements
  radar.setDetectThres(5, 2400, 20);    // min threshold, max range, gain


  // Print current configuration
  sSensorStatus_t data = radar.getStatus();
  Serial.print("Work status: "); Serial.println(data.workStatus);
  Serial.print("Work mode  : "); Serial.println(data.workMode);
  Serial.print("Init status: "); Serial.println(data.initStatus);
  Serial.print("Min range  : "); Serial.println(radar.getTMinRange());
  Serial.print("Max range  : "); Serial.println(radar.getTMaxRange());
  Serial.print("Threshold  : "); Serial.println(radar.getThresRange());
  Serial.print("Fretting   : "); Serial.println(radar.getFrettingDetection());
  Serial.println("--------------------------------------");
}

void loop() {
  int numTargets = radar.getTargetNumber();

  if (numTargets > 0) {
    Serial.print("Target detected: ");
    Serial.print("Speed: "); Serial.print(radar.getTargetSpeed()); Serial.print(" m/s, ");
    Serial.print("Range: "); Serial.print(radar.getTargetRange()); Serial.print(" m, ");
    Serial.print("Energy: "); Serial.println(radar.getTargetEnergy());
  } else {
    Serial.println("No targets detected.");
  }

  delay(100); // ~10 Hz update
}
