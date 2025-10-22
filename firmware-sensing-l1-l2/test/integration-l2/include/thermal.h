#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_AMG88xx.h>

// I2C pins for ESP32-S3
#define I2C_SDA 8
#define I2C_SCL 9

// Sensor object
extern Adafruit_AMG88xx amg;

// Struct for thermal data
struct ThermalFrame {
    float pixels[AMG88xx_PIXEL_ARRAY_SIZE];
    int width;   // 8
    int height;  // 8
};

// Functions
bool setupThermalSensor();
ThermalFrame readThermalFrame();
void printThermalFrame(const ThermalFrame& frame);
