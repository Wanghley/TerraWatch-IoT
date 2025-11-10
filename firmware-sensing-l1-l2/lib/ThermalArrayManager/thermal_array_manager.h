#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_AMG88xx.h>
#include <ArduinoJson.h>

struct ThermalReadings {
    float left[64];
    float center[64];
    float right[64];
};

class ThermalArrayManager {
public:
    ThermalArrayManager(uint8_t leftAddr = 0x68,
                        uint8_t rightAddr = 0x69,
                        uint8_t centerAddr = 0x69,
                        TwoWire &wireLeft = Wire,
                        TwoWire &wireRight = Wire1);

    // Initialize sensors with I2C pins and frequency
    bool begin(uint8_t sda0, uint8_t scl0,
               uint8_t sda1, uint8_t scl1,
               uint32_t freq = 400000);

    // Read raw pixel data
    void readRaw();

    // Read pixels and rotate 270° clockwise
    void readRotated();

    // Return rotated pixels as object
    ThermalReadings getObject();

    // Return rotated pixels as JSON string
    String getJSON();

private:
    TwoWire *_wireLeft;
    TwoWire *_wireRight;

    Adafruit_AMG88xx _amgLeft;
    Adafruit_AMG88xx _amgRight;
    Adafruit_AMG88xx _amgCenter;

    float _pixelsLeft[64];
    float _pixelsRight[64];
    float _pixelsCenter[64];

    float _rotatedLeft[64];
    float _rotatedRight[64];
    float _rotatedCenter[64];

    void rotate270CW(float* src, float* dst);
};
