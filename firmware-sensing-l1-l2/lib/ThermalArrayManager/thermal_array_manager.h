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
    ThermalArrayManager(uint8_t leftAddr, uint8_t rightAddr, uint8_t centerAddr,
                        TwoWire &wireLeft, TwoWire &wireRight, bool debug = false);

    // Initialize sensors with I2C pins and frequency
    bool begin(uint8_t sda0, uint8_t scl0,
               uint8_t sda1, uint8_t scl1,
               uint32_t freq = 400000);

    // Read raw pixel data from sensors
    void readRaw();

    // Read and rotate pixel data
    void readRotated();

    // Get pixel data as ThermalReadings object
    ThermalReadings getObject();

    // Get pixel data as JSON string
    String getJSON();

    // Quick status
    inline bool isReady() const { return _leftOk || _rightOk || _centerOk; }

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

    bool _debug;

    // New: init flags
    bool _leftOk;
    bool _rightOk;
    bool _centerOk;

    void rotate270CW(float* src, float* dst);
};
