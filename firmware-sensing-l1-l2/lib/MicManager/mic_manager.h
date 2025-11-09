#pragma once

#include <Arduino.h>
#include <driver/i2s.h>
#include <ArduinoJson.h>
#include <math.h>

class MicManager {
public:
    // Constructor with optional smoothing alpha
    MicManager(double smoothingAlpha = 0.2);

    // Initialize I2S driver
    void begin();

    // Read audio and get smoothed RMS values
    bool read(double &left, double &right);

    // Send JSON output to Serial
    void sendJSON(double left, double right);

private:
    // I2S configuration
    i2s_config_t makeI2SConfig();
    i2s_pin_config_t makeI2SPins();

    // Compute RMS from interleaved stereo buffer
    void computeStereoRMS(double &leftRMS, double &rightRMS, int32_t *buf, size_t len);

    // I2S port and pins
    static const i2s_port_t I2S_PORT = I2S_NUM_0;
    static const int SAMPLE_COUNT = 128;
    int32_t i2sBuffer[SAMPLE_COUNT * 2];

    // Smoothing
    double prevL;
    double prevR;
    double alpha;

    // I2S pins
    static const int I2S_SCK = 5;
    static const int I2S_WS  = 4;
    static const int I2S_SD  = 21;
};
