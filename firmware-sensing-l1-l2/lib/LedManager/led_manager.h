#pragma once
#include <Arduino.h>

class LedManager {
public:
    // Constructor: Takes the pin and brightness
    LedManager(int pin, uint8_t brightness = 255);

    void begin(); // Call this in setup() to init the pixel
    void setBrightness(uint8_t brightness);
    void setColor(uint8_t r, uint8_t g, uint8_t b);
    void setOff();

private:
    int _pin;
    uint8_t _brightness;
    uint8_t _targetR;
    uint8_t _targetG;
    uint8_t _targetB;
    bool _initialized;

    void writeColor();
};