#include "led_manager.h"

// Constructor
LedManager::LedManager(int pin, uint8_t brightness)
  : _pin(pin), _brightness(brightness),
    _targetR(255), _targetG(255), _targetB(255),
    _initialized(false) {
}

void LedManager::begin() {
    if (_initialized) {
        return;
    }
    pinMode(_pin, OUTPUT);
    _initialized = true;
    setOff();
}

void LedManager::setBrightness(uint8_t brightness) {
    if (_brightness == brightness) {
        return;
    }
    _brightness = brightness;
    writeColor();
}

void LedManager::setColor(uint8_t r, uint8_t g, uint8_t b) {
    if (_targetR == r && _targetG == g && _targetB == b) {
        return;
    }
    _targetR = r;
    _targetG = g;
    _targetB = b;
    writeColor();
}

void LedManager::writeColor() {
    if (!_initialized) {
        return;
    }
    uint8_t scaled_r = (_targetR * _brightness) / 255;
    uint8_t scaled_g = (_targetG * _brightness) / 255;
    uint8_t scaled_b = (_targetB * _brightness) / 255;

    neopixelWrite(_pin, scaled_r, scaled_g, scaled_b);
}

void LedManager::setOff() {
    setColor(0, 0, 0);
}