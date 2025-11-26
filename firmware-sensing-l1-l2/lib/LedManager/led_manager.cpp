#include "led_manager.h"
#include <Arduino.h>

LedManager::LedManager(int pin, uint8_t brightness)
  : _pin(pin), _brightness(brightness),
    _targetR(0), _targetG(0), _targetB(0),
    _initialized(false) {
}

void LedManager::begin() {
    if (_initialized) return;

    pinMode(_pin, OUTPUT);
    digitalWrite(_pin, LOW);       // WS2812 idle low
    delay(10);                     // allow power-up stabilization
    delayMicroseconds(300);        // reset/latch gap before first frame

    _initialized = true;
    writeColor();                  // push current target (default off)
    delayMicroseconds(80);         // ensure latch after first write
}

void LedManager::setBrightness(uint8_t brightness) {
    // Ignored (no global brightness control)
    _brightness = brightness;
}

void LedManager::setColor(uint8_t r, uint8_t g, uint8_t b) {
    _targetR = r;
    _targetG = g;
    _targetB = b;
    writeColor();
}

void LedManager::writeColor() {
    if (!_initialized) return;

    // Directly drive a single NeoPixel on _pin
    neopixelWrite(_pin, _targetR, _targetG, _targetB);
}

void LedManager::setOff() {
    setColor(0, 0, 0);
}