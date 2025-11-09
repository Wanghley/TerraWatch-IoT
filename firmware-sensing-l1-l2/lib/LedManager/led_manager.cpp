#include "led_manager.h"

// Constructor
LedManager::LedManager(int pin, uint8_t brightness)
  : _pin(pin), _brightness(brightness) {
}

void LedManager::begin() {
    // Initialize the LED to off
    // We use neopixelWrite, a built-in helper for the ESP32-S3
    setOff();
}

void LedManager::setBrightness(uint8_t brightness) {
    _brightness = brightness;
}

void LedManager::setColor(uint8_t r, uint8_t g, uint8_t b) {
    // Scale brightness (this is your logic, moved here)
    uint8_t scaled_r = (r * _brightness) / 255;
    uint8_t scaled_g = (g * _brightness) / 255;
    uint8_t scaled_b = (b * _brightness) / 255;
    
    // neopixelWrite is a helper for the onboard S3 LED
    neopixelWrite(_pin, scaled_r, scaled_g, scaled_b);
}

void LedManager::setOff() {
    neopixelWrite(_pin, 0, 0, 0);
}