#include "thermal_array_manager.h"
#include "thermal_array_utils.h"

ThermalArrayManager::ThermalArrayManager(uint8_t leftAddr,
                                         uint8_t rightAddr,
                                         uint8_t centerAddr,
                                         TwoWire &wireLeft,
                                         TwoWire &wireRight,
                                         bool debug)
    : _amgLeft(), _amgRight(), _amgCenter(),
      _wireLeft(&wireLeft), _wireRight(&wireRight),
      _debug(debug)
{ }

bool ThermalArrayManager::begin(uint8_t sda0, uint8_t scl0,
                                uint8_t sda1, uint8_t scl1,
                                uint32_t freq) {

    if (_debug) {
        Serial.println("[thermal] Initializing ThermalArrayManager...");
    }
    _wireLeft->begin(sda0, scl0, freq);
    _wireRight->begin(sda1, scl1, freq);

    bool status;
    status = _amgLeft.begin(0x68, _wireLeft);
    if (!status) {
        if (_debug) {
            Serial.println("[thermal] LEFT sensor not found!");
        }
        return false;
    }

    status = _amgRight.begin(0x69, _wireLeft);
    if (!status) { 
        if (_debug) {
            Serial.println("[thermal] RIGHT sensor not found!");
        }
        return false;
    }

    status = _amgCenter.begin(0x69, _wireRight);
    if (!status) { 
        if (_debug) {
            Serial.println("[thermal] CENTER sensor not found!");
        }
        return false;
    }

    if (_debug) {
        Serial.println("[thermal] All sensors initialized successfully.");
    }

    return true;
}

void ThermalArrayManager::readRaw() {
    if (_debug) {
        Serial.println("[thermal] Reading raw pixel data...");
    }
    _amgLeft.readPixels(_pixelsLeft);
    _amgRight.readPixels(_pixelsRight);
    _amgCenter.readPixels(_pixelsCenter);
}

void ThermalArrayManager::readRotated() {
    readRaw();
    if (_debug) {
        Serial.println("[thermal] Rotating pixel data 270° clockwise...");
    }

    rotateMatrix(_pixelsLeft, _rotatedLeft, MatrixRotation::ROT_180_CW);
    rotateMatrix(_pixelsCenter, _rotatedCenter, MatrixRotation::ROT_180_CW);
    rotateMatrix(_pixelsRight, _rotatedRight, MatrixRotation::ROT_180_CW);
}

ThermalReadings ThermalArrayManager::getObject() {
    if (_debug) {
        Serial.println("[thermal] Getting pixel data as object...");
    }
    ThermalReadings r;
    memcpy(r.left, _rotatedLeft, sizeof(_rotatedLeft));
    memcpy(r.center, _rotatedCenter, sizeof(_rotatedCenter));
    memcpy(r.right, _rotatedRight, sizeof(_rotatedRight));
    return r;
}

String ThermalArrayManager::getJSON() {
    if (_debug) {
        Serial.println("[thermal] Getting pixel data as JSON...");
    }
    StaticJsonDocument<1024> doc;
    JsonArray left = doc.createNestedArray("left");
    JsonArray center = doc.createNestedArray("center");
    JsonArray right = doc.createNestedArray("right");

    for (int i = 0; i < 64; i++) {
        left.add(_rotatedLeft[i]);
        center.add(_rotatedCenter[i]);
        right.add(_rotatedRight[i]);
    }

    String output;
    serializeJson(doc, output);
    return output;
}
