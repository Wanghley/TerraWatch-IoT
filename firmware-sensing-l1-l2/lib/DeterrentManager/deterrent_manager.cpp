/*
 * DeterrentManager.cpp
 * Implementation file for the non-blocking deterrent signal manager.
 */

#include "deterrent_manager.h"

DeterrentManager::DeterrentManager(int signalPin, bool debug)
    : _pin(signalPin),
      _stateStartTime(0),
      _debug(debug),
      _currentState(IDLE) {
        if (_debug) {
            Serial.println("DETERRENT: DeterrentManager created.");
        }
}

void DeterrentManager::begin() {
    pinMode(_pin, OUTPUT);
    digitalWrite(_pin, LOW);
    _currentState = IDLE;
    if (_debug) {
        Serial.println("DETERRENT: DeterrentManager initialized.");
    }
}

bool DeterrentManager::isSignaling() {
    return _currentState != IDLE;
}

void DeterrentManager::signalSureDetection() {
    // Only start a new signal if we are idle
    if (_currentState == IDLE) {
        _currentState = SURE_PULSE;
        _stateStartTime = millis();
        digitalWrite(_pin, HIGH); // Start the pulse
        if (_debug) {
            Serial.println("DETERRENT: Sure detection signal started.");
        }
    }
}

void DeterrentManager::signalUnsureDetection() {
    // Only start a new signal if we are idle
    if (_currentState == IDLE) {
        _currentState = UNSURE_PULSE_1;
        _stateStartTime = millis();
        digitalWrite(_pin, HIGH); // Start the first pulse
        if (_debug) {
            Serial.println("DETERRENT: Unsure detection signal started.");
        }
    }
}

void DeterrentManager::update() {
    // If we are idle, there's nothing to do.
    if (_currentState == IDLE) {
        return;
    }

    unsigned long currentTime = millis();
    unsigned long elapsedTime = currentTime - _stateStartTime;

    // Run the state machine
    switch (_currentState) {

        case SURE_PULSE:
            // This is a "Sure" detection (one 20ms pulse)
            if (elapsedTime >= PULSE_DURATION) {
                digitalWrite(_pin, LOW); // End the pulse
                _currentState = IDLE;    // Go back to idle
            }
            break;

        case UNSURE_PULSE_1:
            // This is the first pulse of an "Unsure" detection
            if (elapsedTime >= PULSE_DURATION) {
                digitalWrite(_pin, LOW);            // End the first pulse
                _currentState = UNSURE_PAUSE;       // Move to the pause state
                _stateStartTime = currentTime;      // Reset the timer for the pause
            }
            break;

        case UNSURE_PAUSE:
            // This is the pause between "Unsure" pulses
            if (elapsedTime >= PAUSE_DURATION) {
                digitalWrite(_pin, HIGH);           // Start the second pulse
                _currentState = UNSURE_PULSE_2;     // Move to the second pulse state
                _stateStartTime = currentTime;      // Reset the timer for the pulse
            }
            break;

        case UNSURE_PULSE_2:
            // This is the second pulse of an "Unsure" detection
            if (elapsedTime >= PULSE_DURATION) {
                digitalWrite(_pin, LOW); // End the second pulse
                _currentState = IDLE;    // Go back to idle
            }
            break;
        
        case IDLE:
        default:
            // Should not be here, but good to have a default
            break;
    }
}