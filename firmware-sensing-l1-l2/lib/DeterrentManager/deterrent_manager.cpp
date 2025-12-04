/*
 * DeterrentManager.cpp
 * Implementation file for the non-blocking deterrent signal manager.
 */

#include "deterrent_manager.h"

DeterrentManager::DeterrentManager(int signalPin, bool debug, bool activeHigh)
    : _pin(signalPin),
      _stateStartTime(0),
      _debug(debug),
      _currentState(IDLE),
      _activeHigh(activeHigh) {
    if (_debug) Serial.println("DETERRENT: DeterrentManager created.");
}

void DeterrentManager::begin() {
    pinMode(_pin, OUTPUT);
    drive(false);          // ensure idle level
    _currentState = IDLE;
    if (_debug) Serial.println("DETERRENT: DeterrentManager initialized.");
}

bool DeterrentManager::isSignaling() {
    return _currentState != IDLE;
}

void DeterrentManager::signalSureDetection() {
    if (_currentState == IDLE) {
        _currentState = SURE_PULSE;
        _stateStartTime = millis();
        drive(true);
        if (_debug) Serial.println("DETERRENT: Sure detection signal started.");
    }
}

void DeterrentManager::signalUnsureDetection() {
    if (_currentState != IDLE) return;
    _currentState = UNSURE_PULSE_1;
    _stateStartTime = millis();
    drive(true);
    if (_debug) Serial.println("DETERRENT: Unsure detection sequence started (pulse 1).");
}

void DeterrentManager::update() {
    
    if (_currentState == IDLE) return;
    
    unsigned long now = millis();
    unsigned long elapsed = now - _stateStartTime;
    
    if (_debug) {
        static unsigned long lastDebugLog = 0;
        if (now - lastDebugLog > 100) {
            Serial.printf("DETERRENT: State=%d, Elapsed=%lu\n", (int)_currentState, elapsed);
            lastDebugLog = now;
        }
    }
    
    switch (_currentState) {
        case SURE_PULSE:
            if (elapsed >= PULSE_DURATION) {
                drive(false);
                _currentState = IDLE;
                if (_debug) Serial.println("DETERRENT: Sure pulse complete, returning to IDLE.");
            }
            break;
            
        case UNSURE_PULSE_1:
            if (elapsed >= PULSE_DURATION) {
                drive(false);
                _currentState = UNSURE_PAUSE;
                _stateStartTime = now;
                if (_debug) Serial.println("DETERRENT: Unsure pulse 1 complete, starting pause.");
            }
            break;
            
        case UNSURE_PAUSE:
            if (elapsed >= PAUSE_DURATION) {
                drive(true);
                _currentState = UNSURE_PULSE_2;
                _stateStartTime = now;
                if (_debug) Serial.println("DETERRENT: Pause complete, starting unsure pulse 2.");
            }
            break;
            
        case UNSURE_PULSE_2:
            if (elapsed >= PULSE_DURATION) {
                drive(false);
                _currentState = IDLE;
                if (_debug) Serial.println("DETERRENT: Unsure pulse 2 complete, returning to IDLE.");
            }
            break;
            
        case SURE_LATCH:
            // Stay driven high
            break;
            
        case UNSURE_LATCH:
            // Stay driven high
            break;
            
        default:
            break;
    }
}

void DeterrentManager::deactivate() {
    drive(false);
    _currentState = IDLE;
    if (_debug) Serial.println("DETERRENT: Deactivated and set to IDLE.");
}