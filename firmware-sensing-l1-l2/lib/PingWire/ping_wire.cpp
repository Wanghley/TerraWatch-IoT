#include "ping_wire.h"

#define PULSE_ON_TIME_US   20000   // 20 ms
#define PULSE_OFF_GAP_US   20000   // 40 ms
#define TIMER_ID           0

SingleWirePulseManager* SingleWirePulseManager::_instance = nullptr;

SingleWirePulseManager::SingleWirePulseManager(uint8_t gpioPin)
: _gpioPin(gpioPin), _timer(nullptr), _busy(false), _pingCount(0), _pulsePhase(0) {}

void SingleWirePulseManager::begin() {
    pinMode(_gpioPin, OUTPUT);
    digitalWrite(_gpioPin, LOW);
    _instance = this;  // assign static instance

    _timer = timerBegin(TIMER_ID, 80, true); // 1 µs tick
    timerAttachInterrupt(_timer, &SingleWirePulseManager::onTimerISR, true);
    timerAlarmDisable(_timer);
}

void SingleWirePulseManager::setOutput(bool high) {
    digitalWrite(_gpioPin, high ? HIGH : LOW);
}

void IRAM_ATTR SingleWirePulseManager::onTimerISR() {
    if (_instance) _instance->handleTimerISR();
}

void SingleWirePulseManager::handleTimerISR() {
    if (!_busy) return;

    if (_pulsePhase == 0) {
        setOutput(true);
        _pulsePhase = 1;
        timerAlarmWrite(_timer, PULSE_ON_TIME_US, false);
    } 
    else if (_pulsePhase == 1) {
        setOutput(false);
        _pingCount--;
        if (_pingCount > 0) {
            _pulsePhase = 2;
            timerAlarmWrite(_timer, PULSE_OFF_GAP_US, false);
        } else {
            _busy = false;
            timerStop(_timer);
            return;
        }
    } 
    else if (_pulsePhase == 2) {
        _pulsePhase = 0;
        timerAlarmWrite(_timer, PULSE_ON_TIME_US, false);
    }

    timerWrite(_timer, 0);
    timerStart(_timer);
}

void SingleWirePulseManager::startPulseSequence(int numPings) {
    if (_busy) return;
    _busy = true;
    _pingCount = numPings;
    _pulsePhase = 0;

    timerStop(_timer);
    timerAlarmWrite(_timer, PULSE_ON_TIME_US, false);
    timerWrite(_timer, 0);
    timerStart(_timer);
}

void SingleWirePulseManager::sendSinglePing() {
    startPulseSequence(1);
}

void SingleWirePulseManager::sendDoublePing() {
    startPulseSequence(2);
}

bool SingleWirePulseManager::isBusy() const {
    return _busy;
}
