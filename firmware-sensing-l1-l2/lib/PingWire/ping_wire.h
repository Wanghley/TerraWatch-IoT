#ifndef PING_WIRE_H
#define PING_WIRE_H

#include <Arduino.h>

class SingleWirePulseManager {
public:
    explicit SingleWirePulseManager(uint8_t gpioPin);
    void begin();
    void sendSinglePing();
    void sendDoublePing();
    bool isBusy() const;

private:
    static void IRAM_ATTR onTimerISR(); // static ISR
    void handleTimerISR();              // instance handler
    void startPulseSequence(int numPings);
    void setOutput(bool high);

    uint8_t _gpioPin;
    hw_timer_t* _timer;
    volatile bool _busy;
    volatile int _pingCount;
    volatile int _pulsePhase;

    static SingleWirePulseManager* _instance; // static self-pointer
};

#endif // PING_WIRE_H
