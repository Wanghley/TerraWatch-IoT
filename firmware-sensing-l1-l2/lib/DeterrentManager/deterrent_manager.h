/*
 * DeterrentManager.h
 *
 * A non-blocking library to signal detection status on a single GPIO pin.
 * Uses a millis() state machine to avoid using delay().
 */

#pragma once
#include <Arduino.h>

class DeterrentManager {
public:
    /**
     * @brief Constructor
     * @param signalPin The GPIO pin to use for signaling.
     */
    DeterrentManager(int signalPin, bool debug = false, bool activeHigh = true);

    /**
     * @brief Initializes the pin. Call this in setup().
     */
    void begin();

    /**
     * @brief Call this in your main loop(). This manages the signal timing.
     */
    void update();

    /**
     * @brief Triggers a "Sure" detection signal (one 20ms pulse).
     * Will be ignored if another signal is already in progress.
     */
    void signalSureDetection();

    /**
     * @brief Triggers an "Unsure" detection signal (two 20ms pulses).
     * Will be ignored if another signal is already in progress.
     */
    void signalUnsureDetection();

    /**
     * @brief Checks if a signal is currently active.
     * @return true if the pin is in the middle of a signal, false if idle.
     */
    bool isSignaling();

    void deactivate(); // Immediately stop any signaling and set pin LOW.
    void setActiveHigh(bool ah) { _activeHigh = ah; }

    // NEW: latch/persistent control
    void enablePersistent(bool p) { _persistent = p; }
    void set(bool on) { drive(on); _currentState = on ? SURE_LATCH : IDLE; }

private:
    void drive(bool on) { digitalWrite(_pin, (_activeHigh ? (on ? HIGH : LOW) : (on ? LOW : HIGH))); }
    int _pin;
    unsigned long _stateStartTime;
    bool _debug;
    bool _activeHigh;
    bool _persistent = false;          // NEW

    // Durations in milliseconds
    const int PULSE_DURATION = 20;
    const int PAUSE_DURATION = 20;

    // State machine states
    enum State {
        IDLE,
        SURE_PULSE,
        UNSURE_PULSE_1,
        UNSURE_PAUSE,
        UNSURE_PULSE_2,
        SURE_LATCH,
        UNSURE_LATCH
    }; // NEW latched states

    State _currentState;
};