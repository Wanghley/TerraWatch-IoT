#ifndef MIC_H
#define MIC_H

#include <Arduino.h>

// API for microphone module
void mic_begin();
double mic_readRMS(); // returns RMS normalized 0..1
int mic_getPeak();     // optional: returns last peak value

#endif
