#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#include <Arduino.h>

// ====== RADAR DATA STRUCTURE ======
struct RadarReading {
    float range_cm;
    float speed_ms;
    float energy;
    unsigned long lastDetection;
    bool isValid;
};

// ====== SENSOR PACKET STRUCTURE ======
struct SensorPacket {
    float thermal_left[64];
    float thermal_center[64];
    float thermal_right[64];
    RadarReading r1;
    RadarReading r2;
    double micL;
    double micR;
};

#endif