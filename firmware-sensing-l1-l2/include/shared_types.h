#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#include <Arduino.h>

// 1. Define Radar Packet Sub-struct
struct RadarPacket {
    float range_cm;
    float speed_ms;
    float energy;
    bool isValid;
    int numTargets; 
    unsigned long lastDetection; // Included to match your existing code
};

// 2. Define the Main Sensor Packet
struct SensorPacket {
    // Thermal Data (64 pixels * 3 sensors)
    float thermal_left[64];
    float thermal_center[64];
    float thermal_right[64];

    // Radar Data
    RadarPacket r1;
    RadarPacket r2;

    // Mic Data
    float micL;
    float micR;

    // Timestamp
    unsigned long timestamp;
};

#endif