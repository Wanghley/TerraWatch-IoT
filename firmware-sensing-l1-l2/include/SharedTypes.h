#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

// Define your radar data struct if it's not defined elsewhere
struct RadarData {
    float energy;
    bool isValid;
    uint32_t lastDetection;
    bool isValid;
};

struct SensorPacket {
    float thermal_left[64];
    float thermal_center[64];
    float thermal_right[64];
    RadarData r1; // Left
    RadarData r2; // Right
    double micL;
    double micR;
};

#endif