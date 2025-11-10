#pragma once
#include <Arduino.h>
#include "DFRobot_C4001.h"
#include <ArduinoJson.h>

struct RadarData {
    const char* name;
    int numTargets;
    float range_cm;
    float speed_ms;
    uint32_t energy;
    unsigned long lastDetection;
    bool isValid;
};

class mmWaveArrayManager {
public:
    mmWaveArrayManager(uint8_t radar1RX, uint8_t radar1TX,
                       uint8_t radar2RX, uint8_t radar2TX,
                       bool debug = false);

    bool begin();                     // Initialize both radars with debug info
    void update();                     // Read and update radar data
    RadarData getRadar1();
    RadarData getRadar2();
    String getJSON();

private:
    HardwareSerial _serial1;
    HardwareSerial _serial2;
    DFRobot_C4001_UART _radar1;
    DFRobot_C4001_UART _radar2;
    RadarData _radar1Data;
    RadarData _radar2Data;

    bool _debug;

    bool beginRadarWithTimeout(DFRobot_C4001_UART &radar, const char* name, unsigned long timeout = 5000);
    bool configureRadar(DFRobot_C4001_UART &radar);
    void updateRadarData(DFRobot_C4001_UART &radar, RadarData &data);

    static constexpr int MIN_RANGE = 5;
    static constexpr int MAX_RANGE = 2000;
    static constexpr int TRIG_RANGE = 10;
    static constexpr int KEEP_SENSITIVITY = 6;
    static constexpr int TRIG_SENSITIVITY = 4;
    static constexpr uint32_t MIN_ENERGY_THRESHOLD = 50;
};
