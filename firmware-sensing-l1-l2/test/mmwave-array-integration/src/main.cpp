#include <Arduino.h>
#include "DFRobot_C4001.h"

// ========================================
// CONFIGURATION PARAMETERS
// ========================================
#define RADAR1_RX 18
#define RADAR1_TX 17
#define RADAR2_RX 15
#define RADAR2_TX 16

#define MIN_RANGE 5        
#define MAX_RANGE 2000     
#define TRIG_RANGE 10     
#define KEEP_SENSITIVITY 6  
#define TRIG_SENSITIVITY 4  
#define LOOP_DELAY_MS 10    
#define MIN_ENERGY_THRESHOLD 50

// ========================================
// RADAR DATA STRUCTURE
// ========================================
struct RadarData {
    const char* name;
    int numTargets;
    float range_cm;
    float speed_ms;
    uint32_t energy;
    unsigned long lastDetection;
    bool isValid;
};

// ========================================
// HARDWARE SETUP
// ========================================
HardwareSerial RadarSerial1(1);
HardwareSerial RadarSerial2(2);

DFRobot_C4001_UART radar1(&RadarSerial1, 9600);
DFRobot_C4001_UART radar2(&RadarSerial2, 9600);

RadarData radar1Data = {"R1", 0, 0, 0, 0, 0, false};
RadarData radar2Data = {"R2", 0, 0, 0, 0, 0, false};

// ========================================
// HELPER FUNCTIONS
// ========================================
bool beginRadarWithTimeout(DFRobot_C4001_UART &radar, const char* name, unsigned long timeout = 5000) {
    unsigned long start = millis();
    while (!radar.begin()) {
        if (millis() - start > timeout) return false;
        delay(200);
    }
    return true;
}

bool configureRadar(DFRobot_C4001_UART &radar) {
    if (!radar.setSensorMode(eSpeedMode)) return false;
    if (!radar.setDetectThres(MIN_RANGE, MAX_RANGE, TRIG_RANGE)) return false;
    if (!radar.setKeepSensitivity(KEEP_SENSITIVITY)) return false;
    if (!radar.setTrigSensitivity(TRIG_SENSITIVITY)) return false;
    radar.setFrettingDetection(eON);
    return true;
}

void updateRadarData(DFRobot_C4001_UART &radar, RadarData &data) {
    data.numTargets = radar.getTargetNumber();
    if (data.numTargets > 0) {
        data.range_cm = radar.getTargetRange();
        data.speed_ms = radar.getTargetSpeed();
        data.energy = radar.getTargetEnergy();
        data.lastDetection = millis();
        data.isValid = (data.energy >= MIN_ENERGY_THRESHOLD);
    } else {
        data.isValid = false;
    }
}

// ========================================
// SETUP
// ========================================
void setup() {
    Serial.begin(115200);
    delay(1000);

    RadarSerial1.begin(9600, SERIAL_8N1, RADAR1_RX, RADAR1_TX);
    RadarSerial2.begin(9600, SERIAL_8N1, RADAR2_RX, RADAR2_TX);
    delay(100);

    while (!beginRadarWithTimeout(radar1, "R1") || !configureRadar(radar1));
    delay(200);
    while (!beginRadarWithTimeout(radar2, "R2") || !configureRadar(radar2));
}

// ========================================
// LOOP
// ========================================
void loop() {
    updateRadarData(radar1, radar1Data);
    updateRadarData(radar2, radar2Data);

    // Build JSON manually
    String json = "{";
    json += "\"R1\":{";
    json += "\"numTargets\":" + String(radar1Data.numTargets);
    json += ",\"range\":" + String(radar1Data.range_cm, 2);
    json += ",\"speed\":" + String(radar1Data.speed_ms, 2);
    json += ",\"energy\":" + String(radar1Data.energy);
    json += ",\"valid\":" + String(radar1Data.isValid ? "true" : "false");
    json += "},";
    json += "\"R2\":{";
    json += "\"numTargets\":" + String(radar2Data.numTargets);
    json += ",\"range\":" + String(radar2Data.range_cm, 2);
    json += ",\"speed\":" + String(radar2Data.speed_ms, 2);
    json += ",\"energy\":" + String(radar2Data.energy);
    json += ",\"valid\":" + String(radar2Data.isValid ? "true" : "false");
    json += "}";
    json += "}";

    Serial.println(json);

    delay(LOOP_DELAY_MS);
}