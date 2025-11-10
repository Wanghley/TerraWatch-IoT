#include "mmWave_array_manager.h"

mmWaveArrayManager::mmWaveArrayManager(uint8_t radar1RX, uint8_t radar1TX,
                                       uint8_t radar2RX, uint8_t radar2TX)
    : _serial1(1), _serial2(2),
      _radar1(&_serial1, 9600), _radar2(&_serial2, 9600)
{
    _radar1Data = {"R1", 0, 0, 0, 0, 0, false};
    _radar2Data = {"R2", 0, 0, 0, 0, 0, false};

    _serial1.begin(9600, SERIAL_8N1, radar1RX, radar1TX);
    _serial2.begin(9600, SERIAL_8N1, radar2RX, radar2TX);
}

bool mmWaveArrayManager::beginRadarWithTimeout(DFRobot_C4001_UART &radar, const char* name, unsigned long timeout) {
    unsigned long start = millis();
    while (!radar.begin()) {
        if (millis() - start > timeout) return false;
        delay(200);
    }
    return true;
}

bool mmWaveArrayManager::configureRadar(DFRobot_C4001_UART &radar) {
    if (!radar.setSensorMode(eSpeedMode)) return false;
    if (!radar.setDetectThres(MIN_RANGE, MAX_RANGE, TRIG_RANGE)) return false;
    if (!radar.setKeepSensitivity(KEEP_SENSITIVITY)) return false;
    if (!radar.setTrigSensitivity(TRIG_SENSITIVITY)) return false;
    radar.setFrettingDetection(eON);
    return true;
}

bool mmWaveArrayManager::begin() {
    if (!beginRadarWithTimeout(_radar1, "R1") || !configureRadar(_radar1)) return false;
    if (!beginRadarWithTimeout(_radar2, "R2") || !configureRadar(_radar2)) return false;
    return true;
}

void mmWaveArrayManager::updateRadarData(DFRobot_C4001_UART &radar, RadarData &data) {
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

void mmWaveArrayManager::update() {
    updateRadarData(_radar1, _radar1Data);
    updateRadarData(_radar2, _radar2Data);
}

RadarData mmWaveArrayManager::getRadar1() {
    return _radar1Data;
}

RadarData mmWaveArrayManager::getRadar2() {
    return _radar2Data;
}

String mmWaveArrayManager::getJSON() {
    StaticJsonDocument<512> doc;

    JsonObject r1 = doc.createNestedObject("R1");
    r1["numTargets"] = _radar1Data.numTargets;
    r1["range"] = _radar1Data.range_cm;
    r1["speed"] = _radar1Data.speed_ms;
    r1["energy"] = _radar1Data.energy;
    r1["valid"] = _radar1Data.isValid;

    JsonObject r2 = doc.createNestedObject("R2");
    r2["numTargets"] = _radar2Data.numTargets;
    r2["range"] = _radar2Data.range_cm;
    r2["speed"] = _radar2Data.speed_ms;
    r2["energy"] = _radar2Data.energy;
    r2["valid"] = _radar2Data.isValid;

    String output;
    serializeJson(doc, output);
    return output;
}
