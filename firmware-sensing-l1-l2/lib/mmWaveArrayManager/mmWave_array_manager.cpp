#include "mmWave_array_manager.h"

mmWaveArrayManager::mmWaveArrayManager(uint8_t radar1RX, uint8_t radar1TX,
                                       uint8_t radar2RX, uint8_t radar2TX,
                                       bool debug)
    : _serial1(1), _serial2(2),
      _radar1(&_serial1, 9600), _radar2(&_serial2, 9600),
      _debug(debug)
{
    _radar1Data = {"R1", 0, 0, 0, 0, 0, false};
    _radar2Data = {"R2", 0, 0, 0, 0, 0, false};

    _serial1.begin(9600, SERIAL_8N1, radar1RX, radar1TX);
    _serial2.begin(9600, SERIAL_8N1, radar2RX, radar2TX);

    if (_debug) {
        Serial.println("[mmWaveArrayManager] Serial ports initialized");
    }
}

bool mmWaveArrayManager::beginRadarWithTimeout(DFRobot_C4001_UART &radar, const char* name, unsigned long timeout) {
    unsigned long start = millis();
    while (!radar.begin()) {
        if (_debug) {
            Serial.printf("[mmWave] Waiting for %s to initialize...\n", name);
        }
        if (millis() - start > timeout) {
            if (_debug) {
                Serial.printf("[mmWave] %s initialization timed out!\n", name);
            }
            return false;
        }
        delay(200);
    }
    if (_debug) {
        Serial.printf("[mmWave] %s initialized successfully!\n", name);
    }
    return true;
}

bool mmWaveArrayManager::configureRadar(DFRobot_C4001_UART &radar) {
    if (_debug) Serial.println("[mmWave] Configuring radar...");

    if (!radar.setSensorMode(eSpeedMode)) return false;
    if (!radar.setDetectThres(MIN_RANGE, MAX_RANGE, TRIG_RANGE)) return false;
    if (!radar.setKeepSensitivity(KEEP_SENSITIVITY)) return false;
    if (!radar.setTrigSensitivity(TRIG_SENSITIVITY)) return false;
    radar.setFrettingDetection(eON);

    if (_debug) Serial.println("[mmWave] Radar configured successfully!");
    return true;
}

bool mmWaveArrayManager::begin() {
    bool ok1 = beginRadarWithTimeout(_radar1, "R1") && configureRadar(_radar1);
    bool ok2 = beginRadarWithTimeout(_radar2, "R2") && configureRadar(_radar2);

    if (_debug) {
        Serial.printf("[mmWave] Radar1 status: %s\n", ok1 ? "OK" : "FAILED");
        Serial.printf("[mmWave] Radar2 status: %s\n", ok2 ? "OK" : "FAILED");
    }

    return ok1 && ok2;
}

void mmWaveArrayManager::updateRadarData(DFRobot_C4001_UART &radar, RadarData &data) {
    // Single call to getTargetNumber() - this caches all target data internally
    data.numTargets = radar.getTargetNumber();
    
    if (data.numTargets > 0) {
        // These calls are FAST - they just return cached values from getTargetNumber()
        data.range_cm = radar.getTargetRange();
        data.speed_ms = radar.getTargetSpeed();
        data.energy = radar.getTargetEnergy();
        data.lastDetection = millis();
        data.isValid = (data.energy >= MIN_ENERGY_THRESHOLD);
    } else {
        // No target detected
        data.range_cm = 0;
        data.speed_ms = 0;
        data.energy = 0;
        data.isValid = false;
    }
}

void mmWaveArrayManager::update() {
    // Read both radars - they use different UART ports so this happens in parallel
    // Radar1 reads while Radar2 reads (hardware level parallelism)
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
