#pragma once
#include <Arduino.h>
#include "thermal_array_manager.h"
#include "mmWave_array_manager.h"

// Forward declare TFLite types
namespace tflite {
    class Model;
    class MicroInterpreter;
    class MicroAllocator; // <-- ADD THIS
}
struct TfLiteTensor;

class InferenceManager {
public:
    InferenceManager(bool debug = false);
    bool begin();
    
    // ... (rest of public methods are fine) ...
    void addFrame(const ThermalReadings& thermal, 
                  const RadarData& r1, 
                  const RadarData& r2,
                  double leftMic, 
                  double rightMic);
    int predict();
    float getHumanConfidence() { return human_confidence; }
    float getAnimalConfidence() { return animal_confidence; }
    float getInferenceTimeMs() { return inference_time_ms; }
    bool isReady() { return buffer_filled; }
    
private:
    // ... (constants are fine) ...
    static constexpr int SEQ_LENGTH = 10;
    static constexpr int THERMAL_CHANNELS = 3;
    static constexpr int THERMAL_H = 8;
    static constexpr int THERMAL_W = 8;
    static constexpr int RADAR_SIZE = 12;
    static constexpr int kTensorArenaSize = 80 * 1024;
    
    bool debug;
    bool buffer_filled;
    int buffer_index;
    
    // Sequence buffers
    float thermal_buffer[SEQ_LENGTH][THERMAL_CHANNELS][THERMAL_H][THERMAL_W];
    float radar_buffer[SEQ_LENGTH][RADAR_SIZE];
    
    // TFLite components
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
    const tflite::Model* model;
    
    tflite::MicroAllocator* micro_allocator; // <-- ADD THIS POINTER
    
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_thermal;
    TfLiteTensor* input_radar;
    TfLiteTensor* output;
    
    // ... (rest of private members/methods are fine) ...
    float human_confidence;
    float animal_confidence;
    float inference_time_ms;
    
    void normalizeFrame(float* data, int size);
    void reshapeThermal(const ThermalReadings& thermal, float output[3][8][8]);
};