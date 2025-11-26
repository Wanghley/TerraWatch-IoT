#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <Arduino.h>
#include "shared_types.h"

// Forward declaration for TFLite classes to keep header clean
namespace tflite {
    class MicroErrorReporter;
    class MicroInterpreter;
    struct Model;
    // class MicroMutableOpResolver; // REMOVED: This is a template and causes errors if fwd declared without args.
}

class Predictor {
public:
    Predictor();
    ~Predictor();

    // Initialize TFLite interpreter
    bool begin();

    // Main inference function
    // Returns probability (0.0 to 1.0)
    float update(SensorPacket pkt);

private:
    // TFLite Pointers
    tflite::MicroErrorReporter* error_reporter;
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    
    // Memory pool for the model
    // 64KB is usually enough for this size model, adjust if needed
    uint8_t* tensor_arena;
    const int kArenaSize = 40 * 1024; 

    // Helpers
    void preprocessThermal(const float* left, const float* center, const float* right, int8_t* output_buffer, float scale, int zero_point);
    void preprocessScalars(const SensorPacket& pkt, int8_t* output_buffer, float scale, int zero_point);
};

#endif