#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <Arduino.h>
#include "shared_types.h" 

// TFLite Micro Includes
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model Architecture Constants
// SEQ_LEN is gone. We only process the "Now".
#define NUM_FEATURES 200 

class Predictor {
public:
    Predictor();
    bool begin(); 
    
    // Accepts a packet, returns probability [0.0 - 1.0]
    // Now returns a valid prediction immediately (no warmup).
    float update(SensorPacket pkt);

private:
    // TFLite pointers
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    tflite::AllOpsResolver resolver;

    // Memory Arena
    uint8_t* tensor_arena = nullptr;
    
    // Optimization: Inverse Standard Deviations for fast normalization
    float inv_std_vals[NUM_FEATURES];

    // State tracking
    float last_prediction = 0.0f;

    // Internal Helpers
    void precomputeInverseStd();
    void flattenPacket(SensorPacket& pkt, float* out_features);
    void copyFeaturesToTensor(float* features);
};

#endif