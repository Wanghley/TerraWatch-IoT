#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <Arduino.h>
#include "shared_types.h" // Must include this to know what SensorPacket is!

// TFLite Micro Includes
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model Architecture Constants
#define SEQ_LEN 198       // Time steps
#define NUM_FEATURES 200  // Features per step

class Predictor {
public:
    Predictor();
    bool begin(); // Returns true on success
    
    // Accepts a packet, returns probability [0.0 - 1.0]
    // Returns -1.0 if the buffer is still warming up.
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
    
    // Rolling History Buffer [Time][Features]
    float sensor_history[SEQ_LEN][NUM_FEATURES];
    
    // Optimization: Inverse Standard Deviations for fast normalization
    float inv_std_vals[NUM_FEATURES];

    // State tracking
    int iteration_count = 0;
    float last_prediction = 0.0f;

    // Internal Helpers
    void precomputeInverseStd();
    void flattenPacket(SensorPacket& pkt, float* out_features);
    void shiftAndAppend(float* new_features);
    void copyHistoryToTensor();
};

#endif