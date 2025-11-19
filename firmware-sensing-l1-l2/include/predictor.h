#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <Arduino.h>
#include "SharedTypes.h" // Access to SensorPacket
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include your model data
#include "model.h" 

class Predictor {
public:
    bool begin();
    // Returns -1.0 if buffering, or 0.0-1.0 if prediction made
    float update(const SensorPacket& pkt); 

private:
    // TFLite boilerplate
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    
    // 40KB Arena should be enough
    static const int kTensorArenaSize = 40 * 1024; 
    uint8_t tensor_arena[kTensorArenaSize];

    // Buffering for Time Series
    // 198 steps x 12 features (Must match Python training)
    static const int SEQ_LEN = 198; 
    static const int NUM_FEATS = 12;
    
    int buffer_index = 0;

    // Helper functions
    float get_max(const float* arr, int len);
    float get_mean(const float* arr, int len);
};

#endif