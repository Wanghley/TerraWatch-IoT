#include "predictor.h"
#include "model.h"                 // Your converted TFLite model array
#include "normalization_values.h"  // Your MEAN_VALS and STD_VALS arrays

// Arena size: 120KB is usually safe for this size model. 
// If you crash on allocation, increase to 130 * 1024.
const int kTensorArenaSize = 120 * 1024; 
const float OUTPUT_SMOOTHING = 0.6f; // Low-pass filter on output (0.0 = no smoothing)

Predictor::Predictor() {}

bool Predictor::begin() {
    Serial.println("[AI] Initializing TensorFlow Lite Micro...");

    precomputeInverseStd();

    // 1. Set up logging
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // 2. Allocate memory (Try PSRAM first for ESP32-S3/WROVER, fallback to internal)
    tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        Serial.println("[AI] Warning: PSRAM allocation failed, trying Internal RAM...");
        tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (!tensor_arena) {
        Serial.println("[AI] ❌ Fatal: Could not allocate Tensor Arena.");
        return false;
    }

    // 3. Load Model
    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[AI] ❌ Schema Mismatch.");
        return false;
    }

    // 4. Init Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // 5. Allocate Tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[AI] ❌ AllocateTensors failed.");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Clear history buffer
    memset(sensor_history, 0, sizeof(sensor_history));

    return true;
}

void Predictor::precomputeInverseStd() {
    // Avoid division by zero at runtime
    for (int i = 0; i < NUM_FEATURES; i++) {
        if (STD_VALS[i] <= 0.00001f) inv_std_vals[i] = 1.0f;
        else inv_std_vals[i] = 1.0f / STD_VALS[i];
    }
}

// Converts struct to flat array [ThermalL(64), ThermalC(64), ThermalR(64), R1(4), R2(4)]
void Predictor::flattenPacket(SensorPacket& pkt, float* f) {
    int idx = 0;

    // 1. Thermal Left
    memcpy(&f[idx], pkt.thermal_left, 64 * sizeof(float));
    idx += 64;

    // 2. Thermal Center
    memcpy(&f[idx], pkt.thermal_center, 64 * sizeof(float));
    idx += 64;

    // 3. Thermal Right
    memcpy(&f[idx], pkt.thermal_right, 64 * sizeof(float));
    idx += 64;

    // 4. Radar 1 (4 Features)
    // IMPORTANT: Match the order used in Python training
    f[idx++] = (float)pkt.r1.numTargets; 
    f[idx++] = pkt.r1.range_cm;
    f[idx++] = pkt.r1.speed_ms;
    f[idx++] = pkt.r1.energy;

    // 5. Radar 2 (4 Features)
    f[idx++] = (float)pkt.r2.numTargets;
    f[idx++] = pkt.r2.range_cm;
    f[idx++] = pkt.r2.speed_ms;
    f[idx++] = pkt.r2.energy;
    
    // Total should be 200
}

void Predictor::shiftAndAppend(float* new_features) {
    // Shift rows back by 1 (Discard oldest)
    // Using memmove is safer for overlapping regions, but manual loop is clear for 2D arrays
    for (int t = 0; t < SEQ_LEN - 1; t++) {
        memcpy(sensor_history[t], sensor_history[t + 1], NUM_FEATURES * sizeof(float));
    }
    // Copy new frame to the last row
    memcpy(sensor_history[SEQ_LEN - 1], new_features, NUM_FEATURES * sizeof(float));
}

void Predictor::copyHistoryToTensor() {
    float scale = input->params.scale;
    int zero_point = input->params.zero_point;
    bool is_quantized = (input->type == kTfLiteInt8);
    
    int tensor_idx = 0;

    for (int t = 0; t < SEQ_LEN; t++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            float raw = sensor_history[t][f];
            
            // Normalize: (Val - Mean) / Std
            float norm = (raw - MEAN_VALS[f]) * inv_std_vals[f];

            if (is_quantized) {
                // Quantize to int8
                int32_t q = (int32_t)(norm / scale) + zero_point;
                // Clamp
                if (q < -128) q = -128;
                if (q > 127) q = 127;
                input->data.int8[tensor_idx++] = (int8_t)q;
            } else {
                input->data.f[tensor_idx++] = norm;
            }
        }
    }
}

float Predictor::update(SensorPacket pkt) {
    float features[NUM_FEATURES];
    
    // 1. Flatten Packet
    flattenPacket(pkt, features);

    // 2. Add to History
    shiftAndAppend(features);
    iteration_count++;

    // 3. Warmup Check
    // If the buffer isn't full of real data yet, don't predict.
    if (iteration_count < SEQ_LEN) {
        return -1.0f; // Signal "Not Ready"
    }

    // 4. Prepare Input Tensor
    copyHistoryToTensor();

    // 5. Run Inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[AI] Invoke failed!");
        return -1.0f;
    }

    // 6. Read Output
    float raw_prob = 0.0f;
    if (output->type == kTfLiteInt8) {
        raw_prob = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
    } else {
        raw_prob = output->data.f[0];
    }

    // Clamp
    if (raw_prob < 0.0f) raw_prob = 0.0f;
    if (raw_prob > 1.0f) raw_prob = 1.0f;

    // 7. Smooth Output (EMA)
    last_prediction = (OUTPUT_SMOOTHING * raw_prob) + ((1.0f - OUTPUT_SMOOTHING) * last_prediction);
    
    return last_prediction;
}