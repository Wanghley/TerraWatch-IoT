#include "predictor.h"
#include "model.h"                 
#include "normalization_values.h" 

// Arena size: Reduced slightly as we don't need the massive input tensor buffer anymore.
// However, Dense layers can be heavy on weights. 
const int kTensorArenaSize = 60 * 1024; 

// SMOOTHING IS IMPORTANT: 
// Since we predict per-frame, the output might jitter. 
// 0.7 means we keep 70% of the previous prediction (inertia).
const float OUTPUT_SMOOTHING = 0.7f; 

Predictor::Predictor() {}

bool Predictor::begin() {
    Serial.println("[AI] Initializing Instant Predictor...");

    precomputeInverseStd();

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // --- FIX 2: ROBUST MEMORY ALLOCATION LOGGING ---
    // Try PSRAM first
    tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    
    if (tensor_arena) {
        Serial.printf("[AI] ✅ Allocated %d bytes in PSRAM\n", kTensorArenaSize);
    } else {
        Serial.println("[AI] ⚠️ PSRAM allocation failed/not available. Trying Internal RAM...");
        tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        
        if (tensor_arena) {
            Serial.printf("[AI] ✅ Allocated %d bytes in Internal RAM\n", kTensorArenaSize);
        } else {
            Serial.println("[AI] ❌ Fatal: Could not allocate Tensor Arena!");
            return false;
        }
    }

    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[AI] ❌ Schema Mismatch. Model: %d, Supported: %d\n", 
            model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // --- FIX 3: SAFE INTERPRETER INIT ---
    // Using 'static' here is okay, but ensure the resolver is ready.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[AI] ❌ AllocateTensors failed.");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Validate Input Shape
    if (input->dims->data[1] != NUM_FEATURES) {
        Serial.printf("[AI] ❌ Input dimension mismatch! Model expects %d, Code uses %d\n", 
                      input->dims->data[1], NUM_FEATURES);
        return false;
    }

    Serial.println("[AI] Predictor Initialized Successfully.");
    return true;
}

void Predictor::precomputeInverseStd() {
    for (int i = 0; i < NUM_FEATURES; i++) {
        if (STD_VALS[i] <= 0.00001f) inv_std_vals[i] = 1.0f;
        else inv_std_vals[i] = 1.0f / STD_VALS[i];
    }
}

// MATCHING PYTHON EXTRACT_FEATURES EXACTLY
void Predictor::flattenPacket(SensorPacket& pkt, float* f) {
    int idx = 0;

    // 1. Thermal (192)
    memcpy(&f[idx], pkt.thermal_left, 64 * sizeof(float)); idx += 64;
    memcpy(&f[idx], pkt.thermal_center, 64 * sizeof(float)); idx += 64;
    memcpy(&f[idx], pkt.thermal_right, 64 * sizeof(float)); idx += 64;

    // 2. Radar 1 (3 Features: Energy, Range, Speed)
    // Ensure your Struct has these fields.
    f[idx++] = pkt.r1.energy;   
    f[idx++] = pkt.r1.range_cm; 
    f[idx++] = pkt.r1.speed_ms;

    // 3. Radar 2 (3 Features)
    f[idx++] = pkt.r2.energy;
    f[idx++] = pkt.r2.range_cm;
    f[idx++] = pkt.r2.speed_ms;
    
    // 4. Mic (2 Features) - Added this to match Python
    f[idx++] = pkt.micL;
    f[idx++] = pkt.micR;
}

void Predictor::copyFeaturesToTensor(float* features) {
    float scale = input->params.scale;
    int zero_point = input->params.zero_point;
    bool is_quantized = (input->type == kTfLiteInt8);
    
    for (int f = 0; f < NUM_FEATURES; f++) {
        // Normalize
        float raw = features[f];
        float norm = (raw - MEAN_VALS[f]) * inv_std_vals[f];

        // Quantize
        if (is_quantized) {
            int32_t q = (int32_t)(norm / scale) + zero_point;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            input->data.int8[f] = (int8_t)q;
        } else {
            input->data.f[f] = norm;
        }
    }
}

float Predictor::update(SensorPacket pkt) {
    float features[NUM_FEATURES];
    
    // 1. Extract Features from struct
    flattenPacket(pkt, features);

    // 2. Fill Tensor directly (No history buffer logic needed)
    copyFeaturesToTensor(features);

    // 3. Run Inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[AI] Invoke failed!");
        return last_prediction; // Return last known good state
    }

    // 4. Read Output
    float raw_prob = 0.0f;
    if (output->type == kTfLiteInt8) {
        raw_prob = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
    } else {
        raw_prob = output->data.f[0];
    }

    // 5. EMA Smoothing (Low Pass Filter)
    // This is very important for frame-by-frame models to reduce noise
    last_prediction = (OUTPUT_SMOOTHING * last_prediction) + ((1.0f - OUTPUT_SMOOTHING) * raw_prob);
    
    return last_prediction;
}