#include "inference_manager.h"

// ===== ADD THESE INCLUDES =====
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h" // <-- ADD
#include "tensorflow/lite/micro/micro_allocator.h"       // <-- ADD
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "human_animal_model_data.h" 
#include <math.h>   
#include <string.h> 
// ================================


InferenceManager::InferenceManager(bool debug) 
    : debug(debug), buffer_filled(false), buffer_index(0),
      model(nullptr), 
      micro_allocator(nullptr), // <-- INITIALIZE NEW POINTER
      interpreter(nullptr),
      input_thermal(nullptr), input_radar(nullptr), output(nullptr),
      human_confidence(0), animal_confidence(0), inference_time_ms(0) {
    // Constructor body is empty, which is fine
}

bool InferenceManager::begin() {
    if (debug) {
        Serial.println("\n=== Initializing ML Inference ===");
    }

    // 1. Error reporter
    static tflite::MicroErrorReporter micro_error_reporter;

    // 2. Load model
    model = tflite::GetModel(human_animal_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        if (debug) {
            // use Serial.printf instead of micro_error_reporter.Report with variadic args
            Serial.printf("Model schema mismatch. Got %d, expected %d\n",
                          model->version(), TFLITE_SCHEMA_VERSION);
        }
        return false;
    }
    if (debug) {
        Serial.println("✓ Model loaded");
    }

    // 3. Ops resolver
    static tflite::MicroMutableOpResolver<15> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddConcatenation();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddPad();
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddPack();
    micro_op_resolver.AddTranspose();
    micro_op_resolver.AddShape();

    if (debug) {
        Serial.println("✓ Op resolver configured");
    }

    // 4. Build Interpreter directly with tensor_arena (common TF Lite Micro constructor)
    //    This avoids the MicroAllocator constructor mismatch.
    interpreter = new tflite::MicroInterpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter
    );
    if (!interpreter) {
        if (debug) Serial.println("Failed to create MicroInterpreter");
        return false;
    }

    // 5. Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        if (debug) {
            Serial.println("✗ AllocateTensors() failed");
        }
        return false;
    }

    if (debug) {
        Serial.println("✓ Tensors allocated");
    }

    // 6. Get input/output tensors
    input_thermal = interpreter->input(0);
    input_radar = interpreter->input(1);
    output = interpreter->output(0);

    if (debug) {
        Serial.printf("✓ ML Model ready!\n");
        Serial.printf("  Arena used: %d / %d bytes\n",
                      interpreter->arena_used_bytes(), kTensorArenaSize);
        Serial.println("=================================\n");
    }

    return true;
}

void InferenceManager::reshapeThermal(const ThermalReadings& thermal, 
                                       float output[3][8][8]) {
    // Reshape from 64-element arrays to 8x8 matrices
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        output[0][row][col] = thermal.left[i];
        output[1][row][col] = thermal.center[i];
        output[2][row][col] = thermal.right[i];
    }
}

void InferenceManager::normalizeFrame(float* data, int size) {
    // Calculate mean and std
    float mean = 0;
    for (int i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= size;
    
    float variance = 0;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    float std = sqrt(variance / size) + 1e-6;
    
    // Normalize
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

void InferenceManager::addFrame(const ThermalReadings& thermal,
                                const RadarData& r1,
                                const RadarData& r2,
                                double leftMic,
                                double rightMic) {
    // 1. Reshape and store thermal data
    float thermal_3d[3][8][8];
    reshapeThermal(thermal, thermal_3d);

    // Normalize and copy
    normalizeFrame(&thermal_3d[0][0][0], 3 * 8 * 8);
    memcpy(thermal_buffer[buffer_index], thermal_3d,
           THERMAL_CHANNELS * THERMAL_H * THERMAL_W * sizeof(float));

    // 2. Build and store radar+mic vector (explicit casts to avoid narrowing)
    float radar_mic[12] = {
        static_cast<float>(r1.numTargets),
        static_cast<float>(r1.range_cm),
        static_cast<float>(r1.speed_ms),
        static_cast<float>(r1.energy),
        r1.isValid ? 1.0f : 0.0f,
        static_cast<float>(r2.numTargets),
        static_cast<float>(r2.range_cm),
        static_cast<float>(r2.speed_ms),
        static_cast<float>(r2.energy),
        r2.isValid ? 1.0f : 0.0f,
        static_cast<float>(leftMic),
        static_cast<float>(rightMic)
    };

    normalizeFrame(radar_mic, 12);
    memcpy(radar_buffer[buffer_index], radar_mic, RADAR_SIZE * sizeof(float));

    // 3. Update buffer state
    buffer_index++;
    if (buffer_index >= SEQ_LENGTH) {
        buffer_index = 0;
        buffer_filled = true;
    }
}

int InferenceManager::predict() {
    if (!buffer_filled) {
        return 2; // Not enough data
    }
    
    // Copy sequence buffers to input tensors in chronological order
    for (int i = 0; i < SEQ_LENGTH; i++) {
        int src_idx = (buffer_index + i) % SEQ_LENGTH;
        
        // Copy thermal: [seq, channels, h, w]
        int thermal_offset = i * THERMAL_CHANNELS * THERMAL_H * THERMAL_W;
        memcpy(input_thermal->data.f + thermal_offset,
               thermal_buffer[src_idx],
               THERMAL_CHANNELS * THERMAL_H * THERMAL_W * sizeof(float));
        
        // Copy radar: [seq, features]
        int radar_offset = i * RADAR_SIZE;
        memcpy(input_radar->data.f + radar_offset,
               radar_buffer[src_idx],
               RADAR_SIZE * sizeof(float));
    }
    
    // Run inference
    unsigned long start = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long elapsed = micros() - start;
    inference_time_ms = elapsed / 1000.0;
    
    if (invoke_status != kTfLiteOk) {
        if (debug) {
            Serial.println("✗ Invoke failed");
        }
        return -1;
    }
    
    // Get predictions
    human_confidence = output->data.f[0];
    animal_confidence = output->data.f[1];
    
    if (debug) {
        Serial.printf("Inference: %.2f ms | Human: %.3f | Animal: %.3f\n",
                     inference_time_ms, human_confidence, animal_confidence);
    }
    
    return (animal_confidence > human_confidence) ? 1 : 0;
}