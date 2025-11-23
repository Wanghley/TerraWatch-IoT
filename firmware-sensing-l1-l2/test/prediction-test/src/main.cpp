#include <Arduino.h>
#include <TensorFlowLite_ESP32.h> 

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h" // <--- Restored this
#include "tensorflow/lite/schema/schema_generated.h"
// Removed "micro_log.h" and "system_setup.h" to fix build error

#include "model.h"
#include "normalization_values.h"

// =============================================================
// 1. CONFIGURATION & TUNING
// =============================================================
// 120KB is usually enough for the model you trained.
const int kTensorArenaSize = 120 * 1024; 
uint8_t* tensor_arena = nullptr; 

const int SEQ_LEN = 198;
const int NUM_FEATURES = 200;

// HYSTERESIS: Prevents flickering. 
const float THRESHOLD_TRIGGER = 0.10f; // <--- DECISION THRESHOLD (Adjust as needed)
const float THRESHOLD_RESET   = 0.30f;

// PERSISTENT STATE (Survives Deep Sleep)
RTC_DATA_ATTR bool alarm_active = false;
RTC_DATA_ATTR float last_smooth_prob = 0.0f;
RTC_DATA_ATTR int consecutive_positives = 0;

// OUTPUT SMOOTHING: Exponential moving average to reduce jitter
const float OUTPUT_SMOOTHING = 0.3f; // 0.0-1.0, higher = less smoothing

// WARMUP: Initial iterations to let buffer stabilize
const int WARMUP_ITERATIONS = SEQ_LEN; // 198 frames at 10ms each = ~2 seconds
int iteration_count = 0;

// =============================================================
// 2. DATA STRUCTURES
// =============================================================
// Rolling Window Buffer
float sensor_history[SEQ_LEN][NUM_FEATURES]; 

// Pre-calculated inverse STD
float inv_std_vals[NUM_FEATURES];

// TFLite Objects
tflite::ErrorReporter* error_reporter = nullptr; // <--- Added back
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
tflite::AllOpsResolver resolver;

// =============================================================
// 3. HELPER FUNCTIONS
// =============================================================

void precompute_inverse_std() {
    for (int i = 0; i < NUM_FEATURES; i++) {
        if (STD_VALS[i] == 0) inv_std_vals[i] = 1.0f;
        else inv_std_vals[i] = 1.0f / STD_VALS[i];
    }
}

void update_rolling_window(float* new_frame_data) {
    // Shift everything back by 1 (Element-wise loop is faster on ESP32 for this size)
    for (int t = 0; t < SEQ_LEN - 1; t++) {
        memcpy(&sensor_history[t][0], &sensor_history[t + 1][0], NUM_FEATURES * sizeof(float));
    }

    // Copy new frame to the end
    memcpy(&sensor_history[SEQ_LEN - 1][0], new_frame_data, NUM_FEATURES * sizeof(float));
}

void copy_history_to_tensor() {
    float scale = input->params.scale;
    int zero_point = input->params.zero_point;
    bool is_quantized = (input->type == kTfLiteInt8);
    
    int tensor_idx = 0;

    for (int t = 0; t < SEQ_LEN; t++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            float raw = sensor_history[t][f];
            
            // FAST NORMALIZATION
            float norm = (raw - MEAN_VALS[f]) * inv_std_vals[f];

            if (is_quantized) {
                int32_t q = (int32_t)(norm / scale) + zero_point;
                if (q < -128) q = -128;
                if (q > 127) q = 127;
                input->data.int8[tensor_idx++] = (int8_t)q;
            } else {
                input->data.f[tensor_idx++] = norm;
            }
        }
    }
}

// =============================================================
// 4. SETUP
// =============================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("--- ESP32 Optimized AI Start ---");

  precompute_inverse_std();

  // 0. SETUP ERROR REPORTER
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // 1. ALLOCATE MEMORY
  tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) {
      Serial.println("PSRAM allocation failed, trying Internal RAM...");
      tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }
  if (!tensor_arena) {
      Serial.println("❌ Fatal: OOM"); while(1);
  }

  // 2. LOAD MODEL
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Error: Schema Mismatch"); while(1);
  }

  // 3. INIT INTERPRETER (With Error Reporter)
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
      Serial.println("Error: AllocateTensors failed"); while(1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✅ System Ready.");
}

// =============================================================
// 5. MAIN LOOP
// =============================================================
void loop() {
  unsigned long t_start = millis();

  // --- A. GENERATE REALISTIC SENSOR DATA ---
  float new_frame[NUM_FEATURES];
  
  if (iteration_count < WARMUP_ITERATIONS) {
      // WARMUP: Feed clean baseline data (no spikes)
      for(int i = 0; i < NUM_FEATURES; i++) {
          float center = MEAN_VALS[i];
          float noise = (random(-50, 50) / 100.0f) * STD_VALS[i]; // ±0.5 Sigma noise
          new_frame[i] = center + noise;
      }
  } else {
      // NORMAL: Mix of baseline + occasional anomalies
      for(int i = 0; i < NUM_FEATURES; i++) {
          float center = MEAN_VALS[i];
          
          // Base trend: smooth variation over time
          float trend = sin(iteration_count / 50.0f) * STD_VALS[i] * 0.3f;
          
          // Random noise
          float noise = (random(-100, 100) / 100.0f) * STD_VALS[i] * 0.8f;
          
          // Occasionally (2% chance) inject strong anomalies
          if (random(0, 100) < 2) {
              // 2-4 Sigma spike
              float spike_strength = (2.0f + (random(0, 20) / 10.0f));
              noise = (random(-1, 2) > 0 ? spike_strength : -spike_strength) * STD_VALS[i];
          }
          
          new_frame[i] = center + trend + noise;
      }
  }

  // --- B. UPDATE SLIDING WINDOW ---
  update_rolling_window(new_frame);

  // --- C. PREPARE TENSOR ---
  copy_history_to_tensor();

  // --- D. INFERENCE ---
  if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed"); 
      return;
  }

  // --- E. PROCESS OUTPUT WITH SMOOTHING ---
  float prob = 0.0;
  if (output->type == kTfLiteInt8) {
      prob = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
  } else {
      prob = output->data.f[0];
  }
  
  // Clamp probability to [0, 1]
  if (prob < 0.0f) prob = 0.0f;
  if (prob > 1.0f) prob = 1.0f;

  // Apply exponential moving average smoothing
  float smooth_prob = (OUTPUT_SMOOTHING * prob) + ((1.0f - OUTPUT_SMOOTHING) * last_smooth_prob);
  last_smooth_prob = smooth_prob;

  unsigned long duration = millis() - t_start;

  // if > THRE

  // --- F. APPLY HYSTERESIS & CONSECUTIVE TRIGGER ---
  bool is_above_threshold = (smooth_prob >= THRESHOLD_TRIGGER);

  if (is_above_threshold) {
      consecutive_positives++;
  } else {
      consecutive_positives = 0;
  }

  if (!alarm_active && consecutive_positives >= 2) {
      alarm_active = true;
      Serial.printf("🚨 ALARM TRIGGERED (2x Positive) | Prob: %.3f | Time: %lums\n", smooth_prob, duration);
  } else if (alarm_active && smooth_prob <= THRESHOLD_RESET) {
      alarm_active = false;
      consecutive_positives = 0; // Reset counter
      Serial.printf("✓ Alarm Reset | Prob: %.3f | Time: %lums\n", smooth_prob, duration);
  } else {
      Serial.printf("Prob: %.3f | Smooth: %.3f | Cons: %d | State: %s | Time: %lums\n", 
                    prob, smooth_prob, consecutive_positives, alarm_active ? "ON" : "OFF", duration);
  }
  
  iteration_count++;
  delay(10); // Run at ~100Hz (10ms)
}