#pragma once

#include <Arduino.h>
#include "feature_extractor.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

#include "model_movement.h"

namespace Inference {

constexpr int kTensorArenaSize = 20 * 1024; // 20KB Arena, tune if needed
extern uint8_t g_tensor_arena[kTensorArenaSize];

// Number of features
constexpr int kNumFeatures = 17;

// Setup / initialization
bool begin();

// Predict function: returns output probability (0..1)
float predict(const Features& f);

// Mean and std arrays from Python StandardScaler (replace with your actual values)
extern const float feature_mean[kNumFeatures];
extern const float feature_std[kNumFeatures];

} // namespace Inference
