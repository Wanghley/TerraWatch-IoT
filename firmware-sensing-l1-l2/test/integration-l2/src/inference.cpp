#include "inference.h"
#include <cmath>
#include <tensorflow/lite/micro/micro_error_reporter.h>

namespace Inference {

// TensorFlow Lite global variables
uint8_t g_tensor_arena[kTensorArenaSize];
tflite::ErrorReporter* g_error_reporter = nullptr;
const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input_tensor = nullptr;
TfLiteTensor* g_output_tensor = nullptr;

// Quantization params (will be set during setup)
float g_input_scale = 0.0f;
int8_t g_input_zero_point = 0;
float g_output_scale = 0.0f;
int8_t g_output_zero_point = 0;

// Replace these values with your Python StandardScaler mean/std
const float feature_mean[kNumFeatures] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

const float feature_std[kNumFeatures] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

// Helper: build input vector and quantize
void buildInputVector(const Features& f, int8_t* input_tensor_data) {
    float raw[kNumFeatures] = {
        f.doppler_speed,
        f.doppler_range,
        f.doppler_energy,
        f.thermal_centroid_y,
        f.thermal_centroid_x,
        f.thermal_vertical_diff,
        f.thermal_horizontal_diff,
        f.thermal_max,
        f.thermal_min,
        f.thermal_mean,
        f.thermal_std,
        f.mic_rms_mean,
        f.mic_peak_mean,
        f.mic_rms_samples_mean,
        f.mic_rms_samples_std,
        f.mic_rms_samples_min,
        f.mic_rms_samples_max
    };

    for (int i = 0; i < kNumFeatures; i++) {
        float scaled = (raw[i] - feature_mean[i]) / feature_std[i];
        input_tensor_data[i] = (int8_t)round(scaled / g_input_scale + g_input_zero_point);
    }
}

// Initialization
bool begin() {
    static tflite::MicroErrorReporter micro_error_reporter;
    g_error_reporter = &micro_error_reporter;

    // Load model
    g_model = tflite::GetModel(model);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        return false;
    }

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddLogistic(); // sigmoid
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();

    g_interpreter = new tflite::MicroInterpreter(
        g_model, resolver, g_tensor_arena, kTensorArenaSize, g_error_reporter);

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor allocation failed!");
        return false;
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);

    g_input_scale = g_input_tensor->params.scale;
    g_input_zero_point = g_input_tensor->params.zero_point;
    g_output_scale = g_output_tensor->params.scale;
    g_output_zero_point = g_output_tensor->params.zero_point;

    Serial.println("TFLite Micro setup complete.");
    return true;
}

// Run inference
float predict(const Features& f) {
    buildInputVector(f, g_input_tensor->data.int8);

    if (g_interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return -1.0f;
    }

    int8_t output_q = g_output_tensor->data.int8[0];
    float output = (output_q - g_output_zero_point) * g_output_scale;
    return output;
}

} // namespace Inference
