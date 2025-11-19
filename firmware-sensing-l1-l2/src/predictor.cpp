#include "predictor.h"

// --- PASTE YOUR PYTHON VALUES HERE ---
// These are placeholders. Replace them!
const float MEAN_VALS[] = { 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 2.5, 1.0, 2.5, 1.0, 0.002, 0.002 };
const float STD_VALS[]  = { 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 0.5, 0.5, 0.5, 0.5, 0.001, 0.001 };
// -------------------------------------

bool Predictor::begin() {
    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema error!");
        return false;
    }

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddLogistic();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMaxPool2D(); // Needed for GlobalMaxPooling
    resolver.AddMean(); 

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed!");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    return true;
}

float Predictor::get_max(const float* arr, int len) {
    float m = arr[0];
    for(int i=1; i<len; i++) if(arr[i] > m) m = arr[i];
    return m;
}

float Predictor::get_mean(const float* arr, int len) {
    float sum = 0;
    for(int i=0; i<len; i++) sum += arr[i];
    return sum / len;
}

float Predictor::update(const SensorPacket& pkt) {
    // 1. Extract Features (12 total)
    float features[NUM_FEATS];

    // Thermal (Max, Mean)
    features[0] = get_max(pkt.thermal_left, 64);
    features[1] = get_mean(pkt.thermal_left, 64);
    features[2] = get_max(pkt.thermal_center, 64);
    features[3] = get_mean(pkt.thermal_center, 64);
    features[4] = get_max(pkt.thermal_right, 64);
    features[5] = get_mean(pkt.thermal_right, 64);

    // Radar (Log1p, Range)
    features[6] = log(pkt.r1.energy + 1.0);
    features[7] = pkt.r1.range_cm;
    features[8] = log(pkt.r2.energy + 1.0);
    features[9] = pkt.r2.range_cm;

    // Mic
    features[10] = (float)pkt.micL;
    features[11] = (float)pkt.micR;

    // 2. Normalize & Quantize into the Input Tensor
    // We fill the tensor one timestep at a time
    // The Tensor acts as our circular buffer
    
    // Shift buffer logic (Simple approach: Fill linear)
    // NOTE: The model expects [198, 12]. 
    // Ideally, we use a circular buffer, but TFLite tensor is flat.
    // For simplicity: We fill it up. Once full, we predict, then reset.
    
    if (buffer_index >= SEQ_LEN) {
        // Buffer is full, we should have predicted already or we reset
        buffer_index = 0; 
    }

    for (int f = 0; f < NUM_FEATS; f++) {
        float norm = (features[f] - MEAN_VALS[f]) / STD_VALS[f];
        int8_t q = (int8_t)(norm / input->params.scale + input->params.zero_point);
        
        int idx = (buffer_index * NUM_FEATS) + f;
        input->data.int8[idx] = q;
    }
    
    buffer_index++;
    
    // 3. Check if ready to predict
    if (buffer_index == SEQ_LEN) {
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Inference Error");
            buffer_index = 0;
            return -1.0;
        }
        
        // Reset buffer for next batch (Overlapping window logic is harder, this is Batch logic)
        buffer_index = 0; 

        // Dequantize output
        int8_t out = output->data.int8[0];
        float prob = (out - output->params.zero_point) * output->params.scale;
        return prob;
    }

    return -1.0; // Not ready
}