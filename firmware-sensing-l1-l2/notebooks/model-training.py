import torch
from torch.utils.data import Dataset, DataLoader
import os, json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

# =====================================================
# STEP 1: Dataset (unchanged)
# =====================================================
class HumanAnimalSeqDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.sessions = []
        self.labels = []
        self.augment = augment

        for subdir in os.listdir(root_dir):
            full_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(full_path):
                continue
            label = 1 if "animal" in subdir.lower() else 0
            seq = []
            for f in os.listdir(full_path):
                if f.endswith(".jsonl"):
                    with open(os.path.join(full_path, f), "r") as jf:
                        for line in jf:
                            data = json.loads(line)
                            left = np.array(data["thermal"]["left"]).reshape(8, 8)
                            center = np.array(data["thermal"]["center"]).reshape(8, 8)
                            right = np.array(data["thermal"]["right"]).reshape(8, 8)
                            thermal = np.stack([left, center, right])
                            thermal = (thermal - np.mean(thermal)) / (np.std(thermal)+1e-6)
                            r1 = data["mmWave"]["R1"]
                            r2 = data["mmWave"]["R2"]
                            radar = np.array([
                                r1["numTargets"], r1["range"], r1["speed"], r1["energy"], float(r1["valid"]),
                                r2["numTargets"], r2["range"], r2["speed"], r2["energy"], float(r2["valid"])
                            ])
                            mic = np.array([data["mic"]["left"], data["mic"]["right"]])
                            radar_mic = np.concatenate([radar, mic])
                            radar_mic = (radar_mic - np.mean(radar_mic)) / (np.std(radar_mic)+1e-6)
                            seq.append((thermal, radar_mic))
            if seq:
                self.sessions.append(seq)
                self.labels.append(label)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        seq = self.sessions[idx]
        thermal_seq = torch.tensor(np.array([s[0] for s in seq]), dtype=torch.float32)
        radar_seq   = torch.tensor(np.array([s[1] for s in seq]), dtype=torch.float32)

        if self.augment:
            thermal_seq += torch.randn_like(thermal_seq) * 0.02
            for i in range(thermal_seq.shape[0]):
                axis = np.random.choice([1,2])
                shift = np.random.randint(-1,2)
                thermal_seq[i] = torch.roll(thermal_seq[i], shifts=shift, dims=axis)
            radar_seq += torch.randn_like(radar_seq) * 0.02

        thermal_seq = (thermal_seq - thermal_seq.mean()) / (thermal_seq.std()+1e-6)
        radar_seq   = (radar_seq - radar_seq.mean()) / (radar_seq.std()+1e-6)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return thermal_seq, radar_seq, label


# =====================================================
# STEP 2: Simple PyTorch Model
# =====================================================
class SimplePyTorchModel(nn.Module):
    """Simplified model for PyTorch training"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.flat_size = 8 * 4 * 4  # 128
        combined_size = self.flat_size + 12  # 140
        
        # Simple feedforward instead of RNN
        self.fc1 = nn.Linear(combined_size, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, thermal_seq, radar_seq):
        batch_size, seq_len, C, H, W = thermal_seq.shape
        
        # Process each frame
        thermal_seq = thermal_seq.view(batch_size*seq_len, C, H, W)
        x = F.relu(self.conv1(thermal_seq))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Combine with radar
        x = torch.cat([x, radar_seq], dim=2)
        
        # Average over sequence
        x = x.mean(dim=1)
        
        # Feedforward layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


# =====================================================
# STEP 3: TensorFlow Keras Model (Direct Build)
# =====================================================
class TimeDistributedCNN(tf.keras.layers.Layer):
    """Custom layer to apply CNN to each timestep"""
    def __init__(self):
        super(TimeDistributedCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(4, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(2, 2)
        self.flatten = tf.keras.layers.Flatten()
    
    def call(self, inputs):
        # Input shape: (batch, seq_len, 3, 8, 8) in NCHW format
        # Need to convert to NHWC and process each timestep
        
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Permute from NCHW to NHWC: (batch, seq, C, H, W) -> (batch, seq, H, W, C)
        x = tf.transpose(inputs, [0, 1, 3, 4, 2])
        
        # Merge batch and sequence: (batch*seq, H, W, C)
        x = tf.reshape(x, [-1, 8, 8, 3])
        
        # Apply CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)  # Now (batch*seq, 128)
        
        # Reshape back to sequence: (batch, seq, 128)
        x = tf.reshape(x, [batch_size, seq_len, 128])
        
        return x


def build_keras_model():
    """Build equivalent model directly in TensorFlow/Keras"""
    
    # Input layers - note: using NCHW format to match PyTorch
    thermal_input = tf.keras.Input(shape=(None, 3, 8, 8), name='thermal_input')
    radar_input = tf.keras.Input(shape=(None, 12), name='radar_input')
    
    # Apply CNN to thermal sequence
    thermal_features = TimeDistributedCNN()(thermal_input)
    
    # Concatenate with radar data
    combined = tf.keras.layers.Concatenate(axis=2)([thermal_features, radar_input])
    
    # Average pooling over time
    x = tf.keras.layers.GlobalAveragePooling1D()(combined)
    
    # Fully connected layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=[thermal_input, radar_input], outputs=output)
    return model


# =====================================================
# STEP 4: Train PyTorch Model
# =====================================================
def train_pytorch_model():
    """Train PyTorch model and return it"""
    full_dataset = HumanAnimalSeqDataset('../tools/dataset')

    num_samples = len(full_dataset)
    indices = torch.randperm(num_samples).tolist()
    split_idx = int(0.8 * num_samples)
    train_indices = indices[:split_idx]
    val_indices   = indices[split_idx:]

    train_ds = torch.utils.data.Subset(
        HumanAnimalSeqDataset('../tools/dataset', augment=True), train_indices
    )
    val_ds = torch.utils.data.Subset(
        HumanAnimalSeqDataset('../tools/dataset', augment=False), val_indices
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplePyTorchModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    epochs = 30

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for thermal_seq, radar_seq, labels in train_loader:
            thermal_seq = thermal_seq.to(device)
            radar_seq   = radar_seq.to(device)
            labels      = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(thermal_seq, radar_seq)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/total:.4f}, Acc: {100*correct/total:.2f}%")

    # Validation
    model.eval()
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for thermal_seq, radar_seq, labels in val_loader:
            thermal_seq = thermal_seq.to(device)
            radar_seq   = radar_seq.to(device)
            labels      = labels.to(device)
            outputs = model(thermal_seq, radar_seq)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
    
    print(f"\nValidation Accuracy: {100*correct/total:.2f}%")
    print(f"Validation Loss: {loss/total:.4f}")
    
    return model, train_loader


# =====================================================
# STEP 5: Transfer Weights PyTorch -> Keras
# =====================================================
def transfer_weights(pytorch_model, keras_model):
    """Transfer weights from PyTorch to Keras model"""
    
    print("\nTransferring weights from PyTorch to Keras...")
    
    pytorch_model.eval()
    pytorch_model.cpu()
    
    # Get PyTorch weights
    state_dict = pytorch_model.state_dict()
    
    # Find the TimeDistributedCNN layer
    time_dist_cnn = None
    for layer in keras_model.layers:
        if isinstance(layer, TimeDistributedCNN):
            time_dist_cnn = layer
            break
    
    if time_dist_cnn is None:
        raise ValueError("Could not find TimeDistributedCNN layer")
    
    # Transfer CNN weights
    # Conv1: PyTorch OIHW (out, in, h, w) -> TF HWIO (h, w, in, out)
    time_dist_cnn.conv1.set_weights([
        state_dict['conv1.weight'].permute(2, 3, 1, 0).numpy(),
        state_dict['conv1.bias'].numpy()
    ])
    
    # Conv2
    time_dist_cnn.conv2.set_weights([
        state_dict['conv2.weight'].permute(2, 3, 1, 0).numpy(),
        state_dict['conv2.bias'].numpy()
    ])
    
    # Find Dense layers
    dense_layers = [l for l in keras_model.layers if isinstance(l, tf.keras.layers.Dense)]
    
    # FC1 (32 units)
    dense_layers[0].set_weights([
        state_dict['fc1.weight'].t().numpy(),  # Transpose for row-major vs col-major
        state_dict['fc1.bias'].numpy()
    ])
    
    # FC2 (2 units - output)
    dense_layers[1].set_weights([
        state_dict['fc2.weight'].t().numpy(),
        state_dict['fc2.bias'].numpy()
    ])
    
    print("✓ Weights transferred successfully!")


# =====================================================
# STEP 6: Convert to TFLite
# =====================================================
def convert_keras_to_tflite(keras_model, output_name='model'):
    """Convert Keras model to TFLite"""
    
    print("\nConverting Keras model to TFLite...")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    tflite_path = f'{output_name}.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ Saved TFLite model: {tflite_path}")
    print(f"  Size: {len(tflite_model) / 1024:.2f} KB")
    
    # Convert to C array
    convert_to_c_array(tflite_model, output_name)
    
    return tflite_path


def convert_to_c_array(tflite_model, output_name):
    """Convert TFLite model to C array for embedding in ESP32 code"""
    
    c_array_path = f'{output_name}_data.h'
    
    with open(c_array_path, 'w') as f:
        f.write(f"// Auto-generated TFLite model data\n")
        f.write(f"#ifndef {output_name.upper()}_DATA_H\n")
        f.write(f"#define {output_name.upper()}_DATA_H\n\n")
        f.write(f"const unsigned int {output_name}_tflite_len = {len(tflite_model)};\n")
        f.write(f"alignas(8) const unsigned char {output_name}_tflite[] = {{\n  ")
        
        # Write bytes in rows of 12
        for i, byte in enumerate(tflite_model):
            f.write(f"0x{byte:02x}")
            if i < len(tflite_model) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0 and i < len(tflite_model) - 1:
                f.write("\n  ")
        
        f.write("\n};\n\n")
        f.write(f"#endif // {output_name.upper()}_DATA_H\n")
    
    print(f"✓ Saved C array header: {c_array_path}")


# =====================================================
# STEP 7: Generate ESP32 Arduino Code
# =====================================================
def generate_esp32_arduino_code(model_name='model'):
    """Generate complete Arduino sketch for ESP32"""
    
    arduino_code = f'''/*
 * ESP32 Human/Animal Detection using TensorFlow Lite
 * Auto-generated code
 */

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "{model_name}_data.h"

// Model parameters
constexpr int kTensorArenaSize = 80 * 1024;  // 80KB tensor arena
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// TFLite globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_thermal = nullptr;
TfLiteTensor* input_radar = nullptr;
TfLiteTensor* output = nullptr;

// Sequence buffer
constexpr int SEQ_LENGTH = 10;
constexpr int THERMAL_CHANNELS = 3;
constexpr int THERMAL_H = 8;
constexpr int THERMAL_W = 8;
constexpr int RADAR_SIZE = 12;

float thermal_buffer[SEQ_LENGTH][THERMAL_CHANNELS][THERMAL_H][THERMAL_W];
float radar_buffer[SEQ_LENGTH][RADAR_SIZE];
int buffer_index = 0;
bool buffer_filled = false;

void setup() {{
  Serial.begin(115200);
  delay(2000);
  while (!Serial) delay(10);
  
  Serial.println("\\n=================================");
  Serial.println("ESP32 TFLite Human/Animal Detector");
  Serial.println("=================================\\n");
  
  // Load model
  model = tflite::GetModel({model_name}_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {{
    Serial.printf("Model schema mismatch. Got %d, expected %d\\n", 
                  model->version(), TFLITE_SCHEMA_VERSION);
    while(1) delay(100);
  }}
  Serial.println("✓ Model loaded");
  
  // Set up ops resolver
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
  
  Serial.println("✓ Op resolver configured");
  
  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize
  );
  interpreter = &static_interpreter;
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {{
    Serial.println("✗ AllocateTensors() failed");
    while(1) delay(100);
  }}
  
  Serial.println("✓ Tensors allocated");
  
  // Get input/output tensors
  input_thermal = interpreter->input(0);
  input_radar = interpreter->input(1);
  output = interpreter->output(0);
  
  Serial.printf("✓ Model ready!\\n");
  Serial.printf("  Arena used: %d / %d bytes\\n", 
                interpreter->arena_used_bytes(), kTensorArenaSize);
  Serial.printf("  Thermal input shape: [");
  for (int i = 0; i < input_thermal->dims->size; i++) {{
    Serial.printf("%d", input_thermal->dims->data[i]);
    if (i < input_thermal->dims->size - 1) Serial.print(", ");
  }}
  Serial.println("]");
  Serial.printf("  Radar input shape: [");
  for (int i = 0; i < input_radar->dims->size; i++) {{
    Serial.printf("%d", input_radar->dims->data[i]);
    if (i < input_radar->dims->size - 1) Serial.print(", ");
  }}
  Serial.println("]\\n");
  
  Serial.println("Waiting for sensor data...\\n");
}}

void addSensorData(float thermal_data[THERMAL_CHANNELS][THERMAL_H][THERMAL_W], 
                   float* radar_data) {{
  // Copy new data to buffers
  memcpy(thermal_buffer[buffer_index], thermal_data, 
         THERMAL_CHANNELS * THERMAL_H * THERMAL_W * sizeof(float));
  memcpy(radar_buffer[buffer_index], radar_data, RADAR_SIZE * sizeof(float));
  
  buffer_index++;
  if (buffer_index >= SEQ_LENGTH) {{
    buffer_index = 0;
    buffer_filled = true;
  }}
}}

int runInference() {{
  if (!buffer_filled) {{
    return -2;  // Not enough data yet
  }}
  
  // Copy sequence buffers to input tensors
  // Arrange in chronological order
  for (int i = 0; i < SEQ_LENGTH; i++) {{
    int src_idx = (buffer_index + i) % SEQ_LENGTH;
    
    // Copy thermal data: [seq, channels, h, w]
    int thermal_offset = i * THERMAL_CHANNELS * THERMAL_H * THERMAL_W;
    memcpy(input_thermal->data.f + thermal_offset,
           thermal_buffer[src_idx],
           THERMAL_CHANNELS * THERMAL_H * THERMAL_W * sizeof(float));
    
    // Copy radar data: [seq, features]
    int radar_offset = i * RADAR_SIZE;
    memcpy(input_radar->data.f + radar_offset,
           radar_buffer[src_idx],
           RADAR_SIZE * sizeof(float));
  }}
  
  // Run inference
  unsigned long start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long elapsed = micros() - start;
  
  if (invoke_status != kTfLiteOk) {{
    Serial.println("✗ Invoke failed");
    return -1;
  }}
  
  // Get prediction
  float human_prob = output->data.f[0];
  float animal_prob = output->data.f[1];
  
  Serial.printf("Inference time: %.2f ms\\n", elapsed / 1000.0);
  Serial.printf("Probabilities - Human: %.3f | Animal: %.3f\\n", 
                human_prob, animal_prob);
  
  return (animal_prob > human_prob) ? 1 : 0;
}}

void loop() {{
  // TODO: Replace with actual sensor readings
  // Example structure for thermal data
  float thermal_data[THERMAL_CHANNELS][THERMAL_H][THERMAL_W];
  float radar_data[RADAR_SIZE];
  
  // Simulate sensor readings (REPLACE THIS WITH REAL SENSOR CODE)
  for (int c = 0; c < THERMAL_CHANNELS; c++) {{
    for (int h = 0; h < THERMAL_H; h++) {{
      for (int w = 0; w < THERMAL_W; w++) {{
        thermal_data[c][h][w] = (random(2000, 3000) / 100.0);  // 20-30°C
      }}
    }}
  }}
  
  for (int i = 0; i < RADAR_SIZE; i++) {{
    radar_data[i] = random(0, 100) / 100.0;
  }}
  
  // Add to buffer
  addSensorData(thermal_data, radar_data);
  
  // Run inference when buffer is full
  if (buffer_filled) {{
    int prediction = runInference();
    
    if (prediction == 0) {{
      Serial.println(">>> 👤 HUMAN DETECTED <<<\\n");
    }} else if (prediction == 1) {{
      Serial.println(">>> 🐾 ANIMAL DETECTED <<<\\n");
    }}
  }} else {{
    Serial.printf("Buffering data... %d/%d frames\\n", buffer_index, SEQ_LENGTH);
  }}
  
  delay(100);  // 10Hz sampling rate
}}
'''
    
    with open('esp32_inference.ino', 'w') as f:
        f.write(arduino_code)
    
    print(f"✓ Generated Arduino sketch: esp32_inference.ino")


# =====================================================
# MAIN EXECUTION
# =====================================================
def main():
    print("="*60)
    print("ESP32 TensorFlow Lite Deployment (No ONNX)")
    print("="*60)
    
    # Step 1: Train PyTorch model
    print("\n[1/5] Training PyTorch model...")
    pytorch_model, train_loader = train_pytorch_model()
    
    # Step 2: Build Keras model
    print("\n[2/5] Building equivalent Keras model...")
    keras_model = build_keras_model()
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    keras_model.summary()
    
    # Step 3: Transfer weights
    print("\n[3/5] Transferring weights...")
    transfer_weights(pytorch_model, keras_model)
    
    # Step 4: Convert to TFLite
    print("\n[4/5] Converting to TFLite...")
    tflite_path = convert_keras_to_tflite(keras_model, output_name='human_animal_model')
    
    # Step 5: Generate Arduino code
    print("\n[5/5] Generating ESP32 code...")
    generate_esp32_arduino_code('human_animal_model')
    
    # Final instructions
    print("\n" + "="*60)
    print("✓ DEPLOYMENT READY!")
    print("="*60)
    print("\n📋 ESP32 Setup Instructions:")
    print("\n1. Install Arduino IDE library:")
    print("   - TensorFlowLite_ESP32")
    print("   - Library Manager → Search 'TensorFlowLite_ESP32'")
    print("   - Or: https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32")
    
    print("\n2. Copy to Arduino sketch folder:")
    print("   ✓ human_animal_model_data.h")
    print("   ✓ esp32_inference.ino")
    
    print("\n3. Arduino IDE settings:")
    print("   - Board: ESP32 Dev Module")
    print("   - Partition: Huge APP (3MB)")
    print("   - Upload Speed: 921600")
    print("   - Flash Frequency: 80MHz")
    
    print("\n4. Modify sensor interface:")
    print("   - Replace dummy data in loop()")
    print("   - Add thermal camera reading")
    print("   - Add mmWave radar reading")
    print("   - Add microphone reading")
    
    print("\n5. Upload & test!")
    
    print(f"\n📊 Model Stats:")
    if os.path.exists(tflite_path):
        print(f"   - TFLite size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
    print(f"   - Tensor arena: 80 KB")
    print(f"   - Est. total RAM: ~100 KB")
    print(f"   - ESP32 has: 520 KB SRAM ✓")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have: pip install torch tensorflow")