#include <driver/i2s.h>
#include <Arduino.h>
#include <math.h>

#define I2S_WS   37  
#define I2S_SD   38 
#define I2S_SCK  36 
#define I2S_PORT I2S_NUM_0
#define SAMPLE_COUNT 128

int32_t i32Buffer[SAMPLE_COUNT];

i2s_config_t makeI2SConfig() {
  i2s_config_t cfg = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 44100,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // hook L/R pin to ground (left)
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = SAMPLE_COUNT,
    .use_apll = false
  };
  return cfg;
}

i2s_pin_config_t makeI2SPins() {
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK,
    .ws_io_num  = I2S_WS,
    .data_out_num = -1,
    .data_in_num  = I2S_SD
  };
  return pins;
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("I2S 24-bit test starting...");

  i2s_config_t cfg = makeI2SConfig();
  esp_err_t r = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  Serial.print("i2s_driver_install: "); Serial.println(r);
  i2s_pin_config_t pins = makeI2SPins();
  r = i2s_set_pin(I2S_PORT, &pins);
  Serial.print("i2s_set_pin: "); Serial.println(r);
  r = i2s_start(I2S_PORT);
  Serial.print("i2s_start: "); Serial.println(r);

  delay(200);
}

void loop() {
  size_t bytesRead = 0;
  size_t bytesToRead = SAMPLE_COUNT * sizeof(int32_t);
  esp_err_t res = i2s_read(I2S_PORT, (void*)i32Buffer, bytesToRead, &bytesRead, pdMS_TO_TICKS(200));
  if (res != ESP_OK || bytesRead < 4) {
    Serial.println("0");
    delay(50);
    return;
  }

  int samples = bytesRead / 4;
  double sumsq = 0.0;
  int32_t peak = 0;

  for (int i = 0; i < samples; ++i) {
    int32_t v = (int32_t)(i32Buffer[i] >> 8); // decode A
    v = (v & 0x00800000) ? (v | 0xFF000000) : (v & 0x00FFFFFF);
    if (abs(v) > peak) peak = abs(v);
    sumsq += (double)v * (double)v;
  }

  double rms = sqrt(sumsq / samples);
  const double MAX24 = (double)((1 << 23) - 1);
  double rms_norm = rms / MAX24;

  // for Serial Plotter:
  // Optionally multiply so plot is more visible: e.g. *1.0 for 0..1, *100 for 0..100
  Serial.print(0); // freeze lower limit
  Serial.print(" ");
  Serial.print(20); // freeze upper limit
  Serial.print(" ");
  Serial.println(rms_norm * 100.0);

  delay(30); // faster refresh for plotter responsiveness
}
