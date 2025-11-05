#include <Arduino.h>
#include <driver/i2s.h>
#include <ArduinoJson.h>
#include <math.h>

// I2S pins
#define I2S_SCK  4
#define I2S_WS   5
#define I2S_SD   20  // Single SD line shared by both mics (stereo)

// I2S port
#define I2S_PORT I2S_NUM_0

// Buffer settings
#define SAMPLE_COUNT 128
int32_t i2sBuffer[SAMPLE_COUNT * 2]; // Stereo: L/R interleaved

// Low-pass smoothing
double prevL = 0;
double prevR = 0;
const double alpha = 0.2;

// ==== I2S Config ====
i2s_config_t makeI2SConfig() {
  i2s_config_t cfg = {};
  cfg.mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = 44100;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT; // Stereo
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = 0;
  cfg.dma_buf_count = 4;
  cfg.dma_buf_len = SAMPLE_COUNT;
  cfg.use_apll = false;
  return cfg;
}

i2s_pin_config_t makeI2SPins() {
  i2s_pin_config_t pins = {};
  pins.bck_io_num = I2S_SCK;
  pins.ws_io_num = I2S_WS;
  pins.data_out_num = -1;
  pins.data_in_num = I2S_SD;
  return pins;
}

// ==== Read RMS from interleaved stereo buffer ====
void computeStereoRMS(double &leftRMS, double &rightRMS, int32_t *buf, size_t len) {
  size_t samples = len * 2; // stereo interleaved
  double sumL = 0, sumR = 0;

  for (size_t i = 0; i < samples; i += 2) {
    int32_t vL = buf[i] >> 8;
    vL = (vL & 0x00800000) ? (vL | 0xFF000000) : (vL & 0x00FFFFFF);
    int32_t vR = buf[i + 1] >> 8;
    vR = (vR & 0x00800000) ? (vR | 0xFF000000) : (vR & 0x00FFFFFF);

    sumL += double(vL) * double(vL);
    sumR += double(vR) * double(vR);
  }

  leftRMS  = sqrt(sumL / len) / ((1 << 23) - 1);
  rightRMS = sqrt(sumR / len) / ((1 << 23) - 1);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("Stereo INMP441 Test");

  // Install I2S driver
  static i2s_config_t cfg = makeI2SConfig();
  static i2s_pin_config_t pins = makeI2SPins();
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
  i2s_start(I2S_PORT);

  Serial.println("I2S initialized in stereo mode.");
}

void loop() {
  size_t bytesRead = 0;
  esp_err_t res = i2s_read(I2S_PORT, i2sBuffer, SAMPLE_COUNT * 2 * sizeof(int32_t), &bytesRead, pdMS_TO_TICKS(50));
  if (res != ESP_OK || bytesRead == 0) {
    Serial.println("Read error or no data");
    delay(50);
    return;
  }

  double rmsL, rmsR;
  computeStereoRMS(rmsL, rmsR, i2sBuffer, SAMPLE_COUNT);

  // Low-pass filter for smoother output
  prevL = alpha * rmsL + (1 - alpha) * prevL;
  prevR = alpha * rmsR + (1 - alpha) * prevR;

  // JSON output
  StaticJsonDocument<128> doc;
  doc["micL"] = prevL * 100;
  doc["micR"] = prevR * 100;
  serializeJson(doc, Serial);
  Serial.println();

  delay(30); // ~33 Hz output
}
