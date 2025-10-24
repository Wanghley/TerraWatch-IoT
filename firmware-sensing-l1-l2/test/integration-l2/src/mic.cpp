#include "mic.h"
#include <driver/i2s.h>
#include <math.h>

// I2S pin and config
#define I2S_WS   5  
#define I2S_SD   4 
#define I2S_SCK  6 
#define I2S_PORT I2S_NUM_0
#define SAMPLE_COUNT 128

static int32_t i32Buffer[SAMPLE_COUNT];
static int lastPeak = 0;

static i2s_config_t makeI2SConfig() {
    i2s_config_t cfg = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 44100,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = 0,
        .dma_buf_count = 4,
        .dma_buf_len = SAMPLE_COUNT,
        .use_apll = false
    };
    return cfg;
}

static i2s_pin_config_t makeI2SPins() {
    i2s_pin_config_t pins = {
        .bck_io_num = I2S_SCK,
        .ws_io_num  = I2S_WS,
        .data_out_num = -1,
        .data_in_num  = I2S_SD
    };
    return pins;
}

// Initialize I2S microphone
void mic_begin() {
    Serial.println("Initializing I2S mic...");
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

// Read I2S mic and return RMS normalized 0..1
double mic_readRMS() {
    size_t bytesRead = 0;
    size_t bytesToRead = SAMPLE_COUNT * sizeof(int32_t);

    esp_err_t res = i2s_read(I2S_PORT, (void*)i32Buffer, bytesToRead, &bytesRead, pdMS_TO_TICKS(200));
    if (res != ESP_OK || bytesRead < 4) return 0.0;

    int samples = bytesRead / 4;
    double sumsq = 0.0;
    lastPeak = 0;

    for (int i = 0; i < samples; ++i) {
        int32_t v = (int32_t)(i32Buffer[i] >> 8);
        v = (v & 0x00800000) ? (v | 0xFF000000) : (v & 0x00FFFFFF);
        if (abs(v) > lastPeak) lastPeak = abs(v);
        sumsq += (double)v * (double)v;
    }

    double rms = sqrt(sumsq / samples);
    const double MAX24 = (double)((1 << 23) - 1);
    return rms / MAX24;
}

// Optional: get last peak value
int mic_getPeak() {
    return lastPeak;
}