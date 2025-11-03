#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_AMG88xx.h>

// I2C pins
#define I2C0_SDA 8
#define I2C0_SCL 9
#define I2C1_SDA 16
#define I2C1_SCL 17

// AMG8833 sensor instances
Adafruit_AMG88xx amgLeft;
Adafruit_AMG88xx amgRight;
Adafruit_AMG88xx amgCenter;

// Pixel arrays
#define AMG88xx_PIXEL_ARRAY_SIZE 64
float pixelsLeft[AMG88xx_PIXEL_ARRAY_SIZE];
float pixelsRight[AMG88xx_PIXEL_ARRAY_SIZE];
float pixelsCenter[AMG88xx_PIXEL_ARRAY_SIZE];

// Rotate 8x8 block 270° clockwise
void rotate270CW(float* src, float* dst) {
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      dst[c * 8 + (7 - r)] = src[r * 8 + c];
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Starting 3x AMG8833 sensors...");

  Wire.begin(I2C0_SDA, I2C0_SCL, 400000);
  Wire1.begin(I2C1_SDA, I2C1_SCL, 400000);

  bool status;
  status = amgLeft.begin(0x68, &Wire);      if (!status) { Serial.println("LEFT sensor not found!"); while(1); }
  status = amgRight.begin(0x69, &Wire);     if (!status) { Serial.println("RIGHT sensor not found!"); while(1); }
  status = amgCenter.begin(0x69, &Wire1);   if (!status) { Serial.println("CENTER sensor not found!"); while(1); }

  Serial.println("All sensors initialized!");
}

void loop() {
  amgLeft.readPixels(pixelsLeft);
  amgRight.readPixels(pixelsRight);
  amgCenter.readPixels(pixelsCenter);

  float rotatedLeft[64], rotatedCenter[64], rotatedRight[64];
  rotate270CW(pixelsLeft, rotatedLeft);
  rotate270CW(pixelsCenter, rotatedCenter);
  rotate270CW(pixelsRight, rotatedRight);

  Serial.println("[");
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) Serial.print(rotatedLeft[row*8 + col], 2), Serial.print(", ");
    for (int col = 0; col < 8; col++) Serial.print(rotatedCenter[row*8 + col], 2), Serial.print(", ");
    for (int col = 0; col < 8; col++) {
      Serial.print(rotatedRight[row*8 + col], 2);
      if (!(row == 7 && col == 7)) Serial.print(", ");
    }
    Serial.println();
  }
  Serial.println("]");
  Serial.println();

  delay(100);
}