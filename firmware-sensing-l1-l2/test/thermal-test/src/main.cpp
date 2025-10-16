#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_AMG88xx.h>

// I2C pins ESP32-S3
#define I2C_SDA 8
#define I2C_SCL 9

// Create an instance of the AMG88xx class
Adafruit_AMG88xx amg;

float pixels[AMG88xx_PIXEL_ARRAY_SIZE];

// put function declarations here:
int myFunction(int, int);

void setup() {
  Serial.begin(115200);
  // Initialize I2C with specified SDA and SCL pins
  Wire.begin(I2C_SDA, I2C_SCL);
  Serial.println(F("AMG8866 Test"));
  bool status;
  // default settings
  status = amg.begin();
  if (!status) {
    Serial.println("Could not find a valid AMG88xx sensor, check wiring!");
    while (1);
  }
  Serial.println(F("AMG8866 connected!"));
}

void loop() {
  amg.readPixels(pixels);
  Serial.print("[");
  for (int i = 0; i < AMG88xx_PIXEL_ARRAY_SIZE; i++) {
    Serial.print(pixels[i]);
    Serial.print(", ");
    if ((i + 1) % 8 == 0) Serial.println();
  }
  Serial.println("]");
  Serial.println();
  delay(15);
}

// put function definitions here:
int myFunction(int x, int y) {
  return x + y;
}