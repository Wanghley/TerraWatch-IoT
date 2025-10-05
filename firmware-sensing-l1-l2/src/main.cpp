/**
 * @file main.cpp
 * @author Wanghley Soares Martins (me@wanghley.com)
 * @brief AMG8833 reading test
 * @version 0.1
 * @date 2025-10-05
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_AMG88xx.h>

Adafruit_AMG88xx amg;

float pixels[AMG88xx_PIXEL_ARRAY_SIZE];
constexpr int MATRIX_SIZE = 8; // AMG8833 is 8x8
float pixels_matrix[MATRIX_SIZE][MATRIX_SIZE];

void setup_amg88833();
void get_amg8833_pixels();
void print_pixel_matrix_2_serial(char separator);

void setup() {
  Serial.begin(9600);
  setup_amg88833();
  delay(100);
}

void loop() {
  get_amg8833_pixels();
  print_pixel_matrix_2_serial(',');
  delay(1000);
}

/**
 * @brief Setup function for the AMG8833 sensor
 * @return void
 */
void setup_amg88833() {
    Serial.begin(9600);
    
    bool status;
    
    // default settings
    status = amg.begin(); // Address 0x69 default
    if (!status) {
        Serial.println("Could not find a valid AMG88xx sensor, check wiring!");
        while (1);
    }
    delay(100); // let sensor boot up
  }

void get_amg8833_pixels() {
    //read all the pixels
    amg.readPixels(pixels);

    // Convert 1D array to 2D matrix
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            pixels_matrix[i][j] = pixels[i * MATRIX_SIZE + j];
        }
    }
}

void print_pixel_matrix_2_serial(char separator = ',') {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            Serial.print(pixels_matrix[i][j], 5); // Print with 5 decimal places
            if (j < MATRIX_SIZE - 1) {
                Serial.print(separator); // Print separator between values
            }
        }
        Serial.println(); // New line after each row
    }
    Serial.println(); // Extra new line after the matrix
}