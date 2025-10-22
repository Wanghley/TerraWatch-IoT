#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_AMG88xx.h>

// Left and right thermal sensors
Adafruit_AMG88xx amgLeft = Adafruit_AMG88xx();
Adafruit_AMG88xx amgRight = Adafruit_AMG88xx();

#define LEFT_SENSOR_ADDR 0x68
#define RIGHT_SENSOR_ADDR 0x69

// I2C pins
#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9

float leftPixels[AMG88xx_PIXEL_ARRAY_SIZE];
float rightPixels[AMG88xx_PIXEL_ARRAY_SIZE];

void setupThermalSensors()
{
    Serial.println("Initializing thermal sensors...");
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

    if (!amgLeft.begin(LEFT_SENSOR_ADDR))
    {
        Serial.println("Could not find left AMG88xx sensor!");
        while (1)
            ;
    }
    Serial.println("Left AMG88xx sensor initialized.");

    if (!amgRight.begin(RIGHT_SENSOR_ADDR))
    {
        Serial.println("Could not find right AMG88xx sensor!");
        while (1)
            ;
    }
    Serial.println("Right AMG88xx sensor initialized.");
}

void readThermalSensors()
{
    amgLeft.readPixels(leftPixels);
    amgRight.readPixels(rightPixels);
}

float *getLeftThermalData()
{
    return leftPixels;
}

float *getRightThermalData()
{
    return rightPixels;
}