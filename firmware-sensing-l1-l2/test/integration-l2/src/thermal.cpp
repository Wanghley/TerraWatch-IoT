#include "thermal.h"

// Sensor object
Adafruit_AMG88xx amg;

bool setupThermalSensor() {
    Serial.println("Initializing AMG88xx thermal sensor...");

    // Initialize I2C
    Wire.begin(I2C_SDA, I2C_SCL);
    delay(100);

    // Initialize sensor
    if (!amg.begin()) {
        Serial.println("Could not find a valid AMG88xx sensor, check wiring!");
        return false;
    }

    Serial.println("AMG88xx sensor connected successfully!");
    return true;
}

ThermalFrame readThermalFrame() {
    ThermalFrame frame;
    frame.width = 8;
    frame.height = 8;

    // Read pixels from sensor
    amg.readPixels(frame.pixels);

    return frame;
}

void printThermalFrame(const ThermalFrame& frame) {
    Serial.println("[");
    for (int y = 0; y < frame.height; y++) {
        for (int x = 0; x < frame.width; x++) {
            int idx = y * frame.width + x;
            Serial.print(frame.pixels[idx], 2);  // two decimals
            if (x < frame.width - 1) Serial.print(", ");
        }
        Serial.println();
    }
    Serial.println("]");
    Serial.println();
}
