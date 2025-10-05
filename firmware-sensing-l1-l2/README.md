# Level 1 & 2 Sensing - ESP32 Firmware

## Overview
Motion detection and ML-based classification using ESP32 microcontroller.

## Hardware
- ESP32 DevKit
- PIR Motion Sensor
- AMG8833 Thermal Array (8x8)
- Microphone
- mmWave Radar

## Features
- Level 1: Motion detection trigger
- Level 2: Multi-sensor fusion and ML classification
- MQTT communication to Level 3

## Getting Started
```bash
pio run -t upload
pio device monitor
```

## Testing
See `test/` directory for individual sensor tests.
