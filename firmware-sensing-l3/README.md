# Level 3 Sensing - Orange Pi Vision System

## Overview
CNN-based image analysis for animal classification using Orange Pi 4A.

## Hardware
- Orange Pi 4A (8GB RAM)
- High-resolution Camera Module: Arducam IMX708 with resolution 2304 (width) x 1296 (height). 
- WiFi Client for communication

## Features
- Camera capture on trigger from L1/L2
- CNN-based animal classification
- Decision making for deterrence activation
- Data logging and analytics

## General Design
- We will have a main script which calls functions that are encapsulated into separate CNN and Capture scripts.
- In the main script, on L1/L2 trigger, we call a capture() function from capture.py which saves an image from the camera module to the captures folder. 
- In the main script, after capture() returns and the image is saved, we call on a classify() function in the cnn.py script which assigns a label to the image.
- In the main script, we then call an activate() function in the wifi.py script which activates our deterrence system via WiFi, sending a JSON message to an REST API endpoint. 

## CNN Design
- For now, we will make use of the YOLO CNN. 
- Once we have sample data collected, we will finetune.

## Getting Started
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

## Model Training
Place trained models in `models/` directory.
