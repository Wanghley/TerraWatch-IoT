# Level 3 Sensing - Orange Pi Vision System

## Overview
CNN-based image analysis for animal classification using Orange Pi 4A.

## Hardware
- Orange Pi 4A (8GB RAM)
- High-resolution Camera Module
- WiFi Client for communication

## Features
- Camera capture on trigger from L1/L2
- CNN-based animal classification
- Decision making for deterrence activation
- Data logging and analytics

## Getting Started
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

## Model Training
Place trained models in `models/` directory.
