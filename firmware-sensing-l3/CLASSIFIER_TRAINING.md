# Training Animal Classification Without Bounding Boxes

## Overview

This approach trains an **image classification model** (not object detection) to classify whether a specific animal is present in an image. Unlike YOLO, this only requires **image-level labels** (e.g., "raccoon" or "no raccoon") - no bounding boxes needed!

## Why This Approach?

1. **No bounding box labeling required** - Your videos are already organized by animal type, which is perfect for this
2. **Faster training** - Classification models train much faster than object detection
3. **Simpler inference** - Just pass an image and get a class prediction
4. **Good accuracy** - Pre-trained models (ResNet, EfficientNet) work well for this task

## Setup

```bash
pip install -r requirements_classifier.txt
```

## Training

### Basic Usage

```bash
python train_classifier.py --data-dir data
```

This will:
1. Extract frames from all videos in your `data/` folder (organized by animal type)
2. Train a ResNet18 model to classify the animals
3. Save the best model as `best_model.pth`

### Options

```bash
python train_classifier.py \
    --data-dir data \
    --frames-per-video 10 \
    --max-frames-per-class 500 \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --model resnet18
```

**Parameters:**
- `--frames-per-video`: Number of frames to extract from each video (default: 10)
- `--max-frames-per-class`: Limit frames per class for balancing (default: None)
- `--epochs`: Training epochs (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--model`: Model architecture - `resnet18` or `efficientnet` (default: resnet18)
- `--skip-extraction`: Skip frame extraction if frames already exist

### Data Structure

Your data should be organized like this:
```
data/
├── groundhog/
│   ├── video1.mp4
│   └── video2.mp4
├── raccoon/
│   ├── video1.mp4
│   └── video2.mp4
├── squirrel/
│   └── video1.mp4
└── none/          # Optional: images with no animals
    └── image1.jpg
```

## Inference

### Single Image

```bash
python inference_classifier.py path/to/image.jpg --model best_model.pth
```

### Integration with Your Existing Script

You can modify `l3.py` to use the classifier instead of YOLO:

```python
from inference_classifier import AnimalClassifier

classifier = AnimalClassifier("best_model.pth")
results = classifier.predict("output.jpg")
top_prediction = results[0]['class']
confidence = results[0]['confidence']

if confidence > 0.7:  # Threshold
    print(f"Detected: {top_prediction} ({confidence*100:.1f}%)")
```

## Alternative: YOLO-Based Approach

If you specifically want to use YOLO architecture, you can:

### Option A: Use YOLO Backbone + Classification Head

1. Extract features from a pre-trained YOLO model
2. Add a classification head on top
3. Train only the classification head with image-level labels

This is more complex but keeps YOLO's architecture. Here's a simplified example:

```python
# Pseudo-code
import torch
from ultralytics import YOLO

# Load pre-trained YOLO (for feature extraction)
yolo = YOLO('yolov8n.pt')
backbone = yolo.model.model[:-1]  # Remove detection head

# Add classification head
classifier_head = nn.Linear(backbone_output_size, num_classes)
model = nn.Sequential(backbone, classifier_head)

# Freeze backbone, train only classifier
for param in backbone.parameters():
    param.requires_grad = False
```

### Option B: Use Pre-trained YOLO + Separate Classifier

1. Use a pre-trained YOLO to detect if ANY animal is present
2. If detected, crop the animal region
3. Pass the cropped region to your classifier to determine the specific animal

This is a two-stage approach that's easier to implement.

## Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **Image Classification** (Recommended) | • Simplest<br>• Fast training<br>• No bbox labels | • Doesn't localize animals<br>• Works best if animal is main subject |
| **YOLO Backbone + Classifier** | • Uses YOLO architecture<br>• Good feature extraction | • More complex<br>• Still need to adapt YOLO code |
| **Pre-trained YOLO + Classifier** | • Reuses YOLO detection<br>• Can localize | • Two-stage inference<br>• More moving parts |

## Recommendations

For your use case (TerraWatch IoT), I recommend:

1. **Start with the image classification approach** - It's the simplest and will work well if your camera frames typically show the animal as the main subject
2. **If you need localization** (to know WHERE the animal is), then consider the two-stage approach (YOLO for detection + classifier for species)

## Next Steps

1. Run the training script on your video data
2. Evaluate the model on test images
3. Integrate into your `l3.py` inference pipeline
4. Deploy to your Orange Pi device

