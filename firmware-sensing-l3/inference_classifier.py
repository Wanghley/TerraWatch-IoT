#!/usr/bin/env python3
"""
Run inference on a single image using the trained classifier model.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path
import json

class AnimalClassifier:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Load trained model"""
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        
        self.label_encoder = checkpoint['label_encoder']
        num_classes = checkpoint['num_classes']
        model_name = checkpoint.get('model_name', 'resnet18')
        
        # Load model architecture
        if model_name == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "efficientnet":
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        self.model = model
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=3):
        """Predict animal class for an image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Get class names
        top_classes = [self.label_encoder.classes_[idx] for idx in top_indices[0].cpu().numpy()]
        top_probs = top_probs[0].cpu().numpy()
        
        results = []
        for class_name, prob in zip(top_classes, top_probs):
            results.append({
                'class': class_name,
                'confidence': float(prob)
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Run animal classification inference")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--top-k", type=int, default=3,
                       help="Number of top predictions to show")
    
    args = parser.parse_args()
    
    # Load classifier
    print(f"Loading model from {args.model}...")
    classifier = AnimalClassifier(args.model)
    
    # Run inference
    print(f"Running inference on {args.image}...")
    results = classifier.predict(args.image, top_k=args.top_k)
    
    # Print results
    print("\nPredictions:")
    print("-" * 40)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['class']}: {result['confidence']*100:.2f}%")
    
    # Return top prediction
    top_prediction = results[0]
    print(f"\nTop prediction: {top_prediction['class']} ({top_prediction['confidence']*100:.2f}%)")
    
    return top_prediction


if __name__ == "__main__":
    main()

