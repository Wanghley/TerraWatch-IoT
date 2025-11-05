#!/usr/bin/env python3
"""
Train an image classification model for animal detection without bounding boxes.
Uses video frames organized by animal type folders.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import argparse

class AnimalDataset(Dataset):
    """Dataset for animal classification from video frames"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def extract_frames_from_video(video_path, output_dir, frames_per_video=10, max_frames=None):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if max_frames:
        frames_per_video = min(frames_per_video, max_frames)
    
    # Sample frames evenly across the video
    frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
    
    extracted = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = output_dir / f"{video_path.stem}_frame_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted.append(frame_path)
    
    cap.release()
    return extracted


def prepare_dataset(data_dir, frames_per_video=10, max_frames_per_class=None):
    """Extract frames from videos and organize by class"""
    data_dir = Path(data_dir)
    all_images = []
    all_labels = []
    
    # Process each animal type folder
    for class_folder in data_dir.iterdir():
        if not class_folder.is_dir() or class_folder.name == 'none':
            continue
        
        class_name = class_folder.name
        print(f"Processing {class_name}...")
        
        # Create output directory for frames
        frames_dir = data_dir / "frames" / class_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames from videos
        video_files = list(class_folder.glob("*.mp4"))
        class_images = []
        
        for video_path in video_files:
            frames = extract_frames_from_video(
                video_path, 
                frames_dir, 
                frames_per_video=frames_per_video,
                max_frames=max_frames_per_class
            )
            class_images.extend(frames)
        
        # Limit frames per class if specified
        if max_frames_per_class and len(class_images) > max_frames_per_class:
            class_images = np.random.choice(class_images, max_frames_per_class, replace=False)
        
        all_images.extend(class_images)
        all_labels.extend([class_name] * len(class_images))
        print(f"  Extracted {len(class_images)} frames from {len(video_files)} videos")
    
    # Handle "none" class if it contains images
    none_dir = data_dir / "none"
    if none_dir.exists():
        none_images = list(none_dir.glob("*.jpg")) + list(none_dir.glob("*.png"))
        if none_images:
            all_images.extend(none_images)
            all_labels.extend(["none"] * len(none_images))
            print(f"  Added {len(none_images)} 'none' images")
    
    return all_images, all_labels


def train_model(
    images, 
    labels, 
    num_epochs=10, 
    batch_size=32, 
    learning_rate=0.001,
    model_name="resnet18",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Train a classification model"""
    
    print(f"Using device: {device}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"\nClasses: {label_encoder.classes_}")
    print(f"Total images: {len(images)}")
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AnimalDataset(X_train, y_train, transform=train_transform)
    val_dataset = AnimalDataset(X_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load pre-trained model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = Path("best_model.pth")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss/(train_total//batch_size):.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'num_classes': num_classes,
                'model_name': model_name
            }, best_model_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        scheduler.step()
        print()
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    
    return model, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Train animal classification model")
    parser.add_argument("--data-dir", type=str, default="data", 
                       help="Directory containing animal type folders")
    parser.add_argument("--frames-per-video", type=int, default=10,
                       help="Number of frames to extract per video")
    parser.add_argument("--max-frames-per-class", type=int, default=None,
                       help="Maximum frames per class (for balancing)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--model", type=str, default="resnet18",
                       choices=["resnet18", "efficientnet"],
                       help="Model architecture")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip frame extraction if frames already exist")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Extract frames from videos
    if not args.skip_extraction:
        print("Extracting frames from videos...")
        images, labels = prepare_dataset(
            data_dir, 
            frames_per_video=args.frames_per_video,
            max_frames_per_class=args.max_frames_per_class
        )
    else:
        # Load existing frames
        print("Loading existing frames...")
        frames_dir = data_dir / "frames"
        images = []
        labels = []
        for class_dir in frames_dir.iterdir():
            if class_dir.is_dir():
                class_images = list(class_dir.glob("*.jpg"))
                images.extend(class_images)
                labels.extend([class_dir.name] * len(class_images))
    
    if len(images) == 0:
        print("Error: No images found. Make sure videos are in the correct folders.")
        return
    
    # Train model
    model, label_encoder = train_model(
        images,
        labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model
    )


if __name__ == "__main__":
    main()

