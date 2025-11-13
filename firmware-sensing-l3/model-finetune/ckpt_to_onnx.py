#!/usr/bin/env python3
"""
Convert a PyTorch checkpoint (.ckpt) to ONNX format.
"""

import argparse
import json
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from torchvision import models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch checkpoint to ONNX format."
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the ONNX file (default: same as input with .onnx extension)",
    )
    parser.add_argument(
        "--arch",
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet50",
        help="Model architecture (default: resnet50)",
    )
    parser.add_argument(
        "--class-index",
        type=Path,
        help="Path to class_index.json file (default: looks in same directory as checkpoint)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size for the model (default: 224)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)",
    )
    return parser.parse_args()


def load_class_index(class_index_path: Path) -> list[str]:
    """Load class names from class_index.json file."""
    if not class_index_path.exists():
        raise FileNotFoundError(f"Class index not found at {class_index_path}")
    
    with class_index_path.open("r", encoding="utf-8") as f:
        class_dict = json.load(f)
    
    # Convert to list, ensuring correct order
    num_classes = len(class_dict)
    class_names = [class_dict[str(i)] for i in range(num_classes)]
    return class_names


def build_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build the model architecture."""
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    in_features = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main() -> None:
    args = parse_args()
    
    # Resolve paths
    checkpoint_path = args.file.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Determine class_index.json path
    if args.class_index:
        class_index_path = args.class_index.resolve()
    else:
        # Look in the same directory as the checkpoint
        class_index_path = checkpoint_path.parent / "class_index.json"
    
    # Load class index to determine number of classes
    print(f"Loading class index from {class_index_path}...")
    class_names = load_class_index(class_index_path)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Determine output path
    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = checkpoint_path.with_suffix(".onnx")
    
    # Build model
    print(f"Building {args.arch} model with {num_classes} classes...")
    model = build_model(args.arch, num_classes=num_classes, pretrained=False)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try loading as state dict directly
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    
    # Export to ONNX
    print(f"Exporting to ONNX format (opset version {args.opset_version})...")
    print(f"Output will be saved to {output_path}")
    
    # Export to a temporary file first
    temp_output = output_path.with_suffix(".onnx.tmp")
    torch.onnx.export(
        model,
        dummy_input,
        str(temp_output),
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    
    # Load and save as a single file (ensures no external data files)
    print("Converting to single-file format...")
    onnx_model = onnx.load(str(temp_output))
    onnx.save(onnx_model, str(output_path))
    
    # Clean up temporary file
    temp_output.unlink()
    
    # Also clean up any external data files that might have been created
    external_data_file = output_path.with_suffix(".onnx.data")
    if external_data_file.exists():
        external_data_file.unlink()
    
    print(f"✓ Successfully exported ONNX model to {output_path} (single file)")
    print(f"  Model: {args.arch}")
    print(f"  Classes: {num_classes} ({', '.join(class_names)})")
    print(f"  Input shape: (batch_size, 3, {args.image_size}, {args.image_size})")
    print(f"  Output shape: (batch_size, {num_classes})")


if __name__ == "__main__":
    main()


