import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

try:
    import kornia.augmentation as K
except ImportError:  # pragma: no cover - Kornia optional
    K = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "frames"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model-finetune" / "output"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PREVIEWS_DIR = PROJECT_ROOT / "model-finetune" / "previews"

MAX_TRAIN_BATCHES: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a ResNet backbone for single-label classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing class/video/frames hierarchy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument("--arch", choices=["resnet18", "resnet34", "resnet50"], default="resnet50")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pre-trained weights.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for 'other_animal' detection. Predictions with max probability below this are classified as 'other_animal'.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to a checkpoint to resume from (expects state dict with model/optimizer).",
    )
    parser.add_argument(
        "--train-restart",
        action="store_true",
        help="Ignore checkpoints and restart training from scratch.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs with no val-loss improvement before early stopping.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Maximum number of training batches processed per epoch (debug/profiling).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run evaluation only on the test split using the latest checkpoint.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Run inference on a single image file. Requires --resume or checkpoint in output-dir.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=50,
        help="Number of augmented training samples to save as previews (0 disables).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_ops: List = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if K is None:
        train_ops.append(normalize)
    train_transform = transforms.Compose(train_ops)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_gpu_augmentations(device: torch.device) -> Optional[nn.Module]:
    if K is None:
        print("Kornia not found; skipping GPU augmentations.")
        return None

    mean = torch.tensor(IMAGENET_MEAN, device=device)
    std = torch.tensor(IMAGENET_STD, device=device)

    augmentations = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(brightness=0.05, contrast=0.001, saturation=0.002, hue=0.02, p=0.8),
        # K.RandomBoxBlur(kernel_size=(3, 3), p=0.2),
        # K.Normalize(mean=mean, std=std),
        data_keys=["input"],
    ).to(device)
    augmentations.train()
    return augmentations


class PreviewSaver:
    def __init__(self, directory: Path, limit: int, mean: Sequence[float], std: Sequence[float]) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit = max(limit, 0)
        self.saved = 0
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def maybe_save(self, images: torch.Tensor) -> None:
        if self.saved >= self.limit or self.limit == 0:
            return

        remaining = self.limit - self.saved
        batch = images.detach().cpu()
        batch = batch * self.std + self.mean
        batch = batch.clamp(0.0, 1.0)
        batch = batch[:remaining]

        for idx, img in enumerate(batch):
            save_path = self.directory / f"preview_{self.saved + idx:03d}.png"
            save_image(img, save_path)

        self.saved += len(batch)


def plot_training_curves(
    epoch_indices: List[int],
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    output_dir: Path,
) -> None:
    if not train_losses:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_loss.plot(epoch_indices, train_losses, label="Train Loss", marker="o")
    ax_loss.plot(epoch_indices, val_losses, label="Val Loss", marker="s")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    ax_acc.plot(epoch_indices, val_accs, label="Val Accuracy", color="tab:green", marker="^")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Validation Accuracy")
    ax_acc.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {output_path}")


# Removed MultiHotTransform - ImageFolder already provides integer labels for single-label classification


def split_dataset(
    dataset: datasets.ImageFolder,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    assert 0.0 <= val_split < 1.0, "Validation split must be in [0, 1)."
    assert 0.0 <= test_split < 1.0, "Test split must be in [0, 1)."
    assert val_split + test_split < 1.0, "Train split must be positive."

    total_len = len(dataset)
    val_len = int(total_len * val_split)
    test_len = int(total_len * test_split)
    train_len = total_len - val_len - test_len

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_transform, eval_transform = build_transforms(image_size)

    # ImageFolder provides integer labels (0, 1, 2, ...) for single-label classification
    base_dataset = datasets.ImageFolder(
        root=str(data_dir),
        transform=train_transform,
    )
    class_names = base_dataset.classes
    # No target_transform needed - ImageFolder already provides integer labels

    train_dataset, val_dataset, test_dataset = split_dataset(
        base_dataset, val_split=val_split, test_split=test_split, seed=seed
    )

    # Override transforms for validation/test to remove augmentation.
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, class_names


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
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


def apply_confidence_threshold(
    probs: torch.Tensor, class_names: List[str], threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply confidence threshold for 'other_animal' detection.
    If max probability < threshold, classify as 'other_animal' (represented as -1).
    
    Returns:
        predicted_classes: Class indices (-1 for 'other_animal')
        max_probs: Maximum probabilities for each sample
    """
    max_probs, predicted_indices = torch.max(probs, dim=1)
    # Mark low-confidence predictions as 'other_animal' (-1)
    other_animal_mask = max_probs < threshold
    predicted_indices = predicted_indices.clone()
    predicted_indices[other_animal_mask] = -1
    return predicted_indices, max_probs


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    threshold: float,
) -> Dict[str, float]:
    """
    Compute single-label classification metrics.
    
    All predictions are included in accuracy calculation, including low-confidence
    'other_animal' predictions. 'other_animal' predictions are treated as incorrect
    since they're never a true class in the dataset.
    
    Args:
        logits: Model logits [batch_size, num_classes]
        targets: Integer class labels [batch_size]
        class_names: List of class names
        threshold: Confidence threshold for 'other_animal' detection
    """
    probs = torch.softmax(logits, dim=1)
    predicted_indices, max_probs = apply_confidence_threshold(probs, class_names, threshold)
    
    targets_int = targets.long()
    
    # Calculate overall accuracy including ALL predictions
    # 'other_animal' predictions (-1) will never match targets (>= 0), so they're counted as incorrect
    correct = (predicted_indices == targets_int).float()
    accuracy = correct.mean().item()
    
    # Compute per-class metrics (excluding 'other_animal')
    num_classes = len(class_names)
    eps = 1e-7
    precision = []
    recall = []
    f1_scores = []
    
    for class_idx in range(num_classes):
        # True positives: predicted this class and target is this class
        tp = ((predicted_indices == class_idx) & (targets_int == class_idx)).sum().item()
        # False positives: predicted this class but target is different
        # Note: 'other_animal' predictions (-1) won't match class_idx, so they're not counted as FP
        fp = ((predicted_indices == class_idx) & (targets_int != class_idx)).sum().item()
        # False negatives: target is this class but predicted something else (including 'other_animal')
        fn = ((predicted_indices != class_idx) & (targets_int == class_idx)).sum().item()
        
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = (2 * prec * rec) / (prec + rec + eps)
        
        precision.append(prec)
        recall.append(rec)
        f1_scores.append(f1)
    
    macro_precision = sum(precision) / num_classes
    macro_recall = sum(recall) / num_classes
    macro_f1 = sum(f1_scores) / num_classes
    
    # Count 'other_animal' predictions
    other_animal_count = (predicted_indices == -1).sum().item()
    other_animal_rate = other_animal_count / len(targets)
    
    # Count correct predictions that were marked as 'other_animal' (false negatives)
    # These are cases where the model was uncertain but the true class was actually correct
    false_other_animal = ((predicted_indices == -1) & (targets_int >= 0)).sum().item()
    
    return {
        "accuracy": accuracy,  # Overall accuracy including all predictions
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "other_animal_rate": other_animal_rate,
        "false_other_animal": false_other_animal,  # Count of false 'other_animal' predictions
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    augmentations: Optional[nn.Module] = None,
    preview_saver: Optional[PreviewSaver] = None,
) -> float:
    model.train()
    running_loss = 0.0
    processed_samples = 0

    total_batches = len(dataloader)
    if MAX_TRAIN_BATCHES is not None:
        total_batches = min(total_batches, MAX_TRAIN_BATCHES)

    progress = tqdm(total=total_batches, desc="Train 0/{}".format(total_batches), leave=False)
    if augmentations is not None:
        augmentations.train()

    for batch_idx, (images, targets) in enumerate(dataloader):
        if MAX_TRAIN_BATCHES is not None and batch_idx >= MAX_TRAIN_BATCHES:
            break
        images = images.to(device, non_blocking=True)
        # CrossEntropyLoss expects integer labels (long dtype)
        targets = targets.to(device, non_blocking=True).long()

        if augmentations is not None:
            images = augmentations(images)
        if preview_saver is not None:
            preview_saver.maybe_save(images)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        processed_samples += images.size(0)
        current_batch = min(batch_idx + 1, total_batches)
        progress.set_description(f"Train {current_batch}/{total_batches}")
        progress.update(1)

    avg_loss = running_loss / max(processed_samples, 1)
    progress.close()
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        # CrossEntropyLoss expects integer labels (long dtype)
        targets = targets.to(device, non_blocking=True).long()

        logits = model(images)
        loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat, class_names, threshold)
    metrics["loss"] = running_loss / len(dataloader.dataset)
    return metrics


def compute_confusion_matrix(
    true_classes: torch.Tensor,
    pred_classes: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(true_classes, pred_classes):
        cm[t.long(), p.long()] += 1
    return cm


def summarize_predictions(
    probs: torch.Tensor,
    true_classes: torch.Tensor,
    pred_classes: torch.Tensor,
    paths: List[str],
    class_names: List[str],
    threshold: float,
    max_incorrect: int = 20,
    max_low_conf_correct: int = 20,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    max_probs, _ = probs.max(dim=1)
    incorrect_samples = []
    correct_samples = []

    for prob, true_idx, pred_idx, path in zip(max_probs, true_classes, pred_classes, paths):
        true_idx_val = true_idx.item()
        pred_idx_val = pred_idx.item()
        
        # Handle 'other_animal' (-1) in predictions
        if pred_idx_val == -1:
            pred_name = "other_animal"
        else:
            pred_name = class_names[pred_idx_val]
        
        true_name = class_names[true_idx_val] if true_idx_val >= 0 else "other_animal"
        
        record = {
            "path": path,
            "true": true_name,
            "pred": pred_name,
            "confidence": f"{prob.item():.4f}",
        }
        
        # Consider it incorrect if prediction doesn't match true class
        # (including 'other_animal' predictions, which are never correct since it's not a true class)
        if pred_idx_val != true_idx_val:
            incorrect_samples.append(record)
        else:
            correct_samples.append(record)

    incorrect_samples.sort(key=lambda x: float(x["confidence"]), reverse=True)
    correct_samples.sort(key=lambda x: float(x["confidence"]))
    return incorrect_samples[:max_incorrect], correct_samples[:max_low_conf_correct]


def report_test_results(
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    paths: List[str],
    threshold: float,
) -> None:
    """
    Report test results for single-label classification.
    Includes handling of 'other_animal' class via confidence threshold.
    """
    num_classes = len(class_names)
    true_classes = targets.long()
    
    # Apply confidence threshold
    predicted_indices, max_probs = apply_confidence_threshold(probs, class_names, threshold)
    
    # Count 'other_animal' predictions
    other_animal_count = (predicted_indices == -1).sum().item()
    
    # Compute confusion matrix (excluding 'other_animal' for now)
    # Create extended class names including 'other_animal'
    extended_class_names = class_names + ["other_animal"]
    extended_num_classes = num_classes + 1
    
    # Map -1 to last index for confusion matrix
    pred_for_cm = predicted_indices.clone()
    pred_for_cm[predicted_indices == -1] = num_classes
    
    cm = compute_confusion_matrix(true_classes, pred_for_cm, extended_num_classes)
    
    # Compute overall accuracy including ALL predictions
    # 'other_animal' predictions (-1) will never match targets (>= 0), so they're counted as incorrect
    overall_accuracy = (predicted_indices == true_classes).float().mean().item()
    
    eps = 1e-7
    precision = []
    recall = []
    for idx in range(num_classes):
        tp = cm[idx, idx].item()
        fp = cm[:, idx].sum().item() - tp
        fn = cm[idx, :].sum().item() - tp
        precision.append(tp / (tp + fp + eps))
        recall.append(tp / (tp + fn + eps))

    macro_precision = sum(precision) / num_classes
    macro_recall = sum(recall) / num_classes
    
    # Create class names for predictions (handle 'other_animal')
    pred_class_names = [
        class_names[idx] if idx >= 0 else "other_animal" for idx in predicted_indices.cpu().numpy()
    ]
    
    incorrect, low_conf_correct = summarize_predictions(
        probs, true_classes, predicted_indices, paths, class_names, threshold
    )

    print("\n=== Test Set Summary ===")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"'Other Animal' Predictions: {other_animal_count} ({other_animal_count/len(targets)*100:.2f}%)")
    print(f"False 'Other Animal' (FN): {((predicted_indices == -1) & (true_classes >= 0)).sum().item()} "
          f"(correct predictions marked as uncertain)")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "Pred".ljust(15) + " ".join(name.ljust(15) for name in extended_class_names)
    print(header)
    for idx, row in enumerate(cm):
        if idx < num_classes:
            row_vals = " ".join(str(val.item()).ljust(15) for val in row)
            print(class_names[idx].ljust(15) + row_vals)

    def print_samples(title: str, samples: List[Dict[str, str]]) -> None:
        print(f"\n{title} (showing {len(samples)}):")
        if not samples:
            print("  None")
            return
        for sample in samples:
            print(
                f"  confidence={sample['confidence']} pred={sample['pred']} "
                f"true={sample['true']} path={sample['path']}"
            )

    print_samples("Most Confident Incorrect Samples", incorrect)
    print_samples("Least Confident Correct Samples", low_conf_correct)


def run_test_only(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
    threshold: float,
    checkpoint_path: Path,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    paths: List[str] = []

    subset = dataloader.dataset
    if isinstance(subset, torch.utils.data.Subset):
        base_dataset = subset.dataset
        subset_indices = subset.indices
    else:
        base_dataset = subset
        subset_indices = list(range(len(subset)))

    offset = 0
    with torch.no_grad():
        for images, targets in dataloader:
            batch_size = images.size(0)
            batch_indices = subset_indices[offset : offset + batch_size]
            offset += batch_size

            for idx in batch_indices:
                paths.append(base_dataset.samples[idx][0])

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    # Use softmax for single-label classification
    probs = torch.softmax(logits_cat, dim=1)

    metrics = compute_metrics(logits_cat, targets_cat, class_names, threshold)
    print(
        "\nSingle-label Metrics - Accuracy: {accuracy:.4f}, "
        "Macro F1: {macro_f1:.4f}, Precision: {macro_precision:.4f}, "
        "Recall: {macro_recall:.4f}, 'Other Animal' Rate: {other_animal_rate:.4f}, "
        "False 'Other Animal': {false_other_animal}".format(**metrics)
    )

    report_test_results(probs, targets_cat, class_names, paths, threshold)


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    metrics: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.ckpt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def save_class_index(output_dir: Path, class_names: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "class_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({idx: name for idx, name in enumerate(class_names)}, f, indent=2)


def load_class_index(output_dir: Path) -> List[str]:
    """Load class names from class_index.json file."""
    index_path = output_dir / "class_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Class index not found at {index_path}. Please train the model first or provide class names."
        )
    with index_path.open("r", encoding="utf-8") as f:
        class_dict = json.load(f)
    # Convert to list, ensuring correct order
    num_classes = len(class_dict)
    class_names = [class_dict[str(i)] for i in range(num_classes)]
    return class_names


def run_inference_on_file(
    model: nn.Module,
    image_path: Path,
    class_names: List[str],
    device: torch.device,
    threshold: float,
    image_size: int = 224,
) -> None:
    """
    Run inference on a single image file.
    
    Args:
        model: Trained model
        image_path: Path to image file
        class_names: List of class names
        device: Device to run inference on
        threshold: Confidence threshold for 'other_animal' detection
        image_size: Size to resize image to
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    # Build transform (same as evaluation)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

    # Get predictions
    max_probs, predicted_indices = torch.max(probs, dim=1)
    max_prob = max_probs[0].item()
    predicted_idx = predicted_indices[0].item()

    # Apply confidence threshold
    if max_prob < threshold:
        predicted_class = "other_animal"
        confidence = max_prob
    else:
        predicted_class = class_names[predicted_idx]
        confidence = max_prob

    # Get all class probabilities
    probs_list = probs[0].cpu().numpy()

    # Print results
    print(f"\n{'='*60}")
    print(f"Inference on: {image_path}")
    print(f"{'='*60}")
    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"Threshold: {threshold:.4f}")
    
    if max_prob < threshold:
        print(f"\n⚠️  Low confidence detected! Classified as 'other_animal'.")
        print(f"   (Max probability {max_prob:.4f} is below threshold {threshold:.4f})")
    
    print(f"\nAll Class Probabilities:")
    print(f"{'Class':<20} {'Probability':<15} {'Percentage':<15}")
    print(f"{'-'*50}")
    for idx, (class_name, prob) in enumerate(zip(class_names, probs_list)):
        marker = " ←" if idx == predicted_idx and max_prob >= threshold else ""
        print(f"{class_name:<20} {prob:.6f}       {prob*100:>6.2f}%{marker}")
    
    if max_prob < threshold:
        print(f"{'other_animal':<20} {'N/A':<15} {'N/A':<15} ← (Low confidence)")
    
    print(f"{'='*60}\n")


def resume_if_available(
    resume_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {resume_path}")

    checkpoint = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    return start_epoch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    global MAX_TRAIN_BATCHES
    MAX_TRAIN_BATCHES = args.max_train_batches

    if not args.data_dir.is_absolute():
        args.data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    if args.resume and not args.resume.is_absolute():
        args.resume = (PROJECT_ROOT / args.resume).resolve()

    checkpoint_to_load: Optional[Path] = None
    if args.resume:
        checkpoint_to_load = args.resume
    else:
        default_ckpt = args.output_dir / "best.ckpt"
        if default_ckpt.exists():
            checkpoint_to_load = default_ckpt

    # Handle single file inference (skip data loading and training setup)
    if args.test_file:
        if checkpoint_to_load is None:
            raise FileNotFoundError(
                "No checkpoint available for --test-file. Provide --resume or ensure checkpoint exists in output-dir."
            )
        
        # Load class names from class_index.json
        try:
            class_names = load_class_index(args.output_dir)
            num_classes = len(class_names)
            print(f"Loaded {num_classes} classes from class_index.json: {class_names}")
        except FileNotFoundError:
            # Fallback: try to load from data directory
            print("Warning: class_index.json not found. Attempting to load classes from data directory...")
            if not args.data_dir.exists():
                raise FileNotFoundError(
                    "Cannot determine class names. Either provide class_index.json in output-dir or --data-dir."
                )
            # Build dataset just to get class names
            _, _, _, class_names = build_dataloaders(
                data_dir=args.data_dir,
                image_size=args.image_size,
                batch_size=1,
                num_workers=0,
                val_split=0.0,
                test_split=0.0,
                seed=args.seed,
            )
            num_classes = len(class_names)
            print(f"Loaded {num_classes} classes from data directory: {class_names}")
        
        # Rebuild model with correct number of classes
        model = build_model(args.arch, num_classes=num_classes, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        
        # Handle relative paths
        if not args.test_file.is_absolute():
            args.test_file = (Path.cwd() / args.test_file).resolve()
        
        # Run inference
        run_inference_on_file(
            model=model,
            image_path=args.test_file,
            class_names=class_names,
            device=device,
            threshold=args.threshold,
            image_size=args.image_size,
        )
        return

    # Build data loaders for training/testing
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    model = build_model(args.arch, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)
    gpu_augmentations = build_gpu_augmentations(device)

    # Use CrossEntropyLoss for single-label classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.test:
        if checkpoint_to_load is None:
            raise FileNotFoundError("No checkpoint available for --test. Provide --resume or train first.")
        run_test_only(
            model=model,
            dataloader=test_loader,
            class_names=class_names,
            device=device,
            threshold=args.threshold,
            checkpoint_path=checkpoint_to_load,
        )
        return

    start_epoch = 0
    best_metric = float("-inf")
    best_val_loss = float("inf")
    patience_counter = 0

    if not args.train_restart:
        if checkpoint_to_load:
            print(f"Resuming from checkpoint: {checkpoint_to_load}")
            start_epoch = resume_if_available(checkpoint_to_load, model, optimizer)
        else:
            print("No checkpoint found; starting from scratch.")
    else:
        print("Restarting training from scratch as requested.")

    save_class_index(args.output_dir, class_names)

    preview_saver: Optional[PreviewSaver] = None
    if args.preview_limit > 0:
        preview_saver = PreviewSaver(PREVIEWS_DIR, args.preview_limit, IMAGENET_MEAN, IMAGENET_STD)

    epoch_indices: List[int] = []
    train_history: List[float] = []
    val_loss_history: List[float] = []
    val_acc_history: List[float] = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            augmentations=gpu_augmentations,
            preview_saver=preview_saver,
        )
        print(f"Train Loss: {train_loss:.4f}")

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            threshold=args.threshold,
        )
        print(
            "Validation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
            "Macro F1: {macro_f1:.4f}, Precision: {macro_precision:.4f}, "
            "Recall: {macro_recall:.4f}, 'Other Animal' Rate: {other_animal_rate:.4f}, "
            "False 'Other Animal': {false_other_animal}".format(**val_metrics)
        )

        epoch_indices.append(epoch + 1)
        train_history.append(train_loss)
        val_loss_history.append(val_metrics["loss"])
        val_acc_history.append(val_metrics["accuracy"])

        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(args.output_dir, epoch, model, optimizer, val_metrics)
            print("Saved new best checkpoint (val loss improved).")
        else:
            patience_counter += 1
            print(f"Val loss did not improve. Patience counter: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                plot_training_curves(
                    epoch_indices=epoch_indices,
                    train_losses=train_history,
                    val_losses=val_loss_history,
                    val_accs=val_acc_history,
                    output_dir=args.output_dir,
                )
                break

        plot_training_curves(
            epoch_indices=epoch_indices,
            train_losses=train_history,
            val_losses=val_loss_history,
            val_accs=val_acc_history,
            output_dir=args.output_dir,
        )

    else:
        plot_training_curves(
            epoch_indices=epoch_indices,
            train_losses=train_history,
            val_losses=val_loss_history,
            val_accs=val_acc_history,
            output_dir=args.output_dir,
        )

    print("\nEvaluating best checkpoint on test set...")
    best_ckpt_path = args.output_dir / "best.ckpt"
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        print("Warning: best checkpoint not found, using final epoch weights.")

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        threshold=args.threshold,
    )
    print(
        "Test - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
        "Macro F1: {macro_f1:.4f}, Precision: {macro_precision:.4f}, "
        "Recall: {macro_recall:.4f}, 'Other Animal' Rate: {other_animal_rate:.4f}, "
        "False 'Other Animal': {false_other_animal}".format(**test_metrics)
    )


if __name__ == "__main__":
    main()

