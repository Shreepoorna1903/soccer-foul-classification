"""
06_train_resnet.py
Fine-tune ResNet-18 on middle frames for binary and multiclass tasks.
Uses PyTorch with MPS (Apple Silicon) backend.

Usage:
    python 06_train_resnet.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from utils import (
    load_dataset, extract_middle_frame,
    RESULTS_DIR, MODELS_DIR, FOUL_TYPE_NAMES, OFFENCE_NAMES,
)

# ── Configuration ──────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # MPS works best with 0 on macOS
SEED = 42

# Hyperparameter search space
LR_OPTIONS = [1e-4, 5e-4, 1e-3]
EPOCHS_OPTIONS = [15, 25]
WEIGHT_DECAY = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Dataset ────────────────────────────────────────────────────
class FoulFrameDataset(Dataset):
    """Dataset that loads middle frames from video clips."""

    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform
        # Pre-extract frames to avoid repeated video reads during training
        self.frames = []
        self.labels = []
        print(f"    Pre-loading {len(data_list)} frames...")
        skipped = 0
        for item in tqdm(data_list, desc="    Loading frames", leave=False):
            frame = extract_middle_frame(item["video_path"])
            if frame is None:
                skipped += 1
                continue
            # Convert BGR to RGB
            frame = frame[:, :, ::-1].copy()
            self.frames.append(frame)
            self.labels.append(item["label"])
        if skipped > 0:
            print(f"    Skipped {skipped} unreadable clips")
        print(f"    Loaded {len(self.frames)} frames")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]

        if self.transform:
            from PIL import Image
            img = Image.fromarray(frame)
            img = self.transform(img)
        else:
            img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        return img, label


def get_transforms(train=True):
    """Get data augmentation transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_model(num_classes, device):
    """Build ResNet-18 with pretrained weights, replace final FC layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze early layers (conv1, bn1, layer1, layer2)
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ["conv1", "bn1", "layer1", "layer2"]):
            param.requires_grad = False

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    return model.to(device)


def compute_class_weights(labels, num_classes, device):
    """Compute inverse-frequency class weights for weighted loss."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    # Avoid division by zero
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, loader, device):
    """Evaluate model, return predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return np.array(all_preds), np.array(all_labels)


def run_task(task, num_classes, label_names, device):
    """Run full training pipeline with hyperparameter search."""

    print(f"\n{'='*60}")
    print(f"  ResNet-18 — {task.upper()}")
    print(f"{'='*60}")

    # ── Load data ──
    print("\n  Loading datasets...")
    train_data = load_dataset("train", task=task)
    valid_data = load_dataset("valid", task=task)
    test_data = load_dataset("test", task=task)

    train_ds = FoulFrameDataset(train_data, transform=get_transforms(train=True))
    valid_ds = FoulFrameDataset(valid_data, transform=get_transforms(train=False))
    test_ds = FoulFrameDataset(test_data, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Class weights
    train_labels = np.array(train_ds.labels)
    class_weights = compute_class_weights(train_labels, num_classes, device)
    print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

    # ── Hyperparameter search ──
    best_val_ba = 0
    best_config = {}
    all_results = []

    for lr in LR_OPTIONS:
        for n_epochs in EPOCHS_OPTIONS:
            config_str = f"lr={lr}, epochs={n_epochs}"
            print(f"\n  --- Config: {config_str} ---")

            model = build_model(num_classes, device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

            best_epoch_ba = 0
            best_epoch_model_state = None

            for epoch in range(n_epochs):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                scheduler.step()

                # Validate
                val_preds, val_labels = evaluate(model, valid_loader, device)
                val_ba = balanced_accuracy_score(val_labels, val_preds)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"    Epoch {epoch+1:>2}/{n_epochs} — loss: {train_loss:.4f}, val BA: {val_ba:.4f}")

                if val_ba > best_epoch_ba:
                    best_epoch_ba = val_ba
                    best_epoch_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            all_results.append({
                "lr": lr, "epochs": n_epochs,
                "val_ba": best_epoch_ba,
            })

            if best_epoch_ba > best_val_ba:
                best_val_ba = best_epoch_ba
                best_config = {"lr": lr, "epochs": n_epochs}
                best_model_state = best_epoch_model_state

            print(f"    Best val BA for this config: {best_epoch_ba:.4f}")

    # ── Results summary ──
    print(f"\n  Hyperparameter Search Results:")
    print(f"  {'LR':<10} {'Epochs':<8} {'Val BA':>8}")
    print(f"  {'-'*28}")
    for r in all_results:
        marker = " <-- best" if r["lr"] == best_config["lr"] and r["epochs"] == best_config["epochs"] else ""
        print(f"  {r['lr']:<10} {r['epochs']:<8} {r['val_ba']:>8.4f}{marker}")

    print(f"\n  Best config: {best_config}")
    print(f"  Best validation BA: {best_val_ba:.4f}")

    # ── Test evaluation with best model ──
    model = build_model(num_classes, device)
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    test_preds, test_labels = evaluate(model, test_loader, device)
    test_ba = balanced_accuracy_score(test_labels, test_preds)

    print(f"\n  TEST Balanced Accuracy: {test_ba:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=label_names, digits=4))

    # ── Confusion matrix ──
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Greens", xticks_rotation=45)
    ax.set_title(f"ResNet-18 — {task}\nTest Balanced Acc: {test_ba:.4f}")
    plt.tight_layout()
    fig_path = RESULTS_DIR / f"resnet_{task}_confusion.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {fig_path}")

    # ── Save model ──
    model_path = MODELS_DIR / f"resnet_{task}.pt"
    torch.save(best_model_state, model_path)
    print(f"  Model saved to {model_path}")

    # ── Save config ──
    config_path = MODELS_DIR / f"resnet_{task}_config.json"
    with open(config_path, "w") as f:
        json.dump({**best_config, "val_ba": best_val_ba, "test_ba": test_ba}, f, indent=2)

    return test_ba


def main():
    device = get_device()
    print(f"  Using device: {device}")

    # ── Task 1: Binary ──
    ba_binary = run_task("binary", num_classes=2, label_names=OFFENCE_NAMES, device=device)

    # ── Task 2: Multiclass ──
    ba_multi = run_task("multiclass", num_classes=7, label_names=FOUL_TYPE_NAMES, device=device)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  ResNet-18 SUMMARY")
    print(f"{'='*60}")
    print(f"  Binary (Offence Detection)   Test BA: {ba_binary:.4f}")
    print(f"  Multiclass (Foul Type)       Test BA: {ba_multi:.4f}")


if __name__ == "__main__":
    main()