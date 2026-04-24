"""
06b_train_r3d.py
Fine-tune R3D-18 (3D ResNet) on video clips for binary and multiclass tasks.
Uses 16 evenly-spaced frames per clip as spatiotemporal input.
Proper hyperparameter search: 3 LRs × 30 epochs.

Usage:
    python 06b_train_r3d.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import json
import time

from utils import (
    load_dataset,
    RESULTS_DIR, MODELS_DIR, FOUL_TYPE_NAMES, OFFENCE_NAMES,
)

# ── Configuration ──────────────────────────────────────────────
NUM_FRAMES = 16
CLIP_SIZE = 112
BATCH_SIZE = 8
NUM_WORKERS = 0
SEED = 42

LR_OPTIONS = [1e-4, 3e-4, 1e-3]
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_frames_from_video(video_path, n_frames=16, size=112):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
    cap.release()
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((size, size, 3), dtype=np.uint8))
    return np.stack(frames[:n_frames])


class FoulVideoDataset(Dataset):
    def __init__(self, data_list, n_frames=16, size=112, augment=False):
        self.clips = []
        self.labels = []
        self.augment = augment
        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

        print(f"    Pre-loading {len(data_list)} video clips ({n_frames} frames each)...")
        skipped = 0
        for item in tqdm(data_list, desc="    Loading clips", leave=False):
            clip = sample_frames_from_video(item["video_path"], n_frames, size)
            if clip is None:
                skipped += 1
                continue
            self.clips.append(clip)
            self.labels.append(item["label"])
        if skipped > 0:
            print(f"    Skipped {skipped} unreadable clips")
        print(f"    Loaded {len(self.clips)} clips")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx].astype(np.float32) / 255.0
        label = self.labels[idx]
        if self.augment and np.random.random() > 0.5:
            clip = clip[:, :, ::-1, :].copy()
        clip = (clip - self.mean) / self.std
        clip = clip.transpose(3, 0, 1, 2)
        clip = torch.from_numpy(clip).float()
        return clip, label


def build_model(num_classes, device):
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)

    # Only freeze stem — unfreeze everything else for deeper learning
    for name, param in model.named_parameters():
        if name.startswith("stem"):
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


def compute_class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, n_epochs):
    model.train()
    total_loss = 0
    n_batches = 0
    pbar = tqdm(loader, desc=f"    Epoch {epoch+1:>2}/{n_epochs} [train]", leave=False)
    for clips, labels in pbar:
        clips = clips.to(device)
        labels = labels.clone().detach().long().to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
    return total_loss / max(n_batches, 1)


def evaluate(model, loader, device, desc="eval"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for clips, labels in tqdm(loader, desc=f"    [{desc}]", leave=False):
            clips = clips.to(device)
            outputs = model(clips)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return np.array(all_preds), np.array(all_labels)


def run_task(task, num_classes, label_names, device):
    print(f"\n{'='*60}")
    print(f"  R3D-18 (3D CNN) — {task.upper()}")
    print(f"  Config: {len(LR_OPTIONS)} LRs × {NUM_EPOCHS} epochs")
    print(f"{'='*60}")

    print("\n  Loading datasets...")
    train_data = load_dataset("train", task=task)
    valid_data = load_dataset("valid", task=task)
    test_data = load_dataset("test", task=task)

    train_ds = FoulVideoDataset(train_data, NUM_FRAMES, CLIP_SIZE, augment=True)
    valid_ds = FoulVideoDataset(valid_data, NUM_FRAMES, CLIP_SIZE, augment=False)
    test_ds = FoulVideoDataset(test_data, NUM_FRAMES, CLIP_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    train_labels = np.array(train_ds.labels)
    class_weights = compute_class_weights(train_labels, num_classes, device)
    print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

    best_val_ba = 0
    best_config = {}
    best_model_state = None
    all_results = []
    all_training_curves = {}

    for lr in LR_OPTIONS:
        print(f"\n  ┌─── LR = {lr} ───────────────────────────────────────")
        start_time = time.time()

        model = build_model(num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_epoch_ba = 0
        best_epoch_model_state = None
        epoch_losses = []
        epoch_val_bas = []

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS)
            scheduler.step()

            val_preds, val_labels = evaluate(model, valid_loader, device, desc="valid")
            val_ba = balanced_accuracy_score(val_labels, val_preds)

            epoch_losses.append(train_loss)
            epoch_val_bas.append(val_ba)

            marker = ""
            if val_ba > best_epoch_ba:
                best_epoch_ba = val_ba
                best_epoch_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = " *"

            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (NUM_EPOCHS - epoch - 1)
            print(f"  │ Epoch {epoch+1:>2}/{NUM_EPOCHS} — loss: {train_loss:.4f}, "
                  f"val BA: {val_ba:.4f}{marker}  "
                  f"[{elapsed/60:.0f}m elapsed, ~{eta/60:.0f}m remaining]")

        elapsed_total = time.time() - start_time
        print(f"  └─── LR={lr} done in {elapsed_total/60:.1f} min — Best val BA: {best_epoch_ba:.4f}")

        all_results.append({"lr": lr, "epochs": NUM_EPOCHS, "val_ba": best_epoch_ba})
        all_training_curves[lr] = {"loss": epoch_losses, "val_ba": epoch_val_bas}

        if best_epoch_ba > best_val_ba:
            best_val_ba = best_epoch_ba
            best_config = {"lr": lr, "epochs": NUM_EPOCHS}
            best_model_state = best_epoch_model_state

    # ── HP search summary ──
    print(f"\n  Hyperparameter Search Results:")
    print(f"  {'LR':<12} {'Epochs':<8} {'Val BA':>8}")
    print(f"  {'-'*30}")
    for r in all_results:
        marker = " <-- best" if r["lr"] == best_config["lr"] else ""
        print(f"  {r['lr']:<12} {r['epochs']:<8} {r['val_ba']:>8.4f}{marker}")

    print(f"\n  Best config: {best_config}")
    print(f"  Best validation BA: {best_val_ba:.4f}")

    # ── Training curves plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for lr, curves in all_training_curves.items():
        epochs_range = range(1, NUM_EPOCHS + 1)
        ax1.plot(epochs_range, curves["loss"], label=f"LR={lr}")
        ax2.plot(epochs_range, curves["val_ba"], label=f"LR={lr}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title(f"R3D-18 Training Loss — {task}")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Balanced Accuracy")
    ax2.set_title(f"R3D-18 Val BA — {task}")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / f"r3d_{task}_training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Training curves saved to {path}")

    # ── Test evaluation ──
    model = build_model(num_classes, device)
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    test_preds, test_labels = evaluate(model, test_loader, device, desc="test")
    test_ba = balanced_accuracy_score(test_labels, test_preds)

    print(f"\n  TEST Balanced Accuracy: {test_ba:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=label_names, digits=4, zero_division=0))

    # ── Confusion matrix ──
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Purples", xticks_rotation=45)
    ax.set_title(f"R3D-18 — {task}\nTest Balanced Acc: {test_ba:.4f}")
    plt.tight_layout()
    path = RESULTS_DIR / f"r3d_{task}_confusion.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {path}")

    # ── Save model + config ──
    model_path = MODELS_DIR / f"r3d_{task}.pt"
    torch.save(best_model_state, model_path)
    config_path = MODELS_DIR / f"r3d_{task}_config.json"
    with open(config_path, "w") as f:
        json.dump({
            **best_config, "val_ba": best_val_ba, "test_ba": test_ba,
            "all_results": all_results,
        }, f, indent=2)
    print(f"  Model saved to {model_path}")

    return test_ba


def main():
    device = get_device()
    print(f"  Using device: {device}")

    ba_binary = run_task("binary", num_classes=2, label_names=OFFENCE_NAMES, device=device)
    ba_multi = run_task("multiclass", num_classes=7, label_names=FOUL_TYPE_NAMES, device=device)

    print(f"\n{'='*60}")
    print(f"  R3D-18 (3D CNN) SUMMARY")
    print(f"{'='*60}")
    print(f"  Binary (Offence Detection)   Test BA: {ba_binary:.4f}")
    print(f"  Multiclass (Foul Type)       Test BA: {ba_multi:.4f}")


if __name__ == "__main__":
    main()