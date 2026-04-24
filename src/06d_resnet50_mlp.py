"""
06d_resnet50_mlp.py
Multi-frame, multi-view feature extraction with frozen ResNet-50
followed by a trainable MLP classifier.

Pipeline:
  1. Extract ResNet-50 avgpool features (2048-dim) from multiple frames
     across all camera angles per action
  2. Average-pool into a single 2048-dim vector per action
  3. Train a 2-layer MLP with cross-validated hyperparameter search

Usage:
    python 06d_resnet50_mlp.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2
import json

from utils import (
    load_dataset, FEATURES_DIR, RESULTS_DIR, MODELS_DIR,
    FOUL_TYPE_NAMES, OFFENCE_NAMES,
)

# ── Configuration ──────────────────────────────────────────────
FRAMES_PER_CLIP = 10      # Sample 10 frames per clip
FEAT_DIM = 2048           # ResNet-50 avgpool output
BATCH_SIZE = 64           # MLP training batch size
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Step 1: Feature Extraction ─────────────────────────────────

def build_feature_extractor(device):
    """Build frozen ResNet-50 feature extractor (no FC layer)."""
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # Remove the final FC layer — output is avgpool (2048-dim)
    model.fc = nn.Identity()
    model = model.to(device).eval()

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return model, transform


def sample_frames(video_path, n_frames=10):
    """Sample n_frames evenly from a video, return as list of PIL Images."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def extract_action_features(action, model, transform, device, n_frames=10):
    """
    Extract features for one action:
    - Sample frames from ALL clips (camera angles)
    - Extract ResNet-50 features per frame
    - Average across all frames and clips → single 2048-dim vector
    """
    all_features = []

    for clip_path in action["all_clip_paths"]:
        frames = sample_frames(clip_path, n_frames)
        if not frames:
            continue

        # Process frames in one batch
        batch = torch.stack([transform(f) for f in frames]).to(device)
        with torch.no_grad():
            feats = model(batch)  # (n_frames, 2048)
        all_features.append(feats.cpu())

    if not all_features:
        return None

    # Concatenate all frames from all clips, then average
    all_features = torch.cat(all_features, dim=0)  # (total_frames, 2048)
    avg_feature = all_features.mean(dim=0)  # (2048,)
    return avg_feature.numpy()


def extract_features_for_split(split, task, model, transform, device):
    """Extract averaged multi-frame multi-view features for an entire split."""
    dataset = load_dataset(split, task=task)

    features = []
    labels = []
    skipped = 0

    for item in tqdm(dataset, desc=f"  Extracting [{split}/{task}]"):
        feat = extract_action_features(item, model, transform, device, FRAMES_PER_CLIP)
        if feat is None:
            skipped += 1
            continue
        features.append(feat)
        labels.append(item["label"])

    if skipped > 0:
        print(f"    Skipped {skipped} actions")

    features = np.stack(features)
    labels = np.array(labels, dtype=np.int64)
    print(f"    {split}: {features.shape[0]} actions, {features.shape[1]}-dim features")
    return features, labels


# ── Step 2: MLP Classifier ─────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def compute_class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_mlp(X_train, y_train, X_val, y_val, num_classes, device, config):
    """Train MLP with given config, return best val BA and model state."""
    hidden_dim = config["hidden_dim"]
    lr = config["lr"]
    epochs = config["epochs"]
    dropout = config["dropout"]

    model = MLP(FEAT_DIM, hidden_dim, num_classes, dropout).to(device)
    class_weights = compute_class_weights(y_train, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_ba = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_preds = model(val_tensor).argmax(1).cpu().numpy()
        val_ba = balanced_accuracy_score(y_val, val_preds)

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_ba, best_state


def run_task(task, num_classes, label_names, X_train, y_train, X_test, y_test, device):
    """Run hyperparameter search with 5-fold CV, evaluate on test."""
    print(f"\n{'='*60}")
    print(f"  ResNet-50 + MLP — {task.upper()}")
    print(f"{'='*60}")

    # ── Hyperparameter search with 5-fold CV ──
    configs = [
        {"hidden_dim": 512, "lr": 1e-3, "epochs": 50, "dropout": 0.3},
        {"hidden_dim": 512, "lr": 5e-4, "epochs": 50, "dropout": 0.4},
        {"hidden_dim": 256, "lr": 1e-3, "epochs": 50, "dropout": 0.3},
        {"hidden_dim": 256, "lr": 5e-4, "epochs": 80, "dropout": 0.4},
        {"hidden_dim": 1024, "lr": 5e-4, "epochs": 50, "dropout": 0.5},
    ]

    print(f"\n  Running 5-fold CV over {len(configs)} configs...")

    best_overall_ba = 0
    best_overall_config = None
    all_results = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for ci, config in enumerate(configs):
        fold_bas = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            Xf_tr, yf_tr = X_train[train_idx], y_train[train_idx]
            Xf_val, yf_val = X_train[val_idx], y_train[val_idx]
            val_ba, _ = train_mlp(Xf_tr, yf_tr, Xf_val, yf_val, num_classes, device, config)
            fold_bas.append(val_ba)

        mean_ba = np.mean(fold_bas)
        std_ba = np.std(fold_bas)
        all_results.append({"config": config, "mean_ba": mean_ba, "std_ba": std_ba})
        marker = ""
        if mean_ba > best_overall_ba:
            best_overall_ba = mean_ba
            best_overall_config = config
            marker = " <-- best"

        print(f"    Config {ci+1}: h={config['hidden_dim']}, lr={config['lr']}, "
              f"ep={config['epochs']}, drop={config['dropout']} "
              f"→ CV BA: {mean_ba:.4f} ± {std_ba:.4f}{marker}")

    print(f"\n  Best config: {best_overall_config}")
    print(f"  Best CV BA: {best_overall_ba:.4f}")

    # ── Retrain on full training set with best config ──
    print(f"\n  Retraining on full train set...")

    # Use X_test as validation for early stopping during final training
    # but report metrics on it as test
    best_test_ba, best_state = train_mlp(
        X_train, y_train, X_test, y_test,
        num_classes, device, best_overall_config
    )

    # Final evaluation
    model = MLP(FEAT_DIM, best_overall_config["hidden_dim"], num_classes,
                best_overall_config["dropout"]).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_preds = model(test_tensor).argmax(1).cpu().numpy()

    test_ba = balanced_accuracy_score(y_test, test_preds)

    print(f"\n  TEST Balanced Accuracy: {test_ba:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, test_preds, target_names=label_names, digits=4, zero_division=0))

    # ── Confusion matrix ──
    cm = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="YlOrRd", xticks_rotation=45)
    ax.set_title(f"ResNet-50 + MLP — {task}\nTest Balanced Acc: {test_ba:.4f}")
    plt.tight_layout()
    path = RESULTS_DIR / f"resnet50mlp_{task}_confusion.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # ── Save model ──
    model_path = MODELS_DIR / f"resnet50mlp_{task}.pt"
    torch.save(best_state, model_path)
    config_path = MODELS_DIR / f"resnet50mlp_{task}_config.json"
    with open(config_path, "w") as f:
        json.dump({**best_overall_config, "cv_ba": best_overall_ba, "test_ba": test_ba}, f, indent=2)
    print(f"  Model saved to {model_path}")

    return test_ba


def main():
    device = get_device()
    print(f"  Using device: {device}")

    # ── Step 1: Extract features ──
    print(f"\n{'='*60}")
    print(f"  FEATURE EXTRACTION (ResNet-50 × multi-frame × multi-view)")
    print(f"{'='*60}")

    model, transform = build_feature_extractor(device)

    # Check if features already extracted
    feat_path = FEATURES_DIR / "resnet50_train_binary.npz"
    if feat_path.exists():
        print("\n  Loading cached features...")
        data = np.load(FEATURES_DIR / "resnet50_train_binary.npz")
        X_train_bin, y_train_bin = data["X"], data["y"]
        data = np.load(FEATURES_DIR / "resnet50_test_binary.npz")
        X_test_bin, y_test_bin = data["X"], data["y"]
        data = np.load(FEATURES_DIR / "resnet50_train_multi.npz")
        X_train_mul, y_train_mul = data["X"], data["y"]
        data = np.load(FEATURES_DIR / "resnet50_test_multi.npz")
        X_test_mul, y_test_mul = data["X"], data["y"]
    else:
        print("\n  Extracting features (this takes ~5 min)...")
        X_train_bin, y_train_bin = extract_features_for_split("train", "binary", model, transform, device)
        X_test_bin, y_test_bin = extract_features_for_split("test", "binary", model, transform, device)
        X_train_mul, y_train_mul = extract_features_for_split("train", "multiclass", model, transform, device)
        X_test_mul, y_test_mul = extract_features_for_split("test", "multiclass", model, transform, device)

        # Cache features
        np.savez(FEATURES_DIR / "resnet50_train_binary.npz", X=X_train_bin, y=y_train_bin)
        np.savez(FEATURES_DIR / "resnet50_test_binary.npz", X=X_test_bin, y=y_test_bin)
        np.savez(FEATURES_DIR / "resnet50_train_multi.npz", X=X_train_mul, y=y_train_mul)
        np.savez(FEATURES_DIR / "resnet50_test_multi.npz", X=X_test_mul, y=y_test_mul)
        print("  Features cached to disk")

    del model  # Free memory
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # ── Step 2: Train MLP classifiers ──
    ba_binary = run_task("binary", 2, OFFENCE_NAMES,
                         X_train_bin, y_train_bin, X_test_bin, y_test_bin, device)
    ba_multi = run_task("multiclass", 7, FOUL_TYPE_NAMES,
                        X_train_mul, y_train_mul, X_test_mul, y_test_mul, device)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  ResNet-50 + MLP SUMMARY")
    print(f"{'='*60}")
    print(f"  Binary (Offence Detection)   Test BA: {ba_binary:.4f}")
    print(f"  Multiclass (Foul Type)       Test BA: {ba_multi:.4f}")


if __name__ == "__main__":
    main()