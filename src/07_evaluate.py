"""
07_evaluate.py
Generate all comparison figures and tables for the final report.
Includes all 6 models: LogReg, SVM, ResNet-18, R3D-18, CLIP, ResNet-50+MLP.

Usage:
    python 07_evaluate.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.video import r3d_18
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import cv2
from PIL import Image
from tqdm import tqdm
import open_clip

from utils import (
    FEATURES_DIR, RESULTS_DIR, MODELS_DIR,
    FOUL_TYPE_NAMES, OFFENCE_NAMES,
    load_dataset, extract_middle_frame,
)

MODEL_NAMES = ["Logistic Reg.", "SVM (RBF)", "ResNet-18", "R3D-18", "CLIP (0-shot)", "ResNet-50+MLP"]
MODEL_KEYS = ["logreg", "svm", "resnet", "r3d", "clip", "resnet50mlp"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
CMAPS = ["Blues", "Oranges", "Greens", "Purples", "Reds", "YlOrRd"]
TASKS = ["binary", "multiclass"]


# ── Prediction Loaders ─────────────────────────────────────────

def load_sklearn_predictions(model_name, task):
    with open(MODELS_DIR / f"{model_name}_{task}.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, scaler = bundle["model"], bundle["scaler"]
    X_test = np.load(FEATURES_DIR / "test_combined.npy")
    if task == "binary":
        y_test = np.load(FEATURES_DIR / "test_labels_binary.npy")
        mask = y_test >= 0
        X_test, y_test = X_test[mask], y_test[mask]
    else:
        y_test = np.load(FEATURES_DIR / "test_labels_multi.npy")
    return y_test, model.predict(scaler.transform(X_test))


def load_resnet_predictions(task):
    num_classes = 2 if task == "binary" else 7
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    state = torch.load(MODELS_DIR / f"resnet_{task}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_data = load_dataset("test", task=task)
    y_true, y_pred = [], []
    with torch.no_grad():
        for item in test_data:
            frame = extract_middle_frame(item["video_path"])
            if frame is None: continue
            img = tf(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0).to(device)
            y_true.append(item["label"])
            y_pred.append(model(img).argmax(1).item())
    return np.array(y_true), np.array(y_pred)


def load_r3d_predictions(task):
    num_classes = 2 if task == "binary" else 7
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = r3d_18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    state = torch.load(MODELS_DIR / f"r3d_{task}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
    test_data = load_dataset("test", task=task)
    y_true, y_pred = [], []
    with torch.no_grad():
        for item in test_data:
            cap = cv2.VideoCapture(str(item["video_path"]))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total - 1, 16, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, f = cap.read()
                if ret:
                    frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (112, 112)))
            cap.release()
            while len(frames) < 16:
                frames.append(frames[-1] if frames else np.zeros((112, 112, 3), dtype=np.uint8))
            clip = np.stack(frames[:16]).astype(np.float32) / 255.0
            clip = ((clip - mean) / std).transpose(3, 0, 1, 2)
            clip = torch.from_numpy(clip).float().unsqueeze(0).to(device)
            y_true.append(item["label"])
            y_pred.append(model(clip).argmax(1).item())
    return np.array(y_true), np.array(y_pred)


def load_clip_predictions(task):
    """Run CLIP zero-shot with soft voting across all frames."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()

    if task == "binary":
        prompts = {
            0: ["a fair challenge in soccer with no foul committed",
                "a clean tackle in a soccer match"],
            1: ["a soccer player committing a foul on an opponent",
                "an illegal tackle in a soccer match"],
        }
    else:
        prompts = {
            0: ["a soccer player performing a standing tackle on an opponent"],
            1: ["a soccer player performing a sliding tackle on the ground"],
            2: ["two soccer players in a physical shoulder to shoulder challenge"],
            3: ["a soccer player holding or grabbing an opponent's shirt"],
            4: ["a soccer player elbowing an opponent"],
            5: ["a soccer player raising their leg dangerously high"],
            6: ["a soccer player pushing an opponent"],
        }

    # Encode text
    class_embs = {}
    with torch.no_grad():
        for label, texts in prompts.items():
            tokens = tokenizer(texts).to(device)
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            class_embs[label] = emb.mean(0)
            class_embs[label] = class_embs[label] / class_embs[label].norm()

    class_labels = sorted(class_embs.keys())
    class_stack = torch.stack([class_embs[c] for c in class_labels])

    test_data = load_dataset("test", task=task)
    y_true, y_pred = [], []

    for item in test_data:
        all_sims = []
        for vpath in item["all_clip_paths"]:
            cap = cv2.VideoCapture(str(vpath))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(0, total, 5):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, f = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                    img_t = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = clip_model.encode_image(img_t)
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                    sims = (feat @ class_stack.T).cpu().numpy()[0]
                    all_sims.append(sims)
            cap.release()

        if all_sims:
            avg = np.mean(all_sims, axis=0)
            y_pred.append(int(avg.argmax()))
        else:
            y_pred.append(0)
        y_true.append(item["label"])

    return np.array(y_true), np.array(y_pred)


def load_resnet50mlp_predictions(task):
    suffix = "binary" if task == "binary" else "multi"
    data = np.load(FEATURES_DIR / f"resnet50_test_{suffix}.npz")
    X_test, y_test = data["X"], data["y"]

    with open(MODELS_DIR / f"resnet50mlp_{task}_config.json") as f:
        config = json.load(f)

    num_classes = 2 if task == "binary" else 7
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    from importlib import import_module
    # Inline MLP definition to avoid circular import
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

    model = MLP(2048, config["hidden_dim"], num_classes, config["dropout"]).to(device)
    state = torch.load(MODELS_DIR / f"resnet50mlp_{task}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()

    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(xt).argmax(1).cpu().numpy()

    return y_test, preds


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  GENERATING EVALUATION RESULTS (6 models)")
    print("=" * 60)

    # ── Collect predictions ──
    loaders = {
        "logreg": lambda t: load_sklearn_predictions("logreg", t),
        "svm": lambda t: load_sklearn_predictions("svm", t),
        "resnet": lambda t: load_resnet_predictions(t),
        "r3d": lambda t: load_r3d_predictions(t),
        "clip": lambda t: load_clip_predictions(t),
        "resnet50mlp": lambda t: load_resnet50mlp_predictions(t),
    }

    predictions = {}
    for task in TASKS:
        predictions[task] = {}
        print(f"\n  --- {task} ---")
        for key, name in zip(MODEL_KEYS, MODEL_NAMES):
            print(f"    Loading {name}...", end=" ", flush=True)
            y_true, y_pred = loaders[key](task)
            predictions[task][key] = (y_true, y_pred)
            ba = balanced_accuracy_score(y_true, y_pred)
            print(f"BA: {ba:.4f}")

    # ── 1. Summary Table ──
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY TABLE")
    print(f"{'='*60}")
    header = f"  {'Model':<20} {'Binary BA':>12} {'Multi BA':>12}"
    sep = f"  {'-'*44}"
    print(header)
    print(sep)

    table_lines = [header, sep]
    for key, name in zip(MODEL_KEYS, MODEL_NAMES):
        ba_bin = balanced_accuracy_score(*predictions["binary"][key])
        ba_mul = balanced_accuracy_score(*predictions["multiclass"][key])
        line = f"  {name:<20} {ba_bin:>12.4f} {ba_mul:>12.4f}"
        print(line)
        table_lines.append(line)

    with open(RESULTS_DIR / "comparison_table.txt", "w") as f:
        f.write("\n".join(table_lines))
    print(f"\n  Saved comparison_table.txt")

    # ── 2. Bar Charts ──
    for task in TASKS:
        scores = [balanced_accuracy_score(*predictions[task][k]) for k in MODEL_KEYS]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(MODEL_NAMES, scores, color=COLORS, edgecolor="white", linewidth=1.2)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        task_label = "Binary Offence Detection" if task == "binary" else "7-Class Foul Type"
        ax.set_title(f"Test Balanced Accuracy — {task_label}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Balanced Accuracy", fontsize=12)
        ax.set_ylim(0, max(scores) + 0.12)
        random_line = 0.5 if task == "binary" else 1 / 7
        ax.axhline(y=random_line, color="gray", linestyle="--", alpha=0.7, label=f"Random ({random_line:.3f})")
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        path = RESULTS_DIR / f"comparison_{task}_bar.png"
        fig.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path.name}")

    # ── 3. All Confusion Matrices (2x6 grid) ──
    fig, axes = plt.subplots(2, 6, figsize=(36, 10))
    for row, task in enumerate(TASKS):
        label_names = OFFENCE_NAMES if task == "binary" else FOUL_TYPE_NAMES
        for col, (key, name) in enumerate(zip(MODEL_KEYS, MODEL_NAMES)):
            y_true, y_pred = predictions[task][key]
            cm = confusion_matrix(y_true, y_pred)
            ba = balanced_accuracy_score(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
            disp.plot(ax=axes[row][col], cmap=CMAPS[col], colorbar=False)
            task_short = "Binary" if task == "binary" else "Multi"
            axes[row][col].set_title(f"{name}\n{task_short} BA: {ba:.3f}", fontsize=9)
            if task == "multiclass":
                axes[row][col].tick_params(axis="x", rotation=45, labelsize=7)
                axes[row][col].tick_params(axis="y", labelsize=7)
    plt.suptitle("Confusion Matrices — All Models × Both Tasks", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = RESULTS_DIR / "all_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")

    # ── 4. Per-Class F1 Grouped Bar Chart (Multiclass) ──
    fig, ax = plt.subplots(figsize=(14, 6))
    n_classes = len(FOUL_TYPE_NAMES)
    x = np.arange(n_classes)
    width = 0.13
    for i, (key, name) in enumerate(zip(MODEL_KEYS, MODEL_NAMES)):
        y_true, y_pred = predictions["multiclass"][key]
        f1s = f1_score(y_true, y_pred, average=None, labels=range(n_classes), zero_division=0)
        ax.bar(x + i * width, f1s, width, label=name, color=COLORS[i], edgecolor="white")
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(FOUL_TYPE_NAMES, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score — 7-Class Foul Type", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = RESULTS_DIR / "per_class_f1_multiclass.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")

    # ── 5. Dataset Class Distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for split, color in [("train", "#4C72B0"), ("valid", "#55A868"), ("test", "#DD8452")]:
        ds = load_dataset(split, "binary")
        labels = [d["label"] for d in ds]
        counts = [labels.count(i) for i in range(2)]
        x = np.arange(2)
        w = 0.25
        offset = {"train": -w, "valid": 0, "test": w}[split]
        axes[0].bar(x + offset, counts, w, label=split, alpha=0.85)
    axes[0].set_xticks(range(2))
    axes[0].set_xticklabels(OFFENCE_NAMES)
    axes[0].set_title("Binary: Offence Detection", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    for split in ["train", "valid", "test"]:
        ds = load_dataset(split, "multiclass")
        labels = [d["label"] for d in ds]
        counts = [labels.count(i) for i in range(7)]
        x = np.arange(7)
        w = 0.25
        offset = {"train": -w, "valid": 0, "test": w}[split]
        axes[1].bar(x + offset, counts, w, label=split, alpha=0.85)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(FOUL_TYPE_NAMES, rotation=30, ha="right", fontsize=9)
    axes[1].set_title("Multiclass: Foul Type", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.suptitle("Class Distribution Across Splits", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = RESULTS_DIR / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path.name}")

    # ── Done ──
    print(f"\n{'='*60}")
    print(f"  ALL OUTPUTS SAVED TO {RESULTS_DIR}/")
    print(f"{'='*60}")
    for f in sorted(RESULTS_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<45} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()