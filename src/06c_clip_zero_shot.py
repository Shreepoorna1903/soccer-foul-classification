"""
06c_clip_zero_shot.py
Zero-shot classification using CLIP with majority voting across all frames.
Two voting strategies: hard vote (majority count) and soft vote (avg similarity).
Also supports multi-view voting across all camera angles.

Usage:
    pip install open-clip-torch
    python 06c_clip_zero_shot.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import open_clip
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from utils import (
    load_dataset, RESULTS_DIR, MODELS_DIR,
    FOUL_TYPE_NAMES, OFFENCE_NAMES,
)

# ── Configuration ──────────────────────────────────────────────
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
USE_ALL_CLIPS = True  # Use all camera angles, not just main camera
FRAME_SAMPLE_STEP = 5  # Sample every 5th frame (25 frames from 125) for speed

# ── Text Prompts ───────────────────────────────────────────────
BINARY_PROMPTS = {
    0: [
        "a fair challenge in soccer with no foul committed",
        "a clean tackle in a soccer match",
        "two soccer players competing fairly for the ball",
        "a legal challenge with no infringement",
    ],
    1: [
        "a soccer player committing a foul on an opponent",
        "an illegal tackle in a soccer match",
        "a player fouling another player during a soccer game",
        "a dangerous or reckless challenge in soccer",
    ],
}

MULTICLASS_PROMPTS = {
    0: [
        "a soccer player performing a standing tackle on an opponent",
        "a standing challenge where a player tackles while on their feet",
    ],
    1: [
        "a soccer player performing a sliding tackle on the ground",
        "a sliding challenge where a player goes to ground to win the ball",
    ],
    2: [
        "two soccer players in a physical shoulder to shoulder challenge",
        "a player using their body to challenge for the ball in soccer",
    ],
    3: [
        "a soccer player holding or grabbing an opponent's shirt or body",
        "a player illegally restraining another player in soccer",
    ],
    4: [
        "a soccer player elbowing an opponent",
        "a player striking another with their elbow during a soccer match",
    ],
    5: [
        "a soccer player raising their leg dangerously high near an opponent",
        "a high boot or high kick endangering another player in soccer",
    ],
    6: [
        "a soccer player pushing an opponent with their hands or arms",
        "a player shoving another player during a soccer match",
    ],
}


def load_clip_model(device):
    """Load CLIP model and tokenizer."""
    print(f"  Loading CLIP model ({CLIP_MODEL})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device).eval()
    print(f"  CLIP loaded on {device}")
    return model, preprocess, tokenizer


def encode_text_prompts(model, tokenizer, prompts_dict, device):
    """Encode all text prompts for each class, return averaged embeddings per class."""
    class_embeddings = {}
    with torch.no_grad():
        for label, prompts in prompts_dict.items():
            tokens = tokenizer(prompts).to(device)
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            # Average across prompts for this class
            avg_embedding = embeddings.mean(dim=0)
            avg_embedding = avg_embedding / avg_embedding.norm()
            class_embeddings[label] = avg_embedding
    return class_embeddings


def extract_all_frames(video_path, step=5):
    """Extract frames from video, sampling every `step` frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def predict_action(model, preprocess, class_embeddings, action, device, use_all_clips=True):
    """
    Predict label for an action using majority/soft voting across all frames.

    Returns:
        hard_pred: majority vote prediction
        soft_pred: highest average similarity prediction
        n_frames: total frames processed
    """
    # Collect frames from all clips or just main clip
    if use_all_clips:
        video_paths = action["all_clip_paths"]
    else:
        video_paths = [action["video_path"]]

    all_similarities = []  # list of (num_classes,) similarity vectors per frame

    for vpath in video_paths:
        frames = extract_all_frames(vpath, step=FRAME_SAMPLE_STEP)
        if not frames:
            continue

        # Process frames in batches of 32
        for i in range(0, len(frames), 32):
            batch_frames = frames[i:i+32]
            batch_tensors = torch.stack([preprocess(f) for f in batch_frames]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(batch_tensors)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity to each class
            class_labels = sorted(class_embeddings.keys())
            class_emb_stack = torch.stack([class_embeddings[c] for c in class_labels])  # (C, D)

            # (batch, D) @ (D, C) -> (batch, C)
            sims = (image_features @ class_emb_stack.T).cpu().numpy()
            all_similarities.append(sims)

    if not all_similarities:
        return 0, 0, 0

    all_similarities = np.vstack(all_similarities)  # (total_frames, num_classes)
    n_frames = all_similarities.shape[0]

    # Hard voting: each frame votes for highest similarity class
    frame_preds = all_similarities.argmax(axis=1)
    vote_counts = Counter(frame_preds)
    hard_pred = vote_counts.most_common(1)[0][0]

    # Soft voting: average similarity across frames, pick highest
    avg_sims = all_similarities.mean(axis=0)
    soft_pred = int(avg_sims.argmax())

    return hard_pred, soft_pred, n_frames


def run_task(task, prompts_dict, label_names, model, preprocess, tokenizer, device):
    """Run zero-shot CLIP classification for a task."""
    print(f"\n{'='*60}")
    print(f"  CLIP Zero-Shot — {task.upper()}")
    print(f"  Voting: all frames (step={FRAME_SAMPLE_STEP}) × {'all clips' if USE_ALL_CLIPS else 'main clip only'}")
    print(f"{'='*60}")

    # Encode text prompts
    class_embeddings = encode_text_prompts(model, tokenizer, prompts_dict, device)
    print(f"  Encoded {len(class_embeddings)} class prompts")

    # Load test data
    test_data = load_dataset("test", task=task)

    y_true = []
    y_hard = []
    y_soft = []
    total_frames = 0

    for item in tqdm(test_data, desc="  Predicting"):
        hard, soft, nf = predict_action(model, preprocess, class_embeddings, item, device, USE_ALL_CLIPS)
        y_true.append(item["label"])
        y_hard.append(hard)
        y_soft.append(soft)
        total_frames += nf

    y_true = np.array(y_true)
    y_hard = np.array(y_hard)
    y_soft = np.array(y_soft)

    avg_frames = total_frames / len(test_data) if test_data else 0
    print(f"\n  Avg frames per action: {avg_frames:.0f}")

    # ── Hard voting results ──
    ba_hard = balanced_accuracy_score(y_true, y_hard)
    print(f"\n  HARD VOTING — Test Balanced Accuracy: {ba_hard:.4f}")
    print(classification_report(y_true, y_hard, target_names=label_names, digits=4, zero_division=0))

    # ── Soft voting results ──
    ba_soft = balanced_accuracy_score(y_true, y_soft)
    print(f"  SOFT VOTING — Test Balanced Accuracy: {ba_soft:.4f}")
    print(classification_report(y_true, y_soft, target_names=label_names, digits=4, zero_division=0))

    # ── Confusion matrices ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_pred, vote_type, ba in [
        (axes[0], y_hard, "Hard Vote", ba_hard),
        (axes[1], y_soft, "Soft Vote", ba_soft),
    ]:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
        disp.plot(ax=ax, cmap="Reds", colorbar=False)
        ax.set_title(f"CLIP {vote_type} — {task}\nBA: {ba:.4f}", fontsize=11)
        if task == "multiclass":
            ax.tick_params(axis="x", rotation=45)

    plt.suptitle(f"CLIP Zero-Shot — {task.title()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = RESULTS_DIR / f"clip_{task}_confusion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    best_ba = max(ba_hard, ba_soft)
    best_method = "hard" if ba_hard >= ba_soft else "soft"
    print(f"\n  Best method: {best_method} voting (BA: {best_ba:.4f})")

    return ba_hard, ba_soft


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, preprocess, tokenizer = load_clip_model(device)

    # ── Binary ──
    ba_bin_hard, ba_bin_soft = run_task(
        "binary", BINARY_PROMPTS, OFFENCE_NAMES,
        model, preprocess, tokenizer, device
    )

    # ── Multiclass ──
    ba_mul_hard, ba_mul_soft = run_task(
        "multiclass", MULTICLASS_PROMPTS, FOUL_TYPE_NAMES,
        model, preprocess, tokenizer, device
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  CLIP ZERO-SHOT SUMMARY")
    print(f"{'='*60}")
    print(f"  Binary  — Hard: {ba_bin_hard:.4f}  Soft: {ba_bin_soft:.4f}")
    print(f"  Multi   — Hard: {ba_mul_hard:.4f}  Soft: {ba_mul_soft:.4f}")


if __name__ == "__main__":
    main()