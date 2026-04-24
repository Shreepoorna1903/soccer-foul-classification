"""
03_extract_features.py
Extract handcrafted features from video clips:
  1. HOG (Histogram of Oriented Gradients) from middle frame
  2. Optical Flow histogram from frame pair around middle
  3. BoVW (Bag of Visual Words) using SIFT descriptors + KMeans

Features are saved per-split as .npy files in features/.

Usage:
    python src/03_extract_features.py

Estimated time: ~10-15 min on M4 MacBook Air
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import pickle

from utils import (
    load_dataset, extract_middle_frame, extract_frame_pair,
    FEATURES_DIR, FOUL_TYPE_NAMES, OFFENCE_NAMES,
)

# ── Configuration ──────────────────────────────────────────────
FRAME_SIZE = (224, 224)        # Resize frames for consistent feature dims
HOG_CELL_SIZE = (16, 16)      # HOG cell size
HOG_BLOCK_SIZE = (2, 2)       # HOG block size (in cells)
HOG_NBINS = 9                 # HOG orientation bins
BOVW_K = 200                  # Number of visual words
FLOW_BINS = 32                # Bins for optical flow magnitude/angle histograms
SIFT_MAX_KEYPOINTS = 200      # Max SIFT keypoints per image


def compute_hog(frame_bgr):
    """Compute HOG descriptor for a BGR frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FRAME_SIZE)

    hog = cv2.HOGDescriptor(
        _winSize=FRAME_SIZE,
        _blockSize=(HOG_CELL_SIZE[0] * HOG_BLOCK_SIZE[0],
                    HOG_CELL_SIZE[1] * HOG_BLOCK_SIZE[1]),
        _blockStride=HOG_CELL_SIZE,
        _cellSize=HOG_CELL_SIZE,
        _nbins=HOG_NBINS,
    )
    descriptor = hog.compute(gray)
    return descriptor.flatten()


def compute_optical_flow_hist(gray1, gray2):
    """Compute histogram of optical flow magnitudes and angles."""
    gray1 = cv2.resize(gray1, FRAME_SIZE)
    gray2 = cv2.resize(gray2, FRAME_SIZE)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Histogram of magnitudes
    mag_hist, _ = np.histogram(mag, bins=FLOW_BINS, range=(0, 20))
    # Histogram of angles
    ang_hist, _ = np.histogram(ang, bins=FLOW_BINS, range=(0, 2 * np.pi))

    # Normalize
    mag_hist = mag_hist.astype(np.float32)
    ang_hist = ang_hist.astype(np.float32)
    mag_sum = mag_hist.sum()
    ang_sum = ang_hist.sum()
    if mag_sum > 0:
        mag_hist /= mag_sum
    if ang_sum > 0:
        ang_hist /= ang_sum

    return np.concatenate([mag_hist, ang_hist])


def extract_sift_descriptors(frame_bgr):
    """Extract SIFT descriptors from a BGR frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FRAME_SIZE)

    sift = cv2.SIFT_create(nfeatures=SIFT_MAX_KEYPOINTS)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        return np.zeros((1, 128), dtype=np.float32)
    return descriptors.astype(np.float32)


def build_bovw_vocabulary(dataset, split="train"):
    """Build BoVW vocabulary using KMeans on SIFT descriptors from training set."""
    print(f"\n  Building BoVW vocabulary (K={BOVW_K}) from {split}...")
    all_descriptors = []

    for item in tqdm(dataset, desc="  Collecting SIFT descriptors"):
        frame = extract_middle_frame(item["video_path"], resize=None)
        if frame is None:
            continue
        desc = extract_sift_descriptors(frame)
        all_descriptors.append(desc)

    all_descriptors = np.vstack(all_descriptors)
    print(f"  Total descriptors: {all_descriptors.shape[0]}")

    kmeans = MiniBatchKMeans(
        n_clusters=BOVW_K,
        batch_size=1000,
        random_state=42,
        n_init=3,
    )
    kmeans.fit(all_descriptors)

    # Save vocabulary
    vocab_path = FEATURES_DIR / "bovw_vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"  Vocabulary saved to {vocab_path}")

    return kmeans


def compute_bovw_histogram(frame_bgr, kmeans):
    """Compute BoVW histogram for a single frame using pre-built vocabulary."""
    desc = extract_sift_descriptors(frame_bgr)
    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=np.arange(BOVW_K + 1))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_all_features(dataset, split, kmeans):
    """Extract HOG, Optical Flow, and BoVW features for a dataset."""
    hog_features = []
    flow_features = []
    bovw_features = []
    binary_labels = []
    multi_labels = []
    valid_indices = []

    for i, item in enumerate(tqdm(dataset, desc=f"  Extracting [{split}]")):
        frame = extract_middle_frame(item["video_path"])
        if frame is None:
            continue

        # HOG
        hog = compute_hog(frame)
        hog_features.append(hog)

        # Optical Flow
        g1, g2 = extract_frame_pair(item["video_path"], offset=5)
        if g1 is not None and g2 is not None:
            of_hist = compute_optical_flow_hist(g1, g2)
        else:
            of_hist = np.zeros(FLOW_BINS * 2, dtype=np.float32)
        flow_features.append(of_hist)

        # BoVW
        bovw = compute_bovw_histogram(frame, kmeans)
        bovw_features.append(bovw)

        valid_indices.append(i)

    hog_features = np.array(hog_features, dtype=np.float32)
    flow_features = np.array(flow_features, dtype=np.float32)
    bovw_features = np.array(bovw_features, dtype=np.float32)

    # Concatenated feature vector
    combined = np.concatenate([hog_features, flow_features, bovw_features], axis=1)

    print(f"\n  Feature dimensions for {split}:")
    print(f"    HOG:      {hog_features.shape}")
    print(f"    Flow:     {flow_features.shape}")
    print(f"    BoVW:     {bovw_features.shape}")
    print(f"    Combined: {combined.shape}")

    return {
        "hog": hog_features,
        "flow": flow_features,
        "bovw": bovw_features,
        "combined": combined,
        "valid_indices": valid_indices,
    }


def save_features(features, labels_binary, labels_multi, split):
    """Save extracted features and labels to disk."""
    prefix = FEATURES_DIR / split

    np.save(f"{prefix}_hog.npy", features["hog"])
    np.save(f"{prefix}_flow.npy", features["flow"])
    np.save(f"{prefix}_bovw.npy", features["bovw"])
    np.save(f"{prefix}_combined.npy", features["combined"])
    np.save(f"{prefix}_labels_binary.npy", labels_binary)
    np.save(f"{prefix}_labels_multi.npy", labels_multi)

    print(f"  Saved features to {FEATURES_DIR}/{split}_*.npy")


def main():
    # ── Step 1: Load datasets ──
    # We load multiclass dataset (superset of binary after label filtering)
    # and derive binary labels from the raw annotations
    print("=" * 60)
    print("  FEATURE EXTRACTION PIPELINE")
    print("=" * 60)

    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = load_dataset(split, task="multiclass")

    # ── Step 2: Build BoVW vocabulary on train set ──
    vocab_path = FEATURES_DIR / "bovw_vocab.pkl"
    if vocab_path.exists():
        print(f"\n  Loading existing vocabulary from {vocab_path}")
        with open(vocab_path, "rb") as f:
            kmeans = pickle.load(f)
    else:
        kmeans = build_bovw_vocabulary(datasets["train"], split="train")

    # ── Step 3: Extract features for each split ──
    for split in ["train", "valid", "test"]:
        print(f"\n{'='*60}")
        print(f"  Processing {split.upper()} split")
        print(f"{'='*60}")

        ds = datasets[split]
        features = extract_all_features(ds, split, kmeans)

        # Build label arrays using valid indices only
        valid_idx = features["valid_indices"]
        labels_multi = np.array([ds[i]["label"] for i in valid_idx], dtype=np.int64)

        # Binary labels: map from raw annotation
        # Actions in multiclass set may have Offence/No offence/Between
        # We map Offence->1, No offence->0, skip others
        labels_binary = []
        binary_mask = []
        for i in valid_idx:
            offence = ds[i]["raw"].get("Offence", "").strip()
            if offence == "Offence":
                labels_binary.append(1)
                binary_mask.append(True)
            elif offence == "No offence":
                labels_binary.append(0)
                binary_mask.append(True)
            else:
                labels_binary.append(-1)  # placeholder, will be filtered at train time
                binary_mask.append(False)

        labels_binary = np.array(labels_binary, dtype=np.int64)

        save_features(features, labels_binary, labels_multi, split)

    print(f"\n{'='*60}")
    print("  FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()