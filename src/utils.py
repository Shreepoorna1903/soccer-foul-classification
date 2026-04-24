"""
utils.py
Shared helpers: data loading, label mapping, frame extraction, metrics.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "mvfoul"
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

for d in [FEATURES_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

# ── Label Mappings ─────────────────────────────────────────────

# Task 1: Binary offence detection
OFFENCE_MAP = {
    "Offence": 1,
    "No offence": 0,
}
# "Between" and empty strings are dropped

# Task 2: 7-class foul type
FOUL_TYPE_MAP = {
    "Standing tackling": 0,
    "Tackling": 1,
    "Challenge": 2,
    "Holding": 3,
    "Elbowing": 4,
    "High leg": 5,
    "Pushing": 6,
}
# "Dont know", "Dive", and empty strings are dropped

FOUL_TYPE_NAMES = [k for k, v in sorted(FOUL_TYPE_MAP.items(), key=lambda x: x[1])]
OFFENCE_NAMES = ["No offence", "Offence"]


def load_annotations(split):
    """Load raw annotations dict for a split."""
    path = DATA_DIR / split / "annotations.json"
    with open(path) as f:
        data = json.load(f)
    return data["Actions"]


def get_main_camera_clip(action):
    """Return the clip dict for the main camera view (clip_0).
    Falls back to first clip if no main camera found."""
    clips = action.get("Clips", [])
    for clip in clips:
        if "Main camera" in clip.get("Camera type", ""):
            return clip
    return clips[0] if clips else None


def get_clip_video_path(split, clip_dict):
    """Convert clip URL to local file path.
    Clip URL format: 'Dataset/Train/action_X/clip_Y'
    """
    url = clip_dict["Url"]
    # Extract action_X/clip_Y from the URL
    parts = url.split("/")
    action_folder = parts[-2]  # action_X
    clip_name = parts[-1]      # clip_Y
    return DATA_DIR / split / action_folder / f"{clip_name}.mp4"


def load_dataset(split, task="binary"):
    """
    Load dataset for a given split and task.

    Args:
        split: "train", "valid", or "test"
        task: "binary" (offence detection) or "multiclass" (foul type)

    Returns:
        actions: list of dicts, each with:
            - "action_id": str
            - "label": int
            - "label_name": str
            - "video_path": Path to main camera clip
            - "all_clip_paths": list of Paths to all clips
            - "raw": original annotation dict
    """
    raw_actions = load_annotations(split)
    label_map = OFFENCE_MAP if task == "binary" else FOUL_TYPE_MAP
    label_key = "Offence" if task == "binary" else "Action class"

    dataset = []
    skipped = 0

    for action_id, action in raw_actions.items():
        label_str = action.get(label_key, "").strip()

        if label_str not in label_map:
            skipped += 1
            continue

        main_clip = get_main_camera_clip(action)
        if main_clip is None:
            skipped += 1
            continue

        video_path = get_clip_video_path(split, main_clip)
        if not video_path.exists():
            skipped += 1
            continue

        # All clip paths
        all_clips = []
        for clip in action.get("Clips", []):
            p = get_clip_video_path(split, clip)
            if p.exists():
                all_clips.append(p)

        dataset.append({
            "action_id": action_id,
            "label": label_map[label_str],
            "label_name": label_str,
            "video_path": video_path,
            "all_clip_paths": all_clips,
            "raw": action,
        })

    print(f"  [{split}/{task}] Loaded {len(dataset)} actions, skipped {skipped}")
    return dataset


def extract_middle_frame(video_path, resize=None):
    """
    Extract the middle frame from a video file.

    Args:
        video_path: Path to .mp4 file
        resize: optional (width, height) tuple to resize

    Returns:
        frame: numpy array (H, W, 3) in BGR, or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = n_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    if resize is not None:
        frame = cv2.resize(frame, resize)

    return frame


def extract_frame_pair(video_path, offset=5):
    """
    Extract two frames separated by `offset` frames around the middle.
    Used for optical flow computation.

    Returns:
        (frame1, frame2): both as grayscale numpy arrays, or (None, None)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = n_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, mid - offset)
    ret1, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid + offset)
    ret2, f2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        return None, None

    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    return gray1, gray2


def print_class_distribution(dataset, task="binary"):
    """Print class distribution for a loaded dataset."""
    names = OFFENCE_NAMES if task == "binary" else FOUL_TYPE_NAMES
    labels = [d["label"] for d in dataset]
    counts = Counter(labels)
    total = len(labels)

    print(f"\n  {'Class':<25} {'Count':>6} {'%':>7}")
    print(f"  {'-'*40}")
    for i, name in enumerate(names):
        c = counts.get(i, 0)
        print(f"  {name:<25} {c:>6} {100*c/total:>6.1f}%")
    print(f"  {'Total':<25} {total:>6}")