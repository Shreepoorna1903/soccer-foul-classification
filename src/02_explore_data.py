"""
02_explore_data.py
Explore the SoccerNet-MVFoul dataset: annotation structure, class distributions,
sample video properties. Saves distribution plots to results/.

Usage:
    python src/02_explore_data.py
"""

import json
import os
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = Path("data/mvfoul")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
SPLITS = ["train", "valid", "test"]


def load_annotations(split):
    """Load annotations for a given split."""
    path = DATA_DIR / split / "annotations.json"
    with open(path) as f:
        data = json.load(f)
    return data


def explore_annotations():
    """Print annotation structure and class distributions for all splits."""

    for split in SPLITS:
        data = load_annotations(split)
        actions = data["Actions"]
        n = data["Number of actions"]

        print(f"\n{'='*60}")
        print(f"  {split.upper()} SPLIT — {n} actions")
        print(f"{'='*60}")

        # ── Collect label distributions ──
        offence_counts = Counter()
        action_class_counts = Counter()
        severity_counts = Counter()
        bodypart_counts = Counter()
        contact_counts = Counter()
        clips_per_action = []

        for idx, action in actions.items():
            offence_counts[action.get("Offence", "Unknown")] += 1
            action_class_counts[action.get("Action class", "Unknown")] += 1
            severity_counts[action.get("Severity", "Unknown")] += 1
            bodypart_counts[action.get("Bodypart", "Unknown")] += 1
            contact_counts[action.get("Contact", "Unknown")] += 1
            clips_per_action.append(len(action.get("Clips", [])))

        # ── Task 1: Binary Offence Detection ──
        print(f"\n  TASK 1 — Binary Offence Detection:")
        print(f"  {'Label':<20} {'Count':>6} {'%':>7}")
        print(f"  {'-'*35}")
        for label, count in offence_counts.most_common():
            print(f"  {label:<20} {count:>6} {100*count/n:>6.1f}%")

        # ── Task 2: Foul Type (Action Class) ──
        print(f"\n  TASK 2 — Foul Type Classification (Action class):")
        print(f"  {'Label':<25} {'Count':>6} {'%':>7}")
        print(f"  {'-'*40}")
        for label, count in action_class_counts.most_common():
            print(f"  {label:<25} {count:>6} {100*count/n:>6.1f}%")

        # ── Additional attributes ──
        print(f"\n  Severity distribution:")
        print(f"  {'Label':<20} {'Count':>6} {'%':>7}")
        print(f"  {'-'*35}")
        for label, count in severity_counts.most_common():
            print(f"  {label:<20} {count:>6} {100*count/n:>6.1f}%")

        print(f"\n  Body part distribution:")
        print(f"  {'Label':<25} {'Count':>6} {'%':>7}")
        print(f"  {'-'*40}")
        for label, count in bodypart_counts.most_common():
            print(f"  {label:<25} {count:>6} {100*count/n:>6.1f}%")

        print(f"\n  Contact distribution:")
        print(f"  {'Label':<25} {'Count':>6} {'%':>7}")
        print(f"  {'-'*40}")
        for label, count in contact_counts.most_common():
            print(f"  {label:<25} {count:>6} {100*count/n:>6.1f}%")

        # ── Clips per action stats ──
        clips_arr = np.array(clips_per_action)
        print(f"\n  Clips per action: min={clips_arr.min()}, max={clips_arr.max()}, "
              f"mean={clips_arr.mean():.1f}, median={np.median(clips_arr):.0f}")
        clip_dist = Counter(clips_per_action)
        for nc in sorted(clip_dist.keys()):
            print(f"    {nc} clips: {clip_dist[nc]} actions")


def inspect_sample_video():
    """Read one sample video and print frame/resolution info."""
    print(f"\n{'='*60}")
    print(f"  SAMPLE VIDEO PROPERTIES")
    print(f"{'='*60}")

    sample_path = DATA_DIR / "train" / "action_0" / "clip_0.mp4"
    if not sample_path.exists():
        print(f"  Sample not found at {sample_path}")
        return

    cap = cv2.VideoCapture(str(sample_path))
    if not cap.isOpened():
        print(f"  Could not open {sample_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"  File: {sample_path}")
    print(f"  Resolution: {width} x {height}")
    print(f"  FPS: {fps}")
    print(f"  Frame count: {frame_count}")
    print(f"  Duration: {duration:.2f}s")

    # Read middle frame as sanity check
    mid = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    if ret:
        print(f"  Middle frame shape: {frame.shape} (H x W x C)")
    cap.release()


def count_total_clips():
    """Count total .mp4 files per split."""
    print(f"\n{'='*60}")
    print(f"  TOTAL CLIP COUNTS")
    print(f"{'='*60}")
    for split in SPLITS:
        split_dir = DATA_DIR / split
        mp4s = list(split_dir.rglob("*.mp4"))
        print(f"  {split}: {len(mp4s)} clips")


if __name__ == "__main__":
    explore_annotations()
    inspect_sample_video()
    count_total_clips()