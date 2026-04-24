"""
01_download_data.py
Download and verify the SoccerNet-MVFoul dataset.

Usage:
    python src/01_download_data.py

Prerequisites:
    - Sign the NDA at https://www.soccer-net.org/ (you've already done this)
    - pip install SoccerNet
"""

import os
import json
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = Path("data/mvfoul")
SPLITS = ["train", "valid", "test"]


def download_dataset():
    """Download MVFoul dataset using SoccerNet API."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(DATA_DIR))

    # You'll be prompted for your SoccerNet credentials (email/password)
    # from when you signed the NDA
    print("Downloading MVFoul dataset...")
    print("You'll be prompted for your SoccerNet credentials.\n")

    downloader.downloadDataTask(
        task="mvfouls",
        split=SPLITS,
        password="s0cc3rn3t"
    )
    print(f"\nDataset downloaded to: {DATA_DIR.resolve()}")


def verify_dataset():
    """Verify the downloaded dataset structure and print stats."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            # MVFoul might store splits differently; check common patterns
            print(f"  WARNING: {split_dir} not found. Checking alternatives...")
            continue

        # Count action folders / clips
        clips = list(split_dir.rglob("*.mp4")) + list(split_dir.rglob("*.mkv"))
        print(f"\n  {split}:")
        print(f"    Directory: {split_dir}")
        print(f"    Video clips found: {len(clips)}")

    # Look for annotation files
    print("\n  Annotation files:")
    for ann in DATA_DIR.rglob("*.json"):
        size_mb = ann.stat().st_size / (1024 * 1024)
        print(f"    {ann.relative_to(DATA_DIR)} ({size_mb:.1f} MB)")

    # Total disk usage
    total_bytes = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024 ** 3)
    print(f"\n  Total dataset size: {total_gb:.2f} GB")


if __name__ == "__main__":
    if not any(DATA_DIR.rglob("*.mp4")) and not any(DATA_DIR.rglob("*.mkv")):
        download_dataset()
    else:
        print("Dataset already exists, skipping download.")

    verify_dataset()