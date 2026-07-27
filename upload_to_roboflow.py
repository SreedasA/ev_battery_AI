"""
upload_to_roboflow.py
---------------------
Uploads pseudo-labelled images to your existing Roboflow project
using the Roboflow Python SDK.

Usage:
    pip install roboflow
    python upload_to_roboflow.py --api-key YOUR_KEY

Find your API key at: https://app.roboflow.com/ → Settings → Roboflow API

What it does:
  - Reads pseudo_labels/images/ and pseudo_labels/labels/
  - Uploads each image paired with its YOLO .txt annotation
  - Skips images with no matching label file
  - Prints a per-image result (uploaded / skipped / error)

After upload:
  - Go to your Roboflow project → Annotate
  - Filter by "Needs Review" to check flagged images (fasteners, cables)
  - Generate a new Dataset Version (v2) with stratified 80/10/10 split
  - Export as YOLOv8 PyTorch for retraining
"""

import argparse
from pathlib import Path

# ── Project config (from EV Battery Dataset 1045/data.yaml) ──────────────────
WORKSPACE   = "ev-battery"
PROJECT     = "ev-battery-component-detection-gqljq"

# Class ID → name mapping (must match your Roboflow project's class list exactly).
# Roboflow needs this to resolve numeric IDs from YOLO .txt files.
# Without it, uploads fail with "locked annotation classes".
LABELMAP = {
    "0": "BMS_Unit",
    "1": "Busbars",
    "2": "Coolant_tubes",
    "3": "battery_module",
    "4": "battery_tray",
    "5": "fasteners",
    "6": "low_volt_cable",
    "7": "pack_cover",
}

PSEUDO_DIR  = Path(__file__).resolve().parent / "pseudo_labels"
IMAGES_DIR  = PSEUDO_DIR / "images"
LABELS_DIR  = PSEUDO_DIR / "labels"


def upload(api_key: str, batch_name: str, split: str):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed.")
        print("Run:  pip install roboflow")
        return

    print("=" * 60)
    print("EV Battery — Roboflow Upload")
    print(f"  Workspace : {WORKSPACE}")
    print(f"  Project   : {PROJECT}")
    print(f"  Split     : {split}")
    print(f"  Batch     : {batch_name}")
    print("=" * 60)

    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    if not image_files:
        print("ERROR: No .jpg files found in pseudo_labels/images/")
        return

    print(f"\nFound {len(image_files)} images to upload\n")

    uploaded = 0
    skipped  = 0
    errors   = 0

    for img_path in image_files:
        label_path = LABELS_DIR / (img_path.stem + ".txt")

        if not label_path.exists():
            print(f"  [SKIP] {img_path.name} — no matching label file")
            skipped += 1
            continue

        # Check for empty label (no detections)
        label_content = label_path.read_text().strip()
        if not label_content:
            print(f"  [SKIP] {img_path.name} — empty label (no detections, needs manual annotation)")
            skipped += 1
            continue

        try:
            project.upload(
                image_path=str(img_path),
                annotation_path=str(label_path),
                annotation_type="yolov8",      # YOLOv8 format = same .txt structure
                annotation_labelmap=LABELMAP,  # maps class IDs 0-7 to names
                split=split,
                tag_names=[batch_name],
                is_prediction=True,            # marks as model-assisted, enables review flag
                overwrite=True,               # overwrite if image was already uploaded without annotations
            )
            print(f"  [OK]   {img_path.name}")
            uploaded += 1

        except Exception as e:
            print(f"  [ERR]  {img_path.name} — {e}")
            errors += 1

    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print(f"  Uploaded : {uploaded}")
    print(f"  Skipped  : {skipped}")
    print(f"  Errors   : {errors}")
    print("=" * 60)
    print("\nNEXT STEPS:")
    print("  1. Go to your Roboflow project → Annotate")
    print("  2. Filter 'Needs Review' → check fasteners and low_volt_cable boxes")
    print("  3. Generate Dataset v2 → stratified 80/10/10 split")
    print("  4. Export as YOLOv8 PyTorch → use for retraining")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload pseudo-labels to Roboflow")
    parser.add_argument(
        "--api-key", required=True,
        help="Your Roboflow API key (Settings → Roboflow API on app.roboflow.com)"
    )
    parser.add_argument(
        "--batch", default="pseudo_label_batch_02",
        help="Tag name for this upload batch (default: pseudo_label_batch_02)"
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "valid", "test"],
        help="Which split to assign images to (default: train)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload(api_key=args.api_key, batch_name=args.batch, split=args.split)
