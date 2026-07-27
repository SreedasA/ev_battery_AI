"""
sample_frames.py
----------------
Samples unannotated frames from the `frames/` folder for annotation.

Dynamically reads the existing dataset to find which frame numbers are
already annotated, then copies only new frames — at a configurable stride
per brand — into `annotation_staging/`.

Usage:
    python sample_frames.py

Output:
    annotation_staging/
        Kia_EV9/        ~150 frames
        Skywell/        ~180 frames
        Arrival_Van/    ~40 frames
        BYD_Shark/      ~33 frames
    annotation_staging/sampling_report.csv
"""

import csv
import os
import re
import shutil
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

FRAMES_DIR   = PROJECT_ROOT / "frames"
DATASET_DIR  = PROJECT_ROOT / "EV Battery Dataset 1045"
STAGING_DIR  = PROJECT_ROOT / "annotation_staging"

# How many frames to skip between each sampled frame (1-in-N sampling).
# Lower = more frames selected. Tune these to hit the target counts.
BRAND_CONFIG = {
    "Kia EV9_raw": {
        "output_folder": "Kia_EV9",
        "stride": 10,          # ~150 frames target
        "priority": 1,
    },
    "Skywell BE11 (2021)_raw": {
        "output_folder": "Skywell",
        "stride": 5,           # ~180 frames target
        "priority": 2,
    },
    "Arrival Van_raw": {
        "output_folder": "Arrival_Van",
        "stride": 10,          # ~40 frames target
        "priority": 3,
    },
    "BYD Shark_raw": {
        "output_folder": "BYD_Shark",
        "stride": 10,          # ~33 frames target
        "priority": 4,
    },
}

# Dataset folder names map to raw frame folder names (for matching frame numbers)
BRAND_DATASET_PREFIX = {
    "Kia EV9_raw":               "Kia-EV9",
    "Skywell BE11 (2021)_raw":   "Skywell-BE11",
    "Arrival Van_raw":           "Arrival-Van",
    "BYD Shark_raw":             "BYD-Shark",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_frame_number(filename: str) -> str:
    """
    Pull the numeric frame identifier from a filename.
    Handles both raw frame names (frame_0177.jpg) and dataset names
    (Arrival-Van_raw_frame_0177_jpg.rf.xxx.jpg).
    Returns the zero-padded number string, e.g. '0177' or '00177'.
    """
    match = re.search(r'frame_(\d+)', filename)
    return match.group(1) if match else ""


def get_annotated_frame_numbers(dataset_prefix: str) -> set:
    """
    Scan train/valid/test image folders in the dataset and return the set of
    frame number strings already annotated for a given brand prefix.
    """
    annotated = set()
    for split in ("train", "valid", "test"):
        img_dir = DATASET_DIR / split / "images"
        if not img_dir.exists():
            continue
        for f in img_dir.iterdir():
            if dataset_prefix in f.name:
                num = extract_frame_number(f.name)
                if num:
                    annotated.add(num)
    return annotated


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    report_rows = []
    grand_total = 0

    print("=" * 60)
    print("EV Battery — Frame Sampling")
    print("=" * 60)

    for brand_folder, config in sorted(BRAND_CONFIG.items(), key=lambda x: x[1]["priority"]):
        src_dir = FRAMES_DIR / brand_folder
        dst_dir = STAGING_DIR / config["output_folder"]
        stride  = config["stride"]

        if not src_dir.exists():
            print(f"\n[SKIP] {brand_folder} — folder not found: {src_dir}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Find already-annotated frame numbers for this brand
        dataset_prefix = BRAND_DATASET_PREFIX[brand_folder]
        annotated = get_annotated_frame_numbers(dataset_prefix)

        all_frames = sorted(src_dir.glob("*.jpg"))
        total_raw       = len(all_frames)
        already_done    = 0
        sampled         = 0
        skipped_stride  = 0

        print(f"\n[{config['priority']}] {brand_folder}")
        print(f"    Raw frames:      {total_raw}")
        print(f"    Already in dataset: {len(annotated)}")

        # Walk frames in order, skip annotated, sample every Nth remaining
        unannotated_counter = 0
        for frame_path in all_frames:
            num = extract_frame_number(frame_path.name)

            if num in annotated:
                already_done += 1
                continue

            unannotated_counter += 1

            if unannotated_counter % stride != 0:
                skipped_stride += 1
                continue

            # Copy to staging
            dest = dst_dir / frame_path.name
            if not dest.exists():
                shutil.copy2(frame_path, dest)
            sampled += 1

            report_rows.append({
                "brand":          brand_folder,
                "output_folder":  config["output_folder"],
                "source_file":    str(frame_path),
                "staged_file":    str(dest),
                "frame_number":   num,
                "status":         "staged",
            })

        grand_total += sampled
        print(f"    Unannotated:     {unannotated_counter}")
        print(f"    Sampled (1-in-{stride}): {sampled}  →  {dst_dir.name}/")

    # Write sampling report
    report_path = STAGING_DIR / "sampling_report.csv"
    if report_rows:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)

    print("\n" + "=" * 60)
    print(f"Total frames staged: {grand_total}")
    print(f"Staging folder:      {STAGING_DIR}")
    print(f"Sampling report:     {report_path}")
    print("=" * 60)
    print("\nNext step: run  python pseudo_label.py")


if __name__ == "__main__":
    main()
