"""
pseudo_label.py
---------------
Runs the trained best.pt model on staged frames, writes YOLO-format
label files, and produces a review manifest CSV.

Auto-accepts high-confidence boxes for strong classes.
Always flags fasteners and low_volt_cable for manual review — these are
the weak classes where the model's box quality cannot be trusted.

Usage:
    python pseudo_label.py [--input annotation_staging] [--output pseudo_labels] [--imgsz 640]

After running:
    pseudo_labels/
        images/               ← copies of staged images (renamed with brand prefix)
        labels/               ← YOLO .txt label files (auto-accepted boxes only)
        review_manifest.csv   ← one row per image: needs_review, reasons, box counts

Upload `images/` + `labels/` to Roboflow as YOLOv9 PyTorch format.
Images marked needs_review=True must be verified in Roboflow Annotate before
generating a new dataset version.
"""

import argparse
import csv
import shutil
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs/detect/ev_battery_loss_exp"
    / "wiou_v2_250/weights/best.pt"
)

CLASS_NAMES = [
    "BMS_Unit",       # 0
    "Busbars",        # 1
    "Coolant_tubes",  # 2
    "battery_module", # 3
    "battery_tray",   # 4
    "fasteners",      # 5
    "low_volt_cable", # 6
    "pack_cover",     # 7
]

# Minimum confidence to WRITE a box to the label file.
# None = always write the box but always flag the image for manual review.
# If confidence < threshold, the box is discarded entirely (not written).
AUTO_ACCEPT_CONF = {
    "BMS_Unit":       0.70,   # strong class, high real-world confidence
    "Busbars":        0.70,   # strong class
    "Coolant_tubes":  0.65,   # moderate — flag if below
    "battery_module": 0.70,   # strong
    "battery_tray":   0.75,   # very strong
    "fasteners":      None,   # ALWAYS manual review — box quality unreliable
    "low_volt_cable": None,   # ALWAYS manual review — data scarce, miss rate high
    "pack_cover":     0.75,   # very strong
}

# Global detection confidence floor — boxes below this are not considered at all.
GLOBAL_CONF_FLOOR = 0.25

# IOU threshold for NMS
IOU_THRESHOLD = 0.50

# ── Helpers ────────────────────────────────────────────────────────────────────

def pick_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert absolute xyxy box to normalised YOLO cx cy w h."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def box_area_fraction(x1, y1, x2, y2, img_w, img_h):
    """Return the box area as a fraction of the image area."""
    return ((x2 - x1) * (y2 - y1)) / (img_w * img_h)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(input_dir: Path, output_dir: Path, imgsz: int):
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    model  = YOLO(str(MODEL_PATH))

    print("=" * 60)
    print("EV Battery — Pseudo-Labeling")
    print(f"Model:   {MODEL_PATH.name}")
    print(f"Device:  {device}")
    print(f"imgsz:   {imgsz}")
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print("=" * 60)

    # Collect all staged images across brand sub-folders
    all_images = []
    for brand_folder in sorted(input_dir.iterdir()):
        if brand_folder.name == "sampling_report.csv":
            continue
        if brand_folder.is_dir():
            imgs = sorted(brand_folder.glob("*.jpg"))
            for img in imgs:
                all_images.append((brand_folder.name, img))

    print(f"\nTotal images to process: {len(all_images)}\n")

    report_rows = []
    auto_accepted_total = 0
    flagged_total = 0

    for brand_name, img_path in all_images:
        # Unique stem: brand_framename to avoid collisions across brands
        stem = f"{brand_name}__{img_path.stem}"

        result = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=GLOBAL_CONF_FLOOR,
            iou=IOU_THRESHOLD,
            device=device,
            verbose=False,
        )[0]

        img_h, img_w = result.orig_shape
        label_lines   = []
        needs_review  = False
        review_reasons = []
        always_review_classes_found = []

        for box in result.boxes:
            cls_id     = int(box.cls[0].item())
            conf       = float(box.conf[0].item())
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            threshold = AUTO_ACCEPT_CONF.get(class_name, 0.60)

            if threshold is None:
                # Always-review class — write the box but flag the image
                needs_review = True
                if class_name not in always_review_classes_found:
                    always_review_classes_found.append(class_name)
                    review_reasons.append(f"{class_name}=always_review")
                cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  # {class_name} conf={conf:.3f} REVIEW")

            elif conf >= threshold:
                # Auto-accepted
                cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            else:
                # Below threshold — discard but note it
                needs_review = True
                review_reasons.append(f"{class_name}@{conf:.2f}<{threshold:.2f}")

        # Strip inline comments from label lines before writing
        # (YOLO format does not support comments — we write a clean file)
        clean_lines = [ln.split("#")[0].strip() for ln in label_lines]
        clean_lines = [ln for ln in clean_lines if ln]  # remove empty

        # Write label file
        label_file = labels_out / f"{stem}.txt"
        label_file.write_text("\n".join(clean_lines), encoding="utf-8")

        # Copy image
        dest_img = images_out / f"{stem}.jpg"
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)

        # Determine primary review reason summary
        if not review_reasons and len(clean_lines) == 0:
            status = "no_detections"
            needs_review = True
            review_reasons = ["no_detections — check manually for missed objects"]
        elif needs_review:
            status = "needs_review"
            flagged_total += 1
        else:
            status = "auto_accepted"
            auto_accepted_total += 1

        report_rows.append({
            "image":            f"{stem}.jpg",
            "brand":            brand_name,
            "original_file":    img_path.name,
            "status":           status,
            "needs_review":     needs_review,
            "total_detections": len(result.boxes),
            "boxes_written":    len(clean_lines),
            "review_reasons":   " | ".join(review_reasons) if review_reasons else "",
        })

        # Progress print
        symbol = "✓" if not needs_review else "⚑"
        print(f"  {symbol} {stem[:55]:<55}  boxes={len(clean_lines)}  {'[REVIEW]' if needs_review else ''}")

    # Write review manifest
    manifest_path = output_dir / "review_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    # ── Write data.yaml ──────────────────────────────────────────────────────
    # Roboflow requires a data.yaml alongside labels so it can map class IDs
    # to class names on import. Without this, class IDs 0-7 appear as unnamed.
    # NOTE: YOLOv9 PyTorch TXT is import-only on Roboflow (no export support).
    # When downloading the retrain dataset from Roboflow, select YOLOv8 format
    # — it is structurally identical and fully exportable.
    data_yaml_content = (
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
        "\n"
        "# Upload instructions for Roboflow:\n"
        "# 1. Zip this file + images/ + labels/ into one archive\n"
        "# 2. In Roboflow: Upload > select format 'YOLOv8 PyTorch'\n"
        "#    (YOLOv9 import is supported but YOLOv8 is preferred — identical format,\n"
        "#     and YOLOv8 supports both import AND export from Roboflow)\n"
        "# 3. After upload, review flagged images in Roboflow Annotate\n"
        "# 4. Generate new dataset version — export as YOLOv8 PyTorch for retraining\n"
    )
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(data_yaml_content, encoding="utf-8")
    print(f"\n  data.yaml → {yaml_path}")

    # ── Summary ──
    total = len(report_rows)
    no_det = sum(1 for r in report_rows if r["status"] == "no_detections")

    print("\n" + "=" * 60)
    print("PSEUDO-LABELING COMPLETE")
    print(f"  Total images processed : {total}")
    print(f"  Auto-accepted          : {auto_accepted_total}")
    print(f"  Flagged for review     : {flagged_total}")
    print(f"  No detections (check!) : {no_det}")
    print(f"\n  Images  → {images_out}")
    print(f"  Labels  → {labels_out}")
    print(f"  Manifest→ {manifest_path}")
    print("=" * 60)
    print("NEXT STEPS:")
    print("  1. Open review_manifest.csv — sort by needs_review=True")
    print("  2. ZIP: pseudo_labels/images + labels + data.yaml into one archive")
    print("  3. Roboflow Upload > format: YOLOv8 PyTorch (NOT YOLOv9 — no export)")
    print("  4. Review flagged images in Roboflow Annotate (fasteners + cables)")
    print("  5. Generate Dataset v2 > stratified 80/10/10 split")
    print("  6. Export as YOLOv8 PyTorch > retrain from best.pt checkpoint")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Pseudo-label staged frames with best.pt")
    parser.add_argument("--input",  default="annotation_staging",
                        help="Folder of staged frames (output of sample_frames.py)")
    parser.add_argument("--output", default="pseudo_labels",
                        help="Output folder for images + labels + manifest")
    parser.add_argument("--imgsz",  type=int, default=640,
                        help="Inference image size. Use 1280 for Skywell 4K frames.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        imgsz=args.imgsz,
    )
