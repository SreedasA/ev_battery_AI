# EV Battery — Exact Annotation Plan
> For expanding the dataset from 1,045 → ~1,450 images using model-assisted annotation
> Based on analysis of the `frames/` folder and current dataset state

---

## Context: Why we annotate this specific way

The current dataset has two structural problems revealed by error analysis:

| Class | Total images | Problem |
|---|---|---|
| low_volt_cable | 83 | **Data scarcity** — not enough examples. Thin object, needs more images. |
| fasteners | 300 | **Small object quality** — 1,307 of 2,924 annotated boxes are flagged "small" at 640px. Boxes are sloppy. Need tighter boxes on small fasteners. |
| Kia EV9 (brand) | 132 frames annotated of 1,523 | **Brand gap** — model barely trained on this vehicle. Only 8.7% coverage. |
| Skywell BE11 (brand) | 178 frames of 1,104 | **Underrepresented** — but frames are 4K (3840×2160), highest annotation value for small objects. |

Everything else (battery_module, battery_tray, BMS_Unit, Busbars, pack_cover) is strong. **Do not waste time annotating frames that only contain strong-class objects.**

---

## PHASE 1 — Frame Sampling (Do this before anything else)

**Goal:** Select ~400 frames from 3,062 unannotated frames. Not all frames — consecutive video frames are nearly identical and add no value.

### Sampling rules per brand

**Kia EV9 — Target: ~150 frames (highest priority)**
- 1,391 unannotated frames across frames 00001–02029
- Large unannotated gaps: 12–203, 223–413, 471–871, 1008–1269, 1292–1500, 1598–1861
- Strategy: **sample every 10th frame from each gap**
- This covers the full timeline of the disassembly without redundancy
- Why priority: brand diversity, model has barely seen this vehicle

**Skywell BE11 — Target: ~180 frames (second priority)**
- 926 unannotated frames, all at **4K resolution (3840×2160)**
- Strategy: **sample every 5th unannotated frame**
- Why priority: 4K gives 4× pixel density — cables and fasteners that are blurry at 1080p are clearly visible here. Best possible source for improving weak classes.
- Note on resolution: keep images at 4K when uploading to Roboflow. Roboflow resizes on export. Training at imgsz=1280 (not 640) will exploit this.

**Arrival Van — Target: ~40 frames (low priority)**
- 416 unannotated frames
- Strategy: sample every 10th unannotated frame
- Only useful if sampled frames show cables or fasteners

**BYD Shark — Target: ~33 frames (lowest priority)**
- 329 unannotated frames
- Strategy: sample every 10th unannotated frame
- Already well-represented in dataset

### Frame sampling script
Run this to extract your sample frames into a staging folder:

```python
# Run from the project root
import os, shutil

BRANDS = {
    "Kia EV9_raw":           ("frames/Kia EV9_raw",            "annotation_staging/Kia_EV9",    10),
    "Skywell BE11 (2021)_raw":("frames/Skywell BE11 (2021)_raw","annotation_staging/Skywell",     5),
    "Arrival Van_raw":        ("frames/Arrival Van_raw",         "annotation_staging/Arrival_Van", 10),
    "BYD Shark_raw":          ("frames/BYD Shark_raw",           "annotation_staging/BYD_Shark",  10),
}

# Frame numbers already annotated — skip these
ALREADY_ANNOTATED = {
    "Kia EV9_raw": {
        "00006","00007","00008","00009","00010","00011","00204","00205","00206","00209",
        "00210","00211","00213","00214","00215","00216","00217","00218","00219","00221",
        "00222","00414","00435","00436","00438","00440","00442","00444","00445","00446",
        "00454","00455","00456","00458","00459","00461","00465","00466","00467","00468",
        "00469","00470","00872","00873","00874","00875","00876","00935","00937","00938",
        "00939","00940","00942","00944","00946","00951","00952","00954","00955","00956",
        "00957","00958","00959","00960","00961","00962","00963","00964","00965","00966",
        "00968","00969","00971","00972","00973","00974","00975","00976","00977","00978",
        "00979","00980","00981","00982","00983","00984","00985","00986","00988","00990",
        "00994","00996","00998","00999","01000","01001","01002","01003","01004","01005",
        "01006","01007","01270","01271","01272","01273","01274","01275","01276","01277",
        "01280","01281","01282","01283","01284","01286","01287","01288","01289","01501",
        "01502","01503","01505","01516","01517","01597","01722","01861","02026","02027",
        "02028","02029"
    },
    # Add Skywell, Arrival Van, BYD Shark sets similarly if needed
}

for brand, (src, dst, stride) in BRANDS.items():
    os.makedirs(dst, exist_ok=True)
    frames = sorted(os.listdir(src))
    annotated = ALREADY_ANNOTATED.get(brand, set())
    count = 0
    for i, f in enumerate(frames):
        num = f.replace("frame_", "").replace(".jpg", "")
        if num in annotated:
            continue
        if i % stride == 0:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
            count += 1
    print(f"{brand}: copied {count} frames to {dst}")
```

---

## PHASE 2 — Pseudo-Labeling (Model Auto-Annotation)

**Goal:** Run `best.pt` on all sampled frames and generate YOLO `.txt` label files. Accept high-confidence predictions automatically; flag the rest for manual review.

### Confidence thresholds per class

These are based on the model's real-world performance on your BYD Shark video:

| Class | Auto-accept threshold | Action if below |
|---|---|---|
| pack_cover | ≥ 0.75 | Discard — model rarely gets this wrong |
| battery_tray | ≥ 0.75 | Discard |
| battery_module | ≥ 0.70 | Discard |
| Busbars | ≥ 0.70 | Discard |
| BMS_Unit | ≥ 0.70 | Discard |
| Coolant_tubes | ≥ 0.65 | Flag for manual review |
| fasteners | **NEVER auto-accept** | Always manual review — small object boxes are poor |
| low_volt_cable | **NEVER auto-accept** | Always manual review — actively look for missed cables |

### Pseudo-label script
Save as `pseudo_label.py` in the project root:

```python
"""
pseudo_label.py
Runs best.pt on staged frames, writes YOLO labels, flags uncertain frames.
Usage: python pseudo_label.py --input annotation_staging --output pseudo_labels
"""
import argparse, csv, os
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = [
    "BMS_Unit","Busbars","Coolant_tubes","battery_module",
    "battery_tray","fasteners","low_volt_cable","pack_cover"
]

# Confidence above which a box is auto-accepted (None = always flag)
AUTO_ACCEPT_CONF = {
    "BMS_Unit": 0.70, "Busbars": 0.70, "Coolant_tubes": 0.65,
    "battery_module": 0.70, "battery_tray": 0.75,
    "fasteners": None,          # always flag
    "low_volt_cable": None,     # always flag
    "pack_cover": 0.75,
}

MODEL_PATH = "runs/detect/ev_battery_loss_exp/wiou_v2_250/weights/best.pt"

def run(input_dir, output_dir):
    model = YOLO(MODEL_PATH)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    review_rows = []

    for brand_folder in sorted(input_dir.iterdir()):
        if not brand_folder.is_dir(): continue
        images = sorted(brand_folder.glob("*.jpg"))
        print(f"\nProcessing {brand_folder.name}: {len(images)} images")

        for img_path in images:
            result = model.predict(
                source=str(img_path), imgsz=640, conf=0.25, iou=0.50,
                verbose=False
            )[0]

            h, w = result.orig_shape
            label_lines = []
            needs_review = False
            review_reasons = []

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = CLASS_NAMES[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Convert to YOLO normalised format
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                threshold = AUTO_ACCEPT_CONF.get(class_name)

                if threshold is None:
                    needs_review = True
                    review_reasons.append(f"{class_name}@{conf:.2f}(always_review)")
                elif conf < threshold:
                    needs_review = True
                    review_reasons.append(f"{class_name}@{conf:.2f}<{threshold}")
                    continue  # don't write low-conf uncertain boxes to label file

                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Write label file
            stem = f"{brand_folder.name}_{img_path.stem}"
            label_path = output_dir / "labels" / f"{stem}.txt"
            label_path.write_text("\n".join(label_lines))

            # Copy image
            import shutil
            shutil.copy(img_path, output_dir / "images" / f"{stem}.jpg")

            # Log review status
            review_rows.append({
                "image": f"{stem}.jpg",
                "brand": brand_folder.name,
                "needs_review": needs_review,
                "reasons": " | ".join(review_reasons) if review_reasons else "auto_accepted",
                "total_boxes": len(result.boxes),
                "written_boxes": len(label_lines),
            })

    # Write review manifest
    manifest_path = output_dir / "review_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review_rows[0].keys())
        writer.writeheader()
        writer.writerows(review_rows)

    total = len(review_rows)
    flagged = sum(1 for r in review_rows if r["needs_review"])
    print(f"\n=== Done ===")
    print(f"Total images processed: {total}")
    print(f"Auto-accepted (no review needed): {total - flagged}")
    print(f"Flagged for manual review: {flagged}")
    print(f"Review manifest saved to: {manifest_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="annotation_staging")
    p.add_argument("--output", default="pseudo_labels")
    args = p.parse_args()
    run(args.input, args.output)
```

Run it:
```bash
python pseudo_label.py --input annotation_staging --output pseudo_labels
```

**Output:** `pseudo_labels/review_manifest.csv` — sorted list of every image, whether it needs review, and why.

---

## PHASE 3 — Manual Verification

### Tool choice: Roboflow Annotate (not CVAT, not LabelImg)

Use Roboflow's built-in annotator for this project because:
- Your project is already on Roboflow (`ev-battery` workspace)
- You can upload images + YOLO labels together and Roboflow renders them instantly
- Roboflow has a "Review" queue that maps directly to this workflow
- For Skywell 4K frames: Roboflow's annotator supports zoom, which is essential for small fastener boxes

CVAT is more powerful but adds overhead — you'd need to manage a separate project and export/import. Roboflow keeps everything in one place for the final Roboflow export.

### Exact verification workflow per image category

**Category A — Auto-accepted frames (no cable or fastener detected)**
- Open in Roboflow annotator
- Scroll through quickly (3–5 seconds per image)
- Check: is there a cable or fastener visible in the frame that the model missed?
- If YES: draw the missed box manually, then accept
- If NO: click Accept
- Target speed: 50–80 images/hour

**Category B — Flagged frames (cable or fastener detected, or low confidence)**
- Open in Roboflow annotator
- For each flagged box: zoom in, check if the box is tight and correct
- **For fasteners specifically:**
  - Box should hug the fastener head tightly — not a loose blob around the area
  - If multiple fasteners are clustered, each needs its own individual box
  - Delete any box where the centre is not clearly on a fastener
  - On Skywell 4K frames: zoom to 200%+ and draw tight boxes
- **For low_volt_cable specifically:**
  - The cable is a thin line — the box should be a tall narrow rectangle following the cable run
  - Check if there are additional cable segments in the frame the model missed
  - If the cable goes off-screen, draw the box up to the image edge
- Target speed: 15–25 images/hour (slower, more careful)

**Category C — Frames with zero detections**
- Check the review_manifest.csv for images where `written_boxes = 0`
- These are frames where the model found nothing above 0.25 confidence
- Quick scan: is there actually nothing to annotate (e.g., blurry transition frame)? → delete from staging
- Or did the model miss something obvious? → annotate manually

### Annotation quality rules (apply consistently)

1. **Tightness:** Box edges should be within 5–10px of the object boundary at 640px scale. On 4K Skywell frames this means within 20–40px at native resolution.
2. **Occlusion:** If a component is partially obscured, still draw the box around the full estimated extent of the object. Do not crop the box to only the visible part.
3. **Class boundaries — common confusions to watch for:**
   - `battery_module` vs `battery_tray`: tray is the structural frame/housing; modules sit inside it
   - `low_volt_cable` vs `Busbars`: busbars are flat metal conductors; cables are round and flexible
   - `fasteners` — only annotate clearly identifiable bolt heads/screws. Do not annotate rivets or structural holes.
4. **Minimum size:** Do not annotate objects smaller than 10×10px at 640px resolution — they add noise.
5. **Consistency:** If you annotate fasteners on one frame, annotate ALL visible fasteners in that frame. Partial annotation of a class is worse than no annotation.

---

## PHASE 4 — Adding to Roboflow Dataset

### Step-by-step Roboflow upload process

**Step 1: Prepare upload batch**
After reviewing, you will have in `pseudo_labels/`:
```
pseudo_labels/
  images/   ← reviewed .jpg files
  labels/   ← corrected YOLO .txt files
```
Delete images from `images/` whose corresponding label file is empty and the frame had nothing to annotate (truly empty frames). Keep empty label files for frames that are genuinely background/transition — they teach the model what NOT to detect.

**Step 2: Upload to Roboflow**

> ⚠️ **Format note:** YOLOv9 PyTorch TXT is **import-only** on Roboflow — there is no YOLOv9 export option. Use **YOLOv8 PyTorch** for both import and export. The `.txt` label structure is identical, so nothing changes in the files — only the format label you select in the Roboflow UI.

1. Go to roboflow.com → workspace `ev-battery` → project `ev-battery-component-detection-gqljq`
2. Click **Upload** (top-right)
3. Create a ZIP of: `pseudo_labels/images/` + `pseudo_labels/labels/` + `pseudo_labels/data.yaml`
   ```
   zip -r batch_02.zip pseudo_labels/images pseudo_labels/labels pseudo_labels/data.yaml
   ```
4. Select upload format: **YOLOv8 PyTorch** (NOT YOLOv9 — no export support)
5. Drag `batch_02.zip` into the upload zone — Roboflow matches images to labels by filename stem
6. Assign batch name: `"batch_02_frames_expansion"` — keeps it separate from the original 1045 images for audit purposes

**Step 3: Review in Roboflow before generating new version**
1. After upload, go to **Annotate** tab → your new batch will appear as "Needs Review"
2. Spot-check 10–15 images per brand to confirm labels uploaded correctly
3. Fix any mismatches (Roboflow annotator makes this quick)

**Step 4: Generate Dataset Version 2**
1. Go to **Generate** tab in your Roboflow project
2. **Do NOT use the same split as v1.** Use:
   - Train: 80% | Valid: 10% | Test: 10%
   - Enable **Stratified split** so new brand images are distributed across splits (not all Kia EV9 in train only)
3. Preprocessing: set resize to **640×640** (same as current training config)
4. **Augmentation — add these for v2:**
   - Flip: horizontal only (vertical flip disabled — batteries have orientation)
   - Brightness: ±15%
   - Blur: up to 1px (simulates out-of-focus frames)
   - **Mosaic: disabled** (Ultralytics handles this in training already)
   - Do NOT add rotation — battery components have strong orientation priors
5. Click **Generate** — Roboflow creates v2
6. Export as **YOLOv8 PyTorch** format — NOT YOLOv9 (Roboflow has no YOLOv9 export). YOLOv8 .txt format is structurally identical to YOLOv9 and works directly with Ultralytics `yolo detect train`

**Step 5: Retrain**
Update `args.yaml` data path from v1 to v2 export, then run:
```bash
# Fine-tune from current best.pt (NOT from scratch)
yolo detect train \
  model=runs/detect/ev_battery_loss_exp/wiou_v2_250/weights/best.pt \
  data=<path_to_v2_data.yaml> \
  epochs=50 \
  imgsz=640 \
  batch=4 \
  name=ev_battery_yolov9s_640_v2 \
  cos_lr=True \
  copy_paste=0.3 \
  lr0=0.001
```
Key changes from original run:
- `copy_paste=0.3` — helps low_volt_cable by pasting cable boxes onto other frames
- `lr0=0.001` — lower LR for fine-tune (not full retrain from COCO weights)
- `epochs=50` — enough for fine-tuning, model already knows the task

---

## Summary: What to annotate, how many, in what order

| Phase | Brand | Frames to annotate | Hours est. | Goal |
|---|---|---|---|---|
| 1 | Kia EV9 | ~150 | 4–6 hrs | Brand diversity |
| 2 | Skywell BE11 | ~180 | 6–8 hrs | Cable + fastener quality (4K) |
| 3 | Arrival Van | ~40 | 1 hr | Fill small gaps |
| 4 | BYD Shark | ~33 | 1 hr | Fill small gaps |
| **Total** | | **~400 frames** | **~12–16 hrs** | **Dataset: 1,045 → ~1,450** |

Do Phase 1 + 2 first. Retrain. Check if weak class mAP improves before doing Phase 3 + 4.

---

## Expected outcome after retraining on expanded dataset

| Class | Current mAP50 | Target after v2 | Strategy |
|---|---|---|---|
| low_volt_cable | 0.363 | 0.50–0.60 | More images + copy_paste aug |
| fasteners | 0.730 | 0.78–0.82 | Tighter boxes on 4K Skywell frames |
| Overall mAP50 | 0.803 | 0.83–0.86 | Combined effect |
| Overall mAP50-95 | 0.610 | 0.63–0.67 | Better localisation |
