# Model Card — EV Battery Component Detector

**Model:** YOLOv9m with WIoU v3 custom loss function  
**Version:** Experiment #13 (`wiou_v2_250`)  
**Task:** Multi-class object detection — EV battery pack components  
**Domain:** Industrial computer vision — EV battery disassembly  
**Date:** 2026-07-02

---

## Architecture

| Property | Value |
|----------|-------|
| Base model | YOLOv9m (ImageNet pretrained) |
| Parameters | 38.8M (20.0M fused) |
| Layers | 151 |
| GFLOPs | 76.5 |
| Input size | 640×640 px |
| Loss function | WIoU v3 (Wise IoU, dynamic focusing via EMA) |
| Framework | ultralytics 8.4.60 |

### WIoU v3 Custom Loss

Standard box regression uses a fixed loss weight per prediction. WIoU v3
maintains a running EMA of recent IoU values and computes a `wise_factor`
that down-weights easy examples and up-weights hard/minority examples
dynamically during training. This is particularly relevant for imbalanced
classes (low_volt_cable, fasteners) where the model would otherwise be
dominated by majority-class gradient signal.

The implementation is a monkey-patch replacing `ultralytics.utils.loss.bbox_iou`
at module level — no forking of the ultralytics library required.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | EV Battery Dataset 1045 v2.0 (951 train / 270 val / 135 test) |
| Epochs | 250 |
| Batch size | 4 |
| Optimizer | Auto (SGD with momentum) |
| LR schedule | Cosine decay, lr0=0.01, lrf=0.01 |
| Image size | 640×640 |
| AMP | True (mixed precision) |
| close_mosaic | 15 (mosaic off for last 15 epochs) |
| copy_paste | 0.3 |
| mixup | 0.1 |
| LVC oversampling | 2× (87 → 174 training images for low_volt_cable) |
| cls_pw | 1.0 (full inverse-frequency class weighting) |
| patience | 50 |
| Seed | 0 |
| Hardware | RTX 3050 Laptop GPU 4 GB, Windows 11 |
| Training time | 6.2 hours |
| Best checkpoint | Epoch 216 / 250 |

---

## Detection Classes

8 EV battery pack disassembly components:

| ID | Class | Notes |
|----|-------|-------|
| 0 | BMS_Unit | Battery Management System unit |
| 1 | Busbars | Copper current collectors |
| 2 | Coolant_tubes | Thermal management tubing |
| 3 | battery_module | Individual battery modules |
| 4 | battery_tray | Pack structural tray |
| 5 | fasteners | Bolts, screws — small dense objects |
| 6 | low_volt_cable | LV wiring harness — thin, elongated |
| 7 | pack_cover | External cover panels |

---

## Validation Performance

Evaluated on 270 validation images, 1,723 instances.
Reported at `conf=0.001`, `iou=0.6` (standard COCO-style mAP evaluation).

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|-------|--------|-----------|---|---|-------|----------|
| **Overall** | **270** | **1723** | **0.729** | **0.727** | **0.752** | **0.554** |
| BMS_Unit | 41 | 42 | 0.756 | 0.739 | 0.811 | 0.566 |
| Busbars | 35 | 72 | 0.756 | 0.816 | 0.838 | 0.651 |
| Coolant_tubes | 24 | 34 | 0.702 | 0.676 | 0.676 | 0.443 |
| battery_module | 177 | 548 | 0.850 | 0.882 | 0.887 | 0.671 |
| battery_tray | 138 | 138 | 0.855 | 0.898 | 0.944 | 0.811 |
| fasteners | 86 | 786 | 0.715 | 0.590 | 0.683 | 0.372 |
| low_volt_cable | 20 | 47 | 0.432 | 0.319 | 0.263 | 0.138 |
| pack_cover | 56 | 56 | 0.768 | 0.893 | 0.915 | 0.778 |

**Inference speed:** 0.3ms preprocess + 11.7ms inference + 1.0ms postprocess
per image on RTX 3050 Laptop GPU.

---

## Experiment Context

This model is Experiment #13 in a systematic series comparing loss functions
and training configurations. See `EXPERIMENTS.md` for the full log.

Summary of progression on val mAP50:

| Experiment | Loss | Epochs | mAP50 | LVC mAP50 |
|------------|------|--------|-------|-----------|
| Dissertation YOLOv9s (v1 dataset) | SIoU | 100 | 0.820 | 0.434 |
| #10 SIoU v2 scratch | SIoU | 150 | 0.726 | 0.247 |
| #11 WIoU v2 scratch | WIoU v3 | 150 | 0.744 | 0.302 |
| #12 WIoU + LVC ×2 + cls_pw | WIoU v3 | 200 | 0.751 | 0.303 |
| **#13 (this model)** | **WIoU v3** | **250** | **0.752** | **0.263** |

Note: v1 and v2 dataset metrics are not directly comparable (different image
distributions and pseudo-labelled additions in v2).

---

## Intended Use

- **Primary use:** Detection of EV battery pack components in disassembly
  video for process monitoring, documentation, and quality assurance.
- **Secondary use:** Research into domain-specific object detection for
  industrial applications; baseline for further dataset expansion and model
  improvement.
- **Deployment context:** Controlled disassembly environment with known camera
  angle and lighting. Per-class confidence thresholds in
  `configs/video_inference.json` are calibrated for this context.

---

## Limitations and Known Failure Modes

**low_volt_cable (mAP50=0.263):** Chronically underperforming. Root cause is
data volume — 87 training images, many with the cable partially obscured.
WIoU v3 dynamic focusing improves the loss weighting but cannot compensate for
data scarcity. Do not rely on this class for production decisions without
further dataset expansion (target: 200+ training images).

**fasteners (mAP50=0.683 val, mAP50-95=0.372):** Found but imprecisely
localised at 640px resolution. Small, dense objects (up to 30+ per frame) with
high inter-instance overlap. SAHI sliced inference is recommended at runtime
to improve small-object recall without retraining.

**Generalization:** Trained on 4 vehicle brands (Kia EV9, Skywell BE11, Arrival
Van, BYD Shark). Detection confidence degrades on unseen vehicle architectures
with significantly different component layouts or camera angles.

**False positives in background:** Small round workshop objects can trigger
`fasteners`. Red pipes or equipment can trigger `Coolant_tubes`. Use
`--roi "x1,y1,x2,y2"` to restrict detection to the workbench area, and
per-class confidence thresholds to filter weak predictions.

**LVC val metric reliability:** The LVC validation set contains only 20 images
(47 boxes). Single missed detections shift the metric by ~2–5 percentage
points. Treat the 0.263 figure as an order-of-magnitude estimate, not a
precise measurement.

---

## Bias and Fairness

The dataset is sourced from public EV disassembly videos. No personal data
or identifiable individuals are included. The model detects mechanical
components only. No fairness concerns identified for the intended use case.

---

## Ethical Considerations

This model is designed for industrial process monitoring, not safety-critical
autonomous control. Human oversight is required for any decision based on
model output. Do not use this model as a sole determinant for:
- Battery cell handling decisions
- Safety clearance or worker proximity guidance
- Automated machinery control

---

## Citation

If you use this work in research, please cite:

```
Sreedas A. (2026). EV Battery Component Detector — YOLOv9m with WIoU v3
Loss for EV Battery Disassembly. GitHub repository.
https://github.com/SreedasA/ev_battery_AI
```

---

## Acknowledgements

- YOLOv9: Wang et al., "YOLOv9: Learning What You Want to Learn Using
  Programmable Gradient Information" (arXiv 2402.13616)
- WIoU: Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic
  Focusing Mechanism" (arXiv 2301.10051)
- Dataset annotated via Roboflow (roboflow.com)
- Dissertation work completed at Aston University (PGDip, 2025)
