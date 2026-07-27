# Reproducibility Notes

This project was developed on a custom EV battery disassembly dataset. The raw
dataset and trained model weights are not included in the GitHub repository.

## Dataset

**v2 dataset:** `EV Battery Dataset 1045 v2.0/` — 951 train / 270 val / 135 test images,
8 classes, verified clean (0 missing labels, 0 invalid class IDs, 0 boundary violations).
Not committed to git (large files). Source: Roboflow project
`ev-battery/ev-battery-component-detection-gqljq`.

## Current Official Model

**Experiment #13** — YOLOv9m, WIoU v3 loss, 250 epochs, v2 dataset.

```text
Weights: runs/detect/ev_battery_loss_exp/wiou_v2_250/weights/best.pt
Best epoch: 216/250
```

Validation metrics (270 images, 1723 instances):

```text
Overall:        mAP50=0.752   mAP50-95=0.554
BMS_Unit:       mAP50=0.811   mAP50-95=0.566
Busbars:        mAP50=0.838   mAP50-95=0.651
Coolant_tubes:  mAP50=0.676   mAP50-95=0.443
battery_module: mAP50=0.887   mAP50-95=0.671
battery_tray:   mAP50=0.944   mAP50-95=0.811
fasteners:      mAP50=0.683   mAP50-95=0.372
low_volt_cable: mAP50=0.263   mAP50-95=0.138
pack_cover:     mAP50=0.915   mAP50-95=0.778
```

Environment: Python 3.10.0, ultralytics 8.4.60, torch 2.7.1+cu118, CUDA 11.8,
RTX 3050 Laptop GPU 4 GB, Windows 11.

## How To Reproduce Training

Install torch with CUDA first (not on PyPI):

```bash
pip install torch==2.7.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Then open `loss_experiment.ipynb` and run all cells in order. Full hyperparameters
are recorded in `runs/detect/ev_battery_loss_exp/wiou_v2_250/args.yaml`.

## How To Run Inference

```bash
python video_annotator.py \
  --source path/to/video.mp4 \
  --model runs/detect/ev_battery_loss_exp/wiou_v2_250/weights/best.pt
```

## Experiment History

Full experiment log with per-class results and decisions: `EXPERIMENTS.md`
