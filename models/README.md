# Model weights

Model weights are not committed to this repository because they are large.

The current best model (Experiment #13, YOLOv9m, WIoU v3, 250 epochs):

```text
runs/detect/ev_battery_loss_exp/wiou_v2_250/weights/best.pt
```

Metrics: mAP50=0.752, mAP50-95=0.554 (validation, 270 images, 8 classes).

Model weights are not committed to git. To use:

1. Obtain the trained weights (from training run or shared storage).
2. Place at the path above, or pass directly:

```bash
python video_annotator.py --source video.mp4 --model path/to/best.pt
```

See `MODEL_CARD.md` for full architecture, training config, and per-class metrics.
