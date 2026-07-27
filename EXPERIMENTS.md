# EV Battery YOLOv9 — Experiment Log
> Tracks every training run: config, results, decisions, and next steps.  
> Last updated: 2026-07-02

---

## How to run `loss_experiment.ipynb` — Step-by-step

### Step 1 — Pre-flight checks (do this before opening Jupyter)

Open PowerShell and run these one by one:

```powershell
# 1. Check Python and ultralytics are installed
python -c "import ultralytics; print(ultralytics.__version__)"

# 2. Check GPU is detected
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 3. Check your VRAM (should show ~4096 MB for RTX 3050)
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // 1024**2, 'MB')"

# 4. Confirm the dataset exists
dir "C:\Users\asree\EV Battery AI\EV Battery Dataset 1045 v2.0\ev_battery_yolov9_v2.yaml"

# 5. Confirm yolov9s.pt exists (or will be auto-downloaded)
#    If you don't have it, ultralytics will download it automatically on first use
python -c "from ultralytics import YOLO; m = YOLO('yolov9s.pt')"
```

**Expected outputs:**
- ultralytics version: `8.x.x` (any recent version is fine)
- CUDA: `True` + `NVIDIA GeForce RTX 3050 Laptop GPU`
- VRAM: `4096 MB`
- Dataset YAML: file found

---

### Step 2 — Open Jupyter Notebook

```powershell
cd "C:\Users\asree\EV Battery AI"
jupyter notebook loss_experiment.ipynb
```

If Jupyter is not installed:
```powershell
pip install notebook
```

---

### Step 3 — Run cells in order

> ⚠️ **Do not skip cells.** Each cell builds on the previous one. Run them top to bottom.

| Cell | What it does | Expected output |
|------|-------------|-----------------|
| **Cell 0 — Config** | Sets paths, model name, hyperparameters | Prints path to data.yaml + "assert passed" |
| **Cell 1 — Implementations** | Loads SIoU and WIoU v3 math | `SIoU and WIoU v3 implementations loaded.` |
| **Cell 2 — Patch** | Defines `activate_loss()` function | `Monkey-patch functions defined.` |
| **Cell 3 — Activate SIoU** | Patches ultralytics, starts training | `[Patch] bbox_iou → SIoU ✓` then training logs scroll |
| **Cell 4 — Validate SIoU** | Runs val on best.pt | mAP table printed |
| **Cell 5 — Activate WIoU** | Patches ultralytics, starts training | `[Patch] bbox_iou → WIoU ✓` then training logs scroll |
| **Cell 6 — Validate WIoU** | Runs val on best.pt | mAP table printed |
| **Cell 7 — Results table** | Parses results.csv, builds comparison | Summary table with all runs |
| **Cell 8 — Bar chart** | Plots mAP50, mAP50-95, per-class | Saves PNG to `ev_battery_loss_exp/` |
| **Cell 9 — Curves** | Plots val mAP50 per epoch for both runs | Saves training curves PNG |

---

### Step 4 — Monitor training (each run)

During the training cells, you will see a log like:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/80     3.52G    0.9453     1.0213     1.1234         45        640
  2/80     3.52G    0.8871     0.9945     1.0987         51        640
  ...
```

**What to watch:**
- `GPU_mem` should stay below `4.00G` — if it goes over, reduce `BATCH` from `4` to `2` in Cell 0 and restart
- `box_loss` should decrease steadily over epochs (from ~1.0 to ~0.3)
- `cls_loss` drops faster than `box_loss` — that is normal
- Early stopping triggers at patience=30 if val mAP stops improving

**Healthy signs at epoch 20:** box_loss around 0.5–0.7, `metrics/mAP50(B)` around 0.5–0.65  
**Healthy signs at epoch 60+:** box_loss below 0.4, mAP50 above 0.75

---

### Step 5 — If something goes wrong

#### OOM (Out of Memory) error
```
RuntimeError: CUDA out of memory
```
**Fix:** In Cell 0, change `BATCH = 4` to `BATCH = 2`. Restart kernel, re-run all cells.

#### YAML file not found
```
FileNotFoundError: ev_battery_yolov9_v2.yaml
```
**Fix:** Open `dataset_clean_v2.ipynb`, run Cell 12 (generates the YAML), then retry.

#### Patch not applying / mAP collapse (drops below 0.3 after epoch 5)
This happens if the patch runs on a **CIoU-pretrained checkpoint** (best.pt). Make sure `MODEL_WEIGHTS = "yolov9s.pt"` — not a previous best.pt file.

#### Training too slow (> 45 min per epoch)
- Check GPU is being used: if `GPU_mem` shows `0.0G`, CUDA is not active
- Run `nvidia-smi` in a separate PowerShell window to confirm GPU utilisation
- If CPU-only, reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118 --break-system-packages`

#### Experiment folder already exists
```
ERROR: Output directory already exists
```
**Fix:** In Cell 0, set `exist_ok = True` in the train call, or rename the output (e.g. `name = "siou_v2_retry"`).

---

### Step 6 — After both runs complete

1. Check `ev_battery_loss_exp/loss_experiment_summary.png` — bar chart of all metrics
2. Check `ev_battery_loss_exp/training_curves.png` — epoch-by-epoch mAP50 curves
3. The best model weights are at:
   - SIoU best: `ev_battery_loss_exp/siou_v2/weights/best.pt`
   - WIoU best: `ev_battery_loss_exp/wiou_v2/weights/best.pt`
4. Update the **Results** section below in this file with your actual numbers
5. Copy the winning `best.pt` to a safe location before starting any further experiments

---

## Experiment Overview

| # | Date | Model | Dataset | Loss | Epochs | mAP50 | mAP50-95 | low_volt_cable | fasteners | Notes |
|---|------|-------|---------|------|--------|-------|----------|---------------|-----------|-------|
| 1 | 2024 | YOLOv9m | v1 (1045) | CIoU | 100 | 0.822 | 0.623 | 0.354 | — | Dissertation baseline |
| 2 | 2024 | YOLOv9m | v1 (1045) | SIoU | 100 | 0.820 | 0.627 | **0.434** | — | Dissertation; +26% cable |
| 3 | 2024 | YOLOv9m | v1 (1045) | CIoU+Bicubic | 100 | ~0.81 | — | — | — | Dissertation; harmful for recall |
| 4 | 2025 | YOLOv9s | v1 (1045) | CIoU | ~80 | 0.803 | 0.610 | 0.363 | 0.730 | Production model (best.pt) |
| 5 | 2025 | YOLOv9s | v2 (1356) | CIoU (finetune) | ~80 | 0.766 | — | 0.259 | — | Fine-tuned from best.pt; cable regressed |
| 6 | 2026-06-21 | YOLOv9m | v2 (1356) | **SIoU** | 80 | **0.720** | 0.494 | — | — | Training complete; validation pending; not converged at ep80 |
| 7 | 2026-06-21 | YOLOv9m | v2 (1356) | **WIoU v3** | 80 | **0.715** | 0.502 | — | — | Training complete; validation pending; not converged at ep80 |
| 8 | 2026-06-22 | YOLOv9m | v2 (1356) | **SIoU** | +70 (→~150) | **0.740** | **0.510** | — | — | Extended from #6 last.pt; best ep59/70; siou_v2_ext ✅ |
| 9 | 2026-06-22 | YOLOv9m | v2 (1356) | **WIoU v3** | +70 (→~150) | **0.735** | **0.513** | — | — | Extended from #7 last.pt; best ep59/70; wiou_v2_ext ✅ |
| 10 | 2026-06-23 | YOLOv9m | v2 (1356) | **SIoU** | 150 (scratch) | **0.726** | **0.537** | ⚠️ 0.247 | 0.663 | Best ep148; LVC cascade oversampled to 61%; siou_v2_150 ✅ |
| 11 | 2026-06-23 | YOLOv9m | v2 (1356) | **WIoU v3** | 150 (scratch) | **0.744** | **0.546** | ⚠️ 0.302 | 0.653 | Best ep148; LVC cascade oversampled to 61%; wiou_v2_150 ✅ |
| 12 | 2026-06-24 | YOLOv9m | v2 (1356) | **WIoU v3** | 200 (scratch) | **0.756** | **0.546** | ⚠️ 0.303 | 0.679 | Best ep174; LVC 2× fixed; cls_pw=1.0; wiou_v2_200 ✅ |
| 13 | 2026-07-02 | YOLOv9m | v2 (1356) | **WIoU v3** | 250 (scratch) | **0.752** | **0.554** | ⚠️ 0.263 | 0.683 | Best ep216; cls_pw=1.0; wiou_v2_250 ✅ · LVC regressed vs #12 |

---

## Experiment #6 — SIoU on v2 dataset

**Status:** ✅ TRAINING COMPLETE — 🔄 VALIDATION PENDING  
**Script:** `loss_experiment.ipynb` → Experiment 1 cells  
**Config:**

```yaml
model:       yolov9m.pt          # fresh pretrained weights (NOT best.pt); matches dissertation
data:        ev_battery_yolov9_v2.yaml
loss:        SIoU                # monkey-patched via activate_loss("SIoU")
epochs:      80
batch:       2                   # batch=2 for yolov9m on 4 GB VRAM
imgsz:       640
patience:    30
copy_paste:  0.1
mixup:       0.1
device:      0 (RTX 3050 45W)
amp:         True
output:      runs/detect/ev_battery_loss_exp/siou_v2/   # ultralytics prepends runs/detect/
```

**Hypothesis:** Dissertation proved SIoU beats CIoU on this exact dataset (+26% low_volt_cable).  
With 311 more images in v2 (especially fasteners +49.7%), SIoU should improve further.

**Results (training run):**

| Metric | Value | Notes |
|--------|-------|-------|
| Best epoch | **80** | ⚠️ Still rising at epoch 80 — model NOT converged |
| mAP50 (all classes) | **0.720** | vs dissertation SIoU 0.820 — gap likely due to non-convergence |
| mAP50-95 (all classes) | **0.494** | — |
| BMS_Unit mAP50 | — | Run validation cell to get per-class breakdown |
| Busbars mAP50 | — | — |
| Coolant_tubes mAP50 | — | — |
| battery_module mAP50 | — | — |
| battery_tray mAP50 | — | — |
| fasteners mAP50 | — | — |
| low_volt_cable mAP50 | — | This is the key metric — fill in from validation |
| pack_cover mAP50 | — | — |
| Training time | ~6–7 hrs | RTX 3050 45W |

**mAP50 progression by epoch:**
| Epoch | mAP50 | mAP50-95 |
|-------|-------|----------|
| 10 | 0.372 | 0.205 |
| 20 | 0.501 | 0.293 |
| 40 | 0.582 | 0.377 |
| 60 | 0.695 | 0.470 |
| 80 | 0.720 | 0.494 |

**⚠️ Important observation:** mAP50 was still climbing at epoch 80 (no plateau, no early stopping).  
With batch=2 from scratch, YOLOv9m needs ~150 total epochs to converge.  
**→ Actioned as Experiment #8:** `loss_experiment.ipynb` configured to run 70 more epochs from `siou_v2/weights/last.pt` (lr0=0.001) → saves to `siou_v2_ext/`. Running overnight.

**Verdict:** Partial — training complete but validation not yet run. Per-class mAP50 (especially low_volt_cable) unknown. Cannot confirm hypothesis until validation cell runs and model converges.

---

## Experiment #7 — WIoU v3 on v2 dataset

**Status:** ✅ TRAINING COMPLETE — 🔄 VALIDATION PENDING  
**Script:** `loss_experiment.ipynb` → Experiment 2 cells  
**Config:**

```yaml
model:       yolov9m.pt          # same as dissertation; matches Experiment #6 for fair comparison
data:        ev_battery_yolov9_v2.yaml
loss:        WIoU v3             # monkey-patched via activate_loss("WIoU")
epochs:      80
batch:       2                   # batch=2 for yolov9m on 4 GB VRAM
imgsz:       640
patience:    30
copy_paste:  0.1
mixup:       0.1
device:      0 (RTX 3050 45W)
amp:         True
output:      runs/detect/ev_battery_loss_exp/wiou_v2/
```

**Hypothesis:** WIoU v3 dynamic focusing down-weights easy boxes and amplifies gradients for  
geometrically hard samples (fasteners, cables). May outperform SIoU on dense small objects.

**Results (training run):**

| Metric | Value | Notes |
|--------|-------|-------|
| Best epoch | **80** | ⚠️ Still rising at epoch 80 — model NOT converged |
| mAP50 (all classes) | **0.715** | Marginally below SIoU (0.720) at 80 epochs |
| mAP50-95 (all classes) | **0.502** | Slightly higher mAP50-95 than SIoU (0.494) |
| BMS_Unit mAP50 | — | Run validation cell to get per-class breakdown |
| Busbars mAP50 | — | — |
| Coolant_tubes mAP50 | — | — |
| battery_module mAP50 | — | — |
| battery_tray mAP50 | — | — |
| fasteners mAP50 | — | — |
| low_volt_cable mAP50 | — | Key metric — fill in from validation |
| pack_cover mAP50 | — | — |
| Training time | ~6–7 hrs | RTX 3050 45W |

**mAP50 progression by epoch:**
| Epoch | mAP50 | mAP50-95 |
|-------|-------|----------|
| 10 | 0.378 | 0.206 |
| 20 | 0.474 | 0.261 |
| 40 | 0.602 | 0.376 |
| 60 | 0.677 | 0.452 |
| 80 | 0.715 | 0.502 |

**Comparison with SIoU at same epochs:**
| Epoch | SIoU mAP50 | WIoU mAP50 | Leader |
|-------|-----------|-----------|--------|
| 10 | 0.372 | 0.378 | WIoU +0.006 |
| 20 | 0.501 | 0.474 | SIoU +0.027 |
| 40 | 0.582 | 0.602 | WIoU +0.020 |
| 60 | 0.695 | 0.677 | SIoU +0.018 |
| 80 | 0.720 | 0.715 | SIoU +0.005 |

**⚠️ Important observation:** Both models are NOT converged at 80 epochs. Gap between SIoU and WIoU  
is only 0.005 mAP50 — too close to call.  
**→ Actioned as Experiment #9:** `loss_experiment.ipynb` configured to run 70 more epochs from `wiou_v2/weights/last.pt` (lr0=0.001) → saves to `wiou_v2_ext/`. Running overnight after #8 completes.

**Verdict:** Inconclusive at 80 epochs. WIoU shows higher mAP50-95 (better box quality), SIoU shows  
higher mAP50. Both need extension to ~150 epochs before a conclusion can be drawn.

---

## Experiment #8 — SIoU Extended (+70 epochs from last.pt)

**Status:** ✅ COMPLETE  
**Weights:** `runs/detect/ev_battery_loss_exp/siou_v2_ext/weights/best.pt`  
**Config:** lr0=0.001, lrf=0.01, batch=2, close_mosaic=10, patience=40, amp=True

| Metric | Value | Notes |
|--------|-------|-------|
| Best epoch (of 70) | **59** | Peaked then plateaued — model converged |
| mAP50 | **0.740** | +0.020 vs #6 (0.720) |
| mAP50-95 | **0.510** | +0.016 vs #6 (0.494) |
| Precision | 0.708 | At best epoch |
| Recall | 0.727 | At best epoch |
| Final epoch (70) mAP50 | 0.728 | Slight decline after peak — converged |
| Final epoch (70) mAP50-95 | 0.519 | Still inching up due to close_mosaic effect |
| low_volt_cable mAP50 | — | Validation cell not yet run |
| fasteners mAP50 | — | Validation cell not yet run |

**mAP50 progression (extension epochs):**
| Ext Epoch | mAP50 | mAP50-95 |
|-----------|-------|----------|
| 1 | 0.639 | 0.437 |
| 10 | 0.607 | 0.394 |
| 20 | 0.648 | 0.427 |
| 40 | 0.692 | 0.459 |
| 59 (best) | 0.740 | 0.510 |
| 70 (last) | 0.728 | 0.519 |

**Note:** Initial dip (ep1 = 0.639 vs ep80 end of 0.720) is caused by optimiser state reset when starting from checkpoint. Recovered and surpassed by ep50.

---

## Experiment #9 — WIoU v3 Extended (+70 epochs from last.pt)

**Status:** ✅ COMPLETE  
**Weights:** `runs/detect/ev_battery_loss_exp/wiou_v2_ext/weights/best.pt`  
**Config:** lr0=0.001, lrf=0.01, batch=2, close_mosaic=10, patience=40, amp=True

| Metric | Value | Notes |
|--------|-------|-------|
| Best epoch (of 70) | **59** | Same convergence point as SIoU ext |
| mAP50 | **0.735** | +0.020 vs #7 (0.715) |
| mAP50-95 | **0.513** | +0.011 vs #7 (0.502) |
| Precision | 0.709 | At best epoch |
| Recall | 0.712 | At best epoch |
| Final epoch (70) mAP50 | 0.722 | Slight decline after peak |
| Final epoch (70) mAP50-95 | 0.522 | Highest mAP50-95 of all runs |
| low_volt_cable mAP50 | — | Validation cell not yet run |
| fasteners mAP50 | — | Validation cell not yet run |

**SIoU vs WIoU summary at ~150 epoch equivalent:**
| Metric | SIoU #8 | WIoU #9 | Winner |
|--------|---------|---------|--------|
| mAP50 (best) | 0.740 | 0.735 | SIoU +0.005 |
| mAP50-95 (best) | 0.510 | 0.513 | WIoU +0.003 |
| mAP50-95 (final) | 0.519 | 0.522 | WIoU +0.003 |
| Recall | 0.727 | 0.712 | SIoU +0.015 |

**Verdict:** SIoU leads on detection rate (mAP50, recall). WIoU leads on box quality (mAP50-95). Gap to dissertation SIoU baseline (0.820 mAP50) remains ~0.080 — addressed by Experiments #10/#11 (150-epoch from scratch with LVC oversampling).

---

## Experiment #10 — SIoU · 150 epochs from scratch + LVC oversample

**Status:** ✅ COMPLETE  
**Weights:** `runs/detect/ev_battery_loss_exp/siou_v2_150/weights/best.pt`  
**Config:** yolov9m.pt (fresh), lr0=0.01, batch=4, close_mosaic=15, copy_paste=0.3, patience=50, LVC ×3 oversample, amp=True

| Metric | Value | Notes |
|--------|-------|-------|
| Best epoch | **148** | Still improving at ep148 — very close to convergence |
| mAP50 (best, from results.csv) | **0.7267** | Below extended run #8 (0.7397) |
| mAP50-95 (best, from results.csv) | **0.5350** | +0.025 vs #8 (0.5104) — better box quality from scratch |
| Precision (best) | 0.7086 | At best epoch |
| Recall (best) | 0.7173 | At best epoch |
| Final epoch mAP50 | 0.7257 | Very stable final 5 epochs (0.723–0.727) |
| Final epoch mAP50-95 | 0.5356 | Highest single-run mAP50-95 for SIoU approach |

**Validation results (best.pt on val split — 270 images, 1723 instances):**
| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| **all** | 270 | 1723 | 0.712 | 0.712 | **0.726** | **0.537** |
| BMS_Unit | 41 | 42 | 0.756 | 0.762 | 0.833 | 0.578 |
| Busbars | 35 | 72 | 0.766 | 0.778 | 0.766 | 0.578 |
| Coolant_tubes | 24 | 34 | 0.691 | 0.588 | 0.613 | 0.432 |
| battery_module | 177 | 548 | 0.846 | 0.878 | 0.869 | 0.658 |
| battery_tray | 138 | 138 | 0.845 | 0.908 | 0.943 | 0.816 |
| fasteners | 86 | 786 | 0.706 | 0.607 | 0.663 | 0.370 |
| **low_volt_cable** | 20 | 47 | 0.351 | 0.340 | **0.247** ⚠️ | 0.119 |
| pack_cover | 56 | 56 | 0.733 | 0.832 | 0.873 | 0.745 |

**Inference speed:** 1.6ms preprocess, 23.4ms inference, 1.2ms postprocess per image

**mAP50 progression (key milestones):**
| Epoch | mAP50 | mAP50-95 |
|-------|-------|----------|
| 10 | 0.388 | 0.220 |
| 30 | 0.573 | 0.375 |
| 50 | 0.644 | 0.449 |
| 70 | 0.683 | 0.487 |
| 100 | 0.683 | 0.490 |
| 120 | 0.706 | 0.505 |
| 140 | 0.721 | 0.527 |
| 148 (best) | 0.727 | 0.535 |
| 150 (final) | 0.726 | 0.534 |

**⚠️ LVC regression:** low_volt_cable mAP50 = 0.247 vs dissertation 0.434. Root cause: LVC oversampling cascaded to 61.3% of training set (1373/2238 images) because previous partial augmentation runs left files in the dataset that were re-detected and re-augmented. Severe over-representation biased the model away from real LVC appearance distribution. **Fix for next run: clean dataset first, cap at 2× oversample, add class weights.**

**Note:** Training converged cleanly with a single LR schedule (no checkpoint reset dip). mAP50 plateau from ep80–110 then broke through with close_mosaic kicking in at ep135.

---

## Experiment #11 — WIoU v3 · 150 epochs from scratch + LVC oversample

**Status:** ✅ COMPLETE  
**Weights:** `runs/detect/ev_battery_loss_exp/wiou_v2_150/weights/best.pt`  
**Config:** yolov9m.pt (fresh), lr0=0.01, batch=4, close_mosaic=15, copy_paste=0.3, patience=50, LVC ×3 oversample, amp=True

| Metric | Value | Notes |
|--------|-------|-------|
| Best epoch | **148** | Same convergence point as SIoU from-scratch run |
| mAP50 (best, from results.csv) | **0.7444** | Best mAP50 of all v2 dataset runs |
| mAP50-95 (best, from results.csv) | **0.5463** | Best mAP50-95 of all v2 runs |
| Precision (best) | 0.7137 | At best epoch |
| Recall (best) | 0.7274 | At best epoch |
| Final epoch mAP50 | 0.7426 | Extremely stable final 5 epochs (0.739–0.744) |
| Final epoch mAP50-95 | 0.5434 | Sustained quality |

**Validation results (best.pt on val split — 270 images, 1723 instances):**
| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| **all** | 270 | 1723 | 0.710 | 0.728 | **0.744** | **0.546** |
| BMS_Unit | 41 | 42 | 0.775 | 0.762 | 0.828 | 0.609 |
| Busbars | 35 | 72 | 0.691 | 0.764 | 0.793 | 0.620 |
| Coolant_tubes | 24 | 34 | 0.751 | 0.620 | 0.655 | 0.419 |
| battery_module | 177 | 548 | 0.848 | 0.893 | 0.866 | 0.648 |
| battery_tray | 138 | 138 | 0.850 | 0.902 | 0.942 | 0.809 |
| fasteners | 86 | 786 | 0.641 | 0.646 | 0.653 | 0.351 |
| **low_volt_cable** | 20 | 47 | 0.299 | 0.362 | **0.302** ⚠️ | 0.133 |
| pack_cover | 56 | 56 | 0.822 | 0.875 | 0.910 | 0.778 |

**Inference speed:** 1.9ms preprocess, 31.5ms inference, 1.0ms postprocess per image

**mAP50 progression (key milestones):**
| Epoch | mAP50 | mAP50-95 |
|-------|-------|----------|
| 10 | 0.419 | 0.243 |
| 30 | 0.583 | 0.365 |
| 50 | 0.653 | 0.437 |
| 70 | 0.660 | 0.466 |
| 100 | 0.702 | 0.504 |
| 120 | 0.722 | 0.522 |
| 140 | 0.723 | 0.526 |
| 148 (best) | 0.744 | 0.546 |
| 150 (final) | 0.743 | 0.544 |

**#10 SIoU vs #11 WIoU from-scratch comparison:**
| Metric | SIoU #10 | WIoU #11 | Winner |
|--------|----------|----------|--------|
| mAP50 (val) | 0.726 | **0.744** | WIoU +0.018 |
| mAP50-95 (val) | 0.537 | **0.546** | WIoU +0.009 |
| Precision | 0.712 | 0.710 | Draw |
| Recall | 0.712 | **0.728** | WIoU +0.016 |
| low_volt_cable mAP50 | 0.247 | **0.302** | WIoU +0.055 |
| fasteners mAP50 | **0.663** | 0.653 | SIoU +0.010 |
| battery_tray mAP50 | **0.943** | 0.942 | Draw |
| pack_cover mAP50 | 0.873 | **0.910** | WIoU +0.037 |

**⚠️ LVC regression:** low_volt_cable mAP50 = 0.302 (WIoU) / 0.247 (SIoU) vs dissertation 0.434. Root cause: LVC oversampling cascaded to 61.3% of training set because previous partial runs left augmented files that were re-detected and re-augmented. Over-representation biased model away from real LVC distribution. High recall (0.362) but low precision (0.299) — model detects in right area but poor confidence/box quality.

**Cleanup (Cell A):** Removed 2,573 augmented files. Dataset restored to 951 images ✅

**Verdict:** WIoU v3 wins on all key metrics. Best model on v2 dataset. LVC regression is the primary problem for next experiment. Both models were still improving at ep148 — training had not fully converged at 150 epochs.

---

## Experiment #12 — WIoU v3 · 200 epochs from scratch + LVC 2× + cls_pw=1.0

**Status:** ✅ COMPLETE  
**Weights:** `runs/detect/ev_battery_loss_exp/wiou_v2_200/weights/best.pt`  
**Config:** yolov9m.pt fresh, lr0=0.01, batch=4, close_mosaic=15, copy_paste=0.3, cls_pw=1.0, LVC 2×, patience=50, amp=True

| Metric | Value | Notes |
|--------|-------|-------|
| Best epoch | **174** | Ran full 200 epochs — patience=50 never triggered |
| mAP50 (best, results.csv) | **0.7564** | +0.012 vs #11 (0.744) — clear improvement |
| mAP50-95 (best, results.csv) | **0.5462** | Flat vs #11 (0.546) |
| Precision (best) | 0.7457 | +0.036 vs #11 |
| Recall (best) | 0.7374 | +0.009 vs #11 |
| Final epoch (200) mAP50 | 0.7475 | Still above 0.74 at final epoch — NOT fully converged |
| Final epoch (200) mAP50-95 | 0.5468 | Very slightly higher than best |

**Validation results (best.pt ep174, val split — 270 images, 1723 instances):**
| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| **all** | 270 | 1723 | 0.760 | 0.740 | **0.751** | **0.552** |
| BMS_Unit | 41 | 42 | 0.841 | 0.786 | 0.835 | 0.597 |
| Busbars | 35 | 72 | 0.765 | 0.875 | 0.852 | 0.661 |
| Coolant_tubes | 24 | 34 | 0.757 | 0.640 | 0.662 | 0.405 |
| battery_module | 177 | 548 | 0.859 | 0.887 | 0.876 | 0.664 |
| battery_tray | 138 | 138 | 0.801 | 0.913 | 0.881 | 0.753 |
| fasteners | 86 | 786 | 0.781 | 0.517 | 0.679 | 0.383 |
| **low_volt_cable** | 20 | 47 | 0.498 | 0.426 | **0.303** ⚠️ | 0.169 |
| pack_cover | 56 | 56 | 0.777 | 0.875 | 0.919 | 0.788 |

**Inference speed:** 1.9ms preprocess, 21.8ms inference, 1.8ms postprocess per image

**Cleanup:** Removed 348 augmented files (87 × 2 copies × 2 files each). Dataset restored to 951 images ✅

**mAP50 progression (key milestones):**
| Epoch | mAP50 | mAP50-95 |
|-------|-------|----------|
| 10 | 0.378 | 0.205 |
| 40 | 0.651 | 0.404 |
| 70 | 0.654 | 0.459 |
| 100 | 0.703 | 0.493 |
| 130 | 0.719 | 0.517 |
| 150 | 0.726 | 0.523 |
| 174 (best) | **0.756** | **0.546** |
| 200 (final) | 0.748 | 0.547 |

**#11 vs #12 comparison:**
| Metric | #11 WIoU 150ep | #12 WIoU 200ep | Delta |
|--------|----------------|----------------|-------|
| mAP50 (best) | 0.744 | **0.756** | +0.012 |
| mAP50-95 (best) | 0.546 | 0.546 | 0.000 |
| Precision | 0.714 | **0.760** | +0.046 |
| Recall | 0.728 | **0.740** | +0.012 |
| low_volt_cable mAP50 | 0.302 | 0.303 | +0.001 |
| fasteners mAP50 | 0.653 | **0.679** | +0.026 |
| Busbars mAP50 | 0.793 | **0.852** | +0.059 |
| battery_tray mAP50 | 0.942 | 0.881 | −0.061 |
| pack_cover mAP50 | 0.910 | **0.919** | +0.009 |

**Key findings:**
- Overall mAP50 improved +0.012 with 50 more epochs — training time is paying off
- **LVC mAP50 did not improve** (0.302 → 0.303): cls_pw=1.0 and 2× oversampling are insufficient. Root issue is the tiny validation set (20 images, 47 boxes) and likely model uncertainty about small cable appearance in diverse backgrounds
- **Model still not converged** — best at ep174, final epoch at 0.748, no early stopping triggered. Still has upward trajectory
- fasteners improved significantly (+0.026), Busbars most improved class (+0.059)
- battery_tray regressed slightly (−0.061) — may be trade-off from LVC class weight shift

**⚠️ LVC persists as the critical problem.** Across all v2 experiments: 0.247/0.302/0.303/0.303 — stuck around 0.30 regardless of loss function or oversampling strategy. Requires a different approach (see Next Steps).

---

## Experiment #13 — WIoU v3 · 250 epochs from scratch · cls_pw=1.0

**Status:** ✅ COMPLETE  
**Weights:** `runs/detect/ev_battery_loss_exp/wiou_v2_250/weights/best.pt`  
**Config:** yolov9m.pt fresh, lr0=0.01, lrf=0.01, batch=4, close_mosaic=15, copy_paste=0.3, cls_pw=1.0, LVC 2×, patience=50, amp=True  
**Note:** `fl_gamma=1.5` was removed — not a valid argument in the installed ultralytics version (throws SyntaxError before training starts).

| Metric | Value | Notes |
|--------|-------|-------|
| Best epoch (results.csv) | **216** | Peaked at ep216 vs ep174 in #12 — extra 50 epochs moved the peak later |
| mAP50 (best, results.csv) | **0.7555** | −0.0009 vs #12 (0.7564) — essentially flat in training metric |
| mAP50-95 (best, results.csv) | **0.5495** | At ep216 |
| Precision (ep216) | 0.7374 | At best epoch |
| Recall (ep216) | 0.7008 | At best epoch |
| Final epoch (ep250) mAP50 | 0.7501 | — |
| Final epoch (ep250) mAP50-95 | 0.5478 | — |
| Training time | **6.2 h** | 22,377 s total (~89 s/epoch) |

**Validation results (best.pt on val split — 270 images, 1723 instances):**
| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| **all** | 270 | 1723 | 0.729 | 0.727 | **0.752** | **0.554** |
| BMS_Unit | 41 | 42 | 0.756 | 0.739 | 0.811 | 0.566 |
| Busbars | 35 | 72 | 0.756 | 0.816 | 0.838 | 0.651 |
| Coolant_tubes | 24 | 34 | 0.702 | 0.676 | 0.676 | 0.443 |
| battery_module | 177 | 548 | 0.850 | 0.882 | 0.887 | 0.671 |
| battery_tray | 138 | 138 | 0.855 | 0.898 | 0.944 | 0.811 |
| fasteners | 86 | 786 | 0.715 | 0.590 | 0.683 | 0.372 |
| **low_volt_cable** | 20 | 47 | 0.432 | 0.319 | **0.263** ⚠️ | 0.138 |
| pack_cover | 56 | 56 | 0.768 | 0.893 | 0.915 | 0.778 |

**Inference speed:** 1.6ms preprocess, 22.1ms inference, 1.0ms postprocess per image

**mAP50 progression (key milestones):**
| Epoch | mAP50 | mAP50-95 | Notes |
|-------|-------|----------|-------|
| 50 | 0.6348 | 0.3881 | — |
| 100 | 0.6952 | 0.4867 | — |
| 150 | 0.7285 | 0.5136 | — |
| 200 | 0.7434 | 0.5448 | Same point #12 finished |
| 216 | **0.7555** | 0.5495 | **Best epoch (results.csv)** |
| 235 | 0.7519 | 0.5521 | Last epoch with mosaic active |
| 236 | 0.7412 | 0.5397 | ⚠️ close_mosaic fires: box_loss 658→282 |
| 250 | 0.7501 | 0.5478 | Final |

**#12 vs #13 full comparison (validation on best.pt):**
| Metric | #12 WIoU 200ep | #13 WIoU 250ep | Delta |
|--------|----------------|----------------|-------|
| mAP50 (val) | 0.751 | **0.752** | +0.001 |
| mAP50-95 (val) | 0.552 | **0.554** | +0.002 |
| Precision | **0.760** | 0.729 | −0.031 |
| Recall | **0.740** | 0.727 | −0.013 |
| BMS_Unit mAP50 | **0.835** | 0.811 | −0.024 |
| Busbars mAP50 | 0.852 | **0.838** | −0.014 |
| Coolant_tubes mAP50 | 0.662 | **0.676** | +0.014 |
| battery_module mAP50 | 0.876 | **0.887** | +0.011 |
| battery_tray mAP50 | 0.881 | **0.944** | **+0.063** |
| fasteners mAP50 | 0.679 | **0.683** | +0.004 |
| **low_volt_cable mAP50** | **0.303** | 0.263 | **−0.040** ⚠️ |
| pack_cover mAP50 | **0.919** | 0.915 | −0.004 |

**Key findings:**
- **Overall mAP50/mAP50-95 essentially flat** (+0.001 / +0.002 vs #12) — training length alone is not moving the needle at this point.
- **LVC REGRESSED: 0.303 → 0.263 (−0.040)** — the extra 50 epochs made LVC detection worse. Precision dropped (0.498→0.432) and recall dropped (0.426→0.319). More training is actively harming the minority class, likely because the model converges deeper into the majority class distribution.
- **battery_tray improved dramatically** (+0.063) — picked up from the extended training.
- **Precision dropped overall** (0.760→0.729) while recall is similar — the model is making more false detections at 250ep vs 200ep.
- **close_mosaic fires at ep236** causing a temporary mAP50 dip (0.752→0.741). Recovery to 0.750 at ep250 but peak ep216 was never recaptured.
- **LVC across all v2 experiments: 0.247 / 0.302 / 0.303 / 0.303 / 0.263** — stuck around 0.28–0.30 and now regressing. Training-only approaches have reached their ceiling for this class.

**⚠️ LVC is the blocking problem.** cls_pw, 2× oversampling, and more epochs have all failed to push LVC past 0.30. The root constraint is the v2 validation set (only 20 images / 47 boxes) — improvements may not be measurable even if the model improves. A fundamentally different approach is needed (see Decision Log / Next Steps).

---

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2025 | Switched from YOLOv9m → YOLOv9s for production | 4 GB VRAM; YOLOv9m needed batch=2 |
| 2025 | Expanded dataset v1→v2 via pseudo-labelling | Fasteners only +35 imgs in manual set; model-assisted annotation added 311 more |
| 2025 | Dropped CIoU fine-tune from best.pt | mAP regressed 0.803→0.766; root cause: switching loss on CIoU checkpoint collapses mAP (GH #8607) |
| 2026-06-21 | Chose monkey-patch approach for loss experiments | Avoids editing ultralytics source; portable across versions; self-contained in one notebook |
| 2026-06-21 | Chose NOT to retrain with bicubic upsampling | Dissertation explicitly proved it hurts recall for small objects on this dataset |
| 2026-06-21 | Set copy_paste=0.1 (not 0.3) for v2 experiments | v2 fine-tune with copy_paste=0.3 caused low_volt_cable regression; conservative setting for from-scratch runs |
| 2026-06-21 | Fixed validation cell path (siou_v2 AssertionError) | Ultralytics always saves to `runs/detect/{project}/{name}/` — validation cell was looking in `{project}/{name}/` (missing prefix). Fix: use `model.trainer.save_dir` + `RUNS_SAVE_BASE` fallback |
| 2026-06-21 | Changed exist_ok=False → True in train() calls | exist_ok=False auto-increments folder (siou_v2 → siou_v22) if folder exists from a test run; hardcoded paths then fail |
| 2026-06-21 | Extended runs start from last.pt with lr0=0.001 (not yolov9m.pt) | Both models not converged at ep80; starting from ep80 weights with lower LR avoids destroying learned features; saves ~6h vs full retrain |
| 2026-06-22 | Switched to 150-epoch from-scratch for #10/#11 | Extension runs peaked at ep59/70 and converged; optimiser reset caused early dip; cleaner to run 150ep from scratch with single LR schedule |
| 2026-06-22 | LVC oversample ×3 added before training #10/#11 | low_volt_cable only in 87/951 train images (9.1%); oversampling to ~348 images (28%) increases gradient signal for this class; uses augmented copies so original data unchanged |
| 2026-06-22 | copy_paste raised 0.1→0.3 for #10/#11 | Dissertion used 0.3; raised back as from-scratch training is more stable under stronger augmentation than fine-tuning |
| 2026-06-23 | Dropped SIoU for #12 onwards | WIoU v3 confirmed better on every metric in #11 vs #10; no value repeating SIoU |
| 2026-06-23 | Epochs raised 150→200 for #12 | Both #10 and #11 peaked at ep148/150 — models cut off before convergence; 200ep with patience=50 lets them plateau naturally |
| 2026-06-23 | LVC copies reduced 3×→2×, cascade guard added | 3× copies + leftover augmented files caused LVC to reach 61% of training set in #10/#11, collapsing LVC mAP50 to 0.247/0.302; cascade guard in find_lvc_images() now skips _lvcaug files and asserts exactly 87 originals |
| 2026-06-23 | cls_pw corrected 2.0→1.0 for #12 | ultralytics 8.4.x validates cls_pw in [0,1] — it is an inverse-frequency power exponent not a free multiplier; 1.0=full inverse-freq weighting |
| 2026-06-24 | Model not converged at 200 epochs | #12 peaked at ep174/200 with full 200 run (no early stopping); final epoch still scoring 0.747 mAP50 — still has headroom |
| 2026-07-02 | Epochs raised 200→250 for #13 | #12 peaked at ep174 and final ep200 still at 0.748 — clearly not converged; extra 50 epochs moved peak to ep216 |
| 2026-07-02 | fl_gamma=1.5 removed from #13 | Installed ultralytics version rejects fl_gamma as an unknown argument (SyntaxError before training); not available in this build |
| 2026-07-02 | close_mosaic=15 fires at ep236 causing dip | With 250 epochs, close_mosaic=15 turns off mosaic at ep235, causing a box_loss drop from 658→282 and temporary mAP50 dip 0.752→0.741; consider raising close_mosaic to 20–25 in future |
| 2026-07-02 | LVC regressed at 250ep (0.303→0.263) | More training is hurting LVC — model converges deeper into majority class distribution; training-length and oversampling approaches exhausted for LVC; requires data or architectural fix |

---

## Next experiment candidates (after #6 and #7)

| Priority | Idea | Expected benefit | Time cost |
|----------|------|-----------------|-----------|
| High | Inner-SIoU (best of #6 + auxiliary scaled box) | +3–4% mAP50 (SGI-YOLOv9 paper) | ~4 hrs |
| High | SAHI inference only (no retrain) on winning model | +10–20% fastener recall at inference time | ~2 hrs setup |
| Medium | Expand low_volt_cable manually (50+ more images) | Targeted cable data; only 271 boxes in v2 | 1 day annotation |
| Medium | MLflow logging (retrospective, all 7 runs) | Recruiter-visible experiment tracking | ~3 hrs |
| Low | YOLOv9m with SIoU from scratch | Potentially +2–3% mAP50 over YOLOv9s | ~8 hrs (slower) |
