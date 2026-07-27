# Project Structure

```text
.
├── app/
│   ├── __init__.py
│   └── main.py                  FastAPI web app — upload video, track job, download output
├── configs/
│   └── video_inference.json     Per-class confidence thresholds, ROI filter, max_boxes
├── docs/
│   ├── PROJECT_STRUCTURE.md     This file
│   └── REPRODUCIBILITY.md       Training environment, how to reproduce Experiment #13
├── k8s/
│   ├── deployment.yaml          Kubernetes deployment template
│   └── service.yaml             NodePort service
├── models/
│   └── README.md                Model weights location and download instructions
├── outputs/
│   └── .gitkeep                 Placeholder — video runs written here at runtime
├── .github/
│   └── workflows/
│       └── syntax-check.yml     CI — Python syntax check on all .py files
├── video_annotator.py           CLI inference — video → annotated MP4 + CSV + JSON
├── pseudo_label.py              Model-assisted annotation pipeline (uses best.pt)
├── sample_frames.py             Frame sampling from raw video for annotation staging
├── upload_to_roboflow.py        Batch upload reviewed labels to Roboflow project
├── Dockerfile                   Container build — CPU inference, health check wired to /health
├── docker-compose.yml           Local Docker run
├── requirements.txt             Core dependencies (ML + web app)
├── requirements-app.txt         FastAPI app dependencies (used in Docker)
├── requirements-video.txt       CLI video annotator only
├── EXPERIMENTS.md               Full experiment log — per-class tables, decision log
├── MODEL_CARD.md                Model architecture, training config, limitations, citations
├── CONTRIBUTING.md              Contribution guidelines
├── LICENSE                      MIT
└── README.md                    Main documentation
```

## Key Files

**`loss_experiment.ipynb`** is the primary research notebook. It documents 13 training
experiments comparing SIoU and WIoU v3 loss functions on the v2 dataset, with controlled
LVC oversampling and class weighting. Run cells in order to reproduce training.
(Notebooks are gitignored by default — use `git add -f loss_experiment.ipynb` to share.)

**`video_annotator.py`** is the shared inference engine. It is used directly from the
command line and imported by the web app (`app/main.py`).

**`EXPERIMENTS.md`** contains the full experiment log including per-class validation tables,
epoch-by-epoch mAP50 progression, and a decision log explaining every configuration change.

**`MODEL_CARD.md`** documents model architecture, training hyperparameters, per-class
performance, intended use, limitations, and citations.

## Deployment Files

`Dockerfile` and `docker-compose.yml` are for local container runs (CPU inference).
The `/health` endpoint in `app/main.py` validates model weights on startup.

`k8s/` contains a Kubernetes deployment template for demonstration purposes.

## Annotation Pipeline

`sample_frames.py` → `pseudo_label.py` → `upload_to_roboflow.py` forms the
model-assisted annotation workflow used to expand the dataset from v1 to v2.

## Ignored Local Files

The following are not committed to git (large or private):

- `EV Battery Dataset 1045 v2.0/` — dataset images and labels (1,356 images)
- `runs/` — training outputs and model weights
- `outputs/` — inference outputs (annotated videos, CSVs)
- `frames/`, `annotation_staging/`, `pseudo_labels/` — annotation pipeline working data
- `*.ipynb` — notebooks (use `git add -f` to share a specific notebook)
- `*.pdf`, `*.pt`, `*.mp4` — private documents, model weights, videos
