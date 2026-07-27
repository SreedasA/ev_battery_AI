from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = [
    "BMS_Unit",
    "Busbars",
    "Coolant_tubes",
    "battery_module",
    "battery_tray",
    "fasteners",
    "low_volt_cable",
    "pack_cover",
]

COLORS = {
    "BMS_Unit": (80, 180, 255),
    "Busbars": (0, 140, 255),
    "Coolant_tubes": (255, 190, 60),
    "battery_module": (80, 220, 120),
    "battery_tray": (180, 120, 255),
    "fasteners": (50, 220, 255),
    "low_volt_cable": (255, 90, 120),
    "pack_cover": (210, 210, 210),
}

MIN_CLASS_CONF = {
    "BMS_Unit": 0.65,
    "Busbars": 0.55,
    "Coolant_tubes": 0.65,
    "battery_module": 0.55,
    "battery_tray": 0.70,
    "fasteners": 0.55,
    "low_volt_cable": 0.55,
    "pack_cover": 0.70,
}

DEFAULT_MODEL = (
    "runs/detect/ev_battery_loss_exp/"
    "wiou_v2_250/weights/best.pt"
)
DEFAULT_CONFIG = "configs/video_inference.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the EV battery detector on a disassembly video."
    )
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights path")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Optional inference config")
    parser.add_argument("--out-dir", default="outputs/video_runs", help="Output folder")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument("--stride", type=int, default=1, help="Use every nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    parser.add_argument("--max-boxes", type=int, default=12, help="Most confident boxes per frame")
    parser.add_argument("--roi", default="", help="Optional x1,y1,x2,y2 filter in pixels")
    parser.add_argument("--loose", action="store_true", help="Show all boxes above --conf")
    parser.add_argument("--no-track", action="store_true", help="Use predict instead of track")
    return parser.parse_args()


def pick_device(requested):
    if requested != "auto":
        return requested

    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_run_dir(out_dir, source):
    source = Path(source)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / f"{source.stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_config(path):
    config = {
        "class_thresholds": MIN_CLASS_CONF.copy(),
        "roi": None,
        "max_boxes": None,
    }
    path = Path(path)

    if not path.exists():
        return config

    user_config = json.loads(path.read_text(encoding="utf-8"))
    config["class_thresholds"].update(user_config.get("class_thresholds", {}))
    config["roi"] = user_config.get("roi")
    config["max_boxes"] = user_config.get("max_boxes")
    return config


def parse_roi(value):
    if not value:
        return None
    parts = [int(float(v.strip())) for v in value.split(",")]
    if len(parts) != 4:
        raise ValueError("roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("roi must have x2 > x1 and y2 > y1")
    return [x1, y1, x2, y2]


def box_center_in_roi(xyxy, roi):
    if roi is None:
        return True
    x1, y1, x2, y2 = xyxy
    rx1, ry1, rx2, ry2 = roi
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def draw_box(cv2, frame, xyxy, label, color):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    tw, th = text_size
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, y_text + th + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )


def draw_roi(cv2, frame, roi):
    if roi is None:
        return
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)


def open_writer(cv2, path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def box_value(value):
    try:
        return value.item()
    except AttributeError:
        return value


def keep_detection(class_name, conf, loose, thresholds):
    if loose:
        return True
    return conf >= thresholds.get(class_name, 0.60)


def run_video(args):
    import cv2
    from ultralytics import YOLO

    source = Path(args.source)
    model_path = Path(args.model)

    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if args.stride < 1:
        raise ValueError("stride must be 1 or higher")

    config = load_config(args.config)
    thresholds = config["class_thresholds"]
    roi = parse_roi(args.roi) if args.roi else config.get("roi")
    max_boxes = config.get("max_boxes") or args.max_boxes

    run_dir = make_run_dir(args.out_dir, source)
    out_video = run_dir / "annotated.mp4"
    out_csv = run_dir / "predictions.csv"
    out_summary = run_dir / "summary.json"

    device = pick_device(args.device)
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(1, src_fps / args.stride)
    progress_callback = getattr(args, "progress_callback", None)

    writer = open_writer(cv2, out_video, out_fps, width, height)

    counts = Counter()
    conf_sum = defaultdict(float)
    frame_count = 0
    used_frames = 0

    columns = [
        "frame",
        "time_sec",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=columns)
        writer_csv.writeheader()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_count % args.stride != 0:
                frame_count += 1
                continue

            if args.max_frames and used_frames >= args.max_frames:
                break

            if args.no_track:
                result = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    verbose=False,
                )[0]
            else:
                result = model.track(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    persist=True,
                    verbose=False,
                )[0]

            frame_detections = []

            for box in result.boxes:
                cls_id = int(box_value(box.cls[0]))
                conf = float(box_value(box.conf[0]))
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                xyxy = [float(v) for v in box.xyxy[0].tolist()]

                if not keep_detection(class_name, conf, args.loose, thresholds):
                    continue
                if not box_center_in_roi(xyxy, roi):
                    continue

                track_id = ""
                if getattr(box, "id", None) is not None:
                    track_id = int(box_value(box.id[0]))

                frame_detections.append((conf, cls_id, class_name, xyxy, track_id))

            frame_detections.sort(reverse=True, key=lambda item: item[0])
            if max_boxes > 0:
                frame_detections = frame_detections[:max_boxes]

            for conf, cls_id, class_name, xyxy, track_id in frame_detections:
                counts[class_name] += 1
                conf_sum[class_name] += conf

                label = f"{class_name} {conf:.2f}"
                if track_id != "":
                    label = f"{class_name} #{track_id} {conf:.2f}"

                draw_box(cv2, frame, xyxy, label, COLORS.get(class_name, (255, 255, 255)))

                writer_csv.writerow(
                    {
                        "frame": frame_count,
                        "time_sec": round(frame_count / src_fps, 3),
                        "track_id": track_id,
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(conf, 5),
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2),
                    }
                )

            draw_roi(cv2, frame, roi)
            writer.write(frame)
            used_frames += 1
            frame_count += 1

            if progress_callback and used_frames % 10 == 0:
                progress_callback(frame_count, total_frames, used_frames)

    cap.release()
    writer.release()

    if progress_callback:
        progress_callback(frame_count, total_frames, used_frames)

    avg_conf = {
        name: round(conf_sum[name] / counts[name], 4)
        for name in sorted(counts)
        if counts[name] > 0
    }

    summary = {
        "source": str(source),
        "model": str(model_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": device,
        "stride": args.stride,
        "roi": roi,
        "max_boxes": max_boxes,
        "class_thresholds": thresholds,
        "source_fps": round(src_fps, 3),
        "source_frames": total_frames,
        "frames_processed": used_frames,
        "detections_by_class": dict(sorted(counts.items())),
        "average_confidence_by_class": avg_conf,
        "annotated_video": str(out_video),
        "predictions_csv": str(out_csv),
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved video: {out_video}")
    print(f"saved csv:   {out_csv}")
    print(f"saved json:  {out_summary}")
    return summary


if __name__ == "__main__":
    run_video(parse_args())
