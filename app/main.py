from __future__ import annotations

import shutil
import threading
import uuid
from html import escape
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from video_annotator import DEFAULT_CONFIG, DEFAULT_MODEL, run_video


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "outputs" / "uploads"
OUTPUT_DIR = ROOT / "outputs" / "video_runs"

# Maximum accepted upload size (bytes). Prevents OOM on large 4K video files.
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="EV Battery Video Annotator")
app.mount("/outputs", StaticFiles(directory=ROOT / "outputs"), name="outputs")

jobs = {}


@app.get("/health")
def health():
    """Liveness probe for Docker/K8s. Also validates the model file is present."""
    model_path = ROOT / DEFAULT_MODEL
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model weights not found: {model_path}",
        )
    return {"status": "ok", "model": str(model_path)}


STYLE = """
body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: #f4f6f8;
  color: #17202a;
}
main {
  max-width: 980px;
  margin: 0 auto;
  padding: 32px 20px;
}
h1 {
  font-size: 28px;
  margin: 0 0 8px;
}
p {
  line-height: 1.5;
}
.panel {
  background: #ffffff;
  border: 1px solid #d7dde3;
  border-radius: 6px;
  padding: 20px;
  margin-top: 18px;
}
label {
  display: block;
  font-weight: 700;
  margin: 14px 0 6px;
}
input {
  box-sizing: border-box;
  width: 100%;
  padding: 10px;
  border: 1px solid #b9c2cb;
  border-radius: 4px;
  font-size: 15px;
}
button,
.button {
  display: inline-block;
  margin-top: 18px;
  padding: 10px 14px;
  border: 0;
  border-radius: 4px;
  background: #1d4ed8;
  color: #ffffff;
  font-weight: 700;
  text-decoration: none;
  cursor: pointer;
}
.muted {
  color: #5d6773;
}
.links a {
  margin-right: 12px;
}
video {
  width: 100%;
  max-height: 560px;
  background: #000000;
}
pre {
  overflow: auto;
  background: #101827;
  color: #e6edf3;
  padding: 14px;
  border-radius: 4px;
}
.progress-wrap {
  height: 18px;
  background: #dce3ea;
  border-radius: 999px;
  overflow: hidden;
  margin: 16px 0 8px;
}
.progress-bar {
  height: 100%;
  width: 0%;
  background: #1d4ed8;
  transition: width 0.3s ease;
}
.status-line {
  font-weight: 700;
}
"""


def page(title, body):
    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>{title}</title>
          <style>{STYLE}</style>
        </head>
        <body>
          <main>{body}</main>
        </body>
        </html>
        """
    )


def output_link(path_text):
    path = Path(path_text)
    try:
        rel = path.resolve().relative_to((ROOT / "outputs").resolve())
    except ValueError:
        return ""
    return "/outputs/" + rel.as_posix()


def progress_update(job_id):
    def update(frame_count, total_frames, used_frames):
        percent = 0
        if total_frames:
            percent = min(99, int((frame_count / total_frames) * 100))
        jobs[job_id]["frame"] = frame_count
        jobs[job_id]["frames_processed"] = used_frames
        jobs[job_id]["total_frames"] = total_frames
        jobs[job_id]["percent"] = percent

    return update


def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return {"found": False}

    data = {
        "found": True,
        "status": job.get("status", "unknown"),
        "percent": job.get("percent", 0),
        "frame": job.get("frame", 0),
        "frames_processed": job.get("frames_processed", 0),
        "total_frames": job.get("total_frames", 0),
        "error": job.get("error", ""),
    }

    if data["status"] == "done":
        summary = job["summary"]
        data["video_url"] = output_link(summary["annotated_video"])
        data["csv_url"] = output_link(summary["predictions_csv"])
        data["json_url"] = output_link(str(Path(summary["predictions_csv"]).with_name("summary.json")))

    return data


def process_video(job_id, source, conf, iou, imgsz, max_boxes, roi, no_track):
    try:
        jobs[job_id]["status"] = "running"
        args = SimpleNamespace(
            source=str(source),
            model=str(ROOT / DEFAULT_MODEL),
            config=str(ROOT / DEFAULT_CONFIG),
            out_dir=str(OUTPUT_DIR),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device="auto",
            stride=1,
            max_frames=0,
            max_boxes=max_boxes,
            roi=roi.strip(),
            loose=False,
            no_track=no_track,
            progress_callback=progress_update(job_id),
        )
        summary = run_video(args)
        jobs[job_id]["percent"] = 100
        jobs[job_id]["status"] = "done"
        jobs[job_id]["summary"] = summary
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


@app.get("/", response_class=HTMLResponse)
def index():
    body = """
    <h1>EV Battery Video Annotator</h1>
    <p class="muted">Upload a disassembly video and run the YOLOv9m EV battery component detector (WIoU v3, mAP50 0.752, 8 classes).</p>

    <section class="panel">
      <form action="/jobs" method="post" enctype="multipart/form-data">
        <label>Video file</label>
        <input type="file" name="video" accept="video/*" required>

        <label>Confidence</label>
        <input type="number" name="conf" value="0.25" min="0.05" max="0.95" step="0.05">

        <label>IoU</label>
        <input type="number" name="iou" value="0.50" min="0.10" max="0.95" step="0.05">

        <label>Image size</label>
        <input type="number" name="imgsz" value="640" min="320" max="1024" step="32">

        <label>Max boxes per frame</label>
        <input type="number" name="max_boxes" value="12" min="1" max="80" step="1">

        <label>ROI filter, optional</label>
        <input type="text" name="roi" placeholder="x1,y1,x2,y2">

        <label>
          <input type="checkbox" name="no_track" value="true" style="width:auto">
          Run without object tracking
        </label>

        <button type="submit">Start annotation</button>
      </form>
    </section>
    """
    return page("EV Battery Video Annotator", body)


@app.post("/jobs", response_class=HTMLResponse)
async def create_job(
    video: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.50),
    imgsz: int = Form(640),
    max_boxes: int = Form(12),
    roi: str = Form(""),
    no_track: bool = Form(False),
):
    job_id = uuid.uuid4().hex[:10]
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    source = UPLOAD_DIR / f"{job_id}{suffix}"

    # Stream upload to disk with size guard to prevent OOM on large videos.
    bytes_written = 0
    chunk_size = 1024 * 1024  # 1 MB
    with source.open("wb") as f:
        while True:
            chunk = await video.read(chunk_size)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                source.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
                )
            f.write(chunk)

    jobs[job_id] = {
        "status": "queued",
        "source": str(source),
        "percent": 0,
        "frame": 0,
        "frames_processed": 0,
        "total_frames": 0,
    }
    worker = threading.Thread(
        target=process_video,
        args=(job_id, source, conf, iou, imgsz, max_boxes, roi, no_track),
        daemon=True,
    )
    worker.start()

    body = f"""
    <h1>Annotation started</h1>
    <p class="muted">Job id: {job_id}</p>
    <section class="panel">
      <p>The video is being processed. The progress page will update automatically.</p>
      <a class="button" href="/jobs/{job_id}">Open job</a>
    </section>
    <script>
      window.location.href = "/jobs/{job_id}";
    </script>
    """
    return page("Annotation started", body)


@app.get("/jobs/{job_id}/status")
def status_api(job_id: str):
    return JSONResponse(job_status(job_id))


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return page("Job not found", "<h1>Job not found</h1><p>No job exists with that id.</p>")

    status = job["status"]
    if status in {"queued", "running"}:
        body = f"""
        <h1>Job {status}</h1>
        <section class="panel">
          <p class="muted">Job id: {job_id}</p>
          <p class="status-line" id="statusText">Preparing video...</p>
          <div class="progress-wrap">
            <div class="progress-bar" id="progressBar"></div>
          </div>
          <p class="muted" id="frameText">Waiting for the first frames.</p>
        </section>
        <script>
          async function checkJob() {{
            const res = await fetch("/jobs/{job_id}/status");
            const data = await res.json();

            if (!data.found) {{
              document.getElementById("statusText").textContent = "Job not found.";
              return;
            }}

            const percent = data.percent || 0;
            document.getElementById("progressBar").style.width = percent + "%";
            document.getElementById("statusText").textContent =
              data.status === "queued" ? "Queued..." : "Processing video... " + percent + "%";

            let frameText = "Frames processed: " + (data.frames_processed || 0);
            if (data.total_frames) {{
              frameText += " / about " + data.total_frames;
            }}
            document.getElementById("frameText").textContent = frameText;

            if (data.status === "done") {{
              window.location.href = "/jobs/{job_id}?view=done";
              return;
            }}

            if (data.status === "failed") {{
              document.getElementById("statusText").textContent = "Job failed.";
              document.getElementById("frameText").textContent = data.error || "Unknown error";
              return;
            }}

            setTimeout(checkJob, 1500);
          }}
          checkJob();
        </script>
        """
        return page("Job running", body)

    if status == "failed":
        body = f"""
        <h1>Job failed</h1>
        <section class="panel">
          <p>{escape(job.get("error", "Unknown error"))}</p>
          <a class="button" href="/">Run another video</a>
        </section>
        """
        return page("Job failed", body)

    summary = job["summary"]
    video_url = output_link(summary["annotated_video"])
    csv_url = output_link(summary["predictions_csv"])
    json_url = output_link(str(Path(summary["predictions_csv"]).with_name("summary.json")))
    pretty_summary = escape(json.dumps(summary, indent=2))

    body = f"""
    <h1>Annotation complete</h1>
    <p class="muted">Job id: {job_id}</p>
    <section class="panel">
      <video controls src="{video_url}"></video>
      <p class="links">
        <a href="{video_url}">Download video</a>
        <a href="{csv_url}">Download CSV</a>
        <a href="{json_url}">Download JSON</a>
      </p>
    </section>
    <section class="panel">
      <h2>Summary</h2>
      <pre>{pretty_summary}</pre>
    </section>
    """
    return page("Annotation complete", body)
