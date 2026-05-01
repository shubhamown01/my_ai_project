"""
training_pipeline.py — Model Training & Video Import Pipeline
===============================================================
Features:
1. Upload sample video → extract frames → train/fine-tune object classifier
2. Provide YouTube URL → yt-dlp download → same pipeline
3. Custom object labeling UI integration
4. YOLOv8 fine-tuning on custom dataset
5. Dataset auto-augmentation (flip, rotate, brightness)
"""

import os
import cv2
import json
import subprocess
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

TRAIN_DIR    = "training_data"
FRAMES_DIR   = os.path.join(TRAIN_DIR, "frames")
LABELS_DIR   = os.path.join(TRAIN_DIR, "labels")
DATASET_DIR  = os.path.join(TRAIN_DIR, "dataset")
MODELS_DIR   = os.path.join(TRAIN_DIR, "models")
LOG_FILE     = os.path.join(TRAIN_DIR, "training_log.json")

for d in [FRAMES_DIR, LABELS_DIR, DATASET_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


# ══ YouTube Downloader ═════════════════════════════════════════

def check_ytdlp():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_youtube(url: str, output_path: str, progress_cb=None) -> str:
    """
    Download YouTube video using yt-dlp.
    Returns: local video filepath or None on error.
    """
    if not check_ytdlp():
        if progress_cb:
            progress_cb("yt-dlp not installed. Run: pip install yt-dlp", -1)
        return None

    out_template = os.path.join(output_path, "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url
    ]

    if progress_cb:
        progress_cb(f"Downloading: {url}", 0)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Find downloaded file
            for f in Path(output_path).glob("*.mp4"):
                if progress_cb:
                    progress_cb(f"Downloaded: {f.name}", 100)
                return str(f)
        else:
            if progress_cb:
                progress_cb(f"Download failed: {result.stderr[:200]}", -1)
    except subprocess.TimeoutExpired:
        if progress_cb:
            progress_cb("Download timed out (>5 min)", -1)
    except Exception as e:
        if progress_cb:
            progress_cb(f"Error: {e}", -1)
    return None


# ══ Frame Extractor ════════════════════════════════════════════

def extract_frames(video_path: str, session_id: str,
                   fps_sample: int = 2,
                   max_frames: int = 500,
                   progress_cb=None) -> list:
    """
    Extract frames from video at fps_sample per second.
    Returns list of saved frame paths.
    """
    cap  = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if progress_cb:
            progress_cb(f"Cannot open video: {video_path}", -1)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    skip         = max(1, int(video_fps / fps_sample))

    session_dir  = os.path.join(FRAMES_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    saved   = []
    count   = 0
    frame_n = 0

    while count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(session_dir, f"frame_{frame_n:06d}.jpg")
        cv2.imwrite(fname, frame)
        saved.append(fname)
        count  += 1
        frame_n += skip

        if progress_cb and count % 20 == 0:
            pct = int(count / max_frames * 100)
            progress_cb(f"Extracted {count} frames...", pct)

    cap.release()
    if progress_cb:
        progress_cb(f"Done. {len(saved)} frames extracted.", 100)

    return saved


# ══ Auto Labeler (YOLO inference → pseudo-labels) ══════════════

def auto_label_frames(frame_paths: list, model_path: str,
                      session_id: str, progress_cb=None,
                      frame_cb=None) -> dict:
    """
    Run existing YOLO model on extracted frames to generate
    pseudo-labels for fine-tuning.
    Returns: {frame_path: [{"class": "bottle", "bbox": [...]}]}
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        if progress_cb:
            progress_cb("ultralytics not installed", -1)
        return {}

    model   = YOLO(model_path)
    labels  = {}
    total   = len(frame_paths)

    label_dir = os.path.join(LABELS_DIR, session_id)
    os.makedirs(label_dir, exist_ok=True)

    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        results = model(frame, verbose=False, conf=0.4)
        detections = []
        yolo_lines = []

        if results[0].boxes is not None:
            for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(),
                                results[0].boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                cx  = ((x1+x2)/2) / w
                cy  = ((y1+y2)/2) / h
                bw  = (x2-x1) / w
                bh  = (y2-y1) / h
                cls_id = int(cls)
                detections.append({"class_id": cls_id, "bbox": [x1,y1,x2,y2]})
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        labels[fp] = detections

        # Send annotated frame to UI preview callback
        if frame_cb is not None:
            try:
                ann_frame = frame.copy()
                for det in detections:
                    x1,y1,x2,y2 = det["bbox"]
                    cv2.rectangle(ann_frame, (x1,y1),(x2,y2),(0,255,180),2)
                    cv2.putText(ann_frame, f"cls:{det['class_id']}",
                                (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0,255,180), 1)
                frame_cb(ann_frame)
            except Exception:
                pass

        # Save YOLO-format label
        lpath = os.path.join(label_dir, Path(fp).stem + ".txt")
        with open(lpath, 'w') as f:
            f.write("\n".join(yolo_lines))

        if progress_cb and i % 10 == 0:
            pct = int(i / total * 100)
            progress_cb(f"Labeling {i}/{total}...", pct)

    if progress_cb:
        progress_cb("Auto-labeling complete.", 100)
    return labels


# ══ Augmentation ═══════════════════════════════════════════════

def augment_frame(frame: np.ndarray) -> list:
    """
    Basic augmentations: flip, brightness, rotate.
    Returns list of augmented frames.
    """
    aug = []
    # Horizontal flip
    aug.append(cv2.flip(frame, 1))
    # Brightness +30
    bright = np.clip(frame.astype(int) + 30, 0, 255).astype(np.uint8)
    aug.append(bright)
    # Brightness -30
    dark = np.clip(frame.astype(int) - 30, 0, 255).astype(np.uint8)
    aug.append(dark)
    # Slight rotation +10°
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
    aug.append(cv2.warpAffine(frame, M, (w, h)))
    return aug


# ══ Training Session ═══════════════════════════════════════════

class TrainingSession:
    """
    Manages a complete training workflow:
    1. Video source (file or YouTube URL)
    2. Frame extraction
    3. Auto-labeling
    4. (Optional) YOLOv8 fine-tuning
    5. Progress reporting to UI
    """

    def __init__(self, source: str, source_type: str = "file"):
        """
        source:      filepath or YouTube URL
        source_type: "file" | "youtube"
        """
        self.source      = source
        self.source_type = source_type
        self.session_id  = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.progress    = 0
        self.status      = "idle"
        self.log         = []
        self._thread     = None
        self._frame_cb   = None   # Optional: callback(frame) for live preview

    def _log(self, msg: str, pct: int = -1):
        entry = {"time": datetime.now().strftime("%H:%M:%S"),
                 "msg": msg, "pct": pct}
        self.log.append(entry)
        if pct >= 0:
            self.progress = pct
        print(f"[TRAIN] {msg}")

    def start(self, model_path: str = "yolov8n.pt",
              fine_tune: bool = False, epochs: int = 10):
        """Launch training in background thread."""
        self._thread = threading.Thread(
            target=self._run,
            args=(model_path, fine_tune, epochs),
            daemon=True
        )
        self._thread.start()

    def _run(self, model_path, fine_tune, epochs):
        self.status = "running"

        # Step 1: Get video file
        if self.source_type == "youtube":
            self._log("Downloading YouTube video...", 5)
            video_path = download_youtube(
                self.source,
                os.path.join(TRAIN_DIR, "downloads"),
                self._log
            )
            if not video_path:
                self.status = "failed"
                return
        else:
            video_path = self.source
            if not os.path.exists(video_path):
                self._log(f"File not found: {video_path}", -1)
                self.status = "failed"
                return

        # Step 2: Extract frames
        self._log("Extracting frames...", 10)
        frames = extract_frames(video_path, self.session_id,
                                fps_sample=2, max_frames=400,
                                progress_cb=self._log)
        if not frames:
            self.status = "failed"
            return

        # Step 3: Auto-label with live preview
        self._log("Auto-labeling with YOLO...", 50)
        labels = auto_label_frames(frames, model_path,
                                   self.session_id, self._log,
                                   frame_cb=self._frame_cb)

        # Step 4: (Optional) Fine-tune
        if fine_tune and labels:
            self._log("Starting YOLOv8 fine-tuning...", 70)
            self._fine_tune(model_path, epochs)

        # Save session log
        log_data = {
            "session_id":   self.session_id,
            "source":       self.source,
            "type":         self.source_type,
            "frames":       len(frames),
            "labeled":      len(labels),
            "completed_at": datetime.now().isoformat(),
        }
        log_path = os.path.join(TRAIN_DIR, f"{self.session_id}_summary.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        self._log(f"Training session complete. {len(frames)} frames, {len(labels)} labeled.", 100)
        self.status = "done"

    def _fine_tune(self, base_model: str, epochs: int):
        try:
            from ultralytics import YOLO
            model = YOLO(base_model)
            yaml_path = self._create_dataset_yaml()
            if yaml_path:
                model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=640,
                    batch=8,
                    project=MODELS_DIR,
                    name=self.session_id,
                    exist_ok=True,
                    verbose=False
                )
                self._log(f"Fine-tuning done. Model saved in {MODELS_DIR}", 95)
        except Exception as e:
            self._log(f"Fine-tuning error: {e}", -1)

    def _create_dataset_yaml(self) -> str:
        yaml_path = os.path.join(DATASET_DIR, f"{self.session_id}.yaml")
        content   = f"""
path: {os.path.abspath(FRAMES_DIR)}/{self.session_id}
train: .
val: .
nc: 80
names: [person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,
        traffic light,fire hydrant,stop sign,parking meter,bench,bird,
        cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,
        umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,
        kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,
        bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,
        orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,
        potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,
        keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,
        book,clock,vase,scissors,teddy bear,hair drier,toothbrush]
"""
        with open(yaml_path, 'w') as f:
            f.write(content.strip())
        return yaml_path

    def is_done(self) -> bool:
        return self.status in ("done", "failed")

    def get_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "status":     self.status,
            "progress":   self.progress,
            "log":        self.log[-10:],
        }