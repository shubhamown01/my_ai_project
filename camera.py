"""
camera.py — Smart Vision Master Camera Thread
===============================================
Integrates all 3 databases + activity engine:
  - PERSON: head tracking, permanent ID, photo management
  - OBJECT: visual DNA, motion direction, use pattern
  - ACTIVITY: pose, behaviour, interaction, social detection
"""

import os
import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

import config
from database_persons  import PersonDatabase
from database_objects  import ObjectDatabase
from database_activity import (ActivityClassifier, ActivityPatternDB,
                                detect_social_activity,
                                estimate_pose_mp, estimate_pose_heuristic,
                                MP_AVAILABLE)
from database_objects  import EXCLUDE_CLASSES, USE_PATTERNS

PERSON_CLASS_ID = 0

# ── Haar Cascades ──────────────────────────────────────────────
face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# ── YOLOv8-face optional ───────────────────────────────────────
_head_model = None
def _get_head_model():
    global _head_model
    if _head_model is None:
        try:
            _head_model = YOLO("yolov8n-face.pt")
            print("[CAMERA] yolov8n-face.pt loaded.")
        except Exception:
            _head_model = False
    return _head_model if _head_model is not False else None


# ══ Head Extraction ════════════════════════════════════════════

def _haar(gray_crop, x1, y1, fs):
    for cas in [face_cascade, profile_cascade]:
        fcs = cas.detectMultiScale(gray_crop, 1.08, 3, minSize=(18,18),
                                   flags=cv2.CASCADE_SCALE_IMAGE)
        if len(fcs) > 0:
            fx,fy,fw,fh = max(fcs, key=lambda f: f[2]*f[3])
            p = 6
            return (max(0,x1+fx-p), max(0,y1+fy-p),
                    min(fs[1],x1+fx+fw+p), min(fs[0],y1+fy+fh+p))
    return None

def _geo(x1,y1,x2,y2,fs):
    ph,pw = y2-y1, x2-x1
    hh,mg = int(ph*0.26), int(pw*0.12)
    hx1,hy1 = max(0,x1+mg), max(0,y1)
    hx2,hy2 = min(fs[1],x2-mg), min(fs[0],y1+hh)
    return (hx1,hy1,hx2,hy2) if (hx2-hx1)>=10 and (hy2-hy1)>=10 else None

def detect_head(frame, bbox, hm=None):
    x1,y1,x2,y2 = bbox
    crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
    if crop.size == 0:
        return _geo(x1,y1,x2,y2,frame.shape)
    # Tier 1: YOLOv8-face
    if hm:
        try:
            r = hm(crop, verbose=False, conf=0.35)
            if r[0].boxes and len(r[0].boxes) > 0:
                best = r[0].boxes[r[0].boxes.conf.argmax()]
                fx1,fy1,fx2,fy2 = best.xyxy[0].cpu().numpy().astype(int)
                p=5
                return (max(0,x1+fx1-p),max(0,y1+fy1-p),
                        min(frame.shape[1],x1+fx2+p),min(frame.shape[0],y1+fy2+p))
        except Exception:
            pass
    # Tier 2: Haar
    g = cv2.equalizeHist(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY))
    r = _haar(g, x1, y1, frame.shape)
    if r: return r
    # Tier 3: Geometric
    return _geo(x1,y1,x2,y2,frame.shape)


# ══ Draw Helpers ═══════════════════════════════════════════════

def _draw_label(frame, text, pt, color, scale=0.45, thick=2):
    fn = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),bl = cv2.getTextSize(text, fn, scale, thick)
    x,y = pt
    cv2.rectangle(frame,(x,y-th-bl-4),(x+tw+4,y+bl),color,-1)
    tc = (0,0,0) if sum(color)>400 else (255,255,255)
    cv2.putText(frame, text, (x+2,y-2), fn, scale, tc, thick)

def draw_person(frame, head_bbox, pid, activity_label, color, has_good):
    hx1,hy1,hx2,hy2 = head_bbox
    cx,cy = (hx1+hx2)//2,(hy1+hy2)//2
    rx,ry = max(1,(hx2-hx1)//2), max(1,(hy2-hy1)//2)
    th = 3 if has_good else 2
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,0,360,color,th)
    # Photo quality dot
    dc = (0,255,0) if has_good else (0,165,255)
    cv2.circle(frame,(cx,hy1+7),4,dc,-1)
    # ID line
    _draw_label(frame, pid, (hx1,hy1), color)
    # Activity line (below ellipse)
    if activity_label:
        _draw_label(frame, activity_label, (hx1, hy2+4), (60,60,60), scale=0.38, thick=1)

def draw_object(frame, bbox, label, use_pattern, motion_info, color):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    # Label
    _draw_label(frame, label, (x1,y1), color)
    # Motion + use pattern below box
    parts = []
    if motion_info.get("state") == "moving":
        parts.append(f"→{motion_info.get('direction','')} ({motion_info.get('speed','')})")
    if use_pattern and use_pattern != "general":
        parts.append(use_pattern)
    if parts:
        _draw_label(frame," | ".join(parts),(x1,y2+2),(80,80,80),scale=0.35,thick=1)


# ══ Camera Thread ══════════════════════════════════════════════

def camera_thread(person_db: PersonDatabase,
                  object_db: ObjectDatabase,
                  activity_db: ActivityPatternDB,
                  state):
    """
    Unified camera thread using all 3 databases.
    """
    print("[CAMERA] Loading model...")
    model      = YOLO(config.MODEL_PATH)
    head_model = _get_head_model()
    classifier = ActivityClassifier()

    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print(f"[CAMERA ERROR] {config.CAMERA_SOURCE}")
        return
    print(f"[CAMERA] Live. {config.CAMERA_SOURCE}")

    prev_bboxes   = {}    # tracker_id → prev bbox (for motion)
    last_items    = []
    activity_log  = defaultdict(lambda: deque(maxlen=5))  # per-person last 5 activities
    RECORD_EVERY  = 30   # frames between activity DB writes

    frame_count = 0

    while state.running:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1

        display = cv2.resize(frame, (640, 480))

        results = model.track(display, persist=True,
                              conf=config.CONFIDENCE,
                              verbose=False, tracker="bytetrack.yaml")

        if results[0].boxes.id is None:
            # Draw last items (momentum)
            for it in last_items:
                _render_item(display, it)
            state.latest_frame = display
            continue

        boxes  = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids    = results[0].boxes.id.cpu().numpy().astype(int)
        clss   = results[0].boxes.cls.cpu().numpy().astype(int)

        current_labels = []
        new_items      = []

        # ── Collect all objects this frame (for interaction analysis) ──
        frame_objects = []
        for box, tid, cls in zip(boxes, ids, clss):
            if cls != PERSON_CLASS_ID:
                yc = model.names[cls]
                if not ObjectDatabase.is_excluded(yc):
                    crop  = display[max(0,box[1]):box[3], max(0,box[0]):box[2]]
                    label, is_known, use_pat = object_db.identify_or_register(
                        crop, yc, int(tid), state)
                    if label:
                        mot = object_db.update_motion(int(tid), tuple(box), prev_bboxes)
                        frame_objects.append({
                            "label":       label,
                            "bbox":        tuple(box),
                            "use_pattern": use_pat or "",
                            "motion":      mot,
                            "is_known":    is_known,
                        })

        # ── Persons ───────────────────────────────────────────
        person_bboxes_for_social = []

        for box, tid, cls in zip(boxes, ids, clss):
            x1,y1,x2,y2 = box
            t_int = int(tid)

            if cls == PERSON_CLASS_ID:
                # ── Head detection ────────────────────────────
                head_bbox = detect_head(display, (x1,y1,x2,y2), head_model)
                if head_bbox is None:
                    continue
                hx1,hy1,hx2,hy2 = head_bbox
                head_crop = display[hy1:hy2, hx1:hx2]

                # ── Permanent recognition ─────────────────────
                cached_pid = state.person_id_map.get(t_int)
                pid, is_recog, blur = person_db.recognize_or_register(head_crop)

                if pid:
                    state.person_id_map[t_int] = pid
                    label = pid
                elif cached_pid:
                    label = cached_pid
                    pid   = cached_pid
                    is_recog = True
                else:
                    label = f"T_{t_int}"
                    pid   = None

                current_labels.append(label)
                person_bboxes_for_social.append((label, (x1,y1,x2,y2)))

                # ── Pose + Activity ───────────────────────────
                pose_data = estimate_pose_mp(display[max(0,y1):y2, max(0,x1):x2]) \
                            if MP_AVAILABLE else {}
                pose_h    = estimate_pose_heuristic((x1,y1,x2,y2), head_bbox)
                motion    = object_db.update_motion(t_int, (x1,y1,x2,y2), prev_bboxes)

                act_info  = classifier.classify(
                    (x1,y1,x2,y2), head_bbox,
                    pose_data, motion,
                    frame_objects,
                    pose_h
                )
                act_label = act_info["label"]

                # Social override (will be applied after loop)
                activity_log[label].append(act_label)

                # Record to activity DB periodically
                if pid and frame_count % RECORD_EVERY == 0:
                    activity_db.record(pid, act_info)

                # ── UI memory ─────────────────────────────────
                has_good  = person_db.has_good_photo(pid) if pid else False
                img_path  = (person_db.get_photo_path(pid) or
                             os.path.join("persons/photos", f"{label}.jpg"))

                if head_crop.size > 0 and pid and not os.path.exists(img_path):
                    cv2.imwrite(img_path, head_crop)

                if label not in state.unique_objects:
                    rec = person_db.get_record(pid) if pid else {}
                    state.unique_objects[label] = {
                        "uid":            label,
                        "display_name":   label,
                        "image_path":     img_path,
                        "type":           "person",
                        "has_good_photo": has_good,
                        "visit_count":    rec.get("visit_count", 1) if rec else 1,
                        "is_recognized":  is_recog,
                        "activity":       act_label,
                    }
                else:
                    state.unique_objects[label]["activity"]       = act_label
                    state.unique_objects[label]["has_good_photo"] = has_good
                    if pid:
                        r = person_db.get_record(pid)
                        if r:
                            state.unique_objects[label]["visit_count"] = r.get("visit_count",1)
                            np2 = person_db.get_photo_path(pid)
                            if np2:
                                state.unique_objects[label]["image_path"] = np2

                color = config.PERSON_RECOGNIZED_COLOR if is_recog else config.PERSON_COLOR
                new_items.append({
                    "type":     "person",
                    "head_bbox": head_bbox,
                    "label":    label,
                    "act":      act_label,
                    "color":    color,
                    "good":     has_good,
                })

        # ── Social detection ──────────────────────────────────
        social = detect_social_activity(person_bboxes_for_social)
        for item in new_items:
            if item["type"] == "person" and item["label"] in social:
                item["act"] = social[item["label"]]
                if item["label"] in state.unique_objects:
                    state.unique_objects[item["label"]]["activity"] = item["act"]

        # ── Objects draw items ────────────────────────────────
        for obj in frame_objects:
            current_labels.append(obj["label"])
            is_known = obj["is_known"]
            color    = config.KNOWN_COLOR if is_known else config.UNKNOWN_COLOR

            if obj["label"] not in state.unique_objects:
                state.unique_objects[obj["label"]] = {
                    "uid":          obj["label"],
                    "display_name": obj["label"],
                    "image_path":   "",
                    "type":         "object",
                    "known":        is_known,
                    "use_pattern":  obj.get("use_pattern",""),
                    "motion":       obj.get("motion",{}),
                }

            new_items.append({
                "type":       "object",
                "bbox":       obj["bbox"],
                "label":      obj["label"],
                "use_pattern": obj.get("use_pattern",""),
                "motion":     obj.get("motion",{}),
                "color":      color,
            })

        with state.lock:
            state.active_detections = current_labels
        last_items = new_items

        # ── Draw ──────────────────────────────────────────────
        for it in last_items:
            _render_item(display, it)

        state.latest_frame = display

    cap.release()
    print("[CAMERA] Stopped.")


def _render_item(frame, item):
    if item["type"] == "person":
        draw_person(frame, item["head_bbox"], item["label"],
                    item.get("act",""), item["color"], item.get("good",False))
    else:
        draw_object(frame, item["bbox"], item["label"],
                    item.get("use_pattern",""), item.get("motion",{}), item["color"])