"""
database_activity.py — Activity & Behaviour Pattern Database
==============================================================
3rd Database: What is the person/object DOING?

Detects:
  Person activities:  sitting, walking, running, eating, drinking,
                      writing, typing, reading, talking, phone_use,
                      driving, dancing, sleeping, group_talking
  Interaction:        person + object → "using laptop", "writing with pen"
  Object motion:      moving direction + speed for vehicles/objects
  Pose estimation:    uses MediaPipe Pose (if available) or bbox-ratio heuristic

Pattern registry stored in: activity_patterns/pattern_registry.json
"""

import os
import cv2
import json
import numpy as np
from datetime import datetime
from collections import deque, defaultdict
from security import get_security

PATTERN_DIR      = "activity_patterns"
PATTERN_REGISTRY = os.path.join(PATTERN_DIR, "pattern_registry.json")
os.makedirs(PATTERN_DIR, exist_ok=True)

# ── MediaPipe optional ────────────────────────────────────────
# Python 3.12+ / 3.14 compatibility: solutions API may not exist
MP_AVAILABLE = False
_POSE        = None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        _mp_pose = mp.solutions.pose
        _POSE    = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        MP_AVAILABLE = True
        print("[ACTIVITY] MediaPipe Pose loaded.")
    else:
        print("[ACTIVITY] MediaPipe solutions API unavailable. Using heuristics.")
except ImportError:
    print("[ACTIVITY] MediaPipe not installed. Using bbox-ratio heuristics.")
except Exception as e:
    print(f"[ACTIVITY] MediaPipe failed ({e}). Using heuristics.")


# ══ Pose Estimation ════════════════════════════════════════════

def estimate_pose_mp(person_crop: np.ndarray) -> dict:
    """Use MediaPipe to get pose landmarks."""
    if not MP_AVAILABLE or person_crop is None or person_crop.size == 0:
        return {}
    rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    res = _POSE.process(rgb)
    if not res.pose_landmarks:
        return {}
    h, w = person_crop.shape[:2]
    lm = res.pose_landmarks.landmark
    pts = {}
    for name, idx in [
        ("nose",        0),  ("l_shoulder", 11), ("r_shoulder", 12),
        ("l_elbow",    13),  ("r_elbow",    14), ("l_wrist",    15),
        ("r_wrist",    16),  ("l_hip",      23), ("r_hip",      24),
        ("l_knee",     25),  ("r_knee",     26), ("l_ankle",    27),
        ("r_ankle",    28),
    ]:
        pts[name] = (lm[idx].x * w, lm[idx].y * h, lm[idx].visibility)
    return pts


def estimate_pose_heuristic(body_bbox, head_bbox) -> dict:
    """
    Fallback: Use relative bbox positions to guess pose.
    body_bbox: (x1,y1,x2,y2)
    head_bbox: (hx1,hy1,hx2,hy2)
    """
    if body_bbox is None:
        return {"pose": "unknown"}
    x1, y1, x2, y2 = body_bbox
    bh = y2 - y1
    bw = x2 - x1
    ar = bh / (bw + 1e-6)

    if ar > 2.5:
        return {"pose": "standing"}
    elif ar > 1.3:
        return {"pose": "sitting"}
    else:
        return {"pose": "lying_or_crouching"}


# ══ Activity Classifier ════════════════════════════════════════

class ActivityClassifier:
    """
    Classify what a person is doing based on:
    1. Pose landmarks (MediaPipe) or bbox heuristic
    2. Nearby object context (what objects are close)
    3. Motion history (speed, direction)
    4. Interaction analysis (person + object overlap)
    """

    # Bbox IoU threshold for person-object interaction
    INTERACTION_IOU = 0.10

    @staticmethod
    def _iou(b1, b2) -> float:
        """Intersection over Union for two bboxes."""
        ax1, ay1, ax2, ay2 = b1
        bx1, by1, bx2, by2 = b2
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw  = max(0, ix2-ix1); ih = max(0, iy2-iy1)
        ia  = iw * ih
        a1  = (ax2-ax1)*(ay2-ay1)
        a2  = (bx2-bx1)*(by2-by1)
        return ia / (a1 + a2 - ia + 1e-6)

    @staticmethod
    def _wrist_near_face(pose: dict, thresh=0.25) -> bool:
        if not pose:
            return False
        h = pose.get("nose")
        for k in ["l_wrist", "r_wrist"]:
            w = pose.get(k)
            if h and w and abs(w[1] - h[1]) / (abs(w[0] - h[0]) + 1e-6) < thresh:
                return True
        return False

    @staticmethod
    def _wrist_near_mouth(pose: dict) -> bool:
        """Detect eating/drinking: wrist close to mouth region."""
        if not pose:
            return False
        nose = pose.get("nose")
        for k in ["l_wrist", "r_wrist"]:
            w = pose.get(k)
            if nose and w:
                dist = ((w[0]-nose[0])**2 + (w[1]-nose[1])**2)**0.5
                crop_h = nose[1] * 4
                if dist < crop_h * 0.20:
                    return True
        return False

    @staticmethod
    def _elbows_raised(pose: dict) -> bool:
        """Detect typing: both elbows at waist height, wrists forward."""
        if not pose:
            return False
        le = pose.get("l_elbow"); re = pose.get("r_elbow")
        lh = pose.get("l_hip");   rh = pose.get("r_hip")
        if le and re and lh and rh:
            avg_elbow_y = (le[1] + re[1]) / 2
            avg_hip_y   = (lh[1] + rh[1]) / 2
            return avg_elbow_y < avg_hip_y * 0.85
        return False

    def classify(self, person_bbox, head_bbox,
                 pose: dict, motion: dict,
                 nearby_objects: list,
                 pose_heuristic: dict = None) -> dict:
        """
        Returns activity dict:
        {
          "activity": "typing",
          "pose":     "sitting",
          "motion":   "stationary",
          "direction": "",
          "interaction": "using laptop",
          "confidence": 0.82
        }
        """
        motion_state = motion.get("state", "stationary")
        direction    = motion.get("direction", "")
        speed        = motion.get("speed", "")
        ph = pose_heuristic or estimate_pose_heuristic(person_bbox, head_bbox)
        pose_label   = ph.get("pose", "unknown")

        # ── Interaction: find closest overlapping object ───────
        interaction = ""
        best_iou    = 0.0
        for obj in nearby_objects:
            iou = self._iou(person_bbox, obj["bbox"])
            if iou > self.INTERACTION_IOU and iou > best_iou:
                best_iou    = iou
                use_pattern = obj.get("use_pattern", "")
                obj_label   = obj.get("label", "")
                # Map use_pattern to interaction string
                pat_map = {
                    "computing/work":   f"using {obj_label}",
                    "computing/input":  f"using {obj_label}",
                    "communication/entertainment": "using phone",
                    "drinking/hydration": "drinking",
                    "drinking":           "drinking",
                    "eating":             "eating",
                    "writing":            "writing",
                    "reading/study":      "reading",
                    "transportation":     f"riding {obj_label}",
                }
                interaction = pat_map.get(use_pattern, f"interacting with {obj_label}")

        # ── Activity from pose + motion + interaction ──────────
        if MP_AVAILABLE and pose:
            if motion_state == "moving" and speed == "fast":
                activity = "running"
            elif motion_state == "moving":
                activity = "walking"
            elif interaction in ("drinking", "eating"):
                activity = interaction
            elif self._wrist_near_mouth(pose) and not interaction:
                activity = "eating/drinking"
            elif interaction.startswith("using phone") or self._wrist_near_face(pose):
                activity = "using phone"
            elif self._elbows_raised(pose) or "laptop" in interaction or "keyboard" in interaction:
                activity = "typing"
            elif "writing" in interaction:
                activity = "writing"
            elif "reading" in interaction:
                activity = "reading"
            elif pose_label == "lying_or_crouching":
                activity = "sleeping/resting"
            elif pose_label == "sitting":
                activity = interaction if interaction else "sitting"
            else:
                activity = interaction if interaction else "standing"
        else:
            # Heuristic-only path
            if motion_state == "moving" and speed == "fast":
                activity = "running"
            elif motion_state == "moving":
                activity = "walking"
            elif interaction:
                activity = interaction
            elif pose_label == "lying_or_crouching":
                activity = "sleeping/resting"
            elif pose_label == "sitting":
                activity = "sitting"
            else:
                activity = "standing"

        # Build label string
        label_parts = [activity]
        if pose_label not in ("unknown", activity.split()[0]):
            label_parts.append(f"({pose_label})")
        if motion_state == "moving" and direction:
            label_parts.append(f"→{direction}")

        return {
            "activity":    activity,
            "pose":        pose_label,
            "motion":      motion_state,
            "direction":   direction,
            "speed":       speed,
            "interaction": interaction,
            "label":       " ".join(label_parts),
            "confidence":  0.85 if MP_AVAILABLE else 0.60,
        }


# ══ Group/Social Detector ══════════════════════════════════════

def detect_social_activity(person_bboxes: list) -> dict:
    """
    Detect group conversations based on spatial proximity.
    Returns: {person_id: social_label}
    """
    if len(person_bboxes) < 2:
        return {}

    def center(b):
        return ((b[0]+b[2])/2, (b[1]+b[3])/2)

    PROXIMITY = 180   # pixels

    groups = {}
    for i, (pid1, bbox1) in enumerate(person_bboxes):
        close = []
        c1    = center(bbox1)
        for j, (pid2, bbox2) in enumerate(person_bboxes):
            if i == j:
                continue
            c2   = center(bbox2)
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
            if dist < PROXIMITY:
                close.append(pid2)
        if close:
            n = len(close) + 1
            groups[pid1] = "group talking" if n >= 3 else "talking (2 people)"

    return groups


# ══ Pattern Registry ═══════════════════════════════════════════

class ActivityPatternDB:
    """
    Stores activity patterns per person_id for future behaviour analysis.
    Helps system learn usual patterns: "P0001 always sits at 10am".
    """

    def __init__(self):
        self._sec      = get_security()
        self._patterns = self._load()
        print(f"[ACTIVITY-DB] {len(self._patterns)} person activity records.")

    def _load(self) -> dict:
        if os.path.exists(PATTERN_REGISTRY):
            try:
                with open(PATTERN_REGISTRY, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(PATTERN_REGISTRY, 'w') as f:
            json.dump(self._patterns, f, indent=2)
        self._sec.record_write(PATTERN_REGISTRY)

    def record(self, person_id: str, activity_info: dict):
        """Log an activity observation for a person."""
        if person_id not in self._patterns:
            self._patterns[person_id] = {
                "person_id":   person_id,
                "observations": [],
                "summary":      {}
            }
        entry = {
            "time":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "activity":    activity_info.get("activity", ""),
            "pose":        activity_info.get("pose", ""),
            "interaction": activity_info.get("interaction", ""),
        }
        # Keep last 200 per person
        obs = self._patterns[person_id]["observations"]
        obs.append(entry)
        if len(obs) > 200:
            obs.pop(0)

        # Update summary counts
        s = self._patterns[person_id]["summary"]
        act = entry["activity"]
        s[act] = s.get(act, 0) + 1

        # Save every 20 observations
        if len(obs) % 20 == 0:
            self._save()

    def get_profile(self, person_id: str) -> dict:
        return self._patterns.get(person_id, {})

    def most_common_activity(self, person_id: str) -> str:
        s = self._patterns.get(person_id, {}).get("summary", {})
        if not s:
            return "unknown"
        return max(s, key=s.get)