"""
database_objects.py — Non-Living Object Visual DNA Database
=============================================================
NO photo storage. Only visual fingerprints:
  - Edge histogram (Canny + HOG-lite)
  - Dominant color palette (K-means, 5 clusters)
  - Aspect ratio + relative size
  - Motion state (moving / stationary)
  - Use-pattern label (what it's typically used for)

Only MOVING / HAND-CARRIED objects tracked:
  Fixed objects (walls, floors, doors, windows) EXCLUDED by EXCLUDE_CLASSES.

Daily-life objects prioritized:
  phone, laptop, bottle, bag, book, pen, cup, keyboard, mouse,
  chair (when someone sits), vehicle (when moving), etc.
"""

import os
import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from security import get_security

OBJ_REGISTRY = "objects/object_registry.json"
OBJ_DNA_DIR  = "objects/dna"
os.makedirs("objects/dna", exist_ok=True)

# ── COCO classes to SKIP (permanent/fixed environment) ────────
EXCLUDE_CLASSES = {
    "wall", "floor", "ceiling", "window", "door", "building",
    "sky", "road", "sidewalk", "grass", "tree", "fence",
    "curtain", "light", "traffic light", "stop sign",
    "fire hydrant", "parking meter", "bench",   # fixed infra
}

# ── Daily-life object priority (these get USE-PATTERN labels) ─
USE_PATTERNS = {
    "cell phone":   "communication/entertainment",
    "laptop":       "computing/work",
    "keyboard":     "computing/input",
    "mouse":        "computing/input",
    "book":         "reading/study",
    "pen":          "writing",
    "pencil":       "writing",
    "notebook":     "writing/study",
    "bottle":       "drinking/hydration",
    "cup":          "drinking",
    "wine glass":   "drinking",
    "fork":         "eating",
    "spoon":        "eating",
    "knife":        "eating/cooking",
    "bowl":         "eating",
    "backpack":     "carrying/storage",
    "handbag":      "carrying/personal",
    "suitcase":     "travel/carrying",
    "umbrella":     "weather protection",
    "tie":          "clothing/formal",
    "scissors":     "cutting/crafts",
    "remote":       "device control",
    "clock":        "timekeeping",
    "vase":         "decoration",
    "car":          "transportation",
    "motorcycle":   "transportation",
    "bicycle":      "transportation/exercise",
    "bus":          "public transport",
    "truck":        "cargo transport",
    "chair":        "seating",
    "couch":        "seating/relaxation",
    "bed":          "sleeping/rest",
    "dining table": "eating/working",
    "tv":           "entertainment/display",
    "toothbrush":   "personal hygiene",
    "hair drier":   "personal grooming",
    "sports ball":  "sports/recreation",
    "skateboard":   "sports/transport",
    "surfboard":    "sports",
    "tennis racket":"sports",
    "baseball bat": "sports",
    "frisbee":      "sports/recreation",
    "kite":         "recreation",
    "snowboard":    "sports",
    "skis":         "sports",
}


# ══ Visual DNA Extraction ══════════════════════════════════════

def extract_dna(crop: np.ndarray, yolo_class: str = "") -> dict:
    """
    Extract visual fingerprint WITHOUT storing photo.
    Returns a dict of features.
    """
    if crop is None or crop.size == 0:
        return {}

    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 1. Edge histogram (16-bin)
    edges      = cv2.Canny(gray, 50, 150)
    edge_hist, _ = np.histogram(edges.flatten(), bins=16, range=(0, 256))
    edge_hist  = (edge_hist / (edge_hist.sum() + 1e-6)).tolist()

    # 2. Dominant colors via K-means (5 clusters)
    pixels = crop.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 500:
        idx     = np.random.choice(len(pixels), 500, replace=False)
        pixels  = pixels[idx]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = min(5, len(pixels))
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    palette = []
    for i in np.argsort(-counts):
        b, g, r = centers[i].astype(int).tolist()
        palette.append({"rgb": [r, g, b], "pct": float(counts[i] / counts.sum())})

    # 3. Aspect ratio + size bucket
    ar = round(w / (h + 1e-6), 2)
    area = h * w
    if area < 3000:
        size = "tiny"
    elif area < 15000:
        size = "small"
    elif area < 60000:
        size = "medium"
    else:
        size = "large"

    # 4. Texture variance
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 5. Brightness
    brightness = float(gray.mean())

    return {
        "edge_hist":   edge_hist,
        "palette":     palette,
        "aspect_ratio": ar,
        "size_bucket": size,
        "texture":     round(texture, 2),
        "brightness":  round(brightness, 2),
        "use_pattern": USE_PATTERNS.get(yolo_class, "general"),
    }


def dna_similarity(dna1: dict, dna2: dict) -> float:
    """
    Cosine similarity between two edge histograms + color match.
    Returns 0.0–1.0 score.
    """
    if not dna1 or not dna2:
        return 0.0

    # Edge histogram cosine
    e1 = np.array(dna1.get("edge_hist", [0]*16))
    e2 = np.array(dna2.get("edge_hist", [0]*16))
    edge_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-6))

    # Dominant color match (compare top-2 colors)
    p1 = dna1.get("palette", [])[:2]
    p2 = dna2.get("palette", [])[:2]
    color_sim = 0.0
    if p1 and p2:
        c1 = np.array(p1[0]["rgb"], dtype=float)
        c2 = np.array(p2[0]["rgb"], dtype=float)
        dist = np.linalg.norm(c1 - c2) / 441.67   # max distance in RGB
        color_sim = 1.0 - dist

    # Size match
    size_sim = 1.0 if dna1.get("size_bucket") == dna2.get("size_bucket") else 0.3

    return 0.5 * edge_sim + 0.3 * color_sim + 0.2 * size_sim


# ══ ObjectDatabase ═════════════════════════════════════════════

class ObjectDatabase:
    """
    Visual DNA registry for non-living objects.
    - No photos stored — only edge/color/size fingerprints
    - Fixed environment objects excluded
    - Motion state tracked per session
    """

    def __init__(self):
        self._sec      = get_security()
        self._registry = {}   # obj_id → record
        self._dna_cache= {}   # obj_id → dna dict (RAM)
        self._motion   = {}   # tracker_id → motion state
        self._load()
        print(f"[OBJ-DB] Ready. {len(self._registry)} object DNA records.")

    # ── Load / Save ───────────────────────────────────────────

    def _load(self):
        if os.path.exists(OBJ_REGISTRY):
            try:
                with open(OBJ_REGISTRY, 'r') as f:
                    data = json.load(f)
                self._registry  = data.get("registry", {})
                self._dna_cache = data.get("dna_cache", {})
            except Exception as e:
                print(f"[OBJ-DB] Load error: {e}")

    def _save(self):
        with open(OBJ_REGISTRY, 'w') as f:
            json.dump({"registry": self._registry, "dna_cache": self._dna_cache},
                      f, indent=2)
        self._sec.record_write(OBJ_REGISTRY)

    # ── Identify / Register ───────────────────────────────────

    def identify_or_register(self, crop: np.ndarray, yolo_class: str,
                              tracker_id: int, state_obj) -> tuple:
        """
        Returns: (obj_label, is_known, use_pattern)

        Skip fixed/excluded classes → returns (None, False, None)
        """
        yc_lower = yolo_class.lower()

        # Skip fixed environment objects
        if yc_lower in EXCLUDE_CLASSES:
            return None, False, None

        # Extract visual DNA (no photo stored)
        dna = extract_dna(crop, yc_lower)
        if not dna:
            return None, False, None

        use_pattern = dna.get("use_pattern", "general")

        # Check object_id_map (session cache)
        tid = int(tracker_id)
        if tid in state_obj.object_id_map:
            cached = state_obj.object_id_map[tid]
            return cached, not cached.startswith("Object_"), use_pattern

        # Match against DNA registry
        best_id, best_sim = None, 0.0
        for oid, rec in self._registry.items():
            if rec.get("yolo_class") != yc_lower:
                continue
            cached_dna = self._dna_cache.get(oid, {})
            sim = dna_similarity(dna, cached_dna)
            if sim > 0.72 and sim > best_sim:
                best_sim, best_id = sim, oid

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if best_id:
            # Known object
            self._registry[best_id]["last_seen"]   = now
            self._registry[best_id]["seen_count"]  = self._registry[best_id].get("seen_count", 1) + 1
            self._save()
            state_obj.object_id_map[tid] = best_id
            return best_id, True, use_pattern
        else:
            # New object
            state_obj.object_counter += 1
            new_label = f"Object_{state_obj.object_counter}"
            self._registry[new_label] = {
                "obj_id":      new_label,
                "yolo_class":  yc_lower,
                "use_pattern": use_pattern,
                "registered":  now,
                "last_seen":   now,
                "seen_count":  1,
            }
            self._dna_cache[new_label] = dna
            self._save()
            state_obj.object_id_map[tid] = new_label
            return new_label, False, use_pattern

    # ── Motion tracking ───────────────────────────────────────

    def update_motion(self, tracker_id: int, bbox, prev_bboxes: dict) -> dict:
        """
        Compare current bbox center with previous to determine:
        - moving / stationary
        - direction (North/South/East/West/NE/NW/SE/SW)
        - speed bucket (slow/fast)
        Returns motion info dict.
        """
        tid = int(tracker_id)
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        motion = {"state": "stationary", "direction": "", "speed": ""}

        if tid in prev_bboxes:
            px1, py1, px2, py2 = prev_bboxes[tid]
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            dx, dy   = cx - pcx, cy - pcy
            dist     = (dx**2 + dy**2) ** 0.5

            if dist > 6:
                motion["state"] = "moving"
                # Direction
                angle = np.degrees(np.arctan2(-dy, dx))
                dirs  = ["East","NE","North","NW","West","SW","South","SE"]
                motion["direction"] = dirs[int((angle + 202.5) / 45) % 8]
                motion["speed"]     = "fast" if dist > 20 else "slow"

        prev_bboxes[tid] = bbox
        self._motion[tid] = motion
        return motion

    def get_motion(self, tracker_id: int) -> dict:
        return self._motion.get(int(tracker_id), {"state": "stationary"})

    # ── Excluded check ────────────────────────────────────────

    @staticmethod
    def is_excluded(yolo_class: str) -> bool:
        return yolo_class.lower() in EXCLUDE_CLASSES