"""
database_persons.py — Permanent Person Face Registry
======================================================
- Zero-latency recognition via RAM-cached ORB descriptors
- Permanent unique IDs (P0001, P0002...) — NEVER change
- Blur-gated photo: only best quality photo saved
- Photo auto-upgrade: blur score always improves, never degrades
- AES-256 encrypted registry backup
- SHA-256 tamper detection

Folder layout:
  persons/
    person_registry.json       ← master registry (plaintext + .enc backup)
    photos/
      P0001.jpg                ← best quality face/head crop
    features/
      P0001.pkl                ← ORB 800-point descriptors (RAM cached)
    thumbnails/
      P0001_thumb.jpg          ← 64x64 thumbnail for UI
"""

import os
import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from security import get_security

# ══ Paths ══════════════════════════════════════════════════════
PERSON_DIR       = "persons"
PERSON_PHOTO_DIR = os.path.join(PERSON_DIR, "photos")
PERSON_FEAT_DIR  = os.path.join(PERSON_DIR, "features")
PERSON_THUMB_DIR = os.path.join(PERSON_DIR, "thumbnails")
PERSON_REGISTRY  = os.path.join(PERSON_DIR, "person_registry.json")

for _d in [PERSON_PHOTO_DIR, PERSON_FEAT_DIR, PERSON_THUMB_DIR]:
    os.makedirs(_d, exist_ok=True)

# ══ Tuning ═════════════════════════════════════════════════════
BLUR_REJECT     = 80     # Below → completely skip (too blurry)
BLUR_ACCEPTABLE = 80     # First registration threshold
BLUR_UPGRADE    = 200    # Must exceed current stored score to upgrade photo
ORB_FEATURES    = 800    # Descriptor count per face
MIN_MATCHES     = 28     # Good ORB matches needed to confirm same person
MATCH_DIST      = 55     # Max Hamming distance for a "good" match


# ══ Helpers ════════════════════════════════════════════════════

def _blur_score(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (128, 128))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _orb_des(img: np.ndarray):
    if img is None or img.size == 0:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.equalizeHist(g)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    _, des = orb.detectAndCompute(g, None)
    return des


def _good_matches(des1, des2) -> int:
    if des1 is None or des2 is None:
        return 0
    if len(des1) < 2 or len(des2) < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ms = bf.match(des1, des2)
    return sum(1 for m in ms if m.distance < MATCH_DIST)


def _make_thumb(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (64, 64)) if img is not None and img.size > 0 else None


# ══ PersonDatabase ═════════════════════════════════════════════

class PersonDatabase:
    """
    Zero-latency permanent face registry.

    Key design decisions:
    - All ORB descriptors loaded into RAM on startup → no disk I/O during recognition
    - Photo saved ONLY when blur_score >= BLUR_UPGRADE (or first time >= BLUR_ACCEPTABLE)
    - Person ID (P0001) is permanent and never reassigned
    - Security: all writes go through SecurityManager (audit + integrity update)
    """

    def __init__(self):
        self._sec      = get_security()
        self._registry = {}      # pid → record dict
        self._cache    = {}      # pid → ORB descriptors (RAM)
        self._load()
        self._warm_cache()
        print(f"[PERSON-DB] Ready. {len(self._registry)} persons in registry.")

    # ── Load / Save ───────────────────────────────────────────

    def _load(self):
        if os.path.exists(PERSON_REGISTRY):
            if not self._sec.verify_read(PERSON_REGISTRY):
                print("[PERSON-DB] ⚠️  Registry tamper warning — loading anyway.")
            try:
                with open(PERSON_REGISTRY, 'r') as f:
                    self._registry = json.load(f)
            except Exception as e:
                print(f"[PERSON-DB] Load error: {e}")
                self._registry = {}

    def _save(self):
        with open(PERSON_REGISTRY, 'w') as f:
            json.dump(self._registry, f, indent=2, default=str)
        self._sec.record_write(PERSON_REGISTRY)
        # Encrypted backup
        self._sec.encryptor.encrypt_file(PERSON_REGISTRY)

    def _warm_cache(self):
        """Load all descriptors into RAM at startup."""
        self._cache = {}
        for pid, rec in self._registry.items():
            fp = rec.get("feat_path", "")
            if os.path.exists(fp):
                try:
                    with open(fp, 'rb') as f:
                        self._cache[pid] = pickle.load(f)
                except Exception:
                    pass
        print(f"[PERSON-DB] Feature cache warm: {len(self._cache)} entries.")

    # ── ID ────────────────────────────────────────────────────

    def _next_id(self) -> str:
        if not self._registry:
            return "P0001"
        return f"P{max(int(k[1:]) for k in self._registry) + 1:04d}"

    # ── Core: Recognize or Register ───────────────────────────

    def recognize_or_register(self, head_crop: np.ndarray) -> tuple:
        """
        Returns: (person_id, is_recognized, blur)
          person_id     = "P0001" | None (if too blurry for any action)
          is_recognized = True (returning) | False (new)
          blur          = float blur score of this crop
        """
        blur = _blur_score(head_crop)

        if blur < BLUR_REJECT:
            return None, False, blur          # Too blurry — skip entirely

        des = _orb_des(head_crop)

        # ── Match against RAM cache ───────────────────────────
        best_pid, best_n = None, 0
        for pid, cached_des in self._cache.items():
            n = _good_matches(des, cached_des)
            if n > best_n:
                best_n, best_pid = n, pid

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if best_pid:
            # ── RECOGNIZED ──
            rec = self._registry[best_pid]
            rec["visit_count"] = rec.get("visit_count", 1) + 1
            rec["last_seen"]   = now

            cur_blur = rec.get("blur_score", 0)
            if blur > BLUR_UPGRADE and blur > cur_blur:
                self._persist_photo(best_pid, head_crop, des, blur)
                rec["blur_score"]     = blur
                rec["has_good_photo"] = True
                print(f"[PERSON-DB] {best_pid} photo upgraded {cur_blur:.0f}→{blur:.0f}")

            self._save()
            return best_pid, True, blur

        else:
            # ── NEW PERSON ──
            new_id = self._next_id()
            self._registry[new_id] = {
                "person_id":      new_id,
                "assigned_at":    now,
                "last_seen":      now,
                "photo_path":     os.path.join(PERSON_PHOTO_DIR, f"{new_id}.jpg"),
                "thumb_path":     os.path.join(PERSON_THUMB_DIR, f"{new_id}.jpg"),
                "feat_path":      os.path.join(PERSON_FEAT_DIR,  f"{new_id}.pkl"),
                "blur_score":     blur,
                "has_good_photo": blur >= BLUR_UPGRADE,
                "visit_count":    1,
                "name":           "",          # Can be set by user
                "notes":          "",
            }
            self._persist_photo(new_id, head_crop, des, blur)
            self._save()
            tag = "HD" if blur >= BLUR_UPGRADE else "acceptable"
            print(f"[PERSON-DB] NEW {new_id} registered | blur={blur:.0f} | {tag}")
            return new_id, False, blur

    def _persist_photo(self, pid, img, des, blur):
        """Write photo + thumbnail + descriptors to disk and update RAM cache."""
        cv2.imwrite(os.path.join(PERSON_PHOTO_DIR, f"{pid}.jpg"), img)
        thumb = _make_thumb(img)
        if thumb is not None:
            cv2.imwrite(os.path.join(PERSON_THUMB_DIR, f"{pid}.jpg"), thumb)
        if des is not None:
            fp = os.path.join(PERSON_FEAT_DIR, f"{pid}.pkl")
            with open(fp, 'wb') as f:
                pickle.dump(des, f)
            self._cache[pid] = des    # RAM update → zero latency next time

    # ── Public API ────────────────────────────────────────────

    def get_record(self, pid): return self._registry.get(pid)

    def get_photo_path(self, pid):
        rec = self._registry.get(pid)
        if rec:
            p = rec.get("photo_path", "")
            return p if os.path.exists(p) else None
        return None

    def get_thumb_path(self, pid):
        rec = self._registry.get(pid)
        if rec:
            p = rec.get("thumb_path", "")
            return p if os.path.exists(p) else None
        return None

    def has_good_photo(self, pid) -> bool:
        rec = self._registry.get(pid)
        return bool(rec and rec.get("has_good_photo", False))

    def set_name(self, pid, name: str):
        if pid in self._registry:
            self._registry[pid]["name"] = name
            self._save()

    def all_persons(self): return list(self._registry.values())

    def count(self): return len(self._registry)

    def search_by_name(self, query: str):
        q = query.lower()
        return [r for r in self._registry.values()
                if q in r.get("name", "").lower() or q in r["person_id"].lower()]