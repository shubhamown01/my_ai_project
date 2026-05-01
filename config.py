import os
from state import SharedState

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = "yolov8n.pt"
CONFIDENCE   = 0.45
CAMERA_SOURCE = "http://192.0.0.4:8080/video"
# CAMERA_SOURCE = 0  # internal webcam

PERSON_COLOR            = (0, 200, 255)
PERSON_RECOGNIZED_COLOR = (0, 255, 180)
KNOWN_COLOR             = (0, 255, 80)
UNKNOWN_COLOR           = (0, 80, 255)

FACE_SCALE_FACTOR  = 1.08
FACE_MIN_NEIGHBORS = 3
FACE_MIN_SIZE      = (18, 18)

STATE = SharedState()
print(f"[CONFIG] Ready. Camera: {CAMERA_SOURCE}")