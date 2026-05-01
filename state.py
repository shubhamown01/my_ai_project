import threading

class SharedState:
    def __init__(self):
        self.lock                = threading.Lock()
        self.latest_frame        = None
        self.active_detections   = []
        self.unique_objects      = {}
        self.person_id_map       = {}   # tracker_id → P0001
        self.object_id_map       = {}   # tracker_id → Object_N or known label
        self.object_counter      = 0
        self.running             = True
        self.active_detection    = None

STATE = SharedState()