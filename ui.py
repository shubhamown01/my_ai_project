"""
ui.py — Smart Vision AI  |  Professional Dashboard v2
=======================================================
Design: Cinematic dark ops — deep navy/charcoal base, electric cyan accents,
        amber activity highlights, crimson alerts.

Features:
  - Video source switcher: Live Camera ↔ Training Video Preview
  - Auto-reconnect camera (retries every 3 sec when offline)
  - Detection list hidden by default; shown only on button click
  - Training panel embedded in main dashboard (not popup)
  - YouTube/local video preview during training (object detection overlay)
  - Professional stat counters (total persons, objects, activities)
  - Glassmorphism-style panels with subtle borders
"""

import customtkinter as ctk
import threading, sys, os, time, json
import cv2
from PIL import Image
from database_persons  import PersonDatabase
from database_objects  import ObjectDatabase
from database_activity import ActivityPatternDB
import config
from camera import camera_thread

# ══ Color Palette ══════════════════════════════════════════════
C = {
    "bg":          "#0B0F1A",   # Deep space navy
    "panel":       "#111827",   # Card background
    "panel2":      "#1A2235",   # Slightly lighter card
    "border":      "#1E3A5F",   # Blue-tinted border
    "accent":      "#00D4FF",   # Electric cyan
    "accent2":     "#7C3AED",   # Purple
    "green":       "#00FF88",   # Neon green
    "amber":       "#FFB300",   # Amber/gold
    "red":         "#FF3B5C",   # Crimson
    "text":        "#E2E8F0",   # Primary text
    "text2":       "#64748B",   # Muted text
    "person_new":  "#1E3A5F",
    "person_rec":  "#0F2E1A",
    "obj_known":   "#0F2A1A",
    "obj_unk":     "#1A1A2E",
}

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_HEAD   = ("Courier New", 13, "bold")
FONT_BODY   = ("Segoe UI",    12)
FONT_MONO   = ("Consolas",    11)
FONT_BADGE  = ("Segoe UI",    9)
FONT_STAT   = ("Courier New", 18, "bold")


# ══ Helpers ════════════════════════════════════════════════════

def _badge(parent, text, fg, text_color="#FFFFFF", padx=8, pady=2):
    return ctk.CTkLabel(parent, text=text, font=FONT_BADGE,
                        fg_color=fg, text_color=text_color,
                        corner_radius=4, padx=padx, pady=pady)

def _divider(parent, color="#1E3A5F"):
    f = ctk.CTkFrame(parent, fg_color=color, height=1)
    f.pack(fill="x", padx=12, pady=4)
    return f


# ══ Main App ═══════════════════════════════════════════════════

class SmartVisionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("SMART VISION AI  |  Neural Surveillance Dashboard")
        self.root.geometry("1600x920")
        self.root.configure(fg_color=C["bg"])
        self.root.minsize(1200, 700)

        # ── Data layer ────────────────────────────────────────
        self.person_db   = PersonDatabase()
        self.object_db   = ObjectDatabase()
        self.activity_db = ActivityPatternDB()
        self.state       = config.STATE

        self.state.active_detections = []
        self.state.unique_objects    = {}
        self.state.latest_frame      = None
        self.state.training_frame    = None   # NEW: training video frame
        self.state.person_id_map     = {}
        self.state.object_id_map     = {}
        self.state.object_counter    = 0

        # ── UI state ──────────────────────────────────────────
        self.rendered_uids   = set()
        self.all_detections  = []          # Background list (never cleared)
        self.show_det_panel  = False       # Detection list shown only on click
        self.preview_mode    = "camera"    # "camera" | "training"
        self.cam_connected   = False
        self.training_sess   = None        # Active TrainingSession
        self._det_window     = None        # Detection list toplevel

        # Stats counters
        self._stat_persons  = ctk.IntVar(value=0)
        self._stat_objects  = ctk.IntVar(value=0)
        self._stat_activity = ctk.IntVar(value=0)

        self._build_layout()
        self._start_camera_thread()
        self._tick_feed()
        self._tick_dashboard()
        self._tick_camera_watchdog()

    # ══ Layout ═════════════════════════════════════════════════

    def _build_layout(self):
        self.root.grid_columnconfigure(0, weight=4)   # Video area
        self.root.grid_columnconfigure(1, weight=3)   # Right panel
        self.root.grid_rowconfigure(1, weight=1)

        self._build_topbar()
        self._build_video_panel()
        self._build_right_panel()
        self._build_statusbar()

    # ── Top Bar ───────────────────────────────────────────────

    def _build_topbar(self):
        bar = ctk.CTkFrame(self.root, fg_color=C["panel"],
                           corner_radius=0, height=56,
                           border_width=0)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        # Logo
        ctk.CTkLabel(bar,
            text="◈  SMART VISION AI",
            font=("Courier New", 20, "bold"),
            text_color=C["accent"]
        ).grid(row=0, column=0, padx=20, pady=14)

        # Center title
        ctk.CTkLabel(bar,
            text="NEURAL SURVEILLANCE DASHBOARD",
            font=("Courier New", 12),
            text_color=C["text2"]
        ).grid(row=0, column=1, pady=14)

        # Right: Stats pills
        stats_fr = ctk.CTkFrame(bar, fg_color="transparent")
        stats_fr.grid(row=0, column=2, padx=20, pady=8)

        for icon, var, color, label in [
            ("👤", self._stat_persons,  C["accent"],  "PERSONS"),
            ("📦", self._stat_objects,  C["amber"],   "OBJECTS"),
            ("⚡", self._stat_activity, C["green"],   "ACTIVITIES"),
        ]:
            pill = ctk.CTkFrame(stats_fr, fg_color=C["panel2"],
                                corner_radius=20, border_width=1,
                                border_color=C["border"])
            pill.pack(side="left", padx=4)
            ctk.CTkLabel(pill, text=icon, font=("Segoe UI",13),
                         width=24).pack(side="left", padx=(8,2), pady=4)
            num_lbl = ctk.CTkLabel(pill, textvariable=var,
                font=("Courier New",14,"bold"), text_color=color, width=36)
            num_lbl.pack(side="left")
            ctk.CTkLabel(pill, text=label, font=("Segoe UI",8),
                         text_color=C["text2"]).pack(side="left", padx=(2,10), pady=4)

        # Clock
        self._clock_lbl = ctk.CTkLabel(bar,
            text="00:00:00",
            font=("Courier New", 14, "bold"),
            text_color=C["amber"]
        )
        self._clock_lbl.grid(row=0, column=3, padx=20)
        self._tick_clock()

    def _tick_clock(self):
        self._clock_lbl.configure(text=time.strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    # ── Video Panel (Left) ────────────────────────────────────

    def _build_video_panel(self):
        left = ctk.CTkFrame(self.root, fg_color=C["bg"], corner_radius=0)
        left.grid(row=1, column=0, sticky="nsew", padx=(12,6), pady=(8,0))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # ── Source switcher bar ───────────────────────────────
        src_bar = ctk.CTkFrame(left, fg_color=C["panel"],
                               corner_radius=10, border_width=1,
                               border_color=C["border"])
        src_bar.grid(row=0, column=0, sticky="ew", pady=(0,6))
        src_bar.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(src_bar, text="VIDEO SOURCE",
            font=("Courier New",10), text_color=C["text2"]
        ).grid(row=0, column=0, padx=12, pady=8)

        self._src_var = ctk.StringVar(value="camera")

        self._btn_cam = ctk.CTkButton(src_bar,
            text="⬡  LIVE CAMERA", width=160, height=32,
            font=("Courier New",11,"bold"),
            fg_color=C["accent"], text_color="#000000",
            hover_color="#00AACC", corner_radius=6,
            command=lambda: self._switch_source("camera")
        )
        self._btn_cam.grid(row=0, column=1, padx=(4,2), pady=6)

        self._btn_train = ctk.CTkButton(src_bar,
            text="◈  TRAINING PREVIEW", width=180, height=32,
            font=("Courier New",11,"bold"),
            fg_color=C["panel2"], text_color=C["text2"],
            hover_color=C["border"], corner_radius=6,
            command=lambda: self._switch_source("training")
        )
        self._btn_train.grid(row=0, column=2, padx=(2,4), pady=6)

        # Camera status indicator
        self._cam_status = ctk.CTkLabel(src_bar,
            text="⬤  OFFLINE", font=("Courier New",10,"bold"),
            text_color=C["red"]
        )
        self._cam_status.grid(row=0, column=3, padx=16)

        # ── Main video canvas ─────────────────────────────────
        vid_wrap = ctk.CTkFrame(left, fg_color=C["panel"],
                                corner_radius=12, border_width=1,
                                border_color=C["border"])
        vid_wrap.grid(row=1, column=0, sticky="nsew")
        vid_wrap.grid_rowconfigure(0, weight=1)
        vid_wrap.grid_columnconfigure(0, weight=1)

        # Offline placeholder
        self._offline_lbl = ctk.CTkLabel(vid_wrap,
            text="◈\n\nCAMERA OFFLINE\nAuto-reconnecting...",
            font=("Courier New", 16, "bold"),
            text_color=C["text2"]
        )
        self._offline_lbl.grid(row=0, column=0, sticky="nsew")

        self._video_lbl = ctk.CTkLabel(vid_wrap, text="")
        self._video_lbl.grid(row=0, column=0, sticky="nsew")

        # ── Activity Feed (below video) ───────────────────────
        act_wrap = ctk.CTkFrame(left, fg_color=C["panel"],
                                corner_radius=10, border_width=1,
                                border_color=C["border"], height=120)
        act_wrap.grid(row=2, column=0, sticky="ew", pady=(6,8))
        act_wrap.grid_propagate(False)
        act_wrap.grid_columnconfigure(0, weight=1)

        act_head = ctk.CTkFrame(act_wrap, fg_color="transparent")
        act_head.grid(row=0, column=0, sticky="ew", padx=10, pady=(6,0))
        ctk.CTkLabel(act_head, text="⚡  LIVE ACTIVITY FEED",
            font=("Courier New",11,"bold"), text_color=C["amber"]
        ).pack(side="left")
        ctk.CTkButton(act_head,
            text="≡  DETECTION LIST", width=150, height=24,
            font=("Courier New",10,"bold"),
            fg_color=C["panel2"], text_color=C["accent"],
            hover_color=C["border"], corner_radius=4,
            command=self._toggle_detection_list
        ).pack(side="right")

        self._act_box = ctk.CTkTextbox(act_wrap,
            font=("Consolas",11), fg_color="transparent",
            text_color=C["amber"], border_width=0, height=72)
        self._act_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(0,4))

    def _switch_source(self, mode):
        self.preview_mode = mode
        if mode == "camera":
            self._btn_cam.configure(
                fg_color=C["accent"], text_color="#000000")
            self._btn_train.configure(
                fg_color=C["panel2"], text_color=C["text2"])
        else:
            self._btn_train.configure(
                fg_color=C["accent2"], text_color="#FFFFFF")
            self._btn_cam.configure(
                fg_color=C["panel2"], text_color=C["text2"])

    # ── Right Panel ───────────────────────────────────────────

    def _build_right_panel(self):
        right = ctk.CTkFrame(self.root, fg_color=C["bg"], corner_radius=0)
        right.grid(row=1, column=1, sticky="nsew", padx=(6,12), pady=(8,0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # ── Tab navigation ────────────────────────────────────
        nav = ctk.CTkFrame(right, fg_color=C["panel"],
                           corner_radius=10, border_width=1,
                           border_color=C["border"])
        nav.grid(row=0, column=0, sticky="ew", pady=(0,6))

        self._active_tab = ctk.StringVar(value="persons")
        self._tab_frames = {}
        self._tab_btns   = {}

        tabs = [
            ("persons",  "👤  PERSONS",  C["accent"]),
            ("objects",  "📦  OBJECTS",  C["amber"]),
            ("training", "◈  TRAINING",  C["accent2"]),
            ("security", "🔒  SECURITY", C["red"]),
        ]
        for i, (key, label, color) in enumerate(tabs):
            btn = ctk.CTkButton(nav,
                text=label, width=1, height=34,
                font=("Courier New",10,"bold"),
                fg_color=color if key=="persons" else C["panel2"],
                text_color="#000000" if key=="persons" else C["text2"],
                hover_color=color, corner_radius=6,
                command=lambda k=key,c=color: self._switch_tab(k,c)
            )
            btn.grid(row=0, column=i, sticky="ew", padx=3, pady=5)
            nav.grid_columnconfigure(i, weight=1)
            self._tab_btns[key] = (btn, color)

        # ── Content area ──────────────────────────────────────
        content = ctk.CTkFrame(right, fg_color=C["panel"],
                               corner_radius=12, border_width=1,
                               border_color=C["border"])
        content.grid(row=1, column=0, sticky="nsew", pady=(0,8))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        # Persons tab
        pf = ctk.CTkFrame(content, fg_color="transparent")
        pf.grid(row=0, column=0, sticky="nsew")
        pf.grid_rowconfigure(1, weight=1)
        pf.grid_columnconfigure(0, weight=1)
        self._person_scroll = ctk.CTkScrollableFrame(pf, fg_color="transparent",
                                                     label_text="")
        self._person_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._tab_frames["persons"] = pf

        # Objects tab
        of = ctk.CTkFrame(content, fg_color="transparent")
        of.grid(row=0, column=0, sticky="nsew")
        of.grid_rowconfigure(1, weight=1)
        of.grid_columnconfigure(0, weight=1)
        self._object_scroll = ctk.CTkScrollableFrame(of, fg_color="transparent",
                                                     label_text="")
        self._object_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._tab_frames["objects"] = of

        # Training tab
        tf = ctk.CTkFrame(content, fg_color="transparent")
        tf.grid(row=0, column=0, sticky="nsew")
        self._tab_frames["training"] = tf
        self._build_training_tab(tf)

        # Security tab
        sf = ctk.CTkFrame(content, fg_color="transparent")
        sf.grid(row=0, column=0, sticky="nsew")
        self._tab_frames["security"] = sf
        self._build_security_tab(sf)

        self._switch_tab("persons", C["accent"])

    def _switch_tab(self, key, color):
        # Hide all
        for k, fr in self._tab_frames.items():
            fr.grid_remove()
        # Show selected
        self._tab_frames[key].grid()
        # Update buttons
        for k, (btn, c) in self._tab_btns.items():
            if k == key:
                btn.configure(fg_color=c, text_color="#000000")
            else:
                btn.configure(fg_color=C["panel2"], text_color=C["text2"])
        self._active_tab.set(key)

    # ── Training Tab (embedded) ───────────────────────────────

    def _build_training_tab(self, parent):
        parent.grid_rowconfigure(4, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(parent, text="◈  TRAINING PIPELINE",
            font=FONT_HEAD, text_color=C["accent2"]
        ).grid(row=0, column=0, padx=14, pady=(14,4), sticky="w")

        # Source input
        src_fr = ctk.CTkFrame(parent, fg_color=C["panel2"],
                              corner_radius=8, border_width=1,
                              border_color=C["border"])
        src_fr.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        src_fr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(src_fr, text="YouTube URL",
            font=FONT_BADGE, text_color=C["text2"]
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8,0))
        self._yt_var = ctk.StringVar()
        ctk.CTkEntry(src_fr, textvariable=self._yt_var,
            placeholder_text="https://www.youtube.com/watch?v=...",
            font=FONT_MONO, height=34, corner_radius=6,
            fg_color=C["panel"], border_color=C["border"]
        ).grid(row=1, column=0, sticky="ew", padx=10, pady=(2,4))

        ctk.CTkLabel(src_fr, text="OR  Local video file",
            font=FONT_BADGE, text_color=C["text2"]
        ).grid(row=2, column=0, sticky="w", padx=10, pady=(4,0))
        frow = ctk.CTkFrame(src_fr, fg_color="transparent")
        frow.grid(row=3, column=0, sticky="ew", padx=10, pady=(2,8))
        frow.grid_columnconfigure(0, weight=1)
        self._file_var = ctk.StringVar()
        ctk.CTkEntry(frow, textvariable=self._file_var,
            placeholder_text="C:/path/to/video.mp4",
            font=FONT_MONO, height=34, corner_radius=6,
            fg_color=C["panel"], border_color=C["border"]
        ).grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(frow, text="Browse", width=80, height=34,
            font=FONT_BODY, fg_color=C["border"],
            hover_color=C["accent"], command=self._browse_file
        ).grid(row=0, column=1, padx=(6,0))

        # Options + start
        opt_fr = ctk.CTkFrame(parent, fg_color="transparent")
        opt_fr.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        self._ft_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(opt_fr, text="Fine-tune YOLOv8  (slower, better accuracy)",
            variable=self._ft_var,
            font=FONT_BODY, text_color=C["text"],
            fg_color=C["accent2"], checkmark_color="#FFFFFF"
        ).pack(side="left")

        ctk.CTkButton(parent,
            text="▶  START TRAINING", height=42,
            font=("Courier New",13,"bold"),
            fg_color=C["accent2"], text_color="#FFFFFF",
            hover_color="#5B21B6", corner_radius=8,
            command=self._start_training
        ).grid(row=3, column=0, sticky="ew", padx=10, pady=6)

        # Progress
        prog_fr = ctk.CTkFrame(parent, fg_color=C["panel2"],
                               corner_radius=8, border_width=1,
                               border_color=C["border"])
        prog_fr.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0,10))
        prog_fr.grid_rowconfigure(1, weight=1)
        prog_fr.grid_columnconfigure(0, weight=1)

        self._prog_lbl = ctk.CTkLabel(prog_fr,
            text="Ready to train.", font=FONT_MONO,
            text_color=C["text2"], anchor="w"
        )
        self._prog_lbl.grid(row=0, column=0, sticky="w", padx=10, pady=(8,4))

        self._prog_bar = ctk.CTkProgressBar(prog_fr,
            fg_color=C["panel"], progress_color=C["accent2"],
            height=8, corner_radius=4
        )
        self._prog_bar.set(0)
        self._prog_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,8))

        self._train_log = ctk.CTkTextbox(prog_fr,
            font=("Consolas",10), fg_color="transparent",
            text_color=C["text2"], border_width=0
        )
        self._train_log.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0,8))
        prog_fr.grid_rowconfigure(2, weight=1)

    def _browse_file(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if p:
            self._file_var.set(p)

    def _start_training(self):
        from training_pipeline import TrainingSession
        url  = self._yt_var.get().strip()
        path = self._file_var.get().strip()
        if not url and not path:
            self._prog_lbl.configure(text="⚠  Provide YouTube URL or file path.")
            return

        src  = url if url else path
        kind = "youtube" if url else "file"
        self.training_sess = TrainingSession(src, kind)

        # Pass training frame callback so video preview works
        self.training_sess._frame_cb = self._on_training_frame

        def _log(msg, pct):
            self._prog_lbl.configure(text=msg[:80])
            if pct >= 0:
                self._prog_bar.set(pct / 100)
            self._train_log.insert("end", f"{msg}\n")
            self._train_log.see("end")

        self.training_sess.start(
            model_path=config.MODEL_PATH,
            fine_tune=self._ft_var.get()
        )

        # Auto-switch to training preview
        self._switch_source("training")
        self._switch_tab("training", C["accent2"])

        def _poll():
            s = self.training_sess.get_summary()
            if s["log"]:
                _log(s["log"][-1]["msg"], s["progress"])
            if not self.training_sess.is_done():
                self.root.after(800, _poll)
            else:
                _log(f"✅  Done!  Status: {s['status']}", 100)
        _poll()

    def _on_training_frame(self, frame):
        """Called from training thread with detected frame."""
        self.state.training_frame = frame

    # ── Security Tab ─────────────────────────────────────────

    def _build_security_tab(self, parent):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(parent, text="🔒  SECURITY & AUDIT",
            font=FONT_HEAD, text_color=C["red"]
        ).grid(row=0, column=0, padx=14, pady=(14,4), sticky="w")

        self._audit_box = ctk.CTkTextbox(parent,
            font=("Consolas",10), fg_color=C["panel2"],
            text_color="#FF8A80", border_color=C["border"],
            border_width=1, corner_radius=8
        )
        self._audit_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)

        ctk.CTkButton(parent, text="🔄  REFRESH AUDIT LOG", height=36,
            font=("Courier New",11,"bold"),
            fg_color=C["red"], hover_color="#B71C1C", corner_radius=6,
            command=self._refresh_audit
        ).grid(row=2, column=0, sticky="ew", padx=10, pady=(0,10))

    def _refresh_audit(self):
        try:
            from security import get_security
            lines = get_security().get_audit_tail(80)
            self._audit_box.delete("1.0","end")
            self._audit_box.insert("end", "\n".join(lines))
        except Exception as e:
            self._audit_box.insert("end", f"Error: {e}\n")

    # ── Status Bar ────────────────────────────────────────────

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self.root, fg_color=C["panel"],
                           corner_radius=0, height=28,
                           border_width=0)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        self._status_lbl = ctk.CTkLabel(bar,
            text="●  System initializing...",
            font=("Courier New",10), text_color=C["text2"]
        )
        self._status_lbl.grid(row=0, column=0, padx=14, pady=4)

        self._fps_lbl = ctk.CTkLabel(bar,
            text="FPS: --", font=("Courier New",10), text_color=C["text2"]
        )
        self._fps_lbl.grid(row=0, column=2, padx=14)
        self._last_frame_t = time.time()

    def _set_status(self, msg, color=None):
        self._status_lbl.configure(
            text=f"●  {msg}",
            text_color=color or C["text2"]
        )

    # ══ Camera Thread ══════════════════════════════════════════

    def _start_camera_thread(self):
        t = threading.Thread(
            target=camera_thread,
            args=(self.person_db, self.object_db, self.activity_db, self.state),
            daemon=True
        )
        t.start()

    def _tick_camera_watchdog(self):
        """Auto-reconnect: if no frame for 3s, restart camera thread."""
        now = time.time()
        if self.state.latest_frame is None:
            if not self.cam_connected:
                self._cam_status.configure(
                    text="⬤  OFFLINE", text_color=C["red"])
                self._offline_lbl.lift()
            # Restart camera thread if camera is offline
            if not hasattr(self, '_last_restart') or now - self._last_restart > 5:
                self._last_restart = now
                self._start_camera_thread()
                self._set_status("Reconnecting to camera...", C["amber"])
        else:
            if not self.cam_connected:
                self.cam_connected = True
                self._cam_status.configure(
                    text="⬤  LIVE", text_color=C["green"])
                self._video_lbl.lift()
                self._set_status("Camera connected", C["green"])
        self.root.after(3000, self._tick_camera_watchdog)

    # ══ Feed Update Loop ═══════════════════════════════════════

    def _tick_feed(self):
        """Update video frame — camera or training preview."""
        frame = None
        if self.preview_mode == "training":
            frame = self.state.training_frame
        else:
            frame = self.state.latest_frame

        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                # Compute display size maintaining aspect ratio
                w, h  = img.size
                max_w, max_h = 740, 480
                ratio = min(max_w/w, max_h/h)
                nw, nh = int(w*ratio), int(h*ratio)
                ci = ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
                self._video_lbl.configure(image=ci, text="")
                self._video_lbl.image = ci
                # FPS
                now = time.time()
                fps = 1.0 / max(now - self._last_frame_t, 0.001)
                self._fps_lbl.configure(text=f"FPS: {fps:.0f}")
                self._last_frame_t = now
            except Exception:
                pass

        self.root.after(28, self._tick_feed)   # ~35fps target

    # ══ Dashboard Update ═══════════════════════════════════════

    def _tick_dashboard(self):
        with self.state.lock:
            detections = list(self.state.active_detections)
            self.state.active_detections = []

        ts = time.strftime("%H:%M:%S")

        if detections:
            for item in detections:
                icon = self._det_icon(item)
                entry = f"[{ts}]  {icon}  {item}"
                self.all_detections.append(entry)

            # Keep max 2000 entries
            if len(self.all_detections) > 2000:
                self.all_detections = self.all_detections[-2000:]

            # Update detection window if open
            if self._det_window and self._det_window.winfo_exists():
                self._refresh_det_window()

        # Activity feed
        act_lines = []
        for uid, info in list(self.state.unique_objects.items()):
            act = info.get("activity","")
            if act and act not in ("","unknown"):
                icon = "👤" if info.get("type") == "person" else "📦"
                act_lines.append(f"{icon}  {uid}: {act}")
        if act_lines:
            self._act_box.delete("1.0","end")
            self._act_box.insert("end", "\n".join(act_lines[-8:]))

        # Right panel rows
        for uid, info in list(self.state.unique_objects.items()):
            if uid not in self.rendered_uids:
                if info.get("type") == "person":
                    self._add_person_card(info)
                else:
                    self._add_object_card(info)
                self.rendered_uids.add(uid)

        # Update stats
        persons = sum(1 for v in self.state.unique_objects.values()
                      if v.get("type") == "person")
        objects = sum(1 for v in self.state.unique_objects.values()
                      if v.get("type") == "object")
        acts    = sum(1 for v in self.state.unique_objects.values()
                      if v.get("activity","") not in ("","unknown"))
        self._stat_persons.set(persons)
        self._stat_objects.set(objects)
        self._stat_activity.set(acts)

        self.root.after(350, self._tick_dashboard)

    def _det_icon(self, item):
        if item.startswith("P") and len(item) == 5:
            rec = self.state.unique_objects.get(item,{}).get("is_recognized",False)
            return "🔄" if rec else "👤"
        if item.startswith("T_"):   return "🔍"
        if item.startswith("Object_"): return "❓"
        return "✅"

    # ══ Detection List Window ══════════════════════════════════

    def _toggle_detection_list(self):
        if self._det_window and self._det_window.winfo_exists():
            self._det_window.destroy()
            self._det_window = None
            return

        win = ctk.CTkToplevel(self.root)
        win.title("Detection Log")
        win.geometry("520x600")
        win.configure(fg_color=C["bg"])
        win.attributes("-topmost", True)
        self._det_window = win

        # Header
        hdr = ctk.CTkFrame(win, fg_color=C["panel"], corner_radius=0, height=44)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr,
            text=f"◈  DETECTION HISTORY  ({len(self.all_detections)} entries)",
            font=("Courier New",12,"bold"), text_color=C["accent"]
        ).pack(side="left", padx=14, pady=10)

        btn_fr = ctk.CTkFrame(hdr, fg_color="transparent")
        btn_fr.pack(side="right", padx=8, pady=6)
        ctk.CTkButton(btn_fr, text="Clear", width=60, height=28,
            fg_color=C["red"], font=FONT_BADGE,
            command=lambda: [self.all_detections.clear(),
                             self._refresh_det_window()]
        ).pack(side="left", padx=2)
        ctk.CTkButton(btn_fr, text="Export", width=70, height=28,
            fg_color=C["border"], font=FONT_BADGE,
            command=self._export_detections
        ).pack(side="left", padx=2)

        # Filter bar
        flt_fr = ctk.CTkFrame(win, fg_color=C["panel2"], height=36)
        flt_fr.pack(fill="x", padx=8, pady=4)
        flt_fr.pack_propagate(False)
        ctk.CTkLabel(flt_fr, text="Filter:",
            font=FONT_BADGE, text_color=C["text2"]
        ).pack(side="left", padx=8, pady=6)
        self._filter_var = ctk.StringVar()
        ctk.CTkEntry(flt_fr, textvariable=self._filter_var,
            width=200, height=26, font=FONT_MONO,
            fg_color=C["panel"], border_color=C["border"],
            placeholder_text="P0001 / Object / person..."
        ).pack(side="left", padx=4)
        ctk.CTkButton(flt_fr, text="Apply", width=60, height=26,
            fg_color=C["accent"], text_color="#000000",
            font=FONT_BADGE, command=self._refresh_det_window
        ).pack(side="left", padx=4)

        # Log box
        self._det_box = ctk.CTkTextbox(win,
            font=("Consolas",11), fg_color=C["panel"],
            text_color=C["text"], border_color=C["border"],
            border_width=1, corner_radius=8
        )
        self._det_box.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self._refresh_det_window()

    def _refresh_det_window(self):
        if not (self._det_window and self._det_window.winfo_exists()):
            return
        flt = self._filter_var.get().strip().lower() if hasattr(self, '_filter_var') else ""
        lines = self.all_detections if not flt else [
            l for l in self.all_detections if flt in l.lower()
        ]
        self._det_box.delete("1.0","end")
        self._det_box.insert("end", "\n".join(lines[-500:]))
        self._det_box.see("end")

    def _export_detections(self):
        from tkinter import filedialog
        p = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text","*.txt"),("All","*.*")],
            initialfile=f"detections_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if p:
            with open(p,"w",encoding="utf-8") as f:
                f.write("\n".join(self.all_detections))
            self._set_status(f"Exported: {os.path.basename(p)}", C["green"])

    # ══ Person Card ════════════════════════════════════════════

    def _add_person_card(self, info):
        uid      = info["uid"]
        is_recog = info.get("is_recognized", False)
        has_good = info.get("has_good_photo", False)
        visits   = info.get("visit_count", 1)
        activity = info.get("activity", "")

        bg = C["person_rec"] if is_recog else C["person_new"]

        card = ctk.CTkFrame(self._person_scroll,
                            fg_color=bg, corner_radius=10,
                            border_width=1,
                            border_color="#00695C" if is_recog else C["border"])
        card.pack(fill="x", padx=6, pady=4)
        card.grid_columnconfigure(1, weight=1)

        # Thumbnail
        img_path = info.get("image_path","")
        if img_path and os.path.exists(img_path):
            try:
                im = Image.open(img_path)
                ci = ctk.CTkImage(light_image=im, dark_image=im, size=(52,52))
                ctk.CTkLabel(card, image=ci, text="",
                             corner_radius=6).grid(row=0,column=0,rowspan=3,
                             padx=(10,8),pady=10,sticky="n")
            except Exception:
                ctk.CTkLabel(card, text="👤", font=("Segoe UI",26)
                             ).grid(row=0,column=0,rowspan=3,padx=(10,8),pady=10)
        else:
            ctk.CTkLabel(card, text="👤", font=("Segoe UI",26)
                         ).grid(row=0,column=0,rowspan=3,padx=(10,8),pady=10)

        # ID + badges
        id_fr = ctk.CTkFrame(card, fg_color="transparent")
        id_fr.grid(row=0, column=1, sticky="ew", pady=(10,2))
        ctk.CTkLabel(id_fr, text=uid,
            font=("Courier New",14,"bold"),
            text_color=C["accent"] if is_recog else C["text"]
        ).pack(side="left")

        b_fr = ctk.CTkFrame(card, fg_color="transparent")
        b_fr.grid(row=1, column=1, sticky="w")

        rec_txt   = f"🔄 KNOWN  ×{visits}" if is_recog else "👤 NEW"
        rec_color = "#00695C" if is_recog else "#1565C0"
        _badge(b_fr, rec_txt, rec_color).pack(side="left", padx=(0,4))

        photo_txt   = "📸 HD" if has_good else "🔍 Capturing"
        photo_color = "#2E7D32" if has_good else "#5D4037"
        _badge(b_fr, photo_txt, photo_color).pack(side="left", padx=(0,4))

        # Activity
        if activity:
            ctk.CTkLabel(card,
                text=f"⚡  {activity}",
                font=("Consolas",10),
                text_color=C["amber"], anchor="w"
            ).grid(row=2, column=1, sticky="w", pady=(0,8))

    # ══ Object Card ════════════════════════════════════════════

    def _add_object_card(self, info):
        uid      = info["uid"]
        is_known = info.get("known", False)
        use_pat  = info.get("use_pattern","")
        motion   = info.get("motion", {})

        bg = C["obj_known"] if is_known else C["obj_unk"]

        card = ctk.CTkFrame(self._object_scroll,
                            fg_color=bg, corner_radius=10,
                            border_width=1,
                            border_color="#2E7D32" if is_known else C["border"])
        card.pack(fill="x", padx=6, pady=4)
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card,
            text="📦" if is_known else "❓",
            font=("Segoe UI",24)
        ).grid(row=0, column=0, rowspan=2, padx=(10,8), pady=8, sticky="n")

        # Name + badges
        ctk.CTkLabel(card, text=uid,
            font=("Courier New",13,"bold"),
            text_color=C["green"] if is_known else C["text"]
        ).grid(row=0, column=1, sticky="w", pady=(8,2))

        b_fr = ctk.CTkFrame(card, fg_color="transparent")
        b_fr.grid(row=1, column=1, sticky="w", pady=(0,8))

        _badge(b_fr, "✅ KNOWN" if is_known else "❓ UNKNOWN",
               "#2E7D32" if is_known else "#424242").pack(side="left",padx=(0,4))

        if use_pat:
            _badge(b_fr, f"🔧 {use_pat}", "#1565C0").pack(side="left", padx=(0,4))

        if motion.get("state") == "moving":
            mv = f"→{motion.get('direction','')} {motion.get('speed','')}"
            _badge(b_fr, mv, "#7C2D12", text_color=C["amber"]
                   ).pack(side="left", padx=(0,4))

        # Edit buttons
        btn_fr = ctk.CTkFrame(card, fg_color="transparent")
        btn_fr.grid(row=0, column=2, rowspan=2, padx=8)
        ctk.CTkButton(btn_fr, text="✏", width=30, height=28,
            fg_color=C["border"], hover_color=C["accent"],
            font=FONT_BODY, command=lambda u=uid,c=card: self._edit_object(u,c)
        ).pack(pady=2)
        ctk.CTkButton(btn_fr, text="✔", width=30, height=28,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=FONT_BODY, command=lambda c=card: c.configure(border_color="#2E7D32")
        ).pack(pady=2)

    def _edit_object(self, uid, card):
        form = ctk.CTkToplevel(self.root)
        form.title(f"Rename — {uid}")
        form.geometry("400x180")
        form.configure(fg_color=C["bg"])
        form.attributes("-topmost", True)
        ctk.CTkLabel(form, text=f"Rename  {uid}",
            font=FONT_HEAD, text_color=C["accent"]
        ).pack(pady=(16,8))
        nv = ctk.StringVar(value=uid)
        ctk.CTkEntry(form, textvariable=nv, width=340, height=36,
            font=FONT_MONO, fg_color=C["panel"],
            border_color=C["border"]
        ).pack(pady=4)
        def _save():
            if uid in self.state.unique_objects:
                self.state.unique_objects[uid]["display_name"] = nv.get().strip()
            card.configure(border_color=C["green"])
            form.destroy()
        ctk.CTkButton(form, text="💾  SAVE",
            height=36, fg_color=C["green"], text_color="#000000",
            font=("Courier New",12,"bold"), command=_save
        ).pack(pady=10)
        form.bind("<Return>", lambda e: _save())

    # ══ Close ══════════════════════════════════════════════════

    def on_closing(self):
        self.state.running = False
        self.root.destroy()
        sys.exit(0)