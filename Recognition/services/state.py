"""
Shared Application State
========================
Thread-safe globals accessed by the inference thread (writer)
and FastAPI routes (readers).

Why module-level state?
  The inference loop runs in a background thread, not inside a request
  context.  Module-level state with explicit locks is the standard pattern
  for this architecture (single-process, multi-threaded).
"""

import threading

# ── Annotated frame buffer ───────────────────────────────────────────
# Written by inference thread, read by MJPEG generator in dashboard router
frame_lock      = threading.Lock()
annotated_frame = None

# ── Dashboard stats / live counters ──────────────────────────────────
# Written by inference thread, read by /api/v1/health and /api/v1/stats
stats_lock = threading.Lock()
stats = {
    "recognized": 0,
    "unknown":    0,
    "total":      0,
    "fps":        0.0,
    "uptime":     0.0,
    "recent_detections": [],
}

# ── Model info ───────────────────────────────────────────────────────
# Set once at startup, read by health endpoint
model_info = {
    "gallery":     None,
    "class_names": [],
    "model_name":  "",
}

# ── Per-employee log cooldown ────────────────────────────────────────
# {emp_no: last_log_unix_timestamp} — prevents duplicate attendance writes
log_cooldown: dict = {}

# ── Folder name → employee lookup ────────────────────────────────────
# Loaded once at startup from the employees table.
# Keys are gallery folder names; values are {"emp_no": ..., "emp_name": ...}
folder_to_emp: dict = {}

# ── Latest face crops per employee ──────────────────────────────────
# {emp_no: JPEG bytes} — updated every time a known face is seen
face_crops_lock = threading.Lock()
face_crops: dict = {}

# ── Currently visible faces ─────────────────────────────────────────
# {emp_no: {emp_name, confidence, last_seen (unix time)}}
# Updated every frame a known face is detected. Dashboard uses this
# to show cards only while the person is in front of the camera.
active_faces_lock = threading.Lock()
active_faces: dict = {}
