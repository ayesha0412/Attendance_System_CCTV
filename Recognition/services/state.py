"""
Shared Application State
========================
Thread-safe globals accessed by the inference thread (writer)
and Flask routes (readers).

Why module-level state?
  The inference loop runs in a background thread, not inside a Flask request.
  Flask's request context doesn't apply.  Module-level state with explicit
  locks is the standard pattern for this architecture — same as gunicorn
  pre-fork workers sharing via multiprocessing, but simpler since we're
  single-process + multi-threaded.
"""

import threading

# ── Frame buffer ─────────────────────────────────────────────────────
# Written by inference thread, read by MJPEG generator
frame_lock      = threading.Lock()
annotated_frame = None

# ── Dashboard stats ──────────────────────────────────────────────────
# Written by inference thread, read by /api/stats
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
# Set once at startup, read by health endpoint and dashboard
model_info = {
    "gallery":     None,
    "class_names": [],
    "model_name":  "",
}

# ── Per-person log cooldown ──────────────────────────────────────────
# {emp_no: last_log_unix_timestamp}
log_cooldown: dict = {}

# ── Folder name → employee lookup ────────────────────────────────────
# Loaded once at startup from the employees table.
# Keys are gallery folder names; values are {"emp_no": ..., "emp_name": ...}
folder_to_emp: dict = {}
