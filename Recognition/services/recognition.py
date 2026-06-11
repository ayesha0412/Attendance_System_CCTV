"""
Inference Loop
==============
The main face recognition pipeline.  Runs in a background thread,
processes camera frames, and updates shared state for the dashboard.

This module is backend-agnostic — it works with ArcFace, AdaFace, or
any other embedding model via the ``embed_fn`` and ``liveness_input_fn``
callables passed by the entry point.
"""

import time
import logging
import numpy as np
from collections import deque, Counter

from core.tracker import CentroidTracker
from core.drawing import draw_box
from core.liveness import check_liveness
from core.classify import cosine_classify
from core.enhance import enhance_clahe
from services import state
from services.attendance import mark_attendance

logger = logging.getLogger(__name__)


def inference_loop(camera, detector, gallery, config, embed_fn, liveness_input_fn):
    """
    Main recognition loop.  Runs until ``camera.running`` is False.

    Args:
        camera:             ThreadedCamera instance.
        detector:           InsightFace FaceAnalysis (detection only).
        gallery:            {name: L2-normalised mean embedding} dict.
        config:             Parsed config dict (from config.yaml + .env).
        embed_fn:           callable(frame, face) -> 512-dim numpy array or None.
        liveness_input_fn:  callable(frame, face) -> BGR image for liveness check.
    """
    rec_cfg = config["recognition"]
    det_cfg = config["detection"]
    trk_cfg = config["tracker"]
    pre_cfg = config.get("preprocessing", {})
    max_log = config["server"]["max_log_entries"]
    target_fps = config["camera"].get("target_fps", 30)
    min_frame_time = 1.0 / target_fps

    tracker = CentroidTracker(
        max_dist=trk_cfg["max_dist"],
        max_age=trk_cfg["max_age"],
    )

    vote_buffers: dict = {}       # {track_id: deque}
    start_time   = time.time()
    frame_times  = []

    logger.info("Inference loop started")

    while camera.running:
        t0 = time.time()
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        # ── Optional CLAHE enhancement ────────────────────────────────
        if pre_cfg.get("enhance_clahe", False):
            frame = enhance_clahe(frame)

        # ── Detect faces ──────────────────────────────────────────────
        faces = detector.get(frame)

        # Filter by score + size (min AND max)
        faces = [
            f for f in faces
            if f.det_score >= det_cfg["score_min"]
            and det_cfg["face_size_min"] <= (f.bbox[2] - f.bbox[0]) <= det_cfg["face_size_max"]
            and det_cfg["face_size_min"] <= (f.bbox[3] - f.bbox[1]) <= det_cfg["face_size_max"]
            and f.kps is not None
        ]

        bboxes    = [f.bbox for f in faces]
        track_ids = tracker.update(bboxes)

        new_log_entries = []
        recognized = 0
        unknown    = 0
        now        = time.time()

        for face, bbox, tid in zip(faces, bboxes, track_ids):

            # ── Liveness check ────────────────────────────────────────
            if rec_cfg["liveness_check"]:
                liveness_img = liveness_input_fn(frame, face)
                if liveness_img is not None and liveness_img.size > 0:
                    if not check_liveness(liveness_img):
                        draw_box(frame, bbox, "SPOOF", 0.0, "unknown")
                        continue

            # ── Get embedding ─────────────────────────────────────────
            emb = embed_fn(frame, face)
            if emb is None:
                continue

            # ── Cosine classify ───────────────────────────────────────
            raw_name, raw_score = cosine_classify(
                emb, gallery,
                threshold=rec_cfg["cosine_threshold"],
                debug=rec_cfg["debug_scores"],
            )

            # ── Temporal voting ───────────────────────────────────────
            if tid not in vote_buffers:
                vote_buffers[tid] = deque(maxlen=rec_cfg["vote_window"])
            vote_buffers[tid].append((raw_name, raw_score))
            buf = vote_buffers[tid]

            if len(buf) < 5:
                draw_box(frame, bbox, "", 0.0, "identifying")
                continue

            voted_name, vote_count = Counter(n for n, _ in buf).most_common(1)[0]

            # Median of top scores — robust to occasional bad frames
            winner_scores = sorted(
                [s for n, s in buf if n == voted_name], reverse=True,
            )
            voted_score = float(np.median(
                winner_scores[:max(3, len(winner_scores) // 2)]
            ))

            if vote_count < rec_cfg["vote_threshold"]:
                draw_box(frame, bbox, voted_name, voted_score, "identifying")
                continue

            # ── Committed identity ────────────────────────────────────
            is_known = voted_name != "Unknown"
            draw_box(frame, bbox, voted_name, voted_score,
                     "known" if is_known else "unknown")

            if is_known:
                recognized += 1
                # voted_name == folder name from gallery; look up emp_no via employees table
                emp_info = state.folder_to_emp.get(voted_name)
                if emp_info is None:
                    # Folder exists in gallery but NOT in employees table → skip logging
                    logger.warning("No employees row for folder %r — skipping log", voted_name)
                else:
                    emp_no   = emp_info["emp_no"]
                    emp_name = emp_info["emp_name"]
                    last_logged = state.log_cooldown.get(emp_no, 0)
                    if now - last_logged >= rec_cfg["log_cooldown_seconds"]:
                        ts         = time.strftime("%Y-%m-%d %H:%M:%S")
                        confidence = int(voted_score * 100)
                        mark_attendance(emp_no, emp_name, ts, confidence)
                        state.log_cooldown[emp_no] = now
                        new_log_entries.append({
                            "emp_no":     emp_no,
                            "name":       emp_name,
                            "confidence": confidence,
                            "time":       ts,
                        })
            else:
                unknown += 1

        # ── FPS cap ───────────────────────────────────────────────────
        elapsed = time.time() - t0
        if elapsed < min_frame_time:
            time.sleep(min_frame_time - elapsed)

        # ── FPS calculation ───────────────────────────────────────────
        frame_times.append(time.time() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

        # ── Update shared state ───────────────────────────────────────
        with state.frame_lock:
            state.annotated_frame = frame

        with state.stats_lock:
            state.stats["recognized"] += recognized
            state.stats["unknown"]    += unknown
            state.stats["total"]      += recognized + unknown
            state.stats["fps"]         = round(fps, 1)
            state.stats["uptime"]      = time.time() - start_time
            if new_log_entries:
                state.stats["recent_detections"] = (
                    new_log_entries + state.stats["recent_detections"]
                )[:max_log]

    logger.info("Inference loop stopped")
