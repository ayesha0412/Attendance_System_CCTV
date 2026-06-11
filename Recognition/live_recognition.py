"""
Live Face Recognition — ArcFace (InsightFace built-in)
======================================================
Entry point.  Uses InsightFace's built-in ArcFace embedding model.

Usage:
    python live_recognition.py

Dashboard: http://localhost:5001
API:       http://localhost:5001/api/v1/health
"""

import os
import sys
import pickle
import logging
import threading
import time

from insightface.app import FaceAnalysis

from core.config import load_config, build_rtsp_url, setup_logging
from core.camera import ThreadedCamera
from services.recognition import inference_loop
from services import state
from services.database import init_db, get_folder_to_emp_map
from services.attendance import start_ams_worker
from api import create_app

_HERE = os.path.dirname(os.path.abspath(__file__))

GALLERY_PATH = os.path.join(_HERE, "face_model.pkl")

PORT = 5001   # ArcFace runs on 5001 (AdaFace on 5002)


# =====================================================================
#  CALLBACKS — passed to the generic inference loop
# =====================================================================

def arcface_embed_fn(frame, face):
    """ArcFace embedding — InsightFace computes it during detection."""
    return face.embedding


def arcface_liveness_fn(frame, face):
    """Liveness input for ArcFace: crop the face region from the frame."""
    x1, y1, x2, y2 = [max(0, int(v)) for v in face.bbox]
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


# =====================================================================
#  MAIN
# =====================================================================

def main():
    # ── Config + Logging ──────────────────────────────────────────────
    config = load_config()
    config["server"]["port"] = PORT

    # ArcFace benefits from CLAHE lighting normalization
    config.setdefault("preprocessing", {})["enhance_clahe"] = True

    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("=" * 55)
    logger.info("  LIVE RECOGNITION — ArcFace (InsightFace)")
    logger.info("=" * 55)

    # ── Initialize SQLite database ────────────────────────────────────
    init_db()

    # ── Load folder->emp_no map from employees table ──────────────────
    state.folder_to_emp = get_folder_to_emp_map()
    logger.info("Loaded %d employee folder mappings", len(state.folder_to_emp))
    if not state.folder_to_emp:
        logger.error("No employees in DB — run: python migrate_db.py")
        sys.exit(1)

    # ── Start AMS push retry worker (no-op if URL not configured) ─────
    start_ams_worker()

    # ── Load gallery ──────────────────────────────────────────────────
    if not os.path.exists(GALLERY_PATH):
        logger.error("Gallery not found: %s — run: python train_embeddings.py", GALLERY_PATH)
        sys.exit(1)

    with open(GALLERY_PATH, "rb") as f:
        data = pickle.load(f)

    if "gallery" not in data:
        logger.error("face_model.pkl is the old SVM format — re-run: python train_embeddings.py")
        sys.exit(1)

    gallery     = data["gallery"]
    class_names = data["class_names"]

    state.model_info = {
        "gallery":     gallery,
        "class_names": class_names,
        "model_name":  "ArcFace (buffalo_l)",
    }
    logger.info("Gallery loaded — %d people: %s", len(gallery), class_names)

    # ── Load InsightFace (detection + ArcFace embedding) ──────────────
    logger.info("Loading InsightFace (buffalo_l)...")
    det_size = tuple(config["detection"]["det_size"])
    face_app = FaceAnalysis(
        name=config["detection"]["model_name"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=det_size)
    logger.info("InsightFace ready")

    # ── Connect camera ────────────────────────────────────────────────
    rtsp_url = build_rtsp_url(config)
    logger.info("RTSP: rtsp://%s:****@%s:%s/Streaming/Channels/%s",
                config["camera"]["username"], config["camera"]["ip"],
                config["camera"]["port"], config["camera"]["channel"])

    camera = ThreadedCamera(rtsp_url, config["camera"].get("buffer_size", 1))
    if not camera.start():
        logger.error("Cannot connect to camera!")
        sys.exit(1)
    logger.info("Camera connected")

    time.sleep(1)

    # ── Start inference thread ────────────────────────────────────────
    threading.Thread(
        target=inference_loop,
        args=(camera, face_app, gallery, config,
              arcface_embed_fn, arcface_liveness_fn),
        daemon=True,
    ).start()

    # ── Start FastAPI (uvicorn) ───────────────────────────────────────
    import uvicorn

    app = create_app(config)
    logger.info("Dashboard -> http://localhost:%d", PORT)
    logger.info("Swagger   -> http://localhost:%d/docs", PORT)
    logger.info("Health    -> http://localhost:%d/api/v1/health", PORT)

    try:
        uvicorn.run(
            app,
            host=config["server"]["host"],
            port=PORT,
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
