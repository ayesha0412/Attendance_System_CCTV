"""
Live Face Recognition — AdaFace IR-50
======================================
Entry point.  Loads AdaFace model, connects camera, starts inference + Flask.

Usage:
    python live_adaface.py

Dashboard: http://localhost:5002
API:       http://localhost:5002/api/v1/health
"""

import os
import sys
import pickle
import logging
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis

from core.config import load_config, build_rtsp_url, setup_logging
from core.camera import ThreadedCamera
from core.alignment import align_112
from services.recognition import inference_loop
from services import state
from services.database import init_db, get_folder_to_emp_map
from services.attendance import start_ams_worker
from api import create_app

_HERE = os.path.dirname(os.path.abspath(__file__))

ASSETS_DIR = os.path.join(_HERE, "adaface_assets")
MODEL_DIR  = os.path.join(ASSETS_DIR, "cvlface_model")
GALLERY_PATH = os.path.join(_HERE, "face_model_adaface.pkl")

PORT = 5002   # AdaFace runs on 5002 (ArcFace on 5001)


# =====================================================================
#  ADAFACE MODEL — load once, reuse in embed_fn
# =====================================================================
_adaface_model  = None
_adaface_device = "cpu"


def _load_adaface():
    """Load AdaFace IR-50 model weights."""
    global _adaface_model, _adaface_device
    logger = logging.getLogger(__name__)

    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        logger.error("AdaFace model not found at %s — run: python setup_adaface.py", MODEL_DIR)
        sys.exit(1)

    _prev_cwd = os.getcwd()
    try:
        os.chdir(MODEL_DIR)
        if MODEL_DIR not in sys.path:
            sys.path.insert(0, MODEL_DIR)
        from models.iresnet import load_model as _load_iresnet
        from omegaconf import OmegaConf
        import yaml
        conf  = OmegaConf.create(yaml.safe_load(open("pretrained_model/model.yaml")))
        model = _load_iresnet(conf)
        model.load_state_dict_from_path("pretrained_model/model.pt")
    finally:
        os.chdir(_prev_cwd)

    model.eval()

    # Try CUDA — fall back to CPU if kernels aren't available
    _adaface_device = "cpu"
    if torch.cuda.is_available():
        try:
            model.cuda()
            with torch.no_grad():
                model(torch.zeros(1, 3, 112, 112).cuda())
            _adaface_device = "cuda"
        except Exception:
            model.cpu()
            _adaface_device = "cpu"
            logger.warning("CUDA not supported — AdaFace running on CPU")

    _adaface_model = model
    logger.info("AdaFace IR-50 loaded on %s", _adaface_device.upper())


def _get_embedding(aligned_bgr):
    """Extract 512-dim L2-normalised embedding from aligned 112x112 BGR face."""
    img    = aligned_bgr[:, :, ::-1].astype(np.float32)     # BGR -> RGB
    img    = (img / 255.0 - 0.5) / 0.5                      # normalise to [-1, 1]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(_adaface_device)

    with torch.no_grad():
        output = _adaface_model(tensor)

    if isinstance(output, (list, tuple)):
        emb = output[0]
    else:
        emb = output
    if emb.dim() > 2:
        emb = emb[:, 0]
    emb = F.normalize(emb, dim=1)
    return emb.squeeze().cpu().numpy()


# =====================================================================
#  CALLBACKS — passed to the generic inference loop
# =====================================================================

def adaface_embed_fn(frame, face):
    """Align face to 112x112, then extract AdaFace embedding."""
    aligned = align_112(frame, face.kps)
    if aligned is None:
        return None
    return _get_embedding(aligned)


def adaface_liveness_fn(frame, face):
    """Liveness input for AdaFace: use the aligned 112x112 crop."""
    return align_112(frame, face.kps)


# =====================================================================
#  MAIN
# =====================================================================

def main():
    # ── Config + Logging ──────────────────────────────────────────────
    config = load_config()
    config["server"]["port"] = PORT
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("=" * 55)
    logger.info("  LIVE RECOGNITION — AdaFace IR-50")
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
        logger.error("Gallery not found: %s — run: python train_adaface.py", GALLERY_PATH)
        sys.exit(1)

    with open(GALLERY_PATH, "rb") as f:
        data = pickle.load(f)
    gallery     = data["gallery"]
    class_names = data["class_names"]

    state.model_info = {
        "gallery":     gallery,
        "class_names": class_names,
        "model_name":  "AdaFace IR-50 MS1MV3",
    }
    logger.info("Gallery loaded — %d people: %s", len(gallery), class_names)

    # ── Load AdaFace model ────────────────────────────────────────────
    _load_adaface()

    # ── Load InsightFace SCRFD (detection only) ───────────────────────
    logger.info("Loading InsightFace SCRFD detector...")
    det_size = tuple(config["detection"]["det_size"])
    detector = FaceAnalysis(
        name=config["detection"]["model_name"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    detector.prepare(ctx_id=0, det_size=det_size)
    logger.info("Detector ready")

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

    time.sleep(1)    # let first frame buffer

    # ── Start inference thread ────────────────────────────────────────
    threading.Thread(
        target=inference_loop,
        args=(camera, detector, gallery, config,
              adaface_embed_fn, adaface_liveness_fn),
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
