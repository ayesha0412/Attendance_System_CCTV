"""
Dashboard Router
================
  /            — HTML dashboard page
  /video_feed  — MJPEG live stream
  /api/stats   — internal stats for dashboard JS polling (no auth)
"""

import os
import time
import cv2
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from services import state

router = APIRouter(tags=["Dashboard"])

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "templates"))
templates = Jinja2Templates(directory=_TEMPLATE_DIR)


def _generate_mjpeg():
    while True:
        with state.frame_lock:
            if state.annotated_frame is None:
                time.sleep(0.05)
                continue
            frame = state.annotated_frame.copy()

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    config = request.app.state.config
    rec = config["recognition"]

    return templates.TemplateResponse(request, "dashboard.html", context={
        "class_names": state.model_info["class_names"],
        "model_name": state.model_info["model_name"],
        "threshold": rec["cosine_threshold"],
        "cooldown": rec["log_cooldown_seconds"],
        "vote_window": rec["vote_window"],
        "port": config["server"].get("port", 5002),
    })


@router.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/stats")
async def get_stats():
    with state.stats_lock:
        return dict(state.stats)
