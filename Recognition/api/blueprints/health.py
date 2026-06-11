"""
Health Check Router
===================
  GET /health  — system status (no auth)

Mounted at /api/v1/health and /api/health (compat).
"""

from datetime import datetime
from fastapi import APIRouter, Request

from services import state
from services.attendance import get_attendance_today

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(request: Request):
    config = request.app.state.config
    rec = config["recognition"]

    with state.stats_lock:
        s = dict(state.stats)

    uptime_sec = int(s.get("uptime", 0))

    return {
        "status":               "ok",
        "camera_connected":     s.get("fps", 0) > 0,
        "fps":                  s.get("fps", 0),
        "uptime_seconds":       uptime_sec,
        "uptime_human":         f"{uptime_sec // 3600}h {(uptime_sec % 3600) // 60}m",
        "model":                state.model_info["model_name"],
        "employees_registered": len(state.model_info["class_names"]),
        "recognised_today":     len(get_attendance_today()),
        "threshold":            rec["cosine_threshold"],
        "liveness_check":       rec["liveness_check"],
        "api_version":          "v1",
        "timestamp":            datetime.now().isoformat(),
    }
