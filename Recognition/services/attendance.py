"""
Attendance Service
==================
Three responsibilities:
  1. SQLITE  — persistent storage (attendance + sightings)
  2. MEMORY  — in-memory recent sightings for fast dashboard polling
  3. SENDER  — pushes every sighting to external AMS API

AMS Push Contract:
  POST <ATTENDANCE_API_URL>
  Headers: Authorization: Bearer <ATTENDANCE_API_KEY>, Content-Type: application/json
  Body:    { "employee_id": "061", "timestamp": "2026-06-11T09:15:32" }

  The AMS system decides check-in vs check-out:
    - First push of the day for an employee  -> check-in
    - All subsequent pushes                  -> check-out (keeps updating)

Retry: Failed pushes stay in the `sightings` table with ams_sent=0.
       A background worker retries every N seconds.
"""

import os
import time
import logging
import threading
from collections import deque
from dotenv import load_dotenv

from services import database as db

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
load_dotenv(os.path.join(_ROOT, "..", ".env"))

ATTENDANCE_API_URL  = os.getenv("ATTENDANCE_API_URL",  "")
ATTENDANCE_API_KEY  = os.getenv("ATTENDANCE_API_KEY",  "")
ATTENDANCE_READ_KEY = os.getenv("ATTENDANCE_READ_KEY", "")

PUSH_TIMEOUT_SEC      = 5
RETRY_INTERVAL_SEC    = 30   # background worker re-checks pending every 30s
MAX_ATTEMPTS_PER_RUN  = 50   # max pending sightings per retry cycle

_recent_sightings: deque = deque(maxlen=100)
_recent_lock = threading.Lock()


# ======================================================================
#  PUBLIC READ INTERFACE
# ======================================================================

def get_attendance_today() -> list:
    return db.get_today()


def get_recent_sightings(n: int = 50) -> list:
    """Fast in-memory recent — for dashboard polling."""
    with _recent_lock:
        items = list(_recent_sightings)
    return items[-n:]


# ======================================================================
#  AMS PUSH — synchronous (called from background thread)
# ======================================================================

def _push_to_ams(emp_no: str, timestamp: str) -> tuple[bool, str]:
    """POST one sighting to AMS. Returns (success, error_msg)."""
    if not ATTENDANCE_API_URL:
        return False, "ATTENDANCE_API_URL not configured"

    import requests
    headers = {"Content-Type": "application/json"}
    if ATTENDANCE_API_KEY:
        headers["Authorization"] = f"Bearer {ATTENDANCE_API_KEY}"

    payload = {
        "employee_id": emp_no,
        "timestamp":   timestamp,
    }

    try:
        resp = requests.post(
            ATTENDANCE_API_URL, json=payload,
            headers=headers, timeout=PUSH_TIMEOUT_SEC,
        )
        if 200 <= resp.status_code < 300:
            return True, ""
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _retry_worker():
    """Background loop: pull pending sightings from DB, push to AMS, mark status."""
    while True:
        try:
            pending = db.get_pending_ams(limit=MAX_ATTEMPTS_PER_RUN)
            if pending and ATTENDANCE_API_URL:
                logger.info("AMS retry — %d pending sightings", len(pending))
                for s in pending:
                    ok, err = _push_to_ams(s["emp_no"], s["timestamp"])
                    if ok:
                        db.mark_ams_processed(s["id"])
                        logger.info("AMS OK: sighting #%d (emp %s)", s["id"], s["emp_no"])
                    else:
                        db.mark_ams_failed(s["id"], err)
                        logger.warning("AMS FAIL: sighting #%d (emp %s) -> %s",
                                       s["id"], s["emp_no"], err)
        except Exception as e:
            logger.error("AMS retry worker error: %s", e)

        time.sleep(RETRY_INTERVAL_SEC)


def start_ams_worker():
    """Call once at startup to kick off the background retry thread."""
    if not ATTENDANCE_API_URL:
        logger.warning("ATTENDANCE_API_URL not set — AMS push disabled.")
        return
    t = threading.Thread(target=_retry_worker, daemon=True, name="ams-worker")
    t.start()
    logger.info("AMS retry worker started (every %ds)", RETRY_INTERVAL_SEC)


# ======================================================================
#  PUBLIC WRITE INTERFACE — called by inference loop
# ======================================================================

def mark_attendance(emp_no: str, emp_name: str, timestamp: str, confidence: int = 0):
    """
    Persist a sighting + update daily attendance + queue AMS push.

    Called from the inference loop every time a recognition event is committed
    (after vote threshold met, subject to LOG_COOLDOWN_SECONDS per person).
    """
    # Parse timestamp into date / time
    parts = timestamp.split(" ", 1)
    dt = parts[0]
    tm = parts[1] if len(parts) > 1 else "00:00:00"
    iso_ts = f"{dt}T{tm}"

    # ── SQLite: persistent storage ────────────────────────────────────
    try:
        db.upsert_attendance(emp_no, dt, tm, confidence)
        sighting_id = db.insert_sighting(emp_no, iso_ts, confidence)
    except Exception as e:
        logger.error("DB write failed for %s: %s", emp_no, e)
        return

    # ── In-memory: dashboard sidebar ──────────────────────────────────
    with _recent_lock:
        _recent_sightings.append({
            "emp_no":     emp_no,
            "emp_name":   emp_name,
            "date":       dt,
            "time":       tm,
            "confidence": confidence,
        })

    logger.info("Sighting #%d: emp %s (%s) @ %s — %d%%",
                sighting_id, emp_no, emp_name, iso_ts, confidence)

    # ── AMS push: try immediately, fall back to retry worker ──────────
    if not ATTENDANCE_API_URL:
        return

    def _async_push():
        ok, err = _push_to_ams(emp_no, iso_ts)
        if ok:
            db.mark_ams_processed(sighting_id)
            logger.info("AMS OK (immediate): sighting #%d", sighting_id)
        else:
            db.mark_ams_failed(sighting_id, err)
            logger.warning("AMS FAIL (immediate): sighting #%d -> %s — will retry",
                           sighting_id, err)

    threading.Thread(target=_async_push, daemon=True).start()
