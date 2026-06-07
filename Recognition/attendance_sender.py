"""
Attendance Sender — fires once per person per day to an external API.
=====================================================================
Configure ATTENDANCE_API_URL in ../.env (or leave blank to log-only).

The external team receives:
    POST  {url}
    Body: {"employee_name": "Muhammad Noman", "date": "2026-06-07", "time": "09:15:32"}

Thread-safe, non-blocking.
"""

import os
import threading
import time
from datetime import date
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

ATTENDANCE_API_URL = os.getenv("ATTENDANCE_API_URL", "")

_lock = threading.Lock()
_sent_today: dict[str, str] = {}


def _reset_if_new_day():
    today = date.today().isoformat()
    with _lock:
        if _sent_today.get("__date__") != today:
            _sent_today.clear()
            _sent_today["__date__"] = today


def _do_send(employee_name: str, dt: str, tm: str):
    import requests
    try:
        resp = requests.post(
            ATTENDANCE_API_URL,
            json={"employee_name": employee_name, "date": dt, "time": tm},
            timeout=5,
        )
        print(f"[ATTENDANCE] Sent {employee_name} @ {dt} {tm} -> {resp.status_code}")
    except Exception as e:
        print(f"[ATTENDANCE] Failed to send {employee_name}: {e}")


def mark_attendance(employee_name: str, timestamp: str):
    """
    Call this whenever a face is committed as recognised.
    timestamp format: "2026-06-07 09:15:32"
    Splits into date and time, sends once per person per calendar day.
    """
    _reset_if_new_day()

    today = date.today().isoformat()
    with _lock:
        if _sent_today.get(employee_name) == today:
            return
        _sent_today[employee_name] = today

    parts = timestamp.split(" ", 1)
    dt = parts[0]
    tm = parts[1] if len(parts) > 1 else "00:00:00"

    print(f"[ATTENDANCE] Marking: {employee_name} | date={dt} | time={tm}")

    if not ATTENDANCE_API_URL:
        print(f"[ATTENDANCE] No API URL configured -- set ATTENDANCE_API_URL in .env")
        return

    threading.Thread(target=_do_send, args=(employee_name, dt, tm), daemon=True).start()
