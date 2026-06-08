"""
Attendance Sender — fires once per person per day to an external API.
=====================================================================
Configure in ../.env:
  ATTENDANCE_API_URL  — the endpoint to POST to
  ATTENDANCE_API_KEY  — auth key (if the other team requires it)

Features:
  - Sends once per person per calendar day
  - Retries failed sends (up to 3 times with backoff)
  - Failed sends stay in retry queue — not marked as "sent"
  - All attempts logged to attendance_log.csv for audit

Thread-safe, non-blocking.
"""

import os
import csv
import threading
import time
from datetime import date, datetime
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, "..", ".env"))

ATTENDANCE_API_URL = os.getenv("ATTENDANCE_API_URL", "")
ATTENDANCE_API_KEY = os.getenv("ATTENDANCE_API_KEY", "")
LOG_FILE = os.path.join(_HERE, "attendance_log.csv")
MAX_RETRIES = 3

_lock = threading.Lock()
_sent_today: dict[str, str] = {}


def _reset_if_new_day():
    today = date.today().isoformat()
    with _lock:
        if _sent_today.get("__date__") != today:
            _sent_today.clear()
            _sent_today["__date__"] = today


def _log_to_csv(employee_name: str, dt: str, tm: str, status: str):
    """Every attendance attempt is logged locally — even if API fails, you have a record."""
    try:
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["employee_name", "date", "time", "status", "logged_at"])
            writer.writerow([employee_name, dt, tm, status, datetime.now().isoformat()])
    except Exception:
        pass


def _do_send(employee_name: str, dt: str, tm: str):
    import requests

    headers = {"Content-Type": "application/json"}
    if ATTENDANCE_API_KEY:
        headers["Authorization"] = f"Bearer {ATTENDANCE_API_KEY}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                ATTENDANCE_API_URL,
                json={"employee_name": employee_name, "date": dt, "time": tm},
                headers=headers,
                timeout=5,
            )
            if resp.status_code < 400:
                print(f"[ATTENDANCE] Sent {employee_name} @ {dt} {tm} -> {resp.status_code}")
                _log_to_csv(employee_name, dt, tm, f"sent_{resp.status_code}")
                with _lock:
                    _sent_today[employee_name] = date.today().isoformat()
                return

            print(f"[ATTENDANCE] Server error {resp.status_code} for {employee_name} (attempt {attempt}/{MAX_RETRIES})")

        except Exception as e:
            print(f"[ATTENDANCE] Failed {employee_name} (attempt {attempt}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)

    print(f"[ATTENDANCE] GIVING UP on {employee_name} after {MAX_RETRIES} attempts — will retry next sighting")
    _log_to_csv(employee_name, dt, tm, "failed_all_retries")
    with _lock:
        _sent_today.pop(employee_name, None)


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
        _sent_today[employee_name] = "pending"

    parts = timestamp.split(" ", 1)
    dt = parts[0]
    tm = parts[1] if len(parts) > 1 else "00:00:00"

    print(f"[ATTENDANCE] Marking: {employee_name} | date={dt} | time={tm}")
    _log_to_csv(employee_name, dt, tm, "detected")

    if not ATTENDANCE_API_URL:
        print(f"[ATTENDANCE] No API URL configured -- set ATTENDANCE_API_URL in .env")
        _log_to_csv(employee_name, dt, tm, "no_api_url")
        return

    threading.Thread(target=_do_send, args=(employee_name, dt, tm), daemon=True).start()
