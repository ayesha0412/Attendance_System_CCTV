"""
SQLite Database Layer
=====================
Local persistent storage for attendance + AMS push queue.
Survives server restarts, power cuts, crashes.

Schema (created by migrate_db.py):
  employees   — master list from employees.csv (emp_no, emp_name, folder_name)
  attendance  — one row per employee per day (first_seen = check-in, last_seen = check-out)
  sightings   — every recognition event + AMS push status
"""

import os
import sqlite3
import threading
import logging
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DB_PATH = os.path.join(_ROOT, "attendance.db")

_conn = None
_lock = threading.Lock()


def init_db(db_path=None):
    """Open the DB created by migrate_db.py. Does NOT create tables."""
    global _conn
    path = db_path or DB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python migrate_db.py"
        )
    _conn = sqlite3.connect(path, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA foreign_keys=ON")

    # Sanity check — fail loudly if migration wasn't run
    tables = {r["name"] for r in _conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    required = {"employees", "attendance", "sightings"}
    if not required.issubset(tables):
        missing = required - tables
        raise RuntimeError(
            f"Missing tables: {missing}. Run: python migrate_db.py"
        )

    logger.info("SQLite database ready: %s", path)


# ======================================================================
#  EMPLOYEE LOOKUPS
# ======================================================================

def load_all_employees() -> list[dict]:
    """Return [{emp_no, emp_name, folder_name}, ...] for all employees."""
    with _lock:
        rows = _conn.execute(
            "SELECT emp_no, emp_name, folder_name FROM employees ORDER BY emp_name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_folder_to_emp_map() -> dict[str, dict]:
    """Return {folder_name: {emp_no, emp_name}} — fast lookup for inference loop."""
    with _lock:
        rows = _conn.execute(
            "SELECT emp_no, emp_name, folder_name FROM employees"
        ).fetchall()
    return {r["folder_name"]: {"emp_no": r["emp_no"], "emp_name": r["emp_name"]}
            for r in rows}


def get_employee(emp_no: str) -> dict | None:
    with _lock:
        row = _conn.execute(
            "SELECT emp_no, emp_name, folder_name FROM employees WHERE emp_no = ?",
            (emp_no,)
        ).fetchone()
    return dict(row) if row else None


# ======================================================================
#  ATTENDANCE — daily summary (first_seen / last_seen)
# ======================================================================

def upsert_attendance(emp_no: str, dt: str, tm: str, confidence: int):
    """
    First sighting of the day -> INSERT new row (sets first_seen).
    Subsequent sightings    -> UPDATE last_seen + increment sightings.
    """
    now = datetime.now().isoformat()
    with _lock:
        cur = _conn.execute(
            """UPDATE attendance
               SET last_seen = ?, total_sightings = total_sightings + 1,
                   confidence = MAX(confidence, ?), updated_at = ?
               WHERE emp_no = ? AND date = ?""",
            (tm, confidence, now, emp_no, dt)
        )
        if cur.rowcount == 0:
            _conn.execute(
                """INSERT INTO attendance
                   (emp_no, date, first_seen, last_seen,
                    confidence, total_sightings, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
                (emp_no, dt, tm, tm, confidence, now, now)
            )
        _conn.commit()


# ======================================================================
#  SIGHTINGS — raw event log + AMS push queue
# ======================================================================

def insert_sighting(emp_no: str, timestamp: str, confidence: int) -> int:
    """Insert a new sighting (ams_sent=0). Returns the row id."""
    now = datetime.now().isoformat()
    with _lock:
        cur = _conn.execute(
            """INSERT INTO sightings
               (emp_no, timestamp, confidence, created_at)
               VALUES (?, ?, ?, ?)""",
            (emp_no, timestamp, confidence, now)
        )
        _conn.commit()
        return cur.lastrowid


def get_pending_ams(limit: int = 50) -> list[dict]:
    """Sightings that haven't been pushed to AMS yet, oldest first."""
    with _lock:
        rows = _conn.execute(
            """SELECT id, emp_no, timestamp, confidence, ams_attempts
               FROM sightings
               WHERE ams_sent = 0
               ORDER BY id ASC
               LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def mark_ams_sent(sighting_id: int):
    with _lock:
        _conn.execute(
            "UPDATE sightings SET ams_sent = 1, ams_attempts = ams_attempts + 1 WHERE id = ?",
            (sighting_id,)
        )
        _conn.commit()


def mark_ams_processed(sighting_id: int):
    with _lock:
        _conn.execute(
            "UPDATE sightings SET ams_sent = 1, ams_processed = 1, ams_attempts = ams_attempts + 1, ams_last_error = NULL WHERE id = ?",
            (sighting_id,)
        )
        _conn.commit()


def mark_ams_failed(sighting_id: int, error: str):
    """Record a failed push attempt; ams_sent stays 0 so it gets retried."""
    with _lock:
        _conn.execute(
            "UPDATE sightings SET ams_attempts = ams_attempts + 1, ams_last_error = ? WHERE id = ?",
            (error[:500], sighting_id)
        )
        _conn.commit()


# ======================================================================
#  REPORTS — joined with employees for human-readable names
# ======================================================================

def get_today() -> list[dict]:
    today = date.today().isoformat()
    return get_by_date(today)


def get_by_date(dt: str) -> list[dict]:
    with _lock:
        rows = _conn.execute(
            """SELECT a.id, a.emp_no, e.emp_name, a.date,
                      a.first_seen, a.last_seen, a.confidence, a.total_sightings
               FROM attendance a
               LEFT JOIN employees e ON a.emp_no = e.emp_no
               WHERE a.date = ?
               ORDER BY a.first_seen""",
            (dt,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_employee_history(emp_no: str, days: int = 30) -> list[dict]:
    since = (date.today() - timedelta(days=days)).isoformat()
    with _lock:
        rows = _conn.execute(
            """SELECT a.id, a.date, a.first_seen, a.last_seen,
                      a.confidence, a.total_sightings, e.emp_name
               FROM attendance a
               LEFT JOIN employees e ON a.emp_no = e.emp_no
               WHERE a.emp_no = ? AND a.date >= ?
               ORDER BY a.date DESC""",
            (emp_no, since)
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_sightings(n: int = 50) -> list[dict]:
    with _lock:
        rows = _conn.execute(
            """SELECT s.id, s.emp_no, e.emp_name, s.timestamp, s.confidence,
                      s.ams_sent, s.ams_processed, s.ams_attempts
               FROM sightings s
               LEFT JOIN employees e ON s.emp_no = e.emp_no
               ORDER BY s.id DESC LIMIT ?""",
            (n,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_stats() -> dict:
    today = date.today().isoformat()
    with _lock:
        present_today = _conn.execute(
            "SELECT COUNT(*) FROM attendance WHERE date = ?", (today,)
        ).fetchone()[0]
        total_sightings = _conn.execute(
            "SELECT COUNT(*) FROM sightings"
        ).fetchone()[0]
        pending_ams = _conn.execute(
            "SELECT COUNT(*) FROM sightings WHERE ams_sent = 0"
        ).fetchone()[0]
        total_employees = _conn.execute(
            "SELECT COUNT(*) FROM employees"
        ).fetchone()[0]
    return {
        "present_today":    present_today,
        "total_employees":  total_employees,
        "total_sightings":  total_sightings,
        "pending_ams_push": pending_ams,
    }
