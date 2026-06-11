"""
Attendance API Router
=====================
Trimmed endpoints for HR/AMS integration:

  GET /attendance/today              — present today (joined with employee names)
  GET /attendance/date/{date}        — specific date (YYYY-MM-DD)
  GET /attendance/employee/{emp_no}  — one person's history (default 30 days)
  GET /attendance/sightings          — recent sightings + AMS push status (debug)

Mounted at /api/v1/ (and /api/ compat).
Auto-documented at /docs.
"""

from datetime import date, datetime
from fastapi import APIRouter, Depends, Query

from api.middleware.auth import require_read_key
from services import database as db

router = APIRouter(
    tags=["Attendance"],
    dependencies=[Depends(require_read_key)],
)


@router.get("/attendance/today")
async def today():
    data = db.get_today()
    return {
        "status":       "ok",
        "date":         date.today().isoformat(),
        "count":        len(data),
        "attendance":   data,
        "generated_at": datetime.now().isoformat(),
    }


@router.get("/attendance/date/{dt}")
async def by_date(dt: str):
    data = db.get_by_date(dt)
    return {
        "status":       "ok",
        "date":         dt,
        "count":        len(data),
        "attendance":   data,
        "generated_at": datetime.now().isoformat(),
    }


@router.get("/attendance/employee/{emp_no}")
async def employee(emp_no: str, days: int = Query(default=30, le=90)):
    info = db.get_employee(emp_no)
    if not info:
        return {
            "status":  "not_found",
            "emp_no":  emp_no,
            "message": "Employee not in database.",
        }
    history = db.get_employee_history(emp_no, days)
    return {
        "status":       "ok",
        "employee":     info,
        "days":         days,
        "count":        len(history),
        "attendance":   history,
        "generated_at": datetime.now().isoformat(),
    }


@router.get("/attendance/sightings")
async def sightings(n: int = Query(default=50, le=200)):
    """Recent recognition events with AMS push status (debug / monitoring)."""
    rows = db.get_recent_sightings(n)
    return {
        "status":       "ok",
        "count":        len(rows),
        "sightings":    rows,
        "generated_at": datetime.now().isoformat(),
    }
