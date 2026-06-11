"""
Backward Compatibility Wrapper
===============================
All attendance logic has moved to services/attendance.py.

This file re-exports the public API so existing imports still work:
    from attendance_sender import mark_attendance, get_attendance_today, ...

For new code, import directly:
    from services.attendance import mark_attendance
"""

from services.attendance import (        # noqa: F401
    mark_attendance,
    get_attendance_today,
    get_recent_sightings,
    ATTENDANCE_READ_KEY,
    ATTENDANCE_API_URL,
    ATTENDANCE_API_KEY,
)
