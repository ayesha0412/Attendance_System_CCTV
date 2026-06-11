"""
API Authentication — FastAPI Dependency
=======================================
Bearer token check for protected endpoints.
If ATTENDANCE_READ_KEY is configured, requests need: Authorization: Bearer <key>
No key configured = open access (development mode).
"""

from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer = HTTPBearer(auto_error=False)


async def require_read_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
):
    config = request.app.state.config
    read_key = config.get("api", {}).get("read_key", "")

    if not read_key:
        return

    if credentials and credentials.credentials == read_key:
        return

    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key. Send: Authorization: Bearer <your-key>",
    )
