"""
FastAPI Application Factory
============================
Creates the FastAPI app with all routers registered.

Swagger UI:  http://localhost:{port}/docs
ReDoc:       http://localhost:{port}/redoc
"""

import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def create_app(config):
    app = FastAPI(
        title="Face Recognition Attendance API",
        description="CCTV face recognition with live dashboard and REST endpoints.",
        version="v1",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.config = config

    from api.blueprints.dashboard import router as dashboard_router
    from api.blueprints.attendance import router as attendance_router
    from api.blueprints.health import router as health_router

    app.include_router(dashboard_router)
    logger.info("Registered: dashboard")

    app.include_router(attendance_router, prefix="/api/v1")
    logger.info("Registered: attendance -> /api/v1")

    app.include_router(health_router, prefix="/api/v1")
    logger.info("Registered: health -> /api/v1")

    # Backward-compatible aliases at /api/ (hidden from Swagger)
    app.include_router(
        attendance_router, prefix="/api",
        tags=["compat"], include_in_schema=False,
    )
    logger.info("Registered: attendance -> /api (compat)")

    app.include_router(
        health_router, prefix="/api",
        tags=["compat"], include_in_schema=False,
    )
    logger.info("Registered: health -> /api (compat)")

    for route in app.routes:
        if hasattr(route, "methods"):
            methods = ",".join(sorted(route.methods - {"HEAD", "OPTIONS"}))
            if methods:
                logger.debug("Route: %s %s", methods, route.path)

    return app
