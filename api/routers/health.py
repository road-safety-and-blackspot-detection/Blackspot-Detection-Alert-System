"""
api/routers/health.py
──────────────────────
GET /health — server and data health check.
Used by the mobile app to verify API connectivity before use.
"""

# from datetime import datetime, timezone

# from fastapi import APIRouter, Depends

# from api.dependencies import get_predictor
# from api.schemas.models import HealthResponse
# from src.predictor import Predictor

# router = APIRouter(tags=["health"])


# @router.get(
#     "/health",
#     response_model=HealthResponse,
#     summary="API health check",
# )
# def health_check(predictor: Predictor = Depends(get_predictor)) -> HealthResponse:
#     """
#     Returns API status, number of loaded blackspots, and server version.
#     Mobile app calls this on startup to confirm backend is reachable.
#     """
#     return HealthResponse(
#         status="ok",
#         blackspots_loaded=len(predictor),
#         version="1.0.0",
#     )


# @router.get(
#     "/ping",
#     summary="Simple ping — returns pong",
#     include_in_schema=False,
# )
# def ping() -> dict:
#     """Lightweight liveness probe — no dependencies."""
#     return {
#         "pong": True,
#         "timestamp": datetime.now(timezone.utc).isoformat(),
#     }


"""
api/routers/health.py  ← FIXED VERSION
────────────────────────────────────────
Enhanced health endpoint that returns:
  - API status
  - Number of blackspots loaded
  - Whether blackspots.json was found
  - Python path info for debugging connection issues
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])

_BLACKSPOTS = (
    Path(__file__).parent.parent.parent
    / "ml-engine" / "data" / "outputs" / "blackspots.json"
)


@router.get("/health", summary="API health check")
def health_check():
    """
    Simple health check. Mobile app calls this first to verify connectivity.
    Returns 200 with JSON even if blackspots aren't loaded yet.
    """
    blackspots_exist = _BLACKSPOTS.exists()

    # Try to get predictor count
    loaded = 0
    error  = None
    try:
        from api.dependencies import get_predictor
        loaded = len(get_predictor())
    except Exception as e:
        error = str(e)

    return JSONResponse({
        "status":              "ok" if loaded > 0 else "degraded",
        "blackspots_loaded":   loaded,
        "blackspots_file":     str(_BLACKSPOTS),
        "blackspots_file_exists": blackspots_exist,
        "version":             "1.0.0",
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "error":               error,
    })


@router.get("/ping", include_in_schema=False)
def ping():
    """Fastest possible liveness check — no dependencies."""
    return {"pong": True}