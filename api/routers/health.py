"""
api/routers/health.py
──────────────────────
GET /health — server and data health check.
Used by the mobile app to verify API connectivity before use.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from api.dependencies import get_predictor
from api.schemas.models import HealthResponse
from src.predictor import Predictor

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
)
def health_check(predictor: Predictor = Depends(get_predictor)) -> HealthResponse:
    """
    Returns API status, number of loaded blackspots, and server version.
    Mobile app calls this on startup to confirm backend is reachable.
    """
    return HealthResponse(
        status="ok",
        blackspots_loaded=len(predictor),
        version="1.0.0",
    )


@router.get(
    "/ping",
    summary="Simple ping — returns pong",
    include_in_schema=False,
)
def ping() -> dict:
    """Lightweight liveness probe — no dependencies."""
    return {
        "pong": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }