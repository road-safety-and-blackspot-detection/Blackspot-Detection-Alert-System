"""
api/routers/blackspots.py
──────────────────────────
Core blackspot endpoints consumed by the mobile app.

Endpoints
─────────
GET /api/v1/nearby          → blackspots near a GPS point + alert check
GET /api/v1/blackspots/all  → all blackspots (for map screen heatmap)
GET /api/v1/blackspots/stats → dashboard statistics
GET /api/v1/blackspots/{id} → single blackspot detail
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_predictor
from api.schemas.models import (
    AllBlackspotsResponse,
    BlackSpot,
    NearbyResponse,
    StatsResponse,
)
from src.predictor import Predictor

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["blackspots"])


# ── GET /api/v1/nearby ────────────────────────────────────────────────────────
@router.get(
    "/nearby",
    response_model=NearbyResponse,
    summary="Get blackspots near a GPS location + alert status",
)
def get_nearby(
    lat:      float = Query(..., ge=-90,  le=90,   description="User latitude"),
    lng:      float = Query(..., ge=-180, le=180,  description="User longitude"),
    radius:   int   = Query(500, ge=50,   le=5000, description="Search radius in metres"),
    limit:    int   = Query(10,  ge=1,    le=50,   description="Max results"),
    min_risk: float = Query(0.0, ge=0,   le=100,  description="Minimum risk score filter"),
    predictor: Predictor = Depends(get_predictor),
) -> NearbyResponse:
    """
    **Core endpoint for the mobile app.**

    Called every 5 seconds as the user moves.
    Returns all blackspot zones within `radius` metres, sorted by risk score.
    Also returns an alert flag and reason string for the Alert Screen.

    - `alert=true` + `alert_level=HIGH`   → trigger full-screen alert
    - `alert=true` + `alert_level=MEDIUM` → show banner warning
    - `alert=false` + `alert_level=SAFE`  → no alert needed
    """
    log.debug(f"GET /nearby lat={lat} lng={lng} radius={radius}m")

    result = predictor.alert(
        lat=lat,
        lng=lng,
        radius_m=radius,
    )

    # Filter by min_risk if specified
    spots = [
        s for s in result["black_spots"]
        if s["risk_score"] >= min_risk
    ][:limit]

    return NearbyResponse(
        alert       = result["alert"],
        alert_level = result["alert_level"] if result["alert"] else "SAFE",
        top_score   = result["top_score"],
        black_spots = spots,
        reason      = result["reason"],
        queried_at  = datetime.now(timezone.utc).isoformat(),
    )


# ── GET /api/v1/blackspots/all ────────────────────────────────────────────────
@router.get(
    "/blackspots/all",
    response_model=AllBlackspotsResponse,
    summary="All blackspots — used for map heatmap overlay",
)
def get_all_blackspots(
    risk_level: Optional[str] = Query(
        None,
        pattern="^(HIGH|MEDIUM|LOW)$",
        description="Filter by risk level: HIGH, MEDIUM, LOW",
    ),
    limit: int = Query(500, ge=1, le=2000, description="Max results"),
    predictor: Predictor = Depends(get_predictor),
) -> AllBlackspotsResponse:
    """
    Returns all (or filtered) blackspot zones.

    **Used by the Map Screen** to draw the heatmap and risk zone circles.
    For full map load, call without filters.
    For HIGH-risk only layer, pass `risk_level=HIGH`.
    """
    spots = predictor.all_blackspots(risk_level=risk_level, limit=limit)
    return AllBlackspotsResponse(
        count      = len(spots),
        risk_level = risk_level,
        blackspots = spots,
    )


# ── GET /api/v1/blackspots/stats ──────────────────────────────────────────────
@router.get(
    "/blackspots/stats",
    response_model=StatsResponse,
    summary="Aggregate statistics — used for dashboard screen",
)
def get_stats(
    predictor: Predictor = Depends(get_predictor),
) -> StatsResponse:
    """
    Returns aggregate statistics about all blackspot zones.

    **Used by the Dashboard Screen** to show:
    - Total accidents and fatalities indexed
    - Risk distribution (HIGH / MEDIUM / LOW counts)
    - Road type breakdown
    - Top 10 most dangerous zones
    """
    stats = predictor.stats()
    if not stats:
        raise HTTPException(status_code=503, detail="No blackspot data available")
    return StatsResponse(**stats)


# ── GET /api/v1/blackspots/{cluster_id} ───────────────────────────────────────
@router.get(
    "/blackspots/{cluster_id}",
    response_model=BlackSpot,
    summary="Single blackspot detail by cluster ID",
)
def get_blackspot_by_id(
    cluster_id: int,
    predictor:  Predictor = Depends(get_predictor),
) -> BlackSpot:
    """
    Returns detailed info for a single blackspot zone.

    **Used when user taps a zone** on the Map Screen to see:
    - Risk score and level
    - Accident count and fatalities
    - Road type, weather conditions
    - Night and monsoon risk percentages
    """
    spots = predictor.all_blackspots(limit=10000)
    spot  = next((s for s in spots if s["cluster_id"] == cluster_id), None)
    if not spot:
        raise HTTPException(
            status_code=404,
            detail=f"Blackspot with cluster_id={cluster_id} not found",
        )
    return BlackSpot(**spot)