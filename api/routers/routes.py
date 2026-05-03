"""
api/routers/routes.py
──────────────────────
Route risk scoring — the "smart" feature that differentiates
this app from a simple accident map.

Endpoints
─────────
POST /api/v1/route-risk
    → Accepts a list of waypoints (lat/lng pairs)
    → Returns overall route risk + all blackspots along the way
    → Used by Route Screen to compare safe vs fast paths

POST /api/v1/route-compare
    → Accepts two routes (route_a and route_b)
    → Returns side-by-side risk comparison
    → Used by Route Screen to highlight the safer option
"""

import logging

from fastapi import APIRouter, Depends

from api.dependencies import get_predictor
from api.schemas.models import RouteRiskRequest, RouteRiskResponse
from api.services.spatial import interpolate_route
from src.predictor import Predictor

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["routes"])


# ── POST /api/v1/route-risk ───────────────────────────────────────────────────
@router.post(
    "/route-risk",
    response_model=RouteRiskResponse,
    summary="Score a route for accident risk",
)
def score_route(
    body:      RouteRiskRequest,
    predictor: Predictor = Depends(get_predictor),
) -> RouteRiskResponse:
    """
    **Route Screen core endpoint.**

    Accepts a list of GPS waypoints (minimum 2) and returns:
    - `overall_risk_score`  : 0–100 weighted average along route
    - `risk_level`          : HIGH / MEDIUM / LOW
    - `total_blackspots`    : number of unique danger zones crossed
    - `high_risk_zones`     : count of HIGH risk zones specifically
    - `blackspots_on_route` : full detail of each zone found

    **How it works:**
    1. Interpolate waypoints into dense 100m steps
    2. For each step, query nearby blackspots within radius_m
    3. Deduplicate zones (same cluster counted once)
    4. Compute weighted risk score across all found zones

    **Request body example:**
    ```json
    {
      "waypoints": [
        {"lat": 30.90, "lng": 75.85},
        {"lat": 30.95, "lng": 75.90},
        {"lat": 31.00, "lng": 75.95}
      ],
      "radius_m": 300,
      "min_risk": 40.0
    }
    ```
    """
    waypoints = [(wp.lat, wp.lng) for wp in body.waypoints]
    log.info(f"POST /route-risk — {len(waypoints)} waypoints "
             f"radius={body.radius_m}m min_risk={body.min_risk}")

    # Densify route for thorough coverage
    dense_waypoints = interpolate_route(waypoints, step_m=100)
    log.debug(f"  Interpolated to {len(dense_waypoints)} points")

    result = predictor.route_risk(
        waypoints=dense_waypoints,
        radius_m=body.radius_m,
        min_risk=body.min_risk,
    )

    return RouteRiskResponse(**result)


# ── POST /api/v1/route-compare ────────────────────────────────────────────────
class RouteCompareRequest:
    pass


from pydantic import BaseModel
from typing import List


class RouteCompareWaypoints(BaseModel):
    waypoints: List[dict]  # [{"lat": ..., "lng": ...}]


class RouteCompareRequest(BaseModel):
    route_a: List[dict]
    route_b: List[dict]
    radius_m: int = 300
    min_risk: float = 40.0


class RouteCompareResponse(BaseModel):
    route_a: RouteRiskResponse
    route_b: RouteRiskResponse
    safer_route: str          # "A" or "B"
    risk_difference: float    # absolute difference in overall_risk_score
    recommendation: str       # human-readable advice


@router.post(
    "/route-compare",
    response_model=RouteCompareResponse,
    summary="Compare two routes by risk score",
)
def compare_routes(
    body:      RouteCompareRequest,
    predictor: Predictor = Depends(get_predictor),
) -> RouteCompareResponse:
    """
    **Route Screen comparison feature.**

    Pass two routes (fastest vs safest from a routing API like
    Google Maps or OSRM) and this endpoint tells you which is safer.

    Returns:
    - Full risk breakdown for both routes
    - Which route is safer (`safer_route`: "A" or "B")
    - Risk difference score
    - Human-readable recommendation

    **Request body example:**
    ```json
    {
      "route_a": [{"lat": 30.90, "lng": 75.85}, {"lat": 31.00, "lng": 75.95}],
      "route_b": [{"lat": 30.88, "lng": 75.83}, {"lat": 31.02, "lng": 75.97}],
      "radius_m": 300
    }
    ```
    """
    def score(waypoint_dicts):
        wps = [(wp["lat"], wp["lng"]) for wp in waypoint_dicts]
        dense = interpolate_route(wps, step_m=100)
        return predictor.route_risk(
            waypoints=dense,
            radius_m=body.radius_m,
            min_risk=body.min_risk,
        )

    result_a = score(body.route_a)
    result_b = score(body.route_b)

    score_a = result_a["overall_risk_score"]
    score_b = result_b["overall_risk_score"]

    if score_a <= score_b:
        safer        = "A"
        diff         = round(score_b - score_a, 1)
        recommendation = (
            f"Route A is safer by {diff} risk points "
            f"({result_a['high_risk_zones']} vs {result_b['high_risk_zones']} high-risk zones)."
        )
    else:
        safer        = "B"
        diff         = round(score_a - score_b, 1)
        recommendation = (
            f"Route B is safer by {diff} risk points "
            f"({result_b['high_risk_zones']} vs {result_a['high_risk_zones']} high-risk zones)."
        )

    return RouteCompareResponse(
        route_a          = RouteRiskResponse(**result_a),
        route_b          = RouteRiskResponse(**result_b),
        safer_route      = safer,
        risk_difference  = diff,
        recommendation   = recommendation,
    )