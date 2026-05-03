"""
api/schemas/models.py
──────────────────────
All Pydantic request/response models for the API.
Every endpoint imports from here — single source of truth.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Shared sub-models ─────────────────────────────────────────────────────────

class BlackSpot(BaseModel):
    """A single blackspot zone returned by the API."""
    cluster_id:           int
    lat:                  float
    lng:                  float
    accident_count:       int
    total_killed:         int
    avg_severity:         float = Field(..., ge=1, le=4)
    avg_env_risk:         float
    state_weight:         float
    night_accident_pct:   float
    monsoon_accident_pct: float
    primary_road_type:    str
    top_weather:          str
    risk_score:           float = Field(..., ge=0, le=100)
    risk_level:           str   = Field(..., pattern="^(HIGH|MEDIUM|LOW)$")
    distance_m:           Optional[int] = None   # added on proximity queries


class Waypoint(BaseModel):
    """A single lat/lng coordinate for route queries."""
    lat: float = Field(..., ge=-90,  le=90)
    lng: float = Field(..., ge=-180, le=180)


# ── /nearby request & response ────────────────────────────────────────────────

class NearbyResponse(BaseModel):
    alert:       bool
    alert_level: str   = Field(..., pattern="^(HIGH|MEDIUM|SAFE)$")
    top_score:   float
    black_spots: List[BlackSpot]
    reason:      str
    queried_at:  Optional[str] = None


# ── /route-risk ───────────────────────────────────────────────────────────────

class RouteRiskRequest(BaseModel):
    """
    List of waypoints defining the route.
    Minimum 2 waypoints required.
    """
    waypoints: List[Waypoint] = Field(..., min_length=2)
    radius_m:  int            = Field(default=300, ge=50, le=2000)
    min_risk:  float          = Field(default=40.0, ge=0, le=100)

    @field_validator("waypoints")
    @classmethod
    def at_least_two(cls, v):
        if len(v) < 2:
            raise ValueError("Route must have at least 2 waypoints")
        return v


class RouteRiskResponse(BaseModel):
    overall_risk_score:  float
    risk_level:          str
    total_blackspots:    int
    high_risk_zones:     int
    blackspots_on_route: List[BlackSpot]


# ── /weather-risk ─────────────────────────────────────────────────────────────

class WeatherRiskResponse(BaseModel):
    lat:             float
    lng:             float
    weather_desc:    str
    temperature_c:   Optional[float] = None
    visibility_m:    Optional[int]   = None
    wind_speed_mps:  Optional[float] = None
    weather_risk:    int             = Field(..., ge=1, le=4)
    risk_label:      str
    advice:          str


# ── /stats ────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_blackspots:        int
    high_risk_count:         int
    medium_risk_count:       int
    low_risk_count:          int
    total_accidents_indexed: int
    total_killed_indexed:    int
    avg_risk_score:          float
    max_risk_score:          float
    min_risk_score:          float
    road_type_distribution:  dict
    top_10_blackspots:       List[BlackSpot]


# ── /all blackspots ───────────────────────────────────────────────────────────

class AllBlackspotsResponse(BaseModel):
    count:       int
    risk_level:  Optional[str] = None
    blackspots:  List[BlackSpot]


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:           str
    blackspots_loaded: int
    version:          str