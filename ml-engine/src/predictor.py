"""
src/predictor.py
─────────────────
Loads blackspots.json at startup and answers:
  • Proximity queries — blackspots near a GPS point
  • Route risk scoring — risk along a list of waypoints
  • Stats queries — summary of all blackspots

This is the bridge between the ML output and the FastAPI layer.
The API imports Predictor and calls its methods — no ML computation
happens at request time. Everything is pre-computed.

Usage
─────
    from src.predictor import Predictor
    predictor = Predictor()          # loads blackspots.json once
    results = predictor.nearby(28.6, 77.2, radius_m=500)
    route   = predictor.route_risk([(28.6, 77.2), (28.7, 77.3)])
    stats   = predictor.stats()
"""

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000

# Default path — matches pipeline.py output
DEFAULT_BLACKSPOTS_PATH = "data/outputs/blackspots.json"


# ── Geometry helpers ──────────────────────────────────────────────────────────
def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Great-circle distance between two GPS points in metres.
    """
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lng2 - lng1)
    a  = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class Predictor:
    """
    Serves blackspot predictions from pre-computed blackspots.json.
    Thread-safe — load once, share across requests.
    """

    def __init__(self, blackspots_path: str = DEFAULT_BLACKSPOTS_PATH):
        path = Path(blackspots_path)
        if not path.exists():
            raise FileNotFoundError(
                f"blackspots.json not found at {path.resolve()}\n"
                "Run pipeline.py first to generate it."
            )
        with open(path) as f:
            self._blackspots: list[dict] = json.load(f)

        log.info(f"Predictor loaded {len(self._blackspots):,} blackspots "
                 f"from {path.name}")

    # ── Core query: nearby blackspots ─────────────────────────────────────────
    def nearby(
        self,
        lat:       float,
        lng:       float,
        radius_m:  int   = 500,
        limit:     int   = 10,
        min_risk:  float = 0.0,
    ) -> list[dict]:
        """
        Return blackspots within radius_m metres of (lat, lng).
        Sorted by risk_score descending.

        Parameters
        ----------
        lat      : user latitude
        lng      : user longitude
        radius_m : search radius in metres
        limit    : max results to return
        min_risk : minimum risk_score to include (0–100)

        Returns
        -------
        List of blackspot dicts with added 'distance_m' field
        """
        results = []
        for spot in self._blackspots:
            dist = haversine(lat, lng, spot["lat"], spot["lng"])
            if dist <= radius_m and spot["risk_score"] >= min_risk:
                results.append({**spot, "distance_m": int(round(dist))})

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results[:limit]

    # ── Alert check ───────────────────────────────────────────────────────────
    def alert(
        self,
        lat:              float,
        lng:              float,
        radius_m:         int   = 500,
        high_threshold:   float = 70.0,
        medium_threshold: float = 40.0,
    ) -> dict:
        """
        Check whether current location triggers an alert.

        Returns
        -------
        {
          "alert":       bool,
          "alert_level": "HIGH" | "MEDIUM" | "SAFE",
          "top_score":   float,
          "black_spots": [...],
          "reason":      str   (human-readable explanation)
        }
        """
        nearby = self.nearby(lat, lng, radius_m=radius_m)

        if not nearby:
            return {
                "alert":       False,
                "alert_level": "SAFE",
                "top_score":   0.0,
                "black_spots": [],
                "reason":      "No accident-prone zones detected nearby.",
            }

        top = nearby[0]
        top_score = top["risk_score"]

        if top_score >= high_threshold:
            level  = "HIGH"
            alert  = True
            reason = self._build_reason(top)
        elif top_score >= medium_threshold:
            level  = "MEDIUM"
            alert  = True
            reason = f"Moderate accident zone {top['distance_m']}m away. Drive carefully."
        else:
            level  = "SAFE"
            alert  = False
            reason = "Low-risk area. Stay alert."

        return {
            "alert":       alert,
            "alert_level": level,
            "top_score":   top_score,
            "black_spots": nearby,
            "reason":      reason,
        }

    # ── Route risk scoring ────────────────────────────────────────────────────
    def route_risk(
        self,
        waypoints:   list[tuple[float, float]],
        radius_m:    int   = 300,
        min_risk:    float = 40.0,
    ) -> dict:
        """
        Score a route defined by a list of (lat, lng) waypoints.

        Parameters
        ----------
        waypoints : list of (lat, lng) tuples along the route
        radius_m  : how wide a corridor to scan per waypoint
        min_risk  : minimum risk to flag a zone

        Returns
        -------
        {
          "overall_risk_score": float,
          "risk_level":         str,
          "total_blackspots":   int,
          "high_risk_zones":    int,
          "blackspots_on_route": [...]  unique blackspots along route
        }
        """
        seen_ids = set()
        route_spots = []

        for lat, lng in waypoints:
            nearby = self.nearby(lat, lng, radius_m=radius_m, min_risk=min_risk)
            for spot in nearby:
                cid = spot["cluster_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    route_spots.append(spot)

        route_spots.sort(key=lambda x: x["risk_score"], reverse=True)

        if route_spots:
            # Weighted average — higher-scored zones count more
            scores  = np.array([s["risk_score"] for s in route_spots])
            weights = scores / scores.sum()
            overall = float(np.dot(scores, weights))
        else:
            overall = 0.0

        high_zones = sum(1 for s in route_spots if s["risk_score"] >= 70)

        return {
            "overall_risk_score":  round(overall, 1),
            "risk_level":          self._risk_label(overall),
            "total_blackspots":    len(route_spots),
            "high_risk_zones":     high_zones,
            "blackspots_on_route": route_spots[:20],   # cap at 20
        }

    # ── All blackspots (for map screen) ───────────────────────────────────────
    def all_blackspots(
        self,
        risk_level: Optional[str] = None,
        limit:      int           = 500,
    ) -> list[dict]:
        """
        Return all (or filtered) blackspots.

        Parameters
        ----------
        risk_level : filter by "HIGH", "MEDIUM", "LOW" (None = all)
        limit      : max results
        """
        spots = self._blackspots
        if risk_level:
            spots = [s for s in spots if s["risk_level"] == risk_level.upper()]
        return spots[:limit]

    # ── Dashboard stats ───────────────────────────────────────────────────────
    def stats(self) -> dict:
        """
        Aggregate statistics for the Dashboard screen.
        """
        spots = self._blackspots
        if not spots:
            return {}

        scores = [s["risk_score"]     for s in spots]
        killed = [s["total_killed"]   for s in spots]
        counts = [s["accident_count"] for s in spots]

        road_types = {}
        for s in spots:
            rt = s.get("primary_road_type", "unknown")
            road_types[rt] = road_types.get(rt, 0) + 1

        return {
            "total_blackspots":        len(spots),
            "high_risk_count":         sum(1 for s in spots if s["risk_level"] == "HIGH"),
            "medium_risk_count":       sum(1 for s in spots if s["risk_level"] == "MEDIUM"),
            "low_risk_count":          sum(1 for s in spots if s["risk_level"] == "LOW"),
            "total_accidents_indexed": sum(counts),
            "total_killed_indexed":    sum(killed),
            "avg_risk_score":          round(float(np.mean(scores)), 1),
            "max_risk_score":          round(float(max(scores)), 1),
            "min_risk_score":          round(float(min(scores)), 1),
            "road_type_distribution":  dict(
                sorted(road_types.items(), key=lambda x: -x[1])
            ),
            "top_10_blackspots": sorted(spots, key=lambda x: -x["risk_score"])[:10],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _build_reason(spot: dict) -> str:
        parts = []
        if spot.get("primary_road_type", "unknown") not in ("unknown", ""):
            parts.append(spot["primary_road_type"].title())
        if spot.get("night_accident_pct", 0) > 40:
            parts.append("high night-time risk")
        if spot.get("monsoon_accident_pct", 0) > 40:
            parts.append("monsoon season hotspot")
        if spot.get("top_weather", "unknown") not in ("unknown", ""):
            parts.append(f"{spot['top_weather']} conditions")
        if spot.get("total_killed", 0) > 5:
            parts.append(f"{spot['total_killed']} fatalities recorded")

        base = f"⚠️ High accident zone {spot.get('distance_m', '?')}m ahead"
        if parts:
            return base + " — " + ", ".join(parts[:3]) + "."
        return base + "."

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= 70: return "HIGH"
        if score >= 40: return "MEDIUM"
        return "LOW"

    def __len__(self):
        return len(self._blackspots)

    def __repr__(self):
        return f"Predictor(blackspots={len(self._blackspots):,})"


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = Predictor()
    print(p)

    # Test with Ludhiana, Punjab coordinates
    test_lat, test_lng = 30.9010, 75.8573
    print(f"\nTest: nearby blackspots to ({test_lat}, {test_lng})")
    nearby = p.nearby(test_lat, test_lng, radius_m=5000)
    print(f"  Found {len(nearby)} within 5km")
    for s in nearby[:3]:
        print(f"    risk={s['risk_score']} dist={s['distance_m']}m "
              f"road={s['primary_road_type']}")

    print("\nStats:")
    stats = p.stats()
    for k, v in stats.items():
        if k != "top_10_blackspots":
            print(f"  {k}: {v}")