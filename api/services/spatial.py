"""
api/services/spatial.py
────────────────────────
Spatial utility functions used across routers.
All geometry is pure Python — no external dependencies.
"""

import math
from typing import List, Tuple

EARTH_RADIUS_M = 6_371_000


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Great-circle distance between two GPS points in metres.
    """
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lng2 - lng1)
    a  = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def filter_by_radius(
    spots:    List[dict],
    lat:      float,
    lng:      float,
    radius_m: int,
) -> List[dict]:
    """
    Filter a list of blackspot dicts to those within radius_m metres.
    Adds 'distance_m' field to each returned spot.
    Results sorted by distance ascending.
    """
    results = []
    for spot in spots:
        dist = haversine(lat, lng, spot["lat"], spot["lng"])
        if dist <= radius_m:
            results.append({**spot, "distance_m": int(round(dist))})
    results.sort(key=lambda x: x["distance_m"])
    return results


def bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Initial compass bearing from point 1 to point 2, in degrees (0–360).
    """
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lng2 - lng1)
    x  = math.sin(Δλ) * math.cos(φ2)
    y  = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    θ  = math.atan2(x, y)
    return (math.degrees(θ) + 360) % 360


def interpolate_route(
    waypoints: List[Tuple[float, float]],
    step_m:    int = 100,
) -> List[Tuple[float, float]]:
    """
    Densify a route by interpolating points every step_m metres.
    Used for more thorough route risk scanning.

    Parameters
    ----------
    waypoints : list of (lat, lng) tuples
    step_m    : interpolation step size in metres

    Returns
    -------
    Dense list of (lat, lng) tuples
    """
    if len(waypoints) < 2:
        return waypoints

    dense = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        lat1, lng1 = waypoints[i]
        lat2, lng2 = waypoints[i + 1]
        dist = haversine(lat1, lng1, lat2, lng2)
        n_steps = max(1, int(dist / step_m))
        for j in range(1, n_steps + 1):
            frac = j / n_steps
            mid_lat = lat1 + frac * (lat2 - lat1)
            mid_lng = lng1 + frac * (lng2 - lng1)
            dense.append((mid_lat, mid_lng))

    return dense


def bounding_box(lat: float, lng: float, radius_m: int) -> dict:
    """
    Compute approximate bounding box for a radius around a point.
    Useful for quick pre-filtering before haversine.
    """
    lat_delta = math.degrees(radius_m / EARTH_RADIUS_M)
    lng_delta = math.degrees(
        radius_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))
    )
    return {
        "min_lat": lat - lat_delta,
        "max_lat": lat + lat_delta,
        "min_lng": lng - lng_delta,
        "max_lng": lng + lng_delta,
    }