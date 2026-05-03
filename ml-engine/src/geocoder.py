"""
src/geocoder.py
───────────────
Static geocoder — converts city/state names → lat/lng.
Zero API calls. Instant. Deterministic (seeded jitter).

Used by:
  data_ingestion.py  (Kaggle India rows that have no coordinates)
  pipeline.py        (end-to-end run)
"""

import numpy as np
from typing import Optional, Tuple

# ── Named cities present in Kaggle India dataset ──────────────────────────────
CITY_COORDS: dict[str, Tuple[float, float]] = {
    "Ahmedabad":      (23.0225,  72.5714),
    "Bangalore":      (12.9716,  77.5946),
    "Chennai":        (13.0827,  80.2707),
    "Coimbatore":     (11.0168,  76.9558),
    "Durgapur":       (23.5204,  87.3119),
    "Dwarka":         (22.2394,  68.9678),
    "Jaipur":         (26.9124,  75.7873),
    "Jodhpur":        (26.2389,  73.0243),
    "Kanpur":         (26.4499,  80.3319),
    "Kolkata":        (22.5726,  88.3639),
    "Lucknow":        (26.8467,  80.9462),
    "Madurai":        (9.9252,   78.1198),
    "Mangalore":      (12.9141,  74.8560),
    "Mumbai":         (19.0760,  72.8777),
    "Mysore":         (12.2958,  76.6394),
    "Nagpur":         (21.1458,  79.0882),
    "New Delhi":      (28.6139,  77.2090),
    "Pune":           (18.5204,  73.8567),
    "Rohini":         (28.7041,  77.1025),
    "Siliguri":       (26.7271,  88.3953),
    "Surat":          (21.1702,  72.8311),
    "Tirupati":       (13.6288,  79.4192),
    "Udaipur":        (24.5854,  73.7125),
    "Vadodara":       (22.3072,  73.1812),
    "Varanasi":       (25.3176,  82.9739),
    "Vijayawada":     (16.5062,  80.6480),
    "Visakhapatnam":  (17.6868,  83.2185),
}

# ── State capitals — fallback for "Unknown" city rows ─────────────────────────
STATE_CAPITALS: dict[str, Tuple[float, float]] = {
    "Andhra Pradesh":       (15.9129,  79.7400),
    "Arunachal Pradesh":    (27.0844,  93.6053),
    "Assam":                (26.1445,  91.7362),
    "Bihar":                (25.5941,  85.1376),
    "Chandigarh":           (30.7333,  76.7794),
    "Chhattisgarh":         (21.2514,  81.6296),
    "Delhi":                (28.6139,  77.2090),
    "Goa":                  (15.2993,  74.1240),
    "Gujarat":              (23.0225,  72.5714),
    "Haryana":              (29.0588,  76.0856),
    "Himachal Pradesh":     (31.1048,  77.1734),
    "Jammu and Kashmir":    (34.0837,  74.7973),
    "Jharkhand":            (23.3441,  85.3096),
    "Karnataka":            (12.9716,  77.5946),
    "Kerala":               (8.5241,   76.9366),
    "Madhya Pradesh":       (23.2599,  77.4126),
    "Maharashtra":          (19.0760,  72.8777),
    "Manipur":              (24.6637,  93.9063),
    "Meghalaya":            (25.5788,  91.8933),
    "Mizoram":              (23.1645,  92.9376),
    "Nagaland":             (25.6751,  94.1086),
    "Odisha":               (20.2961,  85.8245),
    "Puducherry":           (11.9416,  79.8083),
    "Punjab":               (30.7333,  76.7794),
    "Rajasthan":            (26.9124,  75.7873),
    "Sikkim":               (27.3314,  88.6138),
    "Tamil Nadu":           (13.0827,  80.2707),
    "Telangana":            (17.3850,  78.4867),
    "Tripura":              (23.9408,  91.9882),
    "Uttar Pradesh":        (26.8467,  80.9462),
    "Uttarakhand":          (30.3165,  78.0322),
    "West Bengal":          (22.5726,  88.3639),
}

# Seed so jitter is reproducible across runs
_rng = np.random.default_rng(seed=42)


def geocode(
    city: Optional[str],
    state: Optional[str],
    jitter: float = 0.25,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (latitude, longitude) for a city + state pair.

    Resolution order:
      1. Named city in CITY_COORDS dict          → city-level precision
      2. State in STATE_CAPITALS dict            → state-capital fallback
      3. Neither matched                         → (None, None)

    A small random offset (jitter) is added so duplicate city/state rows
    don't all stack on the exact same pixel on the map.

    Parameters
    ----------
    city    : city name string (may be "Unknown")
    state   : state name string
    jitter  : max degrees of random offset (default ±0.25°  ≈ ±28 km)

    Returns
    -------
    (lat, lng) or (None, None) if not resolvable
    """
    city  = str(city).strip()  if city  else ""
    state = str(state).strip() if state else ""

    lat, lng = None, None

    if city and city.lower() != "unknown" and city in CITY_COORDS:
        lat, lng = CITY_COORDS[city]
    elif state and state in STATE_CAPITALS:
        lat, lng = STATE_CAPITALS[state]
    else:
        return None, None

    lat += float(_rng.uniform(-jitter, jitter))
    lng += float(_rng.uniform(-jitter, jitter))
    return round(lat, 6), round(lng, 6)


def geocode_dataframe(df, city_col: str = "City Name",
                       state_col: str = "State Name",
                       jitter: float = 0.25):
    """
    Add latitude/longitude columns to a DataFrame using static geocoding.

    Parameters
    ----------
    df        : pandas DataFrame
    city_col  : column name containing city names
    state_col : column name containing state names
    jitter    : coordinate scatter radius in degrees

    Returns
    -------
    DataFrame with added 'latitude' and 'longitude' columns
    """
    import pandas as pd
    df = df.copy()
    coords = [
        geocode(row.get(city_col, ""), row.get(state_col, ""), jitter)
        for _, row in df.iterrows()
    ]
    df["latitude"]  = [c[0] for c in coords]
    df["longitude"] = [c[1] for c in coords]

    resolved   = df["latitude"].notna().sum()
    unresolved = df["latitude"].isna().sum()
    print(f"  Geocoding complete: {resolved:,} resolved | {unresolved:,} unresolved")
    return df


if __name__ == "__main__":
    # Quick smoke-test
    tests = [
        ("Lucknow",  "Uttar Pradesh"),
        ("Unknown",  "Punjab"),
        ("Unknown",  "Maharashtra"),
        ("Bangalore","Karnataka"),
        ("Unknown",  "INVALID_STATE"),
    ]
    print(f"{'City':<15} {'State':<25} {'Lat':>10} {'Lng':>10}")
    print("-" * 65)
    for city, state in tests:
        lat, lng = geocode(city, state)
        print(f"{city:<15} {state:<25} {str(lat):>10} {str(lng):>10}")