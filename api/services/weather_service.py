"""
api/services/weather_service.py
─────────────────────────────────
Fetches live weather data from OpenWeatherMap free API
and converts it into a risk score (1–4) for the alert system.

Free tier: 60 calls/minute, no credit card needed.
API key goes in api/.env as OPENWEATHER_API_KEY=your_key

Weather → risk mapping
───────────────────────
1 = Low     (clear, sunny, cloudy)
2 = Medium  (overcast, hazy, light rain)
3 = High    (rain, drizzle, mist, windy)
4 = Extreme (fog, storm, snow, thunderstorm)
"""

import logging
import os
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

OWM_BASE_URL  = "https://api.openweathermap.org/data/2.5/weather"
API_KEY       = os.getenv("OPENWEATHER_API_KEY", "")

# OpenWeatherMap condition ID ranges → risk score
# Full list: https://openweathermap.org/weather-conditions
WEATHER_ID_RISK: dict[tuple, int] = {
    # Thunderstorm (2xx)
    (200, 299): 4,
    # Drizzle (3xx)
    (300, 321): 2,
    # Rain (5xx)
    (500, 501): 3,   # light/moderate rain
    (502, 531): 4,   # heavy/extreme rain
    # Snow (6xx)
    (600, 699): 4,
    # Atmosphere (7xx) — fog, mist, haze
    (700, 711): 3,   # mist, smoke
    (712, 721): 2,   # haze
    (731, 741): 4,   # dust, fog
    (751, 781): 4,   # sand, tornado
    # Clear (800)
    (800, 800): 1,
    # Clouds (80x)
    (801, 802): 1,   # few/scattered clouds
    (803, 804): 2,   # broken/overcast
}

RISK_ADVICE: dict[int, str] = {
    1: "Clear conditions — normal alertness.",
    2: "Reduced visibility possible — slow down.",
    3: "Wet or misty roads — increase following distance.",
    4: "Dangerous conditions — consider delaying travel.",
}

RISK_LABELS: dict[int, str] = {
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
    4: "EXTREME",
}


def _condition_id_to_risk(condition_id: int) -> int:
    """Map OpenWeatherMap weather condition ID → risk score 1–4."""
    for (low, high), risk in WEATHER_ID_RISK.items():
        if low <= condition_id <= high:
            return risk
    return 2   # default medium


async def get_weather_risk(lat: float, lng: float) -> dict:
    """
    Fetch current weather for (lat, lng) and return risk data.

    Returns
    -------
    {
        "lat":            float,
        "lng":            float,
        "weather_desc":   str,
        "temperature_c":  float,
        "visibility_m":   int,
        "wind_speed_mps": float,
        "weather_risk":   int (1–4),
        "risk_label":     str,
        "advice":         str,
    }

    Falls back gracefully if API key missing or request fails.
    """
    if not API_KEY:
        log.warning("OPENWEATHER_API_KEY not set — returning default weather risk")
        return _default_response(lat, lng)

    params = {
        "lat":   lat,
        "lon":   lng,
        "appid": API_KEY,
        "units": "metric",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OWM_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        condition_id   = data["weather"][0]["id"]
        weather_desc   = data["weather"][0]["description"].capitalize()
        temperature_c  = round(data["main"]["temp"], 1)
        visibility_m   = data.get("visibility", None)
        wind_speed_mps = round(data["wind"]["speed"], 1)

        # Night-time multiplier: dark + rain = higher risk
        is_night    = _is_night(data)
        weather_risk = _condition_id_to_risk(condition_id)
        if is_night and weather_risk >= 2:
            weather_risk = min(4, weather_risk + 1)

        return {
            "lat":            lat,
            "lng":            lng,
            "weather_desc":   weather_desc,
            "temperature_c":  temperature_c,
            "visibility_m":   visibility_m,
            "wind_speed_mps": wind_speed_mps,
            "weather_risk":   weather_risk,
            "risk_label":     RISK_LABELS[weather_risk],
            "advice":         RISK_ADVICE[weather_risk],
        }

    except httpx.TimeoutException:
        log.warning(f"OpenWeatherMap timeout for ({lat}, {lng})")
        return _default_response(lat, lng)
    except httpx.HTTPStatusError as e:
        log.warning(f"OpenWeatherMap HTTP error: {e.response.status_code}")
        return _default_response(lat, lng)
    except Exception as e:
        log.error(f"Weather service error: {e}")
        return _default_response(lat, lng)


def _is_night(owm_data: dict) -> bool:
    """Return True if current time is between sunset and sunrise."""
    try:
        import time as _time
        now     = int(_time.time())
        sunrise = owm_data["sys"]["sunrise"]
        sunset  = owm_data["sys"]["sunset"]
        return not (sunrise <= now <= sunset)
    except Exception:
        return False


def _default_response(lat: float, lng: float) -> dict:
    """Safe fallback when API is unavailable."""
    return {
        "lat":            lat,
        "lng":            lng,
        "weather_desc":   "Unknown (API unavailable)",
        "temperature_c":  None,
        "visibility_m":   None,
        "wind_speed_mps": None,
        "weather_risk":   2,
        "risk_label":     RISK_LABELS[2],
        "advice":         RISK_ADVICE[2],
    }