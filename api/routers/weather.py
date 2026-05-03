"""
api/routers/weather.py
──────────────────────
Live weather risk endpoint.
Fetches current weather from OpenWeatherMap and returns
a risk score that the mobile app overlays on alerts.

Endpoints
─────────
GET /api/v1/weather-risk
    → Current weather + risk score for a GPS point
    → Used by Alert Screen to show WHY it's dangerous
      e.g. "Foggy conditions ahead — risk level EXTREME"
"""

import logging

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_predictor
from api.schemas.models import WeatherRiskResponse
from api.services.weather_service import get_weather_risk
from src.predictor import Predictor

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["weather"])


@router.get(
    "/weather-risk",
    response_model=WeatherRiskResponse,
    summary="Live weather risk for a GPS location",
)
async def weather_risk(
    lat: float = Query(..., ge=-90,  le=90,  description="Latitude"),
    lng: float = Query(..., ge=-180, le=180, description="Longitude"),
    predictor: Predictor = Depends(get_predictor),   # kept for future combined scoring
) -> WeatherRiskResponse:
    """
    **Alert Screen supporting endpoint.**

    Returns live weather conditions and a risk score (1–4) for any location.
    The mobile app calls this alongside `/nearby` so the alert can say:

    > ⚠️ High Accident Zone Ahead
    > Sharp turn + **Foggy conditions (risk: EXTREME)**

    **Risk levels:**
    | Score | Label   | Conditions                        |
    |-------|---------|-----------------------------------|
    | 1     | LOW     | Clear, sunny, partly cloudy       |
    | 2     | MEDIUM  | Overcast, hazy, light drizzle     |
    | 3     | HIGH    | Rain, mist, strong wind           |
    | 4     | EXTREME | Fog, thunderstorm, snow, tornado  |

    Night-time automatically raises the risk by 1 level.

    **Returns default MEDIUM risk if:**
    - OPENWEATHER_API_KEY is not set in .env
    - API call times out (5 second timeout)
    - Rate limit exceeded
    """
    log.debug(f"GET /weather-risk lat={lat} lng={lng}")
    result = await get_weather_risk(lat=lat, lng=lng)
    return WeatherRiskResponse(**result)