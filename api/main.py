#This file will contain the api 
"""
api/main.py
────────────
FastAPI application entry point.

Run the API
───────────
    cd api/
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    # Access interactive docs at:
    # http://localhost:8000/docs       ← Swagger UI
    # http://localhost:8000/redoc      ← ReDoc

Mobile app connection
─────────────────────
    Your phone and laptop must be on the same WiFi network.
    Find your laptop's local IP:
        Windows: ipconfig  → look for IPv4 Address
        Mac/Linux: ifconfig / ip a → look for 192.168.x.x

    Set API_URL in mobile-app/app/constants/config.ts:
        export const CONFIG = { API_URL: "http://192.168.1.XX:8000" }

API endpoints summary
─────────────────────
    GET  /health                    → server status
    GET  /ping                      → liveness probe
    GET  /api/v1/nearby             → blackspots near GPS + alert
    GET  /api/v1/blackspots/all     → all zones for map screen
    GET  /api/v1/blackspots/stats   → dashboard statistics
    GET  /api/v1/blackspots/{id}    → single zone detail
    POST /api/v1/route-risk         → score a route
    POST /api/v1/route-compare      → compare two routes
    GET  /api/v1/weather-risk       → live weather risk
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Path setup ────────────────────────────────────────────────────────────────
# Add ml-engine to sys.path so routers can import src.predictor
ML_ENGINE_PATH = Path(__file__).parent.parent / "ml-engine"
sys.path.insert(0, str(ML_ENGINE_PATH))

# ── Routers ───────────────────────────────────────────────────────────────────
from api.routers import blackspots, health, routes, weather
from api.dependencies import get_predictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# ── Lifespan — runs at startup & shutdown ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the Predictor (blackspots.json) once at startup.
    If loading fails, the server exits immediately with a clear error.
    """
    log.info("═" * 55)
    log.info("  BLACK SPOT ALERT API — Starting up")
    log.info("═" * 55)
    try:
        predictor = get_predictor()
        log.info(f"  ✅ Predictor loaded — {len(predictor):,} blackspots ready")
    except FileNotFoundError as e:
        log.error(f"  ❌ {e}")
        log.error("  Run ml-engine/pipeline.py first to generate blackspots.json")
        sys.exit(1)

    yield   # ← application runs here

    log.info("BLACK SPOT ALERT API — Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Black Spot Alert API",
    description = (
        "Road accident blackspot detection and real-time risk alert system. "
        "ML-powered using DBSCAN clustering on historical India accident data."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ── CORS — allow mobile app on any local IP ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # fine for local dev; restrict in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Include routers ───────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(blackspots.router)
app.include_router(routes.router)
app.include_router(weather.router)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({
        "name":    "Black Spot Alert API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "endpoints": {
            "nearby":          "GET  /api/v1/nearby?lat=&lng=&radius=500",
            "all_blackspots":  "GET  /api/v1/blackspots/all",
            "stats":           "GET  /api/v1/blackspots/stats",
            "route_risk":      "POST /api/v1/route-risk",
            "route_compare":   "POST /api/v1/route-compare",
            "weather_risk":    "GET  /api/v1/weather-risk?lat=&lng=",
        },
    })


# ── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )