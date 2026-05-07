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

# import logging
# import sys
# from contextlib import asynccontextmanager
# from pathlib import Path

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# # ── Path setup ────────────────────────────────────────────────────────────────
# # Add ml-engine to sys.path so routers can import src.predictor
# ML_ENGINE_PATH = Path(__file__).parent.parent / "ml-engine"
# sys.path.insert(0, str(ML_ENGINE_PATH))

# # ── Routers ───────────────────────────────────────────────────────────────────
# from api.routers import blackspots, health, routes, weather
# from api.dependencies import get_predictor

# # ── Logging ───────────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
# )
# log = logging.getLogger(__name__)


# # ── Lifespan — runs at startup & shutdown ─────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Load the Predictor (blackspots.json) once at startup.
#     If loading fails, the server exits immediately with a clear error.
#     """
#     log.info("═" * 55)
#     log.info("  BLACK SPOT ALERT API — Starting up")
#     log.info("═" * 55)
#     try:
#         predictor = get_predictor()
#         log.info(f"  ✅ Predictor loaded — {len(predictor):,} blackspots ready")
#     except FileNotFoundError as e:
#         log.error(f"  ❌ {e}")
#         log.error("  Run ml-engine/pipeline.py first to generate blackspots.json")
#         sys.exit(1)

#     yield   # ← application runs here

#     log.info("BLACK SPOT ALERT API — Shutting down")


# # ── App ───────────────────────────────────────────────────────────────────────
# app = FastAPI(
#     title       = "Black Spot Alert API",
#     description = (
#         "Road accident blackspot detection and real-time risk alert system. "
#         "ML-powered using DBSCAN clustering on historical India accident data."
#     ),
#     version     = "1.0.0",
#     lifespan    = lifespan,
#     docs_url    = "/docs",
#     redoc_url   = "/redoc",
# )


# # ── CORS — allow mobile app on any local IP ───────────────────────────────────
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins     = ["*"],   # fine for local dev; restrict in production
#     allow_credentials = True,
#     allow_methods     = ["*"],
#     allow_headers     = ["*"],
# )


# # ── Include routers ───────────────────────────────────────────────────────────
# app.include_router(health.router)
# app.include_router(blackspots.router)
# app.include_router(routes.router)
# app.include_router(weather.router)


# # ── Root ──────────────────────────────────────────────────────────────────────
# @app.get("/", include_in_schema=False)
# def root():
#     return JSONResponse({
#         "name":    "Black Spot Alert API",
#         "version": "1.0.0",
#         "docs":    "/docs",
#         "health":  "/health",
#         "endpoints": {
#             "nearby":          "GET  /api/v1/nearby?lat=&lng=&radius=500",
#             "all_blackspots":  "GET  /api/v1/blackspots/all",
#             "stats":           "GET  /api/v1/blackspots/stats",
#             "route_risk":      "POST /api/v1/route-risk",
#             "route_compare":   "POST /api/v1/route-compare",
#             "weather_risk":    "GET  /api/v1/weather-risk?lat=&lng=",
#         },
#     })


# # ── Global exception handler ─────────────────────────────────────────────────
# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     log.error(f"Unhandled exception: {exc}", exc_info=True)
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "Internal server error", "type": type(exc).__name__},
#     )


"""
api/main.py  ← FIXED VERSION
──────────────────────────────
Key fixes:
  1. sys.path adds ml-engine BEFORE any imports so src.predictor always resolves
  2. Lifespan catches errors gracefully, doesn't crash silently
  3. CORS allows all origins explicitly including Expo dev client
  4. Root endpoint shows all working endpoints for easy testing

Run command (from BLACK-SPOT/ root):
    cd BLACK-SPOT
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ── CRITICAL: Add ml-engine to sys.path BEFORE any local imports ──────────────
# This must happen before "from api.routers import ..." lines
_ROOT       = Path(__file__).parent.parent          # BLACK-SPOT/
_ML_ENGINE  = _ROOT / "ml-engine"
sys.path.insert(0, str(_ML_ENGINE))                 # so "from src.predictor import" works
sys.path.insert(0, str(_ROOT))                      # so "from api.routers import" works

# ── Now safe to import local modules ─────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import blackspots, health, routes, weather
from api.dependencies import get_predictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("═" * 55)
    log.info("  BLACK SPOT ALERT API — Starting")
    log.info(f"  ml-engine path : {_ML_ENGINE}")
    log.info(f"  ml-engine exists: {_ML_ENGINE.exists()}")
    log.info("═" * 55)

    try:
        predictor = get_predictor()
        log.info(f"  ✅ Predictor loaded — {len(predictor):,} blackspots ready")
    except FileNotFoundError as e:
        log.error(f"  ❌ {e}")
        log.error("  Run ml-engine/pipeline.py first OR check path to blackspots.json")
        # Don't sys.exit — let server start so /health still responds with error info
    except Exception as e:
        log.error(f"  ❌ Unexpected error loading predictor: {e}", exc_info=True)

    yield

    log.info("BLACK SPOT ALERT API — Shutdown")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Black Spot Alert API",
    description = "ML-powered road accident blackspot detection for India",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — must allow ALL for local dev with Expo ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,          # must be False when allow_origins=["*"]
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["*"],
    expose_headers    = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(blackspots.router)
app.include_router(routes.router)
app.include_router(weather.router)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({
        "status":  "running",
        "version": "1.0.0",
        "test_these_endpoints": {
            "health":       "GET  /health",
            "all_spots":    "GET  /api/v1/blackspots/all",
            "stats":        "GET  /api/v1/blackspots/stats",
            "nearby":       "GET  /api/v1/nearby?lat=28.61&lng=77.20&radius=5000",
            "docs":         "GET  /docs",
        },
    })


# ── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )