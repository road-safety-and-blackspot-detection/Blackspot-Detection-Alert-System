"""
api/dependencies.py
────────────────────
FastAPI dependency injection layer.

The Predictor is loaded ONCE at server startup and reused
across all requests. This means blackspots.json is read from
disk exactly once — not on every API call.

All routers inject the predictor via:
    predictor: Predictor = Depends(get_predictor)
"""

import logging
import sys
from functools import lru_cache
from pathlib import Path

# Add ml-engine to path so we can import Predictor
ML_ENGINE_PATH = Path(__file__).parent.parent / "ml-engine"
sys.path.insert(0, str(ML_ENGINE_PATH))

from src.predictor import Predictor

log = logging.getLogger(__name__)

# Path to blackspots.json produced by the ML pipeline
BLACKSPOTS_PATH = str(
    Path(__file__).parent.parent / "ml-engine" / "data" / "outputs" / "blackspots.json"
)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """
    Returns the singleton Predictor instance.
    lru_cache(maxsize=1) ensures this is created exactly once
    and reused for every subsequent call.

    FastAPI calls this during the lifespan startup event so any
    load error surfaces immediately, not on the first request.
    """
    log.info(f"Loading Predictor from {BLACKSPOTS_PATH}")
    predictor = Predictor(blackspots_path=BLACKSPOTS_PATH)
    log.info(f"Predictor ready — {len(predictor):,} blackspots loaded")
    return predictor