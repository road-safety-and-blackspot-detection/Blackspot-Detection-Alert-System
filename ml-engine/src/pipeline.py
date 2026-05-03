"""
pipeline.py
────────────
End-to-end ML pipeline orchestrator.
Run this once after downloading datasets to regenerate all outputs.

Usage
─────
    cd ml-engine/
    python pipeline.py                    # full run
    python pipeline.py --from feature     # skip ingestion & preprocessing
    python pipeline.py --from cluster     # skip to clustering step
    python pipeline.py --eps 150 --min-samples 3   # custom DBSCAN params

Outputs produced
────────────────
    data/processed/cleaned_spatial.parquet
    data/processed/featured_spatial.parquet
    data/processed/global_feature_stats.json
    data/processed/state_risk_weights.csv
    data/processed/label_encoders.pkl
    data/outputs/blackspots.json          ← consumed by FastAPI
    data/outputs/model_metadata.json
    models/dbscan_model.pkl
    models/risk_scaler.pkl
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ── Add ml-engine root to path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion    import DataIngestion
from src.preprocessor      import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.clustering        import BlackSpotClusterer
from src.risk_scorer       import RiskScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

PROCESSED = Path("data/processed")
OUTPUTS   = Path("data/outputs")
MODELS    = Path("models")


# ── Step functions ────────────────────────────────────────────────────────────

def step_ingest_and_preprocess() -> pd.DataFrame:
    """
    Step 1 + 2: Load raw CSVs → clean → save cleaned_spatial.parquet
    Also builds calibration tables (state weights, global feature stats).
    """
    _banner("STEP 1 + 2 — DATA INGESTION & PREPROCESSING")

    ing = DataIngestion(raw_dir="data/raw")
    pp  = Preprocessor(processed_dir=str(PROCESSED))

    # Load spatial datasets
    frames = []
    for loader_name, loader in [
        ("Mendeley India",  ing.load_mendeley),
        ("Kaggle India",    ing.load_kaggle_india),
    ]:
        try:
            df = loader()
            frames.append(df)
            log.info(f"  ✓ {loader_name}: {len(df):,} rows")
        except Exception as e:
            log.warning(f"  ✗ {loader_name} skipped: {e}")

    if not frames:
        log.error("No spatial data loaded — check data/raw/ directory")
        sys.exit(1)

    merged  = pd.concat(frames, ignore_index=True)
    cleaned = pp.run(merged)
    pp.save(cleaned, "cleaned_spatial.parquet")

    # Calibration tables
    try:
        morth = ing.load_morth()
        pp.build_morth_weights(morth)
    except Exception as e:
        log.warning(f"  MoRTH weights skipped: {e}")

    try:
        global_df = ing.load_global_kaggle()
        pp.build_global_stats(global_df)
    except Exception as e:
        log.warning(f"  Global stats skipped: {e}")

    log.info(f"Step 1+2 complete — {len(cleaned):,} clean spatial rows")
    return cleaned


def step_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Add all ML features → save featured_spatial.parquet."""
    _banner("STEP 3 — FEATURE ENGINEERING")

    fe       = FeatureEngineer(processed_dir=str(PROCESSED))
    featured = fe.run(df)
    fe.save_encoders()
    fe.save(featured, "featured_spatial.parquet")

    new_cols = [c for c in featured.columns if c not in df.columns]
    log.info(f"Step 3 complete — {len(new_cols)} new features: {new_cols}")
    return featured


def step_cluster(df: pd.DataFrame, eps_meters: int, min_samples: int) -> pd.DataFrame:
    """Step 4: DBSCAN clustering → save clustered_spatial.parquet + model."""
    _banner(f"STEP 4 — DBSCAN CLUSTERING (eps={eps_meters}m, min_samples={min_samples})")

    clusterer  = BlackSpotClusterer(eps_meters=eps_meters, min_samples=min_samples)
    clustered  = clusterer.fit(df)
    clusterer.save(str(MODELS / "dbscan_model.pkl"))

    summary = clusterer.summary()
    for k, v in summary.items():
        log.info(f"  {k:<25} {v}")

    clustered.to_parquet(PROCESSED / "clustered_spatial.parquet", index=False)
    log.info(f"Step 4 complete — {summary['n_clusters']:,} blackspot clusters")
    return clustered


def step_score(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Risk score per cluster → save blackspots.json + artifacts."""
    _banner("STEP 5 — RISK SCORING")

    scorer     = RiskScorer()
    blackspots = scorer.score(df)
    scorer.save(
        blackspots,
        json_path=str(OUTPUTS / "blackspots.json"),
        scaler_path=str(MODELS / "risk_scaler.pkl"),
        metadata_path=str(OUTPUTS / "model_metadata.json"),
    )

    log.info(f"Step 5 complete — {len(blackspots):,} blackspots scored")
    return blackspots


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    start_from:  str = "ingest",
    eps_meters:  int = 200,
    min_samples: int = 5,
):
    """
    Run the full pipeline or resume from a given step.

    Parameters
    ----------
    start_from  : "ingest" | "feature" | "cluster" | "score"
    eps_meters  : DBSCAN eps radius in metres
    min_samples : DBSCAN minimum cluster size
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _banner("BLACK SPOT DETECTION — ML PIPELINE", char="═")

    steps = ["ingest", "feature", "cluster", "score"]
    start_idx = steps.index(start_from) if start_from in steps else 0

    df = None

    # ── Step 1+2: Ingest & preprocess ────────────────────────────────────────
    if start_idx <= 0:
        df = step_ingest_and_preprocess()
    else:
        df = pd.read_parquet(PROCESSED / "cleaned_spatial.parquet")
        log.info(f"Loaded cleaned_spatial.parquet: {df.shape}")

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    if start_idx <= 1:
        df = step_feature_engineering(df)
    else:
        df = pd.read_parquet(PROCESSED / "featured_spatial.parquet")
        log.info(f"Loaded featured_spatial.parquet: {df.shape}")

    # ── Step 4: Clustering ────────────────────────────────────────────────────
    if start_idx <= 2:
        df = step_cluster(df, eps_meters, min_samples)
    else:
        df = pd.read_parquet(PROCESSED / "clustered_spatial.parquet")
        log.info(f"Loaded clustered_spatial.parquet: {df.shape}")

    # ── Step 5: Risk scoring ──────────────────────────────────────────────────
    blackspots = step_score(df)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _banner("PIPELINE COMPLETE", char="═")

    with open(OUTPUTS / "model_metadata.json") as f:
        meta = json.load(f)

    log.info(f"  Total blackspots : {meta['total_blackspots']:,}")
    log.info(f"  HIGH risk        : {meta['high_risk_count']:,}")
    log.info(f"  MEDIUM risk      : {meta['medium_risk_count']:,}")
    log.info(f"  LOW risk         : {meta['low_risk_count']:,}")
    log.info(f"  Elapsed          : {elapsed:.1f}s")
    log.info(f"  Output           : data/outputs/blackspots.json  ← API reads this")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str, char: str = "─"):
    width = 60
    log.info(char * width)
    log.info(f"  {text}")
    log.info(char * width)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Black Spot Detection — ML Pipeline"
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        choices=["ingest", "feature", "cluster", "score"],
        default="ingest",
        help="Resume pipeline from this step (default: ingest)",
    )
    parser.add_argument(
        "--eps",
        type=int,
        default=200,
        help="DBSCAN eps radius in metres (default: 200)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples (default: 5)",
    )

    args = parser.parse_args()
    run_pipeline(
        start_from=args.start_from,
        eps_meters=args.eps,
        min_samples=args.min_samples,
    )