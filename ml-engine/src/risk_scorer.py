"""
src/risk_scorer.py
───────────────────
Computes a 0–100 risk score for each DBSCAN cluster.

Risk formula (components sum to 1.0)
─────────────────────────────────────
  accident_frequency   30%   log-scaled count in cluster
  severity_score       25%   avg severity (1–4) normalised
  fatality_score       25%   log-scaled killed count
  env_risk_score       10%   composite of speed/weather/road/lighting
  state_weight_bonus   10%   MoRTH state-level risk multiplier

Output schema per blackspot
────────────────────────────
  cluster_id, lat, lng, accident_count, total_killed,
  avg_severity, avg_env_risk, state_weight,
  night_accident_pct, monsoon_accident_pct,
  primary_road_type, top_weather,
  risk_score (0–100), risk_level (HIGH/MEDIUM/LOW)

Usage
─────
    from src.risk_scorer import RiskScorer
    scorer = RiskScorer()
    blackspots_df = scorer.score(clustered_df)
    scorer.save(blackspots_df)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)

OUTPUTS = Path("data/outputs")
MODELS  = Path("models")


class RiskScorer:
    """
    Scores each DBSCAN cluster with a 0–100 risk index.
    """

    # Risk level thresholds
    HIGH_THRESHOLD   = 70.0
    MEDIUM_THRESHOLD = 40.0

    def __init__(self):
        self._scaler: Optional[MinMaxScaler] = None

    # ── Main entry point ──────────────────────────────────────────────────────
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute risk scores for all clusters in df.
        Noise points (cluster == -1) are excluded.

        Parameters
        ----------
        df : clustered DataFrame with 'cluster' column from BlackSpotClusterer

        Returns
        -------
        DataFrame with one row per blackspot cluster, sorted by risk_score desc
        """
        clustered = df[df["cluster"] >= 0].copy()
        n_clusters = clustered["cluster"].nunique()
        log.info(f"RiskScorer.score() — scoring {n_clusters:,} clusters")

        records = []
        for cid, group in clustered.groupby("cluster"):
            record = self._score_cluster(int(cid), group)
            records.append(record)

        result = pd.DataFrame(records)

        # Normalise raw scores → 0–100
        self._scaler = MinMaxScaler(feature_range=(0, 100))
        result["risk_score"] = (
            self._scaler.fit_transform(result[["raw_score"]])
            .round(1)
            .flatten()
        )
        result = result.drop(columns=["raw_score"])

        # Risk level labels
        result["risk_level"] = result["risk_score"].apply(self._risk_label)

        # Sort highest risk first
        result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)

        # Summary log
        high   = (result["risk_level"] == "HIGH").sum()
        medium = (result["risk_level"] == "MEDIUM").sum()
        low    = (result["risk_level"] == "LOW").sum()
        log.info(f"  HIGH={high} | MEDIUM={medium} | LOW={low}")
        log.info(f"  Risk range: {result['risk_score'].min():.1f} – "
                 f"{result['risk_score'].max():.1f}")
        return result

    # ── Per-cluster scoring ───────────────────────────────────────────────────
    def _score_cluster(self, cid: int, group: pd.DataFrame) -> dict:
        """Compute raw score and metadata for a single cluster."""
        n_accidents    = len(group)
        center_lat     = float(group["latitude"].mean())
        center_lng     = float(group["longitude"].mean())
        avg_severity   = float(group["severity"].mean()) if "severity" in group.columns else 2.0
        total_killed   = int(group["killed"].sum()) if "killed" in group.columns else 0
        avg_env_risk   = float(group["env_risk_score"].mean()) if "env_risk_score" in group.columns else 2.0
        state_weight   = float(group["state_risk_weight"].mean()) if "state_risk_weight" in group.columns else 1.0
        night_pct      = float(group["is_night"].mean() * 100) if "is_night" in group.columns else 0.0
        monsoon_pct    = float(group["is_monsoon"].mean() * 100) if "is_monsoon" in group.columns else 0.0

        # Most common road type and weather condition
        primary_road = "unknown"
        if "road_type" in group.columns and len(group) > 0:
            primary_road = str(group["road_type"].mode().iloc[0])

        top_weather = "unknown"
        if "weather" in group.columns and len(group) > 0:
            top_weather = str(group["weather"].mode().iloc[0])

        # ── Raw score formula ────────────────────────────────────────────────
        freq_score     = np.log1p(n_accidents)            # log scale
        severity_score = avg_severity / 4.0               # 0-1
        fatality_score = np.log1p(total_killed) / 5.0     # 0-1 (approx)
        env_score      = avg_env_risk / 4.0               # 0-1
        state_bonus    = (state_weight - 1.0)             # 0-1

        raw = (
            freq_score     * 0.30 +
            severity_score * 0.25 +
            fatality_score * 0.25 +
            env_score      * 0.10 +
            state_bonus    * 0.10
        )

        return {
            "cluster_id":            cid,
            "lat":                   round(center_lat, 6),
            "lng":                   round(center_lng, 6),
            "accident_count":        n_accidents,
            "total_killed":          total_killed,
            "avg_severity":          round(avg_severity, 2),
            "avg_env_risk":          round(avg_env_risk, 2),
            "state_weight":          round(state_weight, 2),
            "night_accident_pct":    round(night_pct, 1),
            "monsoon_accident_pct":  round(monsoon_pct, 1),
            "primary_road_type":     primary_road,
            "top_weather":           top_weather,
            "raw_score":             raw,
        }

    def _risk_label(self, score: float) -> str:
        if score >= self.HIGH_THRESHOLD:   return "HIGH"
        if score >= self.MEDIUM_THRESHOLD: return "MEDIUM"
        return "LOW"

    # ── Save outputs ──────────────────────────────────────────────────────────
    def save(
        self,
        df: pd.DataFrame,
        json_path:     str = "data/outputs/blackspots.json",
        scaler_path:   str = "models/risk_scaler.pkl",
        metadata_path: str = "data/outputs/model_metadata.json",
    ):
        """
        Save blackspots.json (read by FastAPI),
        risk_scaler.pkl, and model_metadata.json.
        """
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        MODELS.mkdir(parents=True, exist_ok=True)

        # blackspots.json — the main API data file
        records = df.to_dict(orient="records")
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)
        log.info(f"  Saved blackspots.json ({len(records):,} blackspots) → {json_path}")

        # risk_scaler.pkl
        if self._scaler:
            joblib.dump(self._scaler, scaler_path)
            log.info(f"  Saved risk_scaler.pkl → {scaler_path}")

        # model_metadata.json
        metadata = {
            "total_blackspots":  len(df),
            "high_risk_count":   int((df["risk_level"] == "HIGH").sum()),
            "medium_risk_count": int((df["risk_level"] == "MEDIUM").sum()),
            "low_risk_count":    int((df["risk_level"] == "LOW").sum()),
            "risk_score_mean":   round(float(df["risk_score"].mean()), 2),
            "risk_score_max":    round(float(df["risk_score"].max()), 2),
            "risk_score_min":    round(float(df["risk_score"].min()), 2),
            "thresholds": {
                "HIGH":   self.HIGH_THRESHOLD,
                "MEDIUM": self.MEDIUM_THRESHOLD,
            },
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info(f"  Saved model_metadata.json → {metadata_path}")

        return json_path

    def save_scaler(self, path: str = "models/risk_scaler.pkl"):
        if self._scaler:
            joblib.dump(self._scaler, path)

    @classmethod
    def load_blackspots(cls, path: str = "data/outputs/blackspots.json") -> list[dict]:
        """Load blackspots.json as a list of dicts (used by predictor and API)."""
        with open(path) as f:
            return json.load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    df = pd.read_parquet("data/processed/clustered_spatial.parquet")

    scorer     = RiskScorer()
    blackspots = scorer.score(df)
    scorer.save(blackspots)

    print(f"\n── Top 10 highest-risk blackspots ──────────────────────")
    print(blackspots[
        ["cluster_id","lat","lng","accident_count",
         "total_killed","risk_score","risk_level","primary_road_type"]
    ].head(10).to_string(index=False))