"""
src/feature_engineering.py
───────────────────────────
Creates all ML features from the cleaned spatial DataFrame.

Feature groups produced
───────────────────────
Spatial   : h3_r9, h3_r8, hex_density_r9, hex_density_r8,
            hex_avg_severity, hex_total_killed
Time      : hour_bucket, is_night, is_monsoon, is_peak_hour
Risk      : speed_risk, weather_risk, road_cond_risk,
            lighting_risk, road_type_risk, env_risk_score
Calibrated: state_risk_weight, fatality_rate (from MoRTH)
Encoded   : all categorical cols → _enc integer versions

Usage
─────
    from src.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    featured_df = fe.run(cleaned_df)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

PROCESSED = Path("data/processed")

# Monsoon months — higher risk in India
MONSOON_MONTHS = {
    "july", "august", "september", "october",
    "jul", "aug", "sep", "oct",
    "7", "8", "9", "10",
}

# Peak traffic hours (commute morning + evening)
PEAK_HOURS = set(range(8, 11)) | set(range(17, 21))


class FeatureEngineer:
    """
    Transforms cleaned spatial data into ML-ready features.
    Loads global_feature_stats.json and state_risk_weights.csv
    automatically from data/processed/.
    """

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self._global_stats  = self._load_global_stats()
        self._state_weights = self._load_state_weights()
        self._encoders: dict[str, LabelEncoder] = {}

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pipeline.
        Returns df with all new feature columns appended.
        """
        log.info(f"FeatureEngineer.run() — input: {df.shape}")
        df = df.copy()

        df = self._add_h3_features(df)
        df = self._add_time_features(df)
        df = self._add_risk_features(df)
        df = self._add_state_weights(df)
        df = self._add_encoded_categoricals(df)

        log.info(f"FeatureEngineer.run() — output: {df.shape}")
        return df

    # ── 1. H3 / Spatial density features ─────────────────────────────────────
    def _add_h3_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign each accident to an H3 hex cell and compute density metrics.
        Falls back to lat/lng rounding grid if H3 is not installed.
        """
        try:
            import h3
            log.info("  H3 library available — using real hex cells")
            df["h3_r9"] = df.apply(
                lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], 9), axis=1
            )
            df["h3_r8"] = df.apply(
                lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], 8), axis=1
            )
        except ImportError:
            log.warning("  H3 not installed — using grid-cell fallback")
            df["h3_r9"] = (
                df["latitude"].round(2).astype(str) + "_" +
                df["longitude"].round(2).astype(str)
            )
            df["h3_r8"] = (
                df["latitude"].round(1).astype(str) + "_" +
                df["longitude"].round(1).astype(str)
            )

        # Accident density per cell
        for res in ["r9", "r8"]:
            col  = f"h3_{res}"
            dcol = f"hex_density_{res}"
            density = df.groupby(col).size().rename(dcol)
            df = df.merge(density, on=col, how="left")

        # Per-cell severity and fatality aggregates
        sev_map  = df.groupby("h3_r9")["severity"].mean().rename("hex_avg_severity")
        fat_map  = df.groupby("h3_r9")["killed"].sum().rename("hex_total_killed")
        df = df.merge(sev_map, on="h3_r9", how="left")
        df = df.merge(fat_map, on="h3_r9", how="left")

        log.info(f"  H3 features: {df['h3_r9'].nunique():,} r9 cells, "
                 f"{df['h3_r8'].nunique():,} r8 cells")
        return df

    # ── 2. Time features ──────────────────────────────────────────────────────
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hour_bucket"] = df["hour"].apply(self._hour_to_bucket)
        df["is_night"]    = ((df["hour"] >= 20) | (df["hour"].between(0, 5))).astype(int)
        df.loc[df["hour"] < 0, "is_night"] = 0   # unknown hour → not night

        df["is_monsoon"]   = df["month"].str.lower().isin(MONSOON_MONTHS).astype(int)
        df["is_peak_hour"] = df["hour"].apply(
            lambda h: 1 if h in PEAK_HOURS else 0
        )
        log.info(f"  Time features: night={df['is_night'].sum():,} "
                 f"monsoon={df['is_monsoon'].sum():,} "
                 f"peak={df['is_peak_hour'].sum():,}")
        return df

    # ── 3. Risk factor features ───────────────────────────────────────────────
    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 5 component risk scores (each 1–4) and combine
        into a single env_risk_score.
        """
        df["speed_risk"]      = df["speed_limit"].apply(self._speed_risk)
        df["weather_risk"]    = df["weather"].apply(self._weather_risk)
        df["road_cond_risk"]  = df["road_condition"].apply(self._road_cond_risk)
        df["lighting_risk"]   = df["lighting"].apply(self._lighting_risk)
        df["road_type_risk"]  = df["road_type"].apply(self._road_type_risk)

        # Weighted composite (weights sum to 1.0)
        df["env_risk_score"] = (
            df["speed_risk"]     * 0.30 +
            df["weather_risk"]   * 0.20 +
            df["road_cond_risk"] * 0.20 +
            df["lighting_risk"]  * 0.15 +
            df["road_type_risk"] * 0.15
        ).round(3)

        log.info(f"  Risk features: env_risk mean={df['env_risk_score'].mean():.3f}")
        return df

    # ── 4. State risk weights (MoRTH calibration) ─────────────────────────────
    def _add_state_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._state_weights.empty:
            df["state_risk_weight"] = 1.0
            return df

        df["state_norm"] = df["state"].str.strip().str.title()
        self._state_weights["state_norm"] = (
            self._state_weights["state"].str.strip().str.title()
        )
        df = df.merge(
            self._state_weights[["state_norm", "state_risk_weight"]],
            on="state_norm", how="left"
        )
        df["state_risk_weight"] = df["state_risk_weight"].fillna(1.0)
        df = df.drop(columns=["state_norm"], errors="ignore")
        log.info(f"  State weights joined for "
                 f"{(df['state_risk_weight'] != 1.0).sum():,} rows")
        return df

    # ── 5. Encode categoricals ────────────────────────────────────────────────
    def _add_encoded_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = [
            "weather", "road_type", "road_condition", "lighting",
            "cause", "vehicle_type", "crash_type", "hour_bucket",
            "country", "area_type",
        ]
        for col in cat_cols:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(
                df[col].fillna("unknown").astype(str)
            )
            self._encoders[col] = le

        log.info(f"  Encoded {len(self._encoders)} categorical columns")
        return df

    # ── Risk scoring helpers ───────────────────────────────────────────────────
    @staticmethod
    def _hour_to_bucket(h: int) -> str:
        if h < 0:    return "unknown"
        if h < 6:    return "night"
        if h < 12:   return "morning"
        if h < 18:   return "afternoon"
        return "evening"

    @staticmethod
    def _speed_risk(speed) -> int:
        """1=low → 4=extreme. Over-speeding is #1 cause in India (MoRTH 68.4%)."""
        if pd.isna(speed) or speed <= 0: return 2
        if speed >= 100: return 4
        if speed >= 80:  return 3
        if speed >= 60:  return 2
        return 1

    def _weather_risk(self, w: str) -> int:
        """
        Data-driven if global_feature_stats.json exists.
        Manual fallback otherwise.
        """
        w = str(w).lower().strip()
        # Try data-driven lookup first
        for entry in self._global_stats.get("weather_risk", []):
            key = entry.get("Weather Conditions", "").lower()
            if key and key in w:
                score = round(entry.get("avg_severity", 2))
                return int(max(1, min(4, score)))
        # Manual fallback
        manual = {
            "foggy": 4, "fog": 4, "stormy": 4, "storm": 4,
            "snowy": 4, "snow": 4, "icy": 4,
            "rainy": 3, "rain": 3, "mist": 3,
            "hazy": 2, "drizzle": 2, "windy": 2, "overcast": 2,
            "clear": 1, "sunny": 1, "fine": 1,
        }
        for key, score in manual.items():
            if key in w:
                return score
        return 2

    @staticmethod
    def _road_cond_risk(r: str) -> int:
        r = str(r).lower().strip()
        if any(k in r for k in ["construction", "under"]):  return 4
        if any(k in r for k in ["damage", "pothole", "icy"]): return 3
        if any(k in r for k in ["wet", "damp"]):             return 2
        if any(k in r for k in ["dry", "good"]):             return 1
        return 2

    @staticmethod
    def _lighting_risk(l: str) -> int:
        l = str(l).lower().strip()
        if any(k in l for k in ["dark", "night", "unlit"]): return 3
        if any(k in l for k in ["dusk", "dawn", "dim"]):    return 2
        if any(k in l for k in ["day", "daylight"]):         return 1
        return 2

    def _road_type_risk(self, r: str) -> int:
        r = str(r).lower().strip()
        # Data-driven
        for entry in self._global_stats.get("road_type_risk", []):
            key = entry.get("Road Type", "").lower()
            if key and key in r:
                score = round(entry.get("avg_severity", 2))
                return int(max(1, min(4, score)))
        # Manual fallback
        if any(k in r for k in ["national highway", "nh"]): return 4
        if any(k in r for k in ["highway", "state highway"]): return 3
        if any(k in r for k in ["main", "urban", "city"]):   return 2
        if any(k in r for k in ["street", "lane", "village"]): return 1
        return 2

    # ── Loaders ───────────────────────────────────────────────────────────────
    def _load_global_stats(self) -> dict:
        path = self.processed_dir / "global_feature_stats.json"
        if path.exists():
            with open(path) as f:
                stats = json.load(f)
            log.info("  Loaded global_feature_stats.json for calibration")
            return stats
        log.warning("  global_feature_stats.json not found — using manual risk mappings")
        return {}

    def _load_state_weights(self) -> pd.DataFrame:
        path = self.processed_dir / "state_risk_weights.csv"
        if path.exists():
            df = pd.read_csv(path)
            log.info(f"  Loaded state_risk_weights.csv ({len(df)} states)")
            return df
        log.warning("  state_risk_weights.csv not found — state weight = 1.0 for all")
        return pd.DataFrame()

    # ── Persist encoders ──────────────────────────────────────────────────────
    def save_encoders(self, path: Optional[str] = None):
        out = Path(path) if path else self.processed_dir / "label_encoders.pkl"
        joblib.dump(self._encoders, out)
        log.info(f"  Saved label_encoders.pkl → {out}")

    def save(self, df: pd.DataFrame, filename: str = "featured_spatial.parquet"):
        out = self.processed_dir / filename
        df.to_parquet(out, index=False)
        log.info(f"  Saved {filename}: {df.shape} → {out}")
        return out


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    df = pd.read_parquet("data/processed/cleaned_spatial.parquet")
    fe = FeatureEngineer()
    featured = fe.run(df)
    fe.save_encoders()
    fe.save(featured)
    print(f"\n✅ Feature engineering complete — {featured.shape}")
    print(f"   New columns: {[c for c in featured.columns if c not in df.columns]}")