"""
src/preprocessor.py
────────────────────
Cleans and validates the merged spatial DataFrame.
Also builds MoRTH calibration tables saved as CSV/JSON.

Input  : raw DataFrames from DataIngestion
Output : cleaned DataFrame + state_risk_weights.csv
         + global_feature_stats.json

Usage
─────
    from src.preprocessor import Preprocessor
    pp = Preprocessor()
    clean_df = pp.run(merged_df)
    pp.build_morth_weights(morth_dict)
    pp.build_global_stats(global_df)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROCESSED = Path("data/processed")

# India bounding box — drop anything outside
INDIA_LAT = (6.0, 38.0)
INDIA_LNG = (65.0, 98.0)

# Globally valid bounds (for non-India spatial rows if any)
GLOBAL_LAT = (-90.0, 90.0)
GLOBAL_LNG = (-180.0, 180.0)


class Preprocessor:
    """
    Cleans the merged spatial DataFrame produced by DataIngestion.
    All steps are idempotent and logged.
    """

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full cleaning pipeline. Returns cleaned DataFrame.
        Steps:
          1. Drop rows with missing coordinates
          2. Validate coordinate ranges (India bounds)
          3. Normalise severity to 1–4 int
          4. Fill nulls in all columns
          5. Standardise string columns (lowercase, strip)
          6. Parse & normalise numeric columns
          7. Remove exact duplicates
        """
        log.info(f"Preprocessor.run() — input: {df.shape}")
        original_len = len(df)

        df = self._drop_missing_coords(df)
        df = self._validate_coord_range(df)
        df = self._normalise_severity(df)
        df = self._fill_nulls(df)
        df = self._standardise_strings(df)
        df = self._normalise_numerics(df)
        df = self._remove_duplicates(df)

        log.info(f"Preprocessor.run() — output: {len(df):,} rows "
                 f"(removed {original_len - len(df):,})")
        return df

    # ── Step implementations ──────────────────────────────────────────────────
    def _drop_missing_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["latitude", "longitude"])
        log.info(f"  drop_missing_coords: {before - len(df):,} rows removed")
        return df

    def _validate_coord_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows within valid global lat/lng range."""
        before = len(df)
        df = df[
            df["latitude"].between(*GLOBAL_LAT) &
            df["longitude"].between(*GLOBAL_LNG)
        ].copy()
        log.info(f"  validate_coord_range: {before - len(df):,} rows removed")
        return df

    def _normalise_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip severity to 1–4 int. Fill unknowns with 2 (Serious)."""
        df["severity"] = (
            pd.to_numeric(df["severity"], errors="coerce")
            .clip(1, 4)
            .fillna(2)
            .astype(int)
        )
        return df

    def _fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill nulls per column with appropriate defaults."""
        str_cols = [
            "weather", "road_type", "road_condition",
            "lighting", "state", "country", "month",
            "cause", "vehicle_type", "crash_type",
            "area_type", "source",
        ]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        int_cols = {"killed": 0, "injured": 0, "hour": -1, "year": 2022}
        for col, default in int_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)

        if "speed_limit" in df.columns:
            df["speed_limit"] = pd.to_numeric(df["speed_limit"], errors="coerce")
            # leave NaN for speed_limit — handled in feature engineering

        return df

    def _standardise_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase + strip whitespace from all string columns."""
        string_cols = [
            "weather", "road_type", "road_condition",
            "lighting", "cause", "vehicle_type",
            "crash_type", "area_type",
        ]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()

        # Keep state/country in title case
        for col in ["state", "country"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df

    def _normalise_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clamp killed/injured; ensure year is sensible."""
        if "killed" in df.columns:
            df["killed"] = df["killed"].clip(0, 500)
        if "injured" in df.columns:
            df["injured"] = df["injured"].clip(0, 500)
        if "year" in df.columns:
            df["year"] = df["year"].clip(2000, 2025)
        if "hour" in df.columns:
            df["hour"] = df["hour"].clip(-1, 23)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(
            subset=["latitude", "longitude", "severity", "source"]
        ).reset_index(drop=True)
        log.info(f"  remove_duplicates: {before - len(df):,} removed")
        return df

    # ── MoRTH calibration tables ──────────────────────────────────────────────
    def build_morth_weights(self, morth: dict) -> pd.DataFrame:
        """
        Compute state risk weight multipliers from MoRTH statewise data.
        Saves → data/processed/state_risk_weights.csv
        Returns the weights DataFrame.
        """
        if "statewise_accidents" not in morth:
            log.warning("statewise_accidents not in morth dict — skipping weights")
            return pd.DataFrame()

        df_acc = morth["statewise_accidents"].copy()
        df_fat = morth.get("statewise_fatalities", pd.DataFrame())

        state_col = next(
            (c for c in df_acc.columns if "state" in c.lower()), None
        )
        if not state_col:
            log.warning("Could not find state column in MoRTH accidents")
            return pd.DataFrame()

        acc_2023_col = next(
            (c for c in df_acc.columns if "2023" in str(c) and "accident" in str(c).lower()),
            None,
        )
        if not acc_2023_col:
            # try any 2023 col
            acc_2023_col = next(
                (c for c in df_acc.columns if "2023" in str(c)), None
            )

        result = pd.DataFrame()
        result["state"] = df_acc[state_col].astype(str).str.strip()

        if acc_2023_col:
            result["accidents_2023"] = pd.to_numeric(
                df_acc[acc_2023_col].astype(str).str.replace(",", ""),
                errors="coerce",
            )
        else:
            result["accidents_2023"] = np.nan

        # Fatality data
        if not df_fat.empty:
            fat_state_col = next(
                (c for c in df_fat.columns if "state" in c.lower()), None
            )
            fat_2023_col = next(
                (c for c in df_fat.columns if "2023" in str(c) and "kill" in str(c).lower()),
                None,
            )
            if fat_state_col and fat_2023_col:
                fat_vals = pd.to_numeric(
                    df_fat[fat_2023_col].astype(str).str.replace(",", ""),
                    errors="coerce",
                )
                if len(fat_vals) == len(result):
                    result["killed_2023"] = fat_vals.values

        result = result.dropna(subset=["accidents_2023"])

        # Normalise to 1.0–2.0 multiplier
        result["accident_rank"]     = result["accidents_2023"].rank(pct=True)
        result["state_risk_weight"] = (1.0 + result["accident_rank"]).round(3)

        if "killed_2023" in result.columns:
            result["fatality_rate"] = (
                result["killed_2023"] / result["accidents_2023"]
            ).round(4)

        out = self.processed_dir / "state_risk_weights.csv"
        result.to_csv(out, index=False)
        log.info(f"  Saved state_risk_weights.csv ({len(result)} states) → {out}")
        return result

    # ── Global feature stats ──────────────────────────────────────────────────
    def build_global_stats(self, df_global: pd.DataFrame) -> dict:
        """
        Extract feature pattern statistics from the Global Kaggle dataset.
        ⚠️  Global has NO coordinates — this is feature stats only.
        Saves → data/processed/global_feature_stats.json
        Returns the stats dict.
        """
        log.info("Building global feature stats from Global Kaggle (no coords)...")

        sev_map = {"Minor": 1, "Moderate": 2, "Severe": 3, "Fatal": 4}
        df_global = df_global.copy()
        df_global["severity_num"] = df_global["Accident Severity"].map(sev_map).fillna(2)

        def group_stats(df, col):
            return (
                df.groupby(col)["severity_num"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "avg_severity", "count": "n"})
                .round(3)
                .reset_index()
                .to_dict(orient="records")
            )

        stats = {
            "weather_risk":    group_stats(df_global, "Weather Conditions"),
            "road_type_risk":  group_stats(df_global, "Road Type"),
            "cause_risk":      group_stats(df_global, "Accident Cause"),
            "metadata": {
                "source":       "global_kaggle",
                "total_rows":   len(df_global),
                "purpose":      "feature weight calibration — no spatial data",
            },
        }

        # Speed bins
        df_global["speed_bin"] = pd.cut(
            df_global["Speed Limit"],
            bins=[0, 40, 60, 80, 100, 200],
            labels=["0-40", "41-60", "61-80", "81-100", "100+"],
        )
        speed_stats = (
            df_global.groupby("speed_bin", observed=True)["severity_num"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_severity", "count": "n"})
            .round(3)
            .reset_index()
            .rename(columns={"speed_bin": "bin"})
            .to_dict(orient="records")
        )
        stats["speed_risk"] = speed_stats

        out = self.processed_dir / "global_feature_stats.json"
        with open(out, "w") as f:
            json.dump(stats, f, indent=2)
        log.info(f"  Saved global_feature_stats.json → {out}")
        return stats

    # ── Save helpers ──────────────────────────────────────────────────────────
    def save(self, df: pd.DataFrame, filename: str = "cleaned_spatial.parquet"):
        out = self.processed_dir / filename
        df.to_parquet(out, index=False)
        log.info(f"  Saved {filename}: {df.shape} → {out}")
        return out


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    from src.data_ingestion import DataIngestion

    ing = DataIngestion()
    pp  = Preprocessor()

    # Load + merge spatial datasets
    frames = []
    for loader in [ing.load_mendeley, ing.load_kaggle_india]:
        try:
            frames.append(loader())
        except Exception as e:
            log.warning(f"Skipping {loader.__name__}: {e}")

    import pandas as pd
    merged = pd.concat(frames, ignore_index=True)
    cleaned = pp.run(merged)
    pp.save(cleaned)

    # Calibration tables
    morth = ing.load_morth()
    pp.build_morth_weights(morth)

    global_df = ing.load_global_kaggle()
    pp.build_global_stats(global_df)

    print(f"\n✅ Preprocessing complete — {len(cleaned):,} clean spatial rows")