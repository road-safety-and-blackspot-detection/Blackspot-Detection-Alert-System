"""
src/data_ingestion.py
─────────────────────
Loads all raw CSV datasets, renames columns to the standard schema,
and returns individual DataFrames ready for preprocessing.

Standard schema columns
───────────────────────
latitude, longitude, severity (1–4), killed, injured,
weather, road_type, road_condition, lighting, speed_limit,
state, country, year, month, hour, cause, vehicle_type,
crash_type, area_type, source

Dataset roles
─────────────
Mendeley India  : real GPS coords  → spatial clustering
Kaggle India    : geocoded         → spatial clustering
Global Kaggle   : NO coords        → feature pattern stats only
MoRTH 5 files  : aggregates       → risk calibration lookups

Usage
─────
    from src.data_ingestion import DataIngestion
    ing = DataIngestion()
    df_mendeley = ing.load_mendeley()
    df_kaggle   = ing.load_kaggle_india()
    df_global   = ing.load_global_kaggle()   # feature stats only
    morth       = ing.load_morth()           # dict of DataFrames
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.geocoder import geocode_dataframe

log = logging.getLogger(__name__)

# ── Standard column list every spatial dataset must conform to ────────────────
STANDARD_COLS = [
    "latitude", "longitude",
    "severity",
    "killed",
    "injured",
    "weather",
    "road_type",
    "road_condition",
    "lighting",
    "speed_limit",
    "state",
    "country",
    "year",
    "month",
    "hour",
    "cause",
    "vehicle_type",
    "crash_type",
    "area_type",
    "source",
]


class DataIngestion:
    """
    Handles loading of all raw datasets into a standard schema.
    Instantiate with the path to ml-engine/ root directory.
    """

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw = Path(raw_dir)
        if not self.raw.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw.resolve()}\n"
                "Make sure you run this from ml-engine/ directory."
            )
        log.info(f"DataIngestion initialised — raw dir: {self.raw.resolve()}")

    # ── 1. India Mendeley ─────────────────────────────────────────────────────
    def load_mendeley(self) -> pd.DataFrame:
        """
        Load India Mendeley fatal crash dataset.
        Has real GPS in 'LatLong' column: "18.278, 76.009" string format.
        Source: Times of India crash reports 2022–2023.
        """
        path = self.raw / "india_mendeley" / "crashes.csv"
        log.info(f"Loading Mendeley → {path}")
        df = pd.read_csv(path)
        log.info(f"  Raw shape: {df.shape}")

        # Parse LatLong string → separate numeric columns
        coords = df["LatLong"].str.split(",", expand=True)
        df["latitude"]  = pd.to_numeric(coords[0].str.strip(), errors="coerce")
        df["longitude"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")

        # Severity: based on killed count
        df["severity"] = df["Killed"].apply(self._killed_to_severity)

        # Rename to standard schema
        df = df.rename(columns={
            "State":      "state",
            "Killed":     "killed",
            "Injured":    "injured",
            "Road Type":  "road_type",
            "Crash Type": "crash_type",
            "Month":      "month",
        })

        # Fill columns not present in Mendeley
        df["country"]        = "India"
        df["source"]         = "mendeley_india_2022_2023"
        df["weather"]        = "unknown"
        df["road_condition"] = "unknown"
        df["lighting"]       = "unknown"
        df["speed_limit"]    = np.nan
        df["year"]           = 2022
        df["hour"]           = -1
        df["area_type"]      = "unknown"
        df["cause"]          = df.get("crash_type", pd.Series(["unknown"] * len(df))).fillna("unknown")
        df["vehicle_type"]   = df["Vehicle 1"].fillna("unknown") if "Vehicle 1" in df.columns else "unknown"

        df["killed"]  = pd.to_numeric(df["killed"],  errors="coerce").fillna(0).astype(int)
        df["injured"] = pd.to_numeric(df["injured"], errors="coerce").fillna(0).astype(int)

        result = self._enforce_schema(df)
        log.info(f"  Mendeley loaded: {len(result):,} rows")
        return result

    # ── 2. Kaggle India ───────────────────────────────────────────────────────
    def load_kaggle_india(self) -> pd.DataFrame:
        """
        Load Kaggle India Road Accident dataset.
        Has City Name + State Name but NO coordinates.
        Geocoded using static lookup dict (no API required).
        """
        csv_files = list((self.raw / "kaggle_india").glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV found in {self.raw / 'kaggle_india'}")
        path = csv_files[0]
        log.info(f"Loading Kaggle India → {path}")
        df = pd.read_csv(path)
        log.info(f"  Raw shape: {df.shape}")

        # Geocode City Name + State Name → lat/lng
        log.info("  Geocoding rows (static dict — no API)...")
        df = geocode_dataframe(df, city_col="City Name", state_col="State Name")

        # Severity mapping
        sev_map = {"Minor": 1, "Serious": 2, "Fatal": 3}
        df["severity"] = df["Accident Severity"].map(sev_map).fillna(1).astype(int)

        # Hour extraction from "HH:MM" time string
        df["hour"] = (
            pd.to_datetime(df["Time of Day"], format="%H:%M", errors="coerce")
            .dt.hour.fillna(-1).astype(int)
        )

        # Rename to standard schema
        rename_map = {
            "State Name":                "state",
            "Month":                     "month",
            "Year":                      "year",
            "Weather Conditions":        "weather",
            "Road Type":                 "road_type",
            "Road Condition":            "road_condition",
            "Lighting Conditions":       "lighting",
            "Speed Limit (km/h)":        "speed_limit",
            "Number of Fatalities":      "killed",
            "Number of Casualties":      "injured",
            "Vehicle Type Involved":     "vehicle_type",
            "Accident Location Details": "crash_type",
            "Urban/Rural":               "area_type",
            "Accident Cause":            "cause",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        df["country"]        = "India"
        df["source"]         = "kaggle_india_synthetic"
        df["road_condition"] = df.get("road_condition", pd.Series(["unknown"] * len(df))).fillna("unknown")
        df["lighting"]       = df.get("lighting",       pd.Series(["unknown"] * len(df))).fillna("unknown")
        df["area_type"]      = df.get("area_type",      pd.Series(["unknown"] * len(df))).fillna("unknown")
        df["cause"]          = df.get("cause",          pd.Series(["unknown"] * len(df))).fillna("unknown")
        df["speed_limit"]    = pd.to_numeric(df.get("speed_limit"), errors="coerce")
        df["killed"]         = pd.to_numeric(df.get("killed",  pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)
        df["injured"]        = pd.to_numeric(df.get("injured", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)

        result = self._enforce_schema(df)
        log.info(f"  Kaggle India loaded: {len(result):,} rows (with geocoded coords)")
        return result

    # ── 3. Global Kaggle ──────────────────────────────────────────────────────
    def load_global_kaggle(self) -> pd.DataFrame:
        """
        Load Global Kaggle dataset.
        ⚠️  This dataset has NO latitude/longitude columns.
        Returns the raw DataFrame for feature pattern extraction only.
        Do NOT use for spatial clustering.
        """
        csv_files = list((self.raw / "global").glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV found in {self.raw / 'global'}")
        path = csv_files[0]
        log.info(f"Loading Global Kaggle → {path}")
        df = pd.read_csv(path, low_memory=False)
        log.info(f"  Global Kaggle loaded: {df.shape[0]:,} rows")
        log.info("  ⚠️  NO coordinates — for feature pattern stats only")

        # Confirm absence of coords (guard)
        assert "latitude"  not in df.columns, "Unexpected latitude in global dataset!"
        assert "longitude" not in df.columns, "Unexpected longitude in global dataset!"

        return df  # returned as-is — caller extracts stats

    # ── 4. MoRTH Files ───────────────────────────────────────────────────────
    def load_morth(self) -> dict[str, pd.DataFrame]:
        """
        Load all 5 MoRTH CSV files.
        Returns a dict of DataFrames keyed by short name.
        All numeric columns have commas stripped.
        """
        morth_dir = self.raw / "morth"
        file_map = {
            "statewise_accidents":  "statewise_accidents_2019_2023.csv",
            "statewise_fatalities": "statewise_fatalities_2019_2023.csv",
            "collision_types":      "collision_types_2023.csv",
            "violation_types":      "violation_types_2023.csv",
            "large_cities":         "large_cities_2023.csv",
        }

        result = {}
        for key, filename in file_map.items():
            path = morth_dir / filename
            if not path.exists():
                log.warning(f"  MoRTH file not found: {filename}")
                continue
            df = pd.read_csv(path)
            df = self._clean_morth_numerics(df)
            result[key] = df
            log.info(f"  MoRTH '{key}': {df.shape}")

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _killed_to_severity(k) -> int:
        """Map fatality count → severity scale 1–4."""
        k = pd.to_numeric(k, errors="coerce")
        if pd.isna(k): return 2
        if k >= 3:     return 4   # mass fatal
        if k >= 1:     return 3   # fatal
        return 2                   # serious (injured present)

    @staticmethod
    def _clean_morth_numerics(df: pd.DataFrame) -> pd.DataFrame:
        """Strip Indian number formatting (commas) from MoRTH CSVs."""
        for col in df.columns:
            if df[col].dtype == object:
                cleaned = df[col].astype(str).str.replace(",", "").str.strip()
                numeric = pd.to_numeric(cleaned, errors="coerce")
                if numeric.notna().sum() > len(df) * 0.5:
                    df[col] = numeric
        return df

    @staticmethod
    def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has exactly STANDARD_COLS.
        Missing cols are filled with sensible defaults.
        Extra cols are dropped.
        """
        for col in STANDARD_COLS:
            if col not in df.columns:
                if col in ("latitude", "longitude", "speed_limit"):
                    df[col] = np.nan
                elif col in ("severity", "killed", "injured", "year", "hour"):
                    df[col] = 0
                else:
                    df[col] = "unknown"
        return df[STANDARD_COLS].copy()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ing = DataIngestion()

    print("\n── Mendeley ─────────────────────────────")
    m = ing.load_mendeley()
    print(m[["latitude","longitude","severity","state","source"]].head(3).to_string())

    print("\n── Kaggle India ─────────────────────────")
    k = ing.load_kaggle_india()
    print(k[["latitude","longitude","severity","state","source"]].head(3).to_string())

    print("\n── Global Kaggle (no coords) ────────────")
    g = ing.load_global_kaggle()
    print(f"  Shape: {g.shape} | Columns: {list(g.columns[:6])}...")

    print("\n── MoRTH ────────────────────────────────")
    morth = ing.load_morth()
    for k_name, df in morth.items():
        print(f"  {k_name}: {df.shape}")