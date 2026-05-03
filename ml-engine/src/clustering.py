# this will contain ml 
"""
src/clustering.py
──────────────────
DBSCAN-based blackspot detection using haversine distance metric.

Why DBSCAN?
  • No predefined cluster count needed
  • Handles noise — isolated accidents are not forced into clusters
  • Arbitrary cluster shapes — road corridors are not circular
  • Density-based — high-density zones = real blackspots

Usage
─────
    from src.clustering import BlackSpotClusterer
    clusterer = BlackSpotClusterer(eps_meters=200, min_samples=5)
    df_with_labels = clusterer.fit(df)
    print(clusterer.summary())
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000  # metres


def meters_to_radians(meters: float) -> float:
    """Convert distance in metres to radians for haversine DBSCAN."""
    return meters / EARTH_RADIUS_M


class BlackSpotClusterer:
    """
    Wraps DBSCAN with haversine metric for geographic accident clustering.

    Parameters
    ----------
    eps_meters   : cluster radius in metres (default 200m)
    min_samples  : minimum accidents to form a cluster (default 5)
    """

    def __init__(self, eps_meters: int = 200, min_samples: int = 5):
        self.eps_meters   = eps_meters
        self.min_samples  = min_samples
        self._model: Optional[DBSCAN] = None
        self._labels: Optional[np.ndarray] = None
        self._n_clusters: int = 0
        self._n_noise: int    = 0

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run DBSCAN on the latitude/longitude columns of df.
        Adds a 'cluster' column:
            -1  = noise (isolated accident, not part of any blackspot)
            ≥ 0 = cluster ID (blackspot zone)

        Parameters
        ----------
        df : DataFrame with 'latitude' and 'longitude' columns

        Returns
        -------
        df with 'cluster' column added
        """
        df = df.copy()

        # Drop rows with missing coords before clustering
        valid_mask = df["latitude"].notna() & df["longitude"].notna()
        df_valid   = df[valid_mask].copy()

        log.info(f"DBSCAN clustering — {len(df_valid):,} points "
                 f"| eps={self.eps_meters}m "
                 f"| min_samples={self.min_samples}")

        # Convert lat/lng to radians for haversine metric
        coords_rad = np.radians(df_valid[["latitude", "longitude"]].values)

        self._model = DBSCAN(
            eps=meters_to_radians(self.eps_meters),
            min_samples=self.min_samples,
            metric="haversine",
            n_jobs=-1,                  # use all CPU cores
            algorithm="ball_tree",      # efficient for haversine
        )

        labels = self._model.fit_predict(coords_rad)
        df.loc[valid_mask, "cluster"] = labels
        df["cluster"] = df["cluster"].fillna(-1).astype(int)

        # Stats
        self._labels    = labels
        self._n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        self._n_noise    = int((labels == -1).sum())

        log.info(f"  Clusters found : {self._n_clusters:,}")
        log.info(f"  Noise points   : {self._n_noise:,} "
                 f"({self._n_noise / len(labels) * 100:.1f}%)")

        return df

    def summary(self) -> dict:
        """Return a summary dict of the last fit() call."""
        if self._labels is None:
            return {"status": "not fitted"}
        total  = len(self._labels)
        in_cls = int((self._labels >= 0).sum())
        return {
            "eps_meters":          self.eps_meters,
            "min_samples":         self.min_samples,
            "total_points":        total,
            "n_clusters":          self._n_clusters,
            "points_in_clusters":  in_cls,
            "noise_points":        self._n_noise,
            "noise_pct":           round(self._n_noise / total * 100, 1),
            "avg_cluster_size":    round(in_cls / max(self._n_clusters, 1), 1),
        }

    def save(self, path: str = "models/dbscan_model.pkl"):
        """Persist the fitted DBSCAN model to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, out)
        log.info(f"  Saved DBSCAN model → {out}")

    @classmethod
    def load(cls, path: str = "models/dbscan_model.pkl") -> "BlackSpotClusterer":
        """Load a previously saved clusterer (for inspection only)."""
        instance = cls.__new__(cls)
        instance._model = joblib.load(path)
        instance.eps_meters  = instance._model.eps * EARTH_RADIUS_M
        instance.min_samples = instance._model.min_samples
        instance._labels     = None
        instance._n_clusters = 0
        instance._n_noise    = 0
        return instance

    @staticmethod
    def tune(
        df: pd.DataFrame,
        eps_values:        list[int]  = [100, 150, 200, 300],
        min_sample_values: list[int]  = [3, 5, 8, 10],
        sample_n:          int        = 30_000,
    ) -> pd.DataFrame:
        """
        Grid search over eps and min_samples on a random sample.
        Returns a DataFrame with columns:
            eps_m, min_samples, n_clusters, noise_pct, largest_cluster

        Usage
        ─────
            results = BlackSpotClusterer.tune(df)
            print(results.sort_values("n_clusters"))
        """
        log.info(f"Tuning DBSCAN on {min(sample_n, len(df)):,} sample points...")

        sample = df.dropna(subset=["latitude", "longitude"])
        if len(sample) > sample_n:
            sample = sample.sample(sample_n, random_state=42)

        coords = np.radians(sample[["latitude", "longitude"]].values)
        rows   = []

        for eps_m in eps_values:
            for min_s in min_sample_values:
                db = DBSCAN(
                    eps=meters_to_radians(eps_m),
                    min_samples=min_s,
                    metric="haversine",
                    n_jobs=-1,
                )
                labels      = db.fit_predict(coords)
                n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
                noise_pct   = (labels == -1).sum() / len(labels) * 100
                cluster_pts = (labels >= 0).sum()
                largest     = (
                    max((labels == i).sum() for i in set(labels) if i != -1)
                    if n_clusters > 0 else 0
                )
                rows.append({
                    "eps_m":           eps_m,
                    "min_samples":     min_s,
                    "n_clusters":      n_clusters,
                    "noise_pct":       round(noise_pct, 1),
                    "cluster_points":  int(cluster_pts),
                    "largest_cluster": int(largest),
                })
                log.info(f"  eps={eps_m}m min_s={min_s} → "
                         f"{n_clusters} clusters, noise={noise_pct:.1f}%")

        return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    df = pd.read_parquet("data/processed/featured_spatial.parquet")

    # Optional: run tuning first
    # results = BlackSpotClusterer.tune(df)
    # print(results.to_string())

    clusterer = BlackSpotClusterer(eps_meters=200, min_samples=5)
    df = clusterer.fit(df)
    clusterer.save()

    print("\n── Clustering Summary ───────────────────────")
    for k, v in clusterer.summary().items():
        print(f"  {k:<25} {v}")

    df.to_parquet("data/processed/clustered_spatial.parquet", index=False)
    print(f"\n✅ Saved clustered_spatial.parquet")