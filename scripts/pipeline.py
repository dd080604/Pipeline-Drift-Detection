import warnings
warnings.filterwarnings("ignore")

import os, pickle, time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")

#  STAGE LOGGER captures data at every pipeline boundary

@dataclass
class StageLog:
    #A single snapshot captured at one stage boundary for one batch
    batch_id:       int
    stage_name:     str
    n_rows:         int
    n_cols:         int
    numeric_means:  Dict[str, float]
    numeric_stds:   Dict[str, float]
    null_rates:     Dict[str, float]
    timestamp_sec:  float
    dataframe:      Optional[pd.DataFrame] = None   # full snapshot (optional)


class StageLogger:
    """
    Logging framework for stage-level monitoring. After each pipeline stage processes a batch, call .log() to capture summary statistics and (optionally) the full DataFrame snapshot. All logs are held in memory as a list of StageLog objects.
    """

    def __init__(self, store_dataframes: bool = False):
        self.store_dataframes = store_dataframes
        self.logs: List[StageLog] = []

    def log(self, batch_id: int, stage_name: str, df: pd.DataFrame):
        """Record a snapshot of the data after a pipeline stage."""
        numeric_cols = df.select_dtypes(include="number").columns

        entry = StageLog(
            batch_id      = batch_id,
            stage_name    = stage_name,
            n_rows        = len(df),
            n_cols        = len(df.columns),
            numeric_means = df[numeric_cols].mean().to_dict(),
            numeric_stds  = df[numeric_cols].std().to_dict(),
            null_rates    = df.isnull().mean().to_dict(),
            timestamp_sec = time.time(),
            dataframe     = df.copy() if self.store_dataframes else None,
        )
        self.logs.append(entry)

    def get_stage_logs(self, stage_name: str) -> List[StageLog]:
        """Return all logs for a specific stage."""
        return [l for l in self.logs if l.stage_name == stage_name]

    def to_summary_df(self) -> pd.DataFrame:
        """Flatten logs into a tidy DataFrame for analysis."""
        rows = []
        for l in self.logs:
            base = {
                "batch_id": l.batch_id,
                "stage": l.stage_name,
                "n_rows": l.n_rows,
                "n_cols": l.n_cols,
            }
            for feat, val in l.numeric_means.items():
                rows.append({**base, "feature": feat,
                             "mean": val,
                             "std": l.numeric_stds.get(feat),
                             "null_rate": l.null_rates.get(feat, 0.0)})
        return pd.DataFrame(rows)

    def clear(self):
        """Reset all stored logs."""
        self.logs.clear()

    def __len__(self):
        return len(self.logs)

    def __repr__(self):
        stages = set(l.stage_name for l in self.logs)
        return f"StageLogger({len(self.logs)} entries across stages {stages})"



class BatchIngester:
    """
    Simulate streaming data arrival by yielding fixed-size batches
    from a DataFrame partition.
    """

    def __init__(self, df: pd.DataFrame, batch_size: int = 500):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(len(df) / batch_size))

    def __iter__(self):
        for i in range(self.n_batches):
            start = i * self.batch_size
            end   = min(start + self.batch_size, len(self.df))
            yield i, self.df.iloc[start:end].copy()

    def __len__(self):
        return self.n_batches

    def __repr__(self):
        return (f"BatchIngester({len(self.df):,} rows, "
                f"batch_size={self.batch_size}, "
                f"n_batches={self.n_batches})")


class DataCleaner:
    """
    Apply standard preprocessing to a raw batch.

    The reference statistics (medians, clip bounds) are computed once from
    the reference partition and reused for every subsequent batch.
    """

    def __init__(self, reference_df: pd.DataFrame, dataset_name: str = ""):
        self.dataset_name = dataset_name
        feature_cols = [c for c in reference_df.columns if c != "target"]
        self.numeric_cols = (
            reference_df[feature_cols]
            .select_dtypes(include="number")
            .columns.tolist()
        )

        # Compute reference statistics for imputation and clipping
        self.medians = reference_df[self.numeric_cols].median()
        self.clip_lo = reference_df[self.numeric_cols].quantile(0.01)
        self.clip_hi = reference_df[self.numeric_cols].quantile(0.99)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a single batch. Returns a new DataFrame"""
        out = df.copy()

        # 1. Deduplicate
        before = len(out)
        out = out.drop_duplicates()
        n_dupes = before - len(out)

        # 2. Impute missing values with reference medians
        for col in self.numeric_cols:
            if col in out.columns and out[col].isnull().any():
                out[col] = out[col].fillna(self.medians[col])

        # 3. Clip outliers to reference [1st, 99th] percentile
        for col in self.numeric_cols:
            if col in out.columns:
                out[col] = out[col].clip(
                    lower=self.clip_lo[col],
                    upper=self.clip_hi[col],
                )

        # 4. Cast to float64
        for col in self.numeric_cols:
            if col in out.columns:
                out[col] = out[col].astype(np.float64)

        return out

    def __repr__(self):
        return (f"DataCleaner(dataset={self.dataset_name}, "
                f"n_numeric={len(self.numeric_cols)})")



class FeatureEngineer:
    """
    Create derived features from cleaned data.

    The transformations are dataset-specific to produce meaningful derived
    signals.  Each adds nonlinear combinations that can amplify or mask
    upstream drift.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.dataset_name == "electricity":
            return self._featurize_electricity(df)
        elif self.dataset_name == "covertype":
            return self._featurize_covertype(df)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    # ── Electricity features ────────────────────────────────────────────

    def _featurize_electricity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derived features for the Electricity dataset.
        """
        out = df.copy()

        # Identify the core price / demand columns.
        price_cols = [c for c in out.columns
                      if "price" in c.lower() and c != "target"]
        demand_cols = [c for c in out.columns if "demand" in c.lower()]

        # Price ratio (NSW / VIC) — guards against division by zero
        if "nswprice" in out.columns and "vicprice" in out.columns:
            vic = out["vicprice"].replace(0, np.nan).fillna(out["vicprice"].median())
            out["price_ratio_nsw_vic"] = out["nswprice"] / vic

        # Price spread
        if "nswprice" in out.columns and "vicprice" in out.columns:
            out["price_spread"] = out["nswprice"] - out["vicprice"]

        # Demand-weighted price (interaction)
        if "nswprice" in out.columns and "nswdemand" in out.columns:
            out["demand_weighted_price"] = out["nswprice"] * out["nswdemand"]

        # Log-transformed prices (shift to avoid log(0))
        for col in price_cols:
            min_val = out[col].min()
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            out[f"log_{col}"] = np.log1p(out[col] + shift)

        # Per-batch volatility proxy (std within the batch)
        for col in price_cols:
            out[f"vol_{col}"] = out[col].std()

        numeric_cols = out.select_dtypes(include="number").columns
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
        out[numeric_cols] = out[numeric_cols].fillna(0.0)
        return out

    # ── Covertype features ──────────────────────────────────────────────

    def _featurize_covertype(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derived features for the Covertype dataset.
        """
        out = df.copy()

        # Euclidean distance to hydrology
        if ("Horizontal_Distance_To_Hydrology" in out.columns and
                "Vertical_Distance_To_Hydrology" in out.columns):
            out["euclid_dist_hydrology"] = np.sqrt(
                out["Horizontal_Distance_To_Hydrology"] ** 2 +
                out["Vertical_Distance_To_Hydrology"] ** 2
            )

        # Elevation-to-roadway ratio
        if ("Elevation" in out.columns and
                "Horizontal_Distance_To_Roadways" in out.columns):
            denom = out["Horizontal_Distance_To_Roadways"].replace(0, np.nan)
            denom = denom.fillna(denom.median())
            out["elev_road_ratio"] = out["Elevation"] / denom

        # Hillshade summary features
        hs_cols = [c for c in out.columns if c.startswith("Hillshade")]
        if len(hs_cols) >= 2:
            out["hillshade_mean"]  = out[hs_cols].mean(axis=1)
            out["hillshade_range"] = out[hs_cols].max(axis=1) - out[hs_cols].min(axis=1)

        # Slope-Aspect interaction
        if "Slope" in out.columns and "Aspect" in out.columns:
            out["slope_aspect_interaction"] = (
                out["Slope"] * np.sin(np.radians(out["Aspect"]))
            )

        # Elevation bins (quartile-based)
        if "Elevation" in out.columns:
            out["elevation_bin"] = pd.qcut(
                out["Elevation"], q=4, labels=False, duplicates="drop"
            ).astype(float)

        numeric_cols = out.select_dtypes(include="number").columns
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
        out[numeric_cols] = out[numeric_cols].fillna(0.0)
        return out

    def __repr__(self):
        return f"FeatureEngineer(dataset={self.dataset_name})"



class PredictionStage:
    """
    Train a baseline classifier on the reference data and produce
    predictions on incoming batches.

    The model is a Random Forest.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = RANDOM_SEED):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Train on a featurized reference DataFrame.

        Automatically identifies feature columns (everything except 'target')
        and stores the column list for consistent ordering at prediction time.
        """
        self.feature_cols = [c for c in df.columns if c != "target"]
        X = df[self.feature_cols].values
        y = df["target"].values

        self.model.fit(X, y)
        self.is_fitted = True

        train_acc = accuracy_score(y, self.model.predict(X))
        print(f"  Model trained on {len(df):,} rows  "
              f"({len(self.feature_cols)} features)  "
              f"train accuracy = {train_acc:.4f}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for a featurized batch.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

        # Ensure consistent column ordering; fill missing derived features with 0
        X = df.reindex(columns=self.feature_cols, fill_value=0).values

        out = df.copy()
        out["pred_label"] = self.model.predict(X)

        probas = self.model.predict_proba(X)
        for i, cls in enumerate(self.model.classes_):
            out[f"pred_proba_{cls}"] = probas[:, i]

        return out

    def __repr__(self):
        status = "fitted" if self.is_fitted else "unfitted"
        return f"PredictionStage({status}, {len(self.feature_cols)} features)"


class DriftDetectionPipeline:
    """
    Four-stage pipeline:  ingest, clean, featurize, predict

    Each stage boundary is instrumented with a StageLogger that captures
    summary statistics for monitoring.

    Usage
    -----
    1. Instantiate with a dataset name and reference partition.
    2. Call .fit() to train the cleaning stats and prediction model.
    3. Call .run() with a test partition to process all batches.
    4. Inspect .logger for stage-level monitoring data.
    """

    STAGES = ["ingest", "clean", "featurize", "predict"]

    def __init__(
        self,
        dataset_name:     str,
        reference_df:     pd.DataFrame,
        batch_size:       int  = 500,
        store_dataframes: bool = False,
    ):
        self.dataset_name = dataset_name
        self.batch_size   = batch_size

        # Initialise stage components
        self.cleaner  = DataCleaner(reference_df, dataset_name)
        self.engineer = FeatureEngineer(dataset_name)
        self.predictor = PredictionStage()

        # Logger captures snapshots at every stage boundary
        self.logger = StageLogger(store_dataframes=store_dataframes)

        # Prepare reference data through clean → featurize, then train model
        self._fit_reference(reference_df)

    def _fit_reference(self, reference_df: pd.DataFrame):
        """Push reference data through clean → featurize, then train model."""
        print(f"\n{'='*60}")
        print(f"  FITTING PIPELINE — {self.dataset_name}")
        print(f"{'='*60}")

        print("  [1/3] Cleaning reference data …")
        ref_clean = self.cleaner.transform(reference_df)

        print("  [2/3] Featurizing reference data …")
        ref_feat = self.engineer.transform(ref_clean)

        print("  [3/3] Training prediction model …")
        self.predictor.fit(ref_feat)

        # Store the featurized reference for later drift comparison
        self.reference_clean = ref_clean
        self.reference_feat  = ref_feat

        print(f"Pipeline fitted.\n")

    def run(self, test_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Process the test partition in temporal batches through all four stages.

        Returns a list of predicted DataFrames (one per batch).
        """
        self.logger.clear()
        ingester = BatchIngester(test_df, self.batch_size)
        results = []

        print(f"  Processing {len(ingester)} batches "
              f"(batch_size={self.batch_size}) …")

        for batch_id, raw_batch in ingester:

            # Stage 1: Ingest — raw batch as-is
            self.logger.log(batch_id, "ingest", raw_batch)

            # Stage 2: Clean
            cleaned = self.cleaner.transform(raw_batch)
            self.logger.log(batch_id, "clean", cleaned)

            # Stage 3: Featurize
            featured = self.engineer.transform(cleaned)
            self.logger.log(batch_id, "featurize", featured)

            # Stage 4: Predict
            predicted = self.predictor.predict(featured)
            self.logger.log(batch_id, "predict", predicted)

            results.append(predicted)

        print(f" {len(results)} batches processed.  "
              f"{len(self.logger)} log entries captured.")
        return results

    def __repr__(self):
        return (f"DriftDetectionPipeline(dataset={self.dataset_name}, "
                f"batch_size={self.batch_size}, "
                f"predictor={self.predictor})")
