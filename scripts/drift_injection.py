import warnings
warnings.filterwarnings("ignore")

import os, pickle, copy
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")

from pipeline import (
    StageLog, StageLogger, BatchIngester, DataCleaner,
    FeatureEngineer, PredictionStage, DriftDetectionPipeline
)

with open("/content/drive/MyDrive/Project Phases/Phase 1/phase1_artifacts.pkl", "rb") as f:
    phase1 = pickle.load(f)

with open("/content/drive/MyDrive/Project Phases/Phase 2/phase2_outputs/phase2_artifacts.pkl", "rb") as f:
    phase2 = pickle.load(f)

class BaseDriftInjector(ABC):
    """
    Abstract base for all drift injectors.

    Every injector is parameterised by:
      - target_cols   : which feature column(s) to affect
      - severity      : float in [0, 1] controlling drift magnitude
      - onset_batch   : the batch index at which drift begins
      - drift_type    : string label for logging and identification

    Subclasses implement .inject() which takes a batch DataFrame and
    the current batch_id, and returns a (possibly modified) copy.
    Batches before onset_batch are returned unchanged.
    """

    def __init__(
        self,
        target_cols: List[str],
        severity:    float = 0.5,
        onset_batch: int   = 10,
    ):
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {severity}")

        self.target_cols = target_cols
        self.severity    = severity
        self.onset_batch = onset_batch
        self.drift_type  = self.__class__.__name__

    def __call__(self, df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Apply drift if batch_id >= onset_batch, else pass through."""
        if batch_id < self.onset_batch:
            return df.copy()
        return self.inject(df.copy(), batch_id)

    @abstractmethod
    def inject(self, df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Apply the drift transformation.  df is already a copy."""
        ...

    def _batches_since_onset(self, batch_id: int) -> int:
        """How many batches have elapsed since drift started."""
        return max(0, batch_id - self.onset_batch)

    def __repr__(self):
        return (f"{self.drift_type}(cols={self.target_cols}, "
                f"severity={self.severity}, onset={self.onset_batch})")

class GradualCovariateShift(BaseDriftInjector):
    """
    Slowly shift a feature's mean over many batches.
    Simulates: sensor calibration drift, gradual population change.

    The mean shift at batch t is:
        shift(t) = severity × reference_std × (t - onset) / ramp_batches

    The shift grows linearly from zero at onset to its maximum at
    onset + ramp_batches, then stays constant.  This makes the drift
    hard to detect early and increasingly obvious over time.
    """

    def __init__(
        self,
        target_cols:    List[str],
        reference_stds: Dict[str, float],
        severity:       float = 0.5,
        onset_batch:    int   = 10,
        ramp_batches:   int   = 20,
    ):
        super().__init__(target_cols, severity, onset_batch)
        self.reference_stds = reference_stds
        self.ramp_batches   = ramp_batches

    def inject(self, df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        t = self._batches_since_onset(batch_id)
        progress = min(t / max(self.ramp_batches, 1), 1.0)

        for col in self.target_cols:
            if col in df.columns:
                ref_std = self.reference_stds.get(col, df[col].std())
                shift = self.severity * ref_std * progress * 3.0
                df[col] = df[col] + shift

        return df

class SuddenSchemaChange(BaseDriftInjector):
    """
    Abruptly truncate numeric precision at the onset batch.
    Simulates: database migration, upstream system update, encoding change.

    The precision is controlled by severity:
        severity=0.3 (low): round to 1 decimal place
        severity=0.6 (med): round to 0 decimal places (integers)
        severity=1.0 (high): round to nearest 10
    """

    def inject(self, df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        # Map severity to rounding precision
        if self.severity <= 0.3:
            decimals = 1
        elif self.severity <= 0.6:
            decimals = 0
        else:
            decimals = -1   # round to nearest 10

        for col in self.target_cols:
            if col in df.columns:
                df[col] = df[col].round(decimals)

        return df

class IncreasingNullRate(BaseDriftInjector):
    """
    Progressively increase the fraction of missing values over time.
    Simulates: failing data source, broken API connector.

    The null fraction at batch t is:
        null_frac(t) = severity × min((t - onset) / ramp_batches, 1.0)

    At maximum, severity=1.0 means the feature is entirely NaN.
    Nulls are injected at random positions within the batch.
    """

    def __init__(
        self,
        target_cols:  List[str],
        severity:     float = 0.5,
        onset_batch:  int   = 10,
        ramp_batches: int   = 20,
        random_state: int   = RANDOM_SEED,
    ):
        super().__init__(target_cols, severity, onset_batch)
        self.ramp_batches = ramp_batches
        self.rng = np.random.RandomState(random_state)

    def inject(self, df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        t = self._batches_since_onset(batch_id)
        progress = min(t / max(self.ramp_batches, 1), 1.0)
        null_frac = self.severity * progress

        for col in self.target_cols:
            if col in df.columns:
                n = len(df)
                n_nulls = int(n * null_frac)
                if n_nulls > 0:
                    idx = self.rng.choice(df.index, size=n_nulls, replace=False)
                    df.loc[idx, col] = np.nan

        return df

class LabelShift(BaseDriftInjector):
    """
    Alter the class distribution without changing input features.
    Simulates: changing real-world base rate (e.g., fraud rate increases).

    Method: randomly flip a fraction of labels from the majority class
    to the minority class.  The flip fraction ramps with severity and
    batches since onset.
    """

    def __init__(
        self,
        target_cols:    List[str],      # ignored; always operates on 'target'
        severity:       float = 0.5,
        onset_batch:    int   = 10,
        minority_class: int   = 1,
        ramp_batches:   int   = 20,
        random_state:   int   = RANDOM_SEED,
    ):
        super().__init__(["target"], severity, onset_batch)
        self.minority_class = minority_class
        self.ramp_batches   = ramp_batches
        self.rng = np.random.RandomState(random_state)

    def inject(self, df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        if "target" not in df.columns:
            return df

        t = self._batches_since_onset(batch_id)
        progress = min(t / max(self.ramp_batches, 1), 1.0)
        flip_frac = self.severity * progress * 0.5  # cap at 50% flip rate

        # Find majority-class rows and flip a fraction to minority
        majority_mask = df["target"] != self.minority_class
        majority_idx = df.index[majority_mask]
        n_flip = int(len(majority_idx) * flip_frac)

        if n_flip > 0:
            flip_idx = self.rng.choice(majority_idx, size=n_flip, replace=False)
            df.loc[flip_idx, "target"] = self.minority_class

        return df

# Severity presets that map the outline's low/medium/high to numeric values
SEVERITY_PRESETS = {"low": 0.3, "medium": 0.6, "high": 1.0}

DRIFT_REGISTRY = {
    "gradual_covariate_shift": GradualCovariateShift,
    "sudden_schema_change":    SuddenSchemaChange,
    "increasing_null_rate":    IncreasingNullRate,
    "label_shift":             LabelShift
}


def get_default_target_cols(dataset_name: str, drift_type: str) -> List[str]:
    """
    Return sensible default columns to target for each drift type
    and dataset.  These are the columns where drift is most realistic
    and most interesting to study.
    """
    if drift_type == "label_shift":
        return ["target"]

    if dataset_name == "electricity":
        # Price and demand features are the natural candidates
        if drift_type == "feature_rotation":
            return ["nswprice", "vicprice"]
        return ["nswprice", "vicprice"]

    elif dataset_name == "covertype":
        if drift_type == "feature_rotation":
            return ["Elevation", "Horizontal_Distance_To_Roadways"]
        return ["Elevation", "Slope"]

    return []


def create_injector(
    drift_type:    str,
    dataset_name:  str,
    severity:      float | str = "medium",
    onset_batch:   int         = 10,
    reference_df:  Optional[pd.DataFrame] = None,
    target_cols:   Optional[List[str]]    = None,
    **kwargs,
) -> BaseDriftInjector:
    """
    Function to create a configured drift injector.

    Parameters
    ----------
    drift_type   : one of the keys in DRIFT_REGISTRY
    dataset_name : 'electricity' or 'covertype'
    severity     : float in [0,1] or one of 'low', 'medium', 'high'
    onset_batch  : batch index where drift begins
    reference_df : needed for GradualCovariateShift (to compute stds)
    target_cols  : override default column selection
    **kwargs     : additional parameters forwarded to the injector class
    """
    if drift_type not in DRIFT_REGISTRY:
        raise ValueError(f"Unknown drift type: {drift_type}. "
                         f"Choose from {list(DRIFT_REGISTRY.keys())}")

    if isinstance(severity, str):
        severity = SEVERITY_PRESETS[severity]

    if target_cols is None:
        target_cols = get_default_target_cols(dataset_name, drift_type)

    cls = DRIFT_REGISTRY[drift_type]

    # Build kwargs specific to each injector type
    init_kwargs = dict(
        target_cols=target_cols,
        severity=severity,
        onset_batch=onset_batch,
        **kwargs,
    )

    if drift_type == "gradual_covariate_shift":
        if reference_df is None:
            raise ValueError("GradualCovariateShift requires reference_df.")
        ref_stds = reference_df[target_cols].std().to_dict()
        init_kwargs["reference_stds"] = ref_stds

    return cls(**init_kwargs)

def run_pipeline_with_drift(
    pipeline,
    test_df:       pd.DataFrame,
    injector:      BaseDriftInjector,
    inject_stage:  str = "ingest",
) -> Tuple[List[pd.DataFrame], "StageLogger"]:
    """
    Run the Phase 2 pipeline on test data with drift injected at a
    specific stage.

    This is the core experimental function.  It:
      1. Streams test data in temporal batches (via the pipeline's ingester)
      2. At the designated stage, applies the drift injector to the batch
      3. Logs data at every stage boundary (pre- and post-injection)
      4. Returns the per-batch results and the populated logger

    Parameters
    ----------
    pipeline      : a fitted DriftDetectionPipeline from Phase 2
    test_df       : the test partition DataFrame
    injector      : a configured BaseDriftInjector
    inject_stage  : which stage to inject at: 'ingest', 'clean',
                    'featurize', or 'predict'

    Returns
    -------
    results : list of per-batch predicted DataFrames
    logger  : StageLogger with full stage-level snapshots
    """
    valid_stages = ["ingest", "clean", "featurize", "predict"]
    if inject_stage not in valid_stages:
        raise ValueError(f"inject_stage must be one of {valid_stages}")

    # Use a fresh logger for this trial.
    # BatchIngester and StageLogger are already in scope from Phase 2 cells.
    logger = StageLogger(store_dataframes=False)
    ingester = BatchIngester(test_df, pipeline.batch_size)
    results = []

    for batch_id, raw_batch in ingester:

        # ── Stage 1: Ingest ─────────────────────────────────────────────
        if inject_stage == "ingest":
            raw_batch = injector(raw_batch, batch_id)
        logger.log(batch_id, "ingest", raw_batch)

        # ── Stage 2: Clean ──────────────────────────────────────────────
        cleaned = pipeline.cleaner.transform(raw_batch)
        if inject_stage == "clean":
            cleaned = injector(cleaned, batch_id)
        logger.log(batch_id, "clean", cleaned)

        # ── Stage 3: Featurize ──────────────────────────────────────────
        featured = pipeline.engineer.transform(cleaned)
        if inject_stage == "featurize":
            featured = injector(featured, batch_id)
        logger.log(batch_id, "featurize", featured)

        # ── Stage 4: Predict ────────────────────────────────────────────
        predicted = pipeline.predictor.predict(featured)
        if inject_stage == "predict":
            predicted = injector(predicted, batch_id)
        logger.log(batch_id, "predict", predicted)

        results.append(predicted)

    return results, logger

def compute_divergence(
    reference_vals: np.ndarray,
    drifted_vals:   np.ndarray,
    n_bins: int = 50,
) -> Dict[str, float]:
    """
    Compute divergence statistics between a reference and drifted sample.

    Returns:
      - ks_stat      : Kolmogorov-Smirnov test statistic
      - ks_pvalue    : KS test p-value
      - js_divergence: Jensen-Shannon divergence (symmetric, bounded)
      - mean_shift   : absolute difference in means
      - std_ratio    : ratio of standard deviations
    """
    ref_clean = reference_vals[~np.isnan(reference_vals)]
    dri_clean = drifted_vals[~np.isnan(drifted_vals)]

    # KS test
    if len(ref_clean) > 0 and len(dri_clean) > 0:
        ks_stat, ks_pval = ks_2samp(ref_clean, dri_clean)
    else:
        ks_stat, ks_pval = np.nan, np.nan

    # Jensen-Shannon divergence (histogram-based)
    if len(ref_clean) > 0 and len(dri_clean) > 0:
        combined_min = min(ref_clean.min(), dri_clean.min())
        combined_max = max(ref_clean.max(), dri_clean.max())
        bins = np.linspace(combined_min, combined_max, n_bins + 1)

        ref_hist, _ = np.histogram(ref_clean, bins=bins, density=True)
        dri_hist, _ = np.histogram(dri_clean, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        ref_hist = ref_hist + 1e-10
        dri_hist = dri_hist + 1e-10
        ref_hist = ref_hist / ref_hist.sum()
        dri_hist = dri_hist / dri_hist.sum()

        js_div = float(jensenshannon(ref_hist, dri_hist) ** 2)
    else:
        js_div = np.nan

    # Simple statistics
    mean_shift = abs(np.nanmean(drifted_vals) - np.nanmean(reference_vals))
    ref_std = np.nanstd(reference_vals)
    dri_std = np.nanstd(drifted_vals)
    std_ratio = dri_std / ref_std if ref_std > 0 else np.nan

    # Null rate change
    ref_null_rate = np.isnan(reference_vals).mean()
    dri_null_rate = np.isnan(drifted_vals).mean()

    return {
        "ks_stat":       round(ks_stat, 4),
        "ks_pvalue":     round(ks_pval, 6),
        "js_divergence": round(js_div, 4),
        "mean_shift":    round(mean_shift, 6),
        "std_ratio":     round(std_ratio, 4),
        "ref_null_rate": round(ref_null_rate, 4),
        "dri_null_rate": round(dri_null_rate, 4),
    }

def validate_injector(
    injector:      BaseDriftInjector,
    reference_df:  pd.DataFrame,
    dataset_name:  str,
    n_rows:        int = 2000,
    sim_batch_id:  int = 30,
):
    """
    Visual and statistical validation of a single drift injector.

    Takes a sample of reference data, applies the injector as if it were
    at the given sim_batch_id (well past onset), and compares before/after.
    """
    sample = reference_df.head(n_rows).copy()
    drifted = injector(sample, sim_batch_id)

    # Determine which columns to inspect
    if injector.drift_type == "LabelShift":
        inspect_cols = ["target"]
    else:
        inspect_cols = [c for c in injector.target_cols if c in sample.columns]

    if not inspect_cols:
        print(f"  No inspectable columns for {injector}")
        return

    n_plots = len(inspect_cols)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    divergence_results = {}

    for ax, col in zip(axes, inspect_cols):
        ref_vals = sample[col].values.astype(float)
        dri_vals = drifted[col].values.astype(float)

        # Plot overlapping histograms
        ax.hist(ref_vals[~np.isnan(ref_vals)], bins=50, alpha=0.5,
                density=True, color="#2196F3", label="reference", edgecolor="none")
        ax.hist(dri_vals[~np.isnan(dri_vals)], bins=50, alpha=0.5,
                density=True, color="#E91E63", label="drifted", edgecolor="none")
        ax.set_title(f"{col}", fontsize=10)
        ax.legend(fontsize=8)

        # Compute divergence
        div = compute_divergence(ref_vals, dri_vals)
        divergence_results[col] = div
        ax.set_xlabel(f"KS={div['ks_stat']}  JS={div['js_divergence']}", fontsize=8)

    fig.suptitle(
        f"{dataset_name} — {injector.drift_type}  "
        f"(severity={injector.severity}, batch={sim_batch_id})",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    plt.show()

    # Print divergence table
    div_df = pd.DataFrame(divergence_results).T
    print(f"\n  Divergence statistics (simulated at batch {sim_batch_id}):")
    print(div_df.to_string())
    print()

    return divergence_results


elec_ref = phase1["electricity"]["partitions"]["reference"]

elec_injectors = {}
for drift_type in DRIFT_REGISTRY:
    print(f"\n── {drift_type} ──")
    injector = create_injector(
        drift_type=drift_type,
        dataset_name="electricity",
        severity="medium",
        onset_batch=0,
        reference_df=elec_ref,
    )
    elec_injectors[drift_type] = injector

cov_ref = phase1["covertype"]["partitions"]["reference"]

cov_injectors = {}
for drift_type in DRIFT_REGISTRY:
    print(f"\n── {drift_type} ──")
    injector = create_injector(
        drift_type=drift_type,
        dataset_name="covertype",
        severity="medium",
        onset_batch=0,
        reference_df=cov_ref,
    )
    cov_injectors[drift_type] = injector



def plot_drift_propagation(
    clean_logger,
    drift_logger,
    feature:      str,
    onset_batch:  int,
    dataset_name: str,
    drift_label:  str = "drift",
):
    """
    Compare a feature's batch mean across stages between a clean run
    and a drifted run.  Shows how drift propagates (or gets dampened)
    through the pipeline.
    """
    stages = ["ingest", "clean", "featurize", "predict"]
    colors = {"ingest": "#2196F3", "clean": "#FF9800",
              "featurize": "#4CAF50", "predict": "#E91E63"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    clean_summary = clean_logger.to_summary_df()
    drift_summary = drift_logger.to_summary_df()

    for ax, stage in zip(axes, stages):
        # Clean run
        c = clean_summary[
            (clean_summary["stage"] == stage) &
            (clean_summary["feature"] == feature)
        ].sort_values("batch_id")

        # Drifted run
        d = drift_summary[
            (drift_summary["stage"] == stage) &
            (drift_summary["feature"] == feature)
        ].sort_values("batch_id")

        if not c.empty:
            ax.plot(c["batch_id"], c["mean"], color="gray",
                    lw=1.5, alpha=0.7, label="clean")
        if not d.empty:
            ax.plot(d["batch_id"], d["mean"], color=colors[stage],
                    lw=1.5, label=drift_label)

        ax.axvline(onset_batch, color="red", ls="--", lw=1, alpha=0.6,
                   label="drift onset")
        ax.set_title(f"Stage: {stage}", fontsize=10)
        ax.set_ylabel("Batch mean")
        ax.legend(fontsize=8)

    axes[-2].set_xlabel("Batch ID")
    axes[-1].set_xlabel("Batch ID")

    fig.suptitle(
        f"{dataset_name} — '{feature}' drift propagation through stages",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    plt.show()

OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

phase3_artifacts = {
    "drift_registry":        DRIFT_REGISTRY,
    "severity_presets":       SEVERITY_PRESETS,
    "create_injector":       create_injector,         # factory function
    "run_pipeline_with_drift": run_pipeline_with_drift,  # integration fn
    "compute_divergence":    compute_divergence,
    "validation_injectors": {
        "electricity": elec_injectors,
        "covertype":   cov_injectors,
    },
}

save_path = os.path.join(OUTPUT_DIR, "phase3_artifacts.pkl")
with open(save_path, "wb") as f:
    pickle.dump(phase3_artifacts, f)
