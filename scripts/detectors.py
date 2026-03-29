import warnings
warnings.filterwarnings("ignore")

import os, pickle, time, copy
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from pipeline import (
    StageLog, StageLogger, BatchIngester, DataCleaner,
    FeatureEngineer, PredictionStage, DriftDetectionPipeline
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")


with open("/content/drive/MyDrive/Project Phases/Phase 1/phase1_artifacts.pkl", "rb") as f:
    phase1 = pickle.load(f)

with open("/content/drive/MyDrive/Project Phases/Phase 2/phase2_outputs/phase2_artifacts.pkl", "rb") as f:
    phase2 = pickle.load(f)

from collections import deque

class PageHinkleyDetector:
    """
    Page-Hinkley test with sliding-window reference mean and forgetting.

    Problems with the original implementation:
      1. The global running mean (sum_x / n) becomes rigid after many
         batches — each new observation barely moves it, so even normal
         fluctuations accumulate as deviations.
      2. The cumulative sum grows without bound on long stationary runs,
         causing the false alarm rate to increase with run length.

    This version fixes both:
      - The reference mean is computed over a sliding window of the last
        `mean_window` observations (default 20).  This keeps the baseline
        adaptive to recent data without rigidifying over time.
      - The forgetting factor (alpha) decays old cumulative sum entries,
        preventing unbounded growth.

    Together, these ensure the PH statistic stays bounded on stationary
    data regardless of run length, while still accumulating evidence
    quickly during a real sustained shift.

    Parameters
    ----------
    threshold   : alarm threshold for the PH statistic.
    delta       : minimum change magnitude to detect (tolerance band).
    alpha       : forgetting factor in (0, 1].  Default 0.99.
    mean_window : number of recent observations for the rolling mean.
                  Default 20.
    """

    def __init__(
        self,
        threshold:   float = 50.0,
        delta:       float = 0.005,
        alpha:       float = 0.99,
        mean_window: int   = 20,
    ):
        self.threshold   = threshold
        self.delta       = delta
        self.alpha       = alpha
        self.mean_window = mean_window
        self.name        = "PageHinkley"
        self.reset()

    def reset(self):
        """Clear all internal state for a fresh trial."""
        self.window    = deque(maxlen=self.mean_window)
        self.cum_sum   = 0.0
        self.min_cum   = 0.0
        self.ph_values = []
        self.alarms    = []

    def update(self, value: float, batch_id: int = None) -> bool:
        """
        Ingest one observation (e.g., a batch mean).

        Returns True if an alarm fires on this step.
        """
        self.window.append(value)

        # Rolling mean over the sliding window only
        rolling_mean = sum(self.window) / len(self.window)

        # Forgetting factor decays old accumulations
        self.cum_sum = (
            self.alpha * self.cum_sum +
            (value - rolling_mean - self.delta)
        )
        self.min_cum = min(self.min_cum, self.cum_sum)

        ph_stat = self.cum_sum - self.min_cum
        self.ph_values.append(ph_stat)

        alarm = ph_stat > self.threshold
        if alarm and batch_id is not None:
            self.alarms.append(batch_id)

        return alarm

    def get_statistic_history(self) -> List[float]:
        """Return the full PH statistic time series."""
        return self.ph_values

    def __repr__(self):
        return (f"PageHinkleyDetector(threshold={self.threshold}, "
                f"delta={self.delta}, alpha={self.alpha}, "
                f"mean_window={self.mean_window})")

class WindowedKSDetector:
    """
    Windowed two-sample Kolmogorov-Smirnov test.

    Compares the empirical distribution of the most recent `window_size`
    batch-level observations against a stored reference window.  Fires
    an alarm when the KS statistic exceeds a threshold (or equivalently,
    when the p-value falls below a significance level).

    Unlike the Page-Hinkley test, this method is sensitive to *any* kind
    of distributional change — not just shifts in the mean — because it
    compares entire CDFs.

    Parameters
    ----------
    reference_values : 1-D array of values from the reference partition.
                       This is the "known-good" distribution.
    window_size      : number of recent observations to include in the
                       current window.
    threshold        : KS statistic above which an alarm fires.

    Usage: call .update(value) once per batch.
    """

    def __init__(
        self,
        reference_values: np.ndarray,
        window_size:      int   = 5,
        threshold:        float = 0.3,
    ):
        self.reference   = np.asarray(reference_values, dtype=np.float64)
        self.window_size = window_size
        self.threshold   = threshold
        self.name        = "WindowedKS"
        self.reset()

    def reset(self):
        """Clear batch window and history for a fresh trial."""
        self.batch_window = []   # stores arrays, not scalars
        self.ks_values    = []
        self.p_values     = []
        self.alarms       = []

    def update(self, value: float, batch_id: int = None) -> bool:
        """
        Scalar fallback — wraps the value in an array.
        Prefer update_batch() when full batch data is available.
        """
        return self.update_batch(np.array([value]), batch_id)

    def update_batch(
        self,
        batch_values: np.ndarray,
        batch_id: int = None,
    ) -> bool:
        """
        Ingest a full batch of raw feature values.

        Pools the last window_size batches and runs KS against reference.
        Returns True if an alarm fires.
        """
        clean_vals = batch_values[~np.isnan(batch_values)]
        self.batch_window.append(clean_vals)

        # Only test once we have enough batches
        if len(self.batch_window) < self.window_size:
            self.ks_values.append(0.0)
            self.p_values.append(1.0)
            return False

        # Keep only the most recent window_size batches
        recent = self.batch_window[-self.window_size:]

        # Pool all values from recent batches into one sample
        pooled = np.concatenate(recent)

        if len(pooled) < 5:
            self.ks_values.append(0.0)
            self.p_values.append(1.0)
            return False

        ks_stat, p_val = ks_2samp(self.reference, pooled)
        self.ks_values.append(ks_stat)
        self.p_values.append(p_val)

        alarm = ks_stat > self.threshold
        if alarm and batch_id is not None:
            self.alarms.append(batch_id)

        return alarm

    def get_statistic_history(self) -> List[float]:
        """Return the full KS statistic time series."""
        return self.ks_values

    def __repr__(self):
        return (f"WindowedKSDetector(window={self.window_size}, "
                f"threshold={self.threshold})")

class PSIDetector:
    """
    Population Stability Index (PSI) drift detector.

    PSI compares the current distribution against a reference by dividing
    both into bins and computing:

        PSI = Σ (p_i - q_i) × ln(p_i / q_i)

    where p is the current proportion and q is the reference proportion
    in each bin.

    Standard industry thresholds:
        PSI < 0.10  → no significant drift
        0.10 ≤ PSI < 0.25  → moderate drift
        PSI ≥ 0.25  → significant drift

    Parameters
    ----------
    reference_values : 1-D array from the reference partition.
    n_bins           : number of bins (quantile-based by default).
    threshold        : PSI value above which an alarm fires.
    binning          : 'quantile' or 'uniform'.

    Usage: call .update(value) once per batch.  Unlike the other detectors
    which take scalar summary stats, PSI's .update_batch() can take a full
    array for higher fidelity.  The scalar .update() method accumulates a
    window and computes PSI when the window is full.
    """

    def __init__(
        self,
        reference_values: np.ndarray,
        n_bins:           int   = 10,
        threshold:        float = 0.25,
        binning:          str   = "quantile",
        window_size:      int   = 10,
    ):
        self.reference   = np.asarray(reference_values, dtype=np.float64)
        self.n_bins      = n_bins
        self.threshold   = threshold
        self.binning     = binning
        self.window_size = window_size
        self.name        = "PSI"

        # Precompute bin edges and reference proportions
        self._compute_reference_bins()
        self.reset()

    def _compute_reference_bins(self):
        """Compute bin edges and reference bin proportions."""
        ref_clean = self.reference[~np.isnan(self.reference)]

        if self.binning == "quantile":
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges = np.percentile(ref_clean, quantiles)
            # Ensure unique bin edges
            self.bin_edges = np.unique(self.bin_edges)
        else:
            self.bin_edges = np.linspace(
                ref_clean.min(), ref_clean.max(), self.n_bins + 1
            )

        # Reference bin proportions
        ref_counts, _ = np.histogram(ref_clean, bins=self.bin_edges)
        self.ref_proportions = ref_counts / ref_counts.sum()
        # Avoid zeros for numerical stability
        self.ref_proportions = np.clip(self.ref_proportions, 1e-8, None)

    def reset(self):
        """Clear window and history for a fresh trial."""
        self.window     = []
        self.psi_values = []
        self.alarms     = []

    def _compute_psi(self, current_values: np.ndarray) -> float:
        """Compute PSI between reference and current distributions."""
        cur_clean = current_values[~np.isnan(current_values)]
        if len(cur_clean) < 2:
            return 0.0

        cur_counts, _ = np.histogram(cur_clean, bins=self.bin_edges)
        cur_proportions = cur_counts / max(cur_counts.sum(), 1)
        cur_proportions = np.clip(cur_proportions, 1e-8, None)

        # PSI formula
        psi = np.sum(
            (cur_proportions - self.ref_proportions) *
            np.log(cur_proportions / self.ref_proportions)
        )
        return float(psi)

    def update(self, value: float, batch_id: int = None) -> bool:
        """
        Ingest one scalar observation (e.g. batch mean).

        Accumulates a window and computes PSI when the window is full.
        """
        self.window.append(value)

        if len(self.window) < self.window_size:
            self.psi_values.append(0.0)
            return False

        current = np.array(self.window[-self.window_size:])
        psi = self._compute_psi(current)
        self.psi_values.append(psi)

        alarm = psi > self.threshold
        if alarm and batch_id is not None:
            self.alarms.append(batch_id)

        return alarm

    def update_batch(self, batch_values: np.ndarray, batch_id: int = None) -> bool:
        """
        Ingest a full batch of raw values (higher fidelity than scalar).
        Computes PSI directly from the batch distribution.
        """
        psi = self._compute_psi(batch_values)
        self.psi_values.append(psi)

        alarm = psi > self.threshold
        if alarm and batch_id is not None:
            self.alarms.append(batch_id)

        return alarm

    def get_statistic_history(self) -> List[float]:
        """Return the full PSI time series."""
        return self.psi_values

    def __repr__(self):
        return (f"PSIDetector(n_bins={self.n_bins}, "
                f"threshold={self.threshold}, "
                f"binning={self.binning})")

class FeatureMonitor:
    """
    Monitors a single feature using all three detectors in parallel.

    Updated to pass the mean_window parameter to PageHinkley.
    """

    def __init__(
        self,
        feature_name:     str,
        reference_values: np.ndarray,
        burn_in:          Optional[int] = None,
        ph_threshold:     float = 50.0,
        ph_delta:         float = 0.005,
        ph_alpha:         float = 0.99,
        ph_mean_window:   int   = 20,
        ks_window:        int   = 5,
        ks_threshold:     float = 0.3,
        psi_n_bins:       int   = 10,
        psi_threshold:    float = 0.25,
        psi_window:       int   = 10,
    ):
        self.feature_name = feature_name

        # Default burn-in = longest warm-up across all detectors
        if burn_in is None:
            self.burn_in = max(ph_mean_window, ks_window, psi_window)
        else:
            self.burn_in = burn_in

        self.n_updates = 0

        self.detectors = {
            "PageHinkley": PageHinkleyDetector(
                threshold=ph_threshold,
                delta=ph_delta,
                alpha=ph_alpha,
                mean_window=ph_mean_window,
            ),
            "WindowedKS": WindowedKSDetector(
                reference_values=reference_values,
                window_size=ks_window,
                threshold=ks_threshold,
            ),
            "PSI": PSIDetector(
                reference_values=reference_values,
                n_bins=psi_n_bins,
                threshold=psi_threshold,
                window_size=psi_window,
            ),
        }


        self._burn_in_complete = False

    def _check_burn_in(self):
        """
        At the end of burn-in, reset cumulative state in each detector
        while preserving their filled windows.
        """
        if not self._burn_in_complete and self.n_updates > self.burn_in:
            self._burn_in_complete = True

            # Page-Hinkley: reset cumulative sum but keep the filled window
            ph = self.detectors["PageHinkley"]
            ph.cum_sum = 0.0
            ph.min_cum = 0.0
            ph.alarms.clear()

            # WindowedKS: clear any alarms from warm-up
            ks = self.detectors["WindowedKS"]
            ks.batch_window.clear()
            ks.alarms.clear()

            # PSI: clear any alarms from warm-up
            psi = self.detectors["PSI"]
            psi.alarms.clear()

    def update(self, value: float, batch_id: int) -> Dict[str, bool]:
        self.n_updates += 1

        if self.n_updates <= self.burn_in:
            # During burn-in: compute stats, suppress alarms
            for det in self.detectors.values():
                det.update(value, None)
            return {name: False for name in self.detectors}

        self._check_burn_in()

        results = {}
        for name, det in self.detectors.items():
            results[name] = det.update(value, batch_id)
        return results

    def update_with_batch(self, scalar_value, batch_values, batch_id):
        self.n_updates += 1

        if self.n_updates <= self.burn_in:
            self.detectors["PageHinkley"].update(scalar_value, None)
            self.detectors["WindowedKS"].update_batch(batch_values, None)
            self.detectors["PSI"].update_batch(batch_values, None)
            return {name: False for name in self.detectors}

        self._check_burn_in()

        results = {}
        results["PageHinkley"] = self.detectors["PageHinkley"].update(
            scalar_value, batch_id)
        results["WindowedKS"] = self.detectors["WindowedKS"].update_batch(
            batch_values, batch_id)
        results["PSI"] = self.detectors["PSI"].update_batch(
            batch_values, batch_id)
        return results

    def reset(self):
        self.n_updates = 0
        self._burn_in_complete = False
        for det in self.detectors.values():
            det.reset()

    def get_alarms(self) -> Dict[str, List[int]]:
        return {name: det.alarms for name, det in self.detectors.items()}

    def __repr__(self):
        return f"FeatureMonitor({self.feature_name}, burn_in={self.burn_in})"

class StageMonitor:
    """
    Monitors all tracked features at a single pipeline stage.

    Wraps one FeatureMonitor per feature, all sharing the same detector
    configuration.  Call .update_from_log() with a StageLog entry to
    automatically route each feature's batch mean to its monitor.

    Parameters
    ----------
    stage_name       : which pipeline stage this monitor observes.
    feature_names    : list of features to track.
    reference_df     : reference partition DataFrame (for extracting
                       per-feature reference distributions).
    detector_params  : dict of threshold/window parameters forwarded to
                       each FeatureMonitor.
    """

    def __init__(
        self,
        stage_name:      str,
        feature_names:   List[str],
        reference_df:    pd.DataFrame,
        detector_params: Optional[Dict] = None,
    ):
        self.stage_name    = stage_name
        self.feature_names = feature_names
        params = detector_params or {}

        self.feature_monitors: Dict[str, FeatureMonitor] = {}
        for feat in feature_names:
            if feat in reference_df.columns:
                ref_vals = reference_df[feat].dropna().values
                self.feature_monitors[feat] = FeatureMonitor(
                    feature_name=feat,
                    reference_values=ref_vals,
                    **params,
                )

    def update_from_log(self, log_entry) -> Dict[str, Dict[str, bool]]:
        """
        Feed a StageLog entry to all feature monitors.

        If the log entry contains a stored DataFrame, the KS and PSI
        detectors receive the full batch column.  Otherwise, all
        detectors receive the scalar batch mean.
        """
        results = {}
        has_df = log_entry.dataframe is not None

        for feat, monitor in self.feature_monitors.items():
            scalar = log_entry.numeric_means.get(feat)

            # If feature is entirely null, fall back to null rate as signal
            if scalar is None or (isinstance(scalar, float) and np.isnan(scalar)):
                scalar = log_entry.null_rates.get(feat, 0.0)
                # No meaningful batch array for KS/PSI when feature is null
                results[feat] = monitor.update(scalar, log_entry.batch_id)
                continue

            if has_df and feat in log_entry.dataframe.columns:
                batch_arr = log_entry.dataframe[feat].values.astype(np.float64)
                results[feat] = monitor.update_with_batch(
                    scalar, batch_arr, log_entry.batch_id
                )
            else:
                results[feat] = monitor.update(scalar, log_entry.batch_id)

        return results

    def reset(self):
        """Reset all feature monitors for a fresh trial."""
        for monitor in self.feature_monitors.values():
            monitor.reset()

    def get_all_alarms(self) -> Dict[str, Dict[str, List[int]]]:
        """Return all alarms: {feature: {detector: [batch_ids]}}."""
        return {
            feat: monitor.get_alarms()
            for feat, monitor in self.feature_monitors.items()
        }

    def __repr__(self):
        return (f"StageMonitor(stage={self.stage_name}, "
                f"features={len(self.feature_monitors)})")

class PipelineMonitor:
    """
    Orchestrates drift detection across all four pipeline stages.

    Creates a StageMonitor at each stage boundary.  After a pipeline run,
    call .process_logs() with the StageLogger to feed all batch data
    through the monitors and collect alarms.

    Parameters
    ----------
    reference_data   : dict mapping stage_name → reference DataFrame at
                       that stage boundary.
    feature_names    : features to monitor (same across stages, though
                       featurize/predict stages may have additional cols).
    detector_params  : shared detector configuration.
    stages           : which stages to monitor (default: all four).
    """

    STAGES = ["ingest", "clean", "featurize", "predict"]

    def __init__(
        self,
        reference_data:  Dict[str, pd.DataFrame],
        feature_names:   List[str],
        detector_params: Optional[Dict] = None,
        stages:          Optional[List[str]] = None,
    ):
        self.stages = stages or self.STAGES
        self.feature_names = feature_names

        self.stage_monitors: Dict[str, StageMonitor] = {}
        for stage in self.stages:
            ref_df = reference_data.get(stage)
            if ref_df is not None:
                self.stage_monitors[stage] = StageMonitor(
                    stage_name=stage,
                    feature_names=feature_names,
                    reference_df=ref_df,
                    detector_params=detector_params,
                )

    def process_logs(self, logger) -> Dict[str, Dict]:
        """
        Feed all StageLog entries from a completed pipeline run through
        the corresponding stage monitors.

        Parameters
        ----------
        logger : a StageLogger with logs from a pipeline run.

        Returns
        -------
        Nested dict: {stage: {feature: {detector: [alarm_batch_ids]}}}
        """
        for log_entry in logger.logs:
            stage = log_entry.stage_name
            if stage in self.stage_monitors:
                self.stage_monitors[stage].update_from_log(log_entry)

        return self.get_all_alarms()

    def get_all_alarms(self) -> Dict[str, Dict]:
        """Return alarms across all stages, features, and detectors."""
        return {
            stage: monitor.get_all_alarms()
            for stage, monitor in self.stage_monitors.items()
        }

    def reset(self):
        """Reset all monitors for a fresh trial."""
        for monitor in self.stage_monitors.values():
            monitor.reset()

    def get_first_alarm_batch(self) -> Dict[str, Dict[str, Optional[int]]]:
        """
        For each stage × detector, return the earliest alarm batch.
        Returns None if no alarm was fired.

        Structure: {stage: {detector_name: first_alarm_batch_or_None}}
        """
        result = {}
        all_alarms = self.get_all_alarms()

        for stage, features in all_alarms.items():
            result[stage] = {}
            # Aggregate across all features: earliest alarm per detector
            detector_first = defaultdict(lambda: float("inf"))

            for feat, detectors in features.items():
                for det_name, batch_ids in detectors.items():
                    if batch_ids:
                        detector_first[det_name] = min(
                            detector_first[det_name], min(batch_ids)
                        )

            for det_name in ["PageHinkley", "WindowedKS", "PSI"]:
                val = detector_first.get(det_name, float("inf"))
                result[stage][det_name] = int(val) if val < float("inf") else None

        return result

    def __repr__(self):
        n_monitors = len(self.stage_monitors)
        return f"PipelineMonitor({n_monitors} stages)"

def build_stage_reference_data(pipeline, reference_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Push the reference partition through the pipeline and capture
    the transformed DataFrame at each stage boundary.

    This gives each StageMonitor the correct reference distribution
    for its position in the pipeline — the clean stage monitor
    compares against cleaned reference data, the featurize monitor
    against featurized reference data, etc.

    Parameters
    ----------
    pipeline     : a fitted DriftDetectionPipeline.
    reference_df : the raw reference partition.

    Returns
    -------
    Dict mapping stage name → reference DataFrame at that boundary.
    """
    ref_clean = pipeline.cleaner.transform(reference_df)
    ref_feat  = pipeline.engineer.transform(ref_clean)
    ref_pred  = pipeline.predictor.predict(ref_feat)

    return {
        "ingest":    reference_df,
        "clean":     ref_clean,
        "featurize": ref_feat,
        "predict":   ref_pred,
    }


def get_monitor_features(dataset_name: str) -> List[str]:
    """
    Return the list of features to monitor for a given dataset.
    Focuses on the most informative numeric features rather than
    monitoring every column (which would inflate false alarm rates
    via multiple testing).
    """
    if dataset_name == "electricity":
        return [
            "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer",
        ]
    elif dataset_name == "covertype":
        return [
            "Elevation", "Aspect", "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
    return []

def calibrate_thresholds(
    pipeline,
    calibration_df:  pd.DataFrame,
    reference_df:    pd.DataFrame,
    dataset_name:    str,
    target_far:      float = 0.05,
    detector_params: Optional[Dict] = None,
) -> Dict:
    """
    Calibrate thresholds on drift-free data.
    Updated to pass ph_alpha, ph_mean_window, and ph_delta through.
    """
    params = detector_params or {}
    features = get_monitor_features(dataset_name)
    stage_refs = build_stage_reference_data(pipeline, reference_df)

    # Run calibration data through the pipeline
    ingester = BatchIngester(calibration_df, pipeline.batch_size)
    logger = StageLogger(store_dataframes=True)

    for batch_id, raw_batch in ingester:
        logger.log(batch_id, "ingest", raw_batch)
        cleaned = pipeline.cleaner.transform(raw_batch)
        logger.log(batch_id, "clean", cleaned)
        featured = pipeline.engineer.transform(cleaned)
        logger.log(batch_id, "featurize", featured)
        predicted = pipeline.predictor.predict(featured)
        logger.log(batch_id, "predict", predicted)

    # Loose thresholds to record all statistics
    loose_params = {
        "ph_threshold":    1e12,
        "ph_delta":        params.get("ph_delta", 0.005),
        "ph_alpha":        params.get("ph_alpha", 0.99),
        "ph_mean_window":  params.get("ph_mean_window", 20),
        "ks_threshold":    1e12,
        "ks_window":       params.get("ks_window", 5),
        "psi_threshold":   1e12,
        "psi_n_bins":      params.get("psi_n_bins", 10),
        "psi_window":      params.get("psi_window", 10),
    }

    monitor = PipelineMonitor(
        reference_data=stage_refs,
        feature_names=features,
        detector_params=loose_params,
    )
    monitor.process_logs(logger)
    logger.clear()

    # Collect statistics
    all_stats = {"PageHinkley": [], "WindowedKS": [], "PSI": []}

    for stage, stage_mon in monitor.stage_monitors.items():
        for feat, feat_mon in stage_mon.feature_monitors.items():
            for det_name, det in feat_mon.detectors.items():
                history = det.get_statistic_history()
                burn_in = max(
                    params.get("ph_mean_window", 20),
                    params.get("ks_window", 5),
                    params.get("psi_window", 10),
                  )
                valid_stats = history[burn_in:]
                all_stats[det_name].extend(valid_stats)

    # Thresholds at (1 - target_far) quantile
    calibrated = {}
    quantile = 1.0 - target_far

    for det_name, stats in all_stats.items():
        stats_arr = np.array(stats)
        stats_arr = stats_arr[~np.isnan(stats_arr)]
        if len(stats_arr) > 0:
            calibrated[det_name] = float(np.quantile(stats_arr, quantile))
        else:
            calibrated[det_name] = float("inf")

    calibrated_params = {
        "ph_threshold":    calibrated["PageHinkley"],
        "ph_delta":        params.get("ph_delta", 0.005),
        "ph_alpha":        params.get("ph_alpha", 0.99),
        "ph_mean_window":  params.get("ph_mean_window", 20),
        "ks_window":       params.get("ks_window", 5),
        "ks_threshold":    calibrated["WindowedKS"],
        "psi_n_bins":      params.get("psi_n_bins", 10),
        "psi_threshold":   calibrated["PSI"],
        "psi_window":      params.get("psi_window", 10),
    }

    print(f"\n  {'='*55}")
    print(f"  CALIBRATION — {dataset_name}  (target FAR ≤ {target_far:.0%})")
    print(f"  {'='*55}")
    print(f"  Statistics collected per detector:")
    for det, stats in all_stats.items():
        if len(stats) > 0:
            print(f"    {det:14s}: {len(stats):>5} values  "
                  f"(median={np.median(stats):.4f}, "
                  f"q95={np.quantile(stats, 0.95):.4f})")
    print(f"\n  Calibrated thresholds:")
    for det, thr in calibrated.items():
        print(f"    {det:14s}: {thr:.6f}")
    print()

    return {
        "thresholds":        calibrated,
        "calibrated_params": calibrated_params,
        "statistics":        all_stats,
    }
"""
elec_pipeline = phase2["electricity"]["pipeline"]
elec_ref  = phase1["electricity"]["partitions"]["reference"]
elec_cal  = phase1["electricity"]["partitions"]["calibration"]

elec_calibration = calibrate_thresholds(
    pipeline=elec_pipeline,
    calibration_df=elec_cal,
    reference_df=elec_ref,
    dataset_name="electricity",
    target_far=0.05,
)

cov_pipeline = phase2["covertype"]["pipeline"]
cov_ref  = phase1["covertype"]["partitions"]["reference"]
cov_cal  = phase1["covertype"]["partitions"]["calibration"]

cov_calibration = calibrate_thresholds(
    pipeline=cov_pipeline,
    calibration_df=cov_cal,
    reference_df=cov_ref,
    dataset_name="covertype",
    target_far=0.05,
)
"""
def validate_false_alarm_rate(
    pipeline,
    test_df:           pd.DataFrame,
    reference_df:      pd.DataFrame,
    dataset_name:      str,
    calibrated_params: Dict,
    n_batches_max:     int = 200,
) -> pd.DataFrame:
    """
    Run the full monitoring system on drift-free test data and measure
    the empirical false alarm rate.

    Updated: uses store_dataframes=True so KS gets full batch distributions.
    """
    features = get_monitor_features(dataset_name)
    stage_refs = build_stage_reference_data(pipeline, reference_df)

    test_subset = test_df.head(n_batches_max * pipeline.batch_size)
    ingester = BatchIngester(test_subset, pipeline.batch_size)
    logger = StageLogger(store_dataframes=True)

    n_batches = 0
    for batch_id, raw_batch in ingester:
        logger.log(batch_id, "ingest", raw_batch)
        cleaned = pipeline.cleaner.transform(raw_batch)
        logger.log(batch_id, "clean", cleaned)
        featured = pipeline.engineer.transform(cleaned)
        logger.log(batch_id, "featurize", featured)
        predicted = pipeline.predictor.predict(featured)
        logger.log(batch_id, "predict", predicted)
        n_batches += 1

    # Deploy monitors with calibrated thresholds
    monitor = PipelineMonitor(
        reference_data=stage_refs,
        feature_names=features,
        detector_params=calibrated_params,
    )
    monitor.process_logs(logger)

    # Free stored DataFrames after processing
    logger.clear()

    # Compute false alarm rates
    rows = []
    all_alarms = monitor.get_all_alarms()

    for stage, features_dict in all_alarms.items():
        for feat, detectors_dict in features_dict.items():
            for det_name, alarm_batches in detectors_dict.items():
                n_alarms = len(alarm_batches)
                far = n_alarms / max(n_batches, 1)
                rows.append({
                    "stage":     stage,
                    "feature":   feat,
                    "detector":  det_name,
                    "n_alarms":  n_alarms,
                    "n_batches": n_batches,
                    "FAR":       round(far, 4),
                })

    far_df = pd.DataFrame(rows)

    print(f"\n  {'='*55}")
    print(f"  FALSE ALARM VALIDATION — {dataset_name}")
    print(f"  {'='*55}")
    print(f"  Batches processed: {n_batches}")

    summary = far_df.groupby(["stage", "detector"])["FAR"].agg(["mean", "max"])
    print(f"\n  Mean & max FAR by stage × detector:")
    print(summary.round(4).to_string())
    print()

    return far_df
"""
elec_far = validate_false_alarm_rate(
    pipeline=elec_pipeline,
    test_df=phase1["electricity"]["partitions"]["test"],
    reference_df=elec_ref,
    dataset_name="electricity",
    calibrated_params=elec_calibration["calibrated_params"],
)

cov_far = validate_false_alarm_rate(
    pipeline=cov_pipeline,
    test_df=phase1["covertype"]["partitions"]["test"],
    reference_df=cov_ref,
    dataset_name="covertype",
    calibrated_params=cov_calibration["calibrated_params"],
)
"""
from drift_injection import create_injector, run_pipeline_with_drift, BaseDriftInjector
def smoke_test_detection(
    pipeline,
    test_df:           pd.DataFrame,
    reference_df:      pd.DataFrame,
    dataset_name:      str,
    calibrated_params: Dict,
    drift_type:        str = "gradual_covariate_shift",
    severity:          str = "medium",
    onset_batch:       int = 10,
    inject_stage:      str = "ingest",
):
    """
    Quick end-to-end test: inject known drift and verify that at least
    one detector fires an alarm after the onset batch.
    """
    features = get_monitor_features(dataset_name)
    stage_refs = build_stage_reference_data(pipeline, reference_df)

    # Create injector
    injector = create_injector(
        drift_type=drift_type,
        dataset_name=dataset_name,
        severity=severity,
        onset_batch=onset_batch,
        reference_df=reference_df,
    )

    # Run pipeline with drift
    results, drift_logger = run_pipeline_with_drift(
        pipeline=pipeline,
        test_df=test_df,
        injector=injector,
        inject_stage=inject_stage,
    )

    # Deploy calibrated monitors
    monitor = PipelineMonitor(
        reference_data=stage_refs,
        feature_names=features,
        detector_params=calibrated_params,
    )
    monitor.process_logs(drift_logger)

    # Report first alarm per stage × detector
    first_alarms = monitor.get_first_alarm_batch()

    print(f"\n  {'='*55}")
    print(f"  SMOKE TEST — {dataset_name}")
    print(f"  {'='*55}")
    print(f"  Drift type    : {drift_type}")
    print(f"  Severity      : {severity}")
    print(f"  Onset batch   : {onset_batch}")
    print(f"  Inject stage  : {inject_stage}")
    print(f"\n  First alarm batch per stage × detector:")
    print(f"  {'':14s} {'PageHinkley':>14s} {'WindowedKS':>14s} {'PSI':>14s}")
    print(f"  {'-'*56}")

    any_detected = False
    for stage in ["ingest", "clean", "featurize", "predict"]:
        detectors = first_alarms.get(stage, {})
        ph = detectors.get("PageHinkley")
        ks = detectors.get("WindowedKS")
        psi = detectors.get("PSI")
        ph_str  = str(ph) if ph is not None else "—"
        ks_str  = str(ks) if ks is not None else "—"
        psi_str = str(psi) if psi is not None else "—"

        marker = " ← inject" if stage == inject_stage else ""
        print(f"  {stage:14s} {ph_str:>14s} {ks_str:>14s} {psi_str:>14s}{marker}")

        if any(v is not None for v in [ph, ks, psi]):
            any_detected = True

    status = "✓ DETECTED" if any_detected else "✗ MISSED"
    print(f"\n  {status}")
    print()

    return first_alarms, monitor

class NullInjector(BaseDriftInjector):
    """Does nothing. Used to measure baseline false alarm behaviour."""
    def inject(self, df, batch_id):
        return df

from collections import defaultdict

def null_smoke_test(pipeline, test_df, reference_df, dataset_name, calibrated_params):
    """Run the full monitoring pipeline with NO drift injected."""
    features = get_monitor_features(dataset_name)
    stage_refs = build_stage_reference_data(pipeline, reference_df)

    # Run pipeline with null injector (no drift)
    null_inj = NullInjector(target_cols=[], severity=0.0, onset_batch=30)
    results, logger = run_pipeline_with_drift(
        pipeline=pipeline,
        test_df=test_df,
        injector=null_inj,
        inject_stage="ingest",
    )

    # Deploy calibrated monitors
    monitor = PipelineMonitor(
        reference_data=stage_refs,
        feature_names=features,
        detector_params=calibrated_params,
    )
    monitor.process_logs(logger)
    logger.clear()

    first_alarms = monitor.get_first_alarm_batch()

    print(f"\n  {'='*55}")
    print(f"  NULL SMOKE TEST — {dataset_name} (no drift)")
    print(f"  {'='*55}")
    print(f"\n  First alarm batch per stage × detector:")
    print(f"  {'':14s} {'PageHinkley':>14s} {'WindowedKS':>14s} {'PSI':>14s}")
    print(f"  {'-'*56}")

    for stage in ["ingest", "clean", "featurize", "predict"]:
        d = first_alarms.get(stage, {})
        ph  = d.get("PageHinkley")
        ks  = d.get("WindowedKS")
        psi = d.get("PSI")
        print(f"  {stage:14s} "
              f"{str(ph) if ph is not None else '—':>14s} "
              f"{str(ks) if ks is not None else '—':>14s} "
              f"{str(psi) if psi is not None else '—':>14s}")

    print()
    return first_alarms, monitor
"""
OUTPUT_DIR = "phase4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

phase4_artifacts = {
    "electricity": {
        "calibration":      elec_calibration,
        "calibrated_params": elec_calibration["calibrated_params"],
        "false_alarm_df":   elec_far,
    },
    "covertype": {
        "calibration":      cov_calibration,
        "calibrated_params": cov_calibration["calibrated_params"],
        "false_alarm_df":   cov_far,
    },
    # Store detector class references for Phase 5+ reconstruction
    "detector_classes": {
        "PageHinkleyDetector": PageHinkleyDetector,
        "WindowedKSDetector":  WindowedKSDetector,
        "PSIDetector":         PSIDetector,
    },
    "monitor_classes": {
        "FeatureMonitor":  FeatureMonitor,
        "StageMonitor":    StageMonitor,
        "PipelineMonitor": PipelineMonitor,
    },
    "helper_functions": {
        "build_stage_reference_data": build_stage_reference_data,
        "get_monitor_features":       get_monitor_features,
        "calibrate_thresholds":       calibrate_thresholds,
    },
}

save_path = os.path.join(OUTPUT_DIR, "phase4_artifacts.pkl")
with open(save_path, "wb") as f:
    pickle.dump(phase4_artifacts, f)
"""
