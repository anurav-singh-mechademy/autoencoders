"""Cleaning pipeline orchestrator -- runs the full sequence per regime."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from autoencoder.cleaning.isolation_forest import (
    apply_isolation_forest,
    clean_isolation_forest,
    IsolationForestResult,
)
from autoencoder.cleaning.mahalanobis import (
    apply_mahalanobis_cleaning,
    clean_mahalanobis,
    MahalanobisResult,
)
from autoencoder.cleaning.pca import (
    apply_pca_cleaning,
    clean_pca,
    PCACleaningResult,
)
from autoencoder.cleaning.validation import validate_cleaning, ValidationReport
from autoencoder.data.regime import RegimeResult

logger = logging.getLogger(__name__)


@dataclass
class CleaningStepLog:
    """Log entry for a single cleaning step."""

    step_name: str
    regime: str
    windows_before: int
    windows_after: int
    windows_removed: int


@dataclass
class CleaningPipelineResult:
    """Full result of the cleaning pipeline."""

    cleaned_windows: list[np.ndarray]
    cleaned_regime_labels: list[int]
    original_count: int
    cleaned_count: int
    # Windows removed by isolation_forest/pca/mahalanobis (NOT the sensor-dropout
    # filter upstream of this pipeline, which runs on possibly-null-containing
    # windows -- these are always full, valid numeric arrays). Kept out of
    # autoencoder training as designed, but useful elsewhere as real examples of
    # sensor-level deviation -- e.g. FastSHAP's explainer training set, which
    # otherwise only ever sees close-to-median "normal" rows and has nothing to
    # learn row-specific attribution from (see explain/fastshap.py).
    rejected_windows: list[np.ndarray] = field(default_factory=list)
    step_logs: list[CleaningStepLog] = field(default_factory=list)
    per_step_removals: dict[str, int] = field(default_factory=dict)
    per_regime_results: dict[str, dict] = field(default_factory=dict)
    validation_report: Optional[ValidationReport] = None

    # Detailed results for reporting
    if_results: dict[str, IsolationForestResult] = field(default_factory=dict)
    pca_results: dict[str, PCACleaningResult] = field(default_factory=dict)
    md_results: dict[str, MahalanobisResult] = field(default_factory=dict)


DEFAULT_STEPS = ("isolation_forest", "pca", "mahalanobis")


def run_cleaning_pipeline(
    windows: list[np.ndarray],
    regime_result: RegimeResult,
    sensor_columns: list[str],
    output_dir: Optional[str | Path] = None,
    # Isolation Forest params
    if_contamination: float = 0.05,
    if_n_estimators: int = 200,
    if_random_state: int = 42,
    # PCA params
    pca_variance_threshold: float = 0.95,
    pca_outlier_std: float = 3.0,
    # Mahalanobis params
    md_chi2_percentile: float = 97.5,
    # Validation params
    max_removal_pct: float = 40.0,
    n_sensor_pairs: int = 10,
    # Which steps to run, and in what order. Default matches the historical
    # hardcoded behaviour: IF -> PCA -> Mahalanobis, all three.
    steps: tuple[str, ...] | list[str] = DEFAULT_STEPS,
) -> CleaningPipelineResult:
    """Run the cleaning pipeline (subset/order of IF, PCA, Mahalanobis), per regime.

    Cleaning is applied separately within each operating regime, then results are
    recombined. This prevents healthy low-load windows from being incorrectly
    flagged when the majority of data is at high load.

    `cleaned_windows` (and the returned `cleaned_regime_labels`) are ordered by
    regime, not by time -- callers that split train/val/test must pass
    `cleaned_regime_labels` to `split_windows` to avoid cutting the split along
    regime boundaries.

    Args:
        windows: All windows after transient removal.
        regime_result: Result of regime segmentation (labels per window).
        sensor_columns: Sensor column names for validation plots.
        output_dir: Directory to save validation plots. If None, skips validation plots.
        if_contamination: Isolation Forest contamination parameter.
        if_n_estimators: Isolation Forest number of trees.
        if_random_state: Random seed.
        pca_variance_threshold: Cumulative variance to retain in PCA.
        pca_outlier_std: Std multiplier for PCA outlier threshold.
        md_chi2_percentile: Chi-squared percentile for Mahalanobis threshold.
        max_removal_pct: Warn if more than this % removed overall.
        n_sensor_pairs: Number of sensor pairs for scatter plots.
        steps: Ordered subset of ("isolation_forest", "pca", "mahalanobis") to run.
            Lets callers disable or reorder stages (e.g. for ablation studies)
            without touching the per-stage cleaning logic.

    Returns:
        CleaningPipelineResult with cleaned windows and full audit trail.
    """
    unknown = set(steps) - set(DEFAULT_STEPS)
    if unknown:
        raise ValueError(f"Unknown cleaning step(s): {sorted(unknown)}. Valid: {DEFAULT_STEPS}")

    original_count = len(windows)
    all_cleaned: list[np.ndarray] = []
    all_cleaned_regimes: list[int] = []
    all_rejected: list[np.ndarray] = []
    step_logs: list[CleaningStepLog] = []
    per_step_removals: dict[str, int] = {
        "isolation_forest": 0,
        "pca": 0,
        "mahalanobis": 0,
    }
    if_results_dict: dict[str, IsolationForestResult] = {}
    pca_results_dict: dict[str, PCACleaningResult] = {}
    md_results_dict: dict[str, MahalanobisResult] = {}
    per_regime: dict[str, dict] = {}

    for regime_idx in range(regime_result.n_clusters):
        regime_name = regime_result.regime_names[regime_idx]
        regime_mask = regime_result.labels == regime_idx
        regime_windows = [w for w, m in zip(windows, regime_mask) if m]

        if len(regime_windows) < 10:
            logger.warning(
                "Regime '%s' has only %d windows. Skipping cleaning, keeping all.",
                regime_name, len(regime_windows),
            )
            all_cleaned.extend(regime_windows)
            all_cleaned_regimes.extend([regime_idx] * len(regime_windows))
            per_regime[regime_name] = {"original": len(regime_windows), "cleaned": len(regime_windows)}
            continue

        logger.info("Cleaning regime '%s' (%d windows)...", regime_name, len(regime_windows))
        current = regime_windows

        for step_name in steps:
            if len(current) < 10:
                logger.warning(
                    "Regime '%s' has only %d windows left before '%s'. Skipping remaining steps.",
                    regime_name, len(current), step_name,
                )
                break

            if step_name == "isolation_forest":
                if_result = clean_isolation_forest(
                    current, contamination=if_contamination,
                    n_estimators=if_n_estimators, random_state=if_random_state,
                )
                before = len(current)
                all_rejected.extend(w for w, is_out in zip(current, if_result.is_outlier) if is_out)
                current = apply_isolation_forest(current, if_result)
                removed = before - len(current)
                per_step_removals["isolation_forest"] += removed
                if_results_dict[regime_name] = if_result
                step_logs.append(CleaningStepLog("isolation_forest", regime_name, before, len(current), removed))

            elif step_name == "pca":
                pca_result = clean_pca(
                    current, variance_threshold=pca_variance_threshold,
                    outlier_std=pca_outlier_std,
                )
                before = len(current)
                all_rejected.extend(w for w, is_out in zip(current, pca_result.is_outlier) if is_out)
                current = apply_pca_cleaning(current, pca_result)
                removed = before - len(current)
                per_step_removals["pca"] += removed
                pca_results_dict[regime_name] = pca_result
                step_logs.append(CleaningStepLog("pca", regime_name, before, len(current), removed))

            elif step_name == "mahalanobis":
                md_result = clean_mahalanobis(
                    current, chi2_percentile=md_chi2_percentile,
                )
                before = len(current)
                all_rejected.extend(w for w, is_out in zip(current, md_result.is_outlier) if is_out)
                current = apply_mahalanobis_cleaning(current, md_result)
                removed = before - len(current)
                per_step_removals["mahalanobis"] += removed
                md_results_dict[regime_name] = md_result
                step_logs.append(CleaningStepLog("mahalanobis", regime_name, before, len(current), removed))

        all_cleaned.extend(current)
        all_cleaned_regimes.extend([regime_idx] * len(current))
        per_regime[regime_name] = {"original": len(regime_windows), "cleaned": len(current)}

    # Validation
    validation_report = None
    if output_dir:
        validation_report = validate_cleaning(
            original_windows=windows,
            cleaned_windows=all_cleaned,
            per_step_removals=per_step_removals,
            sensor_columns=sensor_columns,
            output_dir=output_dir,
            n_sensor_pairs=n_sensor_pairs,
            max_removal_pct=max_removal_pct,
        )

    logger.info(
        "Cleaning pipeline complete: %d -> %d windows (%.1f%% removed).",
        original_count, len(all_cleaned),
        (1 - len(all_cleaned) / original_count) * 100 if original_count > 0 else 0,
    )

    return CleaningPipelineResult(
        cleaned_windows=all_cleaned,
        cleaned_regime_labels=all_cleaned_regimes,
        original_count=original_count,
        cleaned_count=len(all_cleaned),
        rejected_windows=all_rejected,
        step_logs=step_logs,
        per_step_removals=per_step_removals,
        per_regime_results=per_regime,
        validation_report=validation_report,
        if_results=if_results_dict,
        pca_results=pca_results_dict,
        md_results=md_results_dict,
    )
