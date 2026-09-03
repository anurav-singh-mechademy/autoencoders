"""Step 1: Isolation Forest -- global multivariate outlier detection per regime."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class IsolationForestResult:
    """Result of Isolation Forest cleaning."""

    is_outlier: np.ndarray        # Boolean mask -- True = outlier
    scores: np.ndarray            # Anomaly scores (lower = more anomalous)
    n_removed: int
    n_total: int
    contamination: float


def clean_isolation_forest(
    windows: list[np.ndarray],
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForestResult:
    """Flag outlier windows using Isolation Forest on per-window mean sensor values.

    Each window is represented by its mean across all 120 rows (one feature vector
    per window). The Isolation Forest identifies windows whose mean sensor profile
    deviates from the majority.

    Args:
        windows: List of (window_size, n_sensors) arrays.
        contamination: Expected fraction of outliers (default 5%).
        n_estimators: Number of trees.
        random_state: Random seed.

    Returns:
        IsolationForestResult with boolean mask and anomaly scores.
    """
    # Represent each window by its mean sensor values
    window_means = np.array([w.mean(axis=0) for w in windows])

    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    predictions = iso.fit_predict(window_means)  # +1 = inlier, -1 = outlier
    scores = iso.decision_function(window_means)

    is_outlier = predictions == -1
    n_removed = int(np.sum(is_outlier))

    logger.info(
        "Isolation Forest: %d / %d windows flagged as outliers (contamination=%.2f).",
        n_removed, len(windows), contamination,
    )

    return IsolationForestResult(
        is_outlier=is_outlier,
        scores=scores,
        n_removed=n_removed,
        n_total=len(windows),
        contamination=contamination,
    )


def apply_isolation_forest(
    windows: list[np.ndarray],
    result: IsolationForestResult,
) -> list[np.ndarray]:
    """Remove windows flagged as outliers by Isolation Forest."""
    clean = [w for w, is_out in zip(windows, result.is_outlier) if not is_out]
    logger.info(
        "Isolation Forest: removed %d windows. %d remaining.",
        result.n_removed, len(clean),
    )
    return clean
