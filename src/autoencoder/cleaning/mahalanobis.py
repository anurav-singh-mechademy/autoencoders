"""Step 3: Mahalanobis distance -- flag windows far from the multivariate mean."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

logger = logging.getLogger(__name__)


@dataclass
class MahalanobisResult:
    """Result of Mahalanobis distance-based cleaning."""

    is_outlier: np.ndarray          # Boolean mask -- True = outlier
    distances: np.ndarray           # Per-window Mahalanobis distance
    n_removed: int
    n_total: int
    threshold: float
    chi2_percentile: float


def clean_mahalanobis(
    windows: list[np.ndarray],
    chi2_percentile: float = 97.5,
) -> MahalanobisResult:
    """Flag windows whose Mahalanobis distance exceeds a chi-squared threshold.

    Works best after Isolation Forest and PCA have already removed gross outliers,
    since the covariance matrix estimate is more reliable on cleaner data.

    Args:
        windows: List of (window_size, n_sensors) arrays.
        chi2_percentile: Percentile of the chi-squared distribution for the threshold.

    Returns:
        MahalanobisResult with boolean mask and distances.
    """
    window_means = np.array([w.mean(axis=0) for w in windows])
    n_windows, n_features = window_means.shape

    mean_vec = np.mean(window_means, axis=0)
    cov_matrix = np.cov(window_means, rowvar=False)

    # Regularise covariance to avoid singularity
    cov_matrix += np.eye(n_features) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix singular even after regularisation. Using pseudo-inverse.")
        cov_inv = np.linalg.pinv(cov_matrix)

    distances = np.array([
        mahalanobis(row, mean_vec, cov_inv) for row in window_means
    ])

    # Threshold from chi-squared distribution with n_features degrees of freedom
    threshold = np.sqrt(chi2.ppf(chi2_percentile / 100.0, df=n_features))

    is_outlier = distances > threshold
    n_removed = int(np.sum(is_outlier))

    logger.info(
        "Mahalanobis cleaning: %d / %d windows flagged (threshold=%.2f, chi2_pct=%.1f).",
        n_removed, len(windows), threshold, chi2_percentile,
    )

    return MahalanobisResult(
        is_outlier=is_outlier,
        distances=distances,
        n_removed=n_removed,
        n_total=len(windows),
        threshold=threshold,
        chi2_percentile=chi2_percentile,
    )


def apply_mahalanobis_cleaning(
    windows: list[np.ndarray],
    result: MahalanobisResult,
) -> list[np.ndarray]:
    """Remove windows flagged by Mahalanobis distance cleaning."""
    clean = [w for w, is_out in zip(windows, result.is_outlier) if not is_out]
    logger.info(
        "Mahalanobis cleaning: removed %d windows. %d remaining.",
        result.n_removed, len(clean),
    )
    return clean
