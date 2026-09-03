"""Step 2: PCA-based cleaning -- remove windows deviating from principal sensor structure."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class PCACleaningResult:
    """Result of PCA-based cleaning."""

    is_outlier: np.ndarray          # Boolean mask -- True = outlier
    reconstruction_errors: np.ndarray  # Per-window reconstruction error
    n_removed: int
    n_total: int
    n_components: int
    explained_variance_ratio: np.ndarray
    threshold: float


def clean_pca(
    windows: list[np.ndarray],
    variance_threshold: float = 0.95,
    outlier_std: float = 3.0,
) -> PCACleaningResult:
    """Flag windows that deviate from the principal sensor structure.

    Approach:
        1. Represent each window by its mean sensor values.
        2. Fit PCA retaining components explaining variance_threshold of variance.
        3. Project to PCA space and reconstruct.
        4. Compute reconstruction error per window.
        5. Flag windows with error > mean + outlier_std * std.

    Args:
        windows: List of (window_size, n_sensors) arrays.
        variance_threshold: Cumulative variance to retain (default 0.95).
        outlier_std: Number of standard deviations for the outlier threshold.

    Returns:
        PCACleaningResult with boolean mask and reconstruction errors.
    """
    window_means = np.array([w.mean(axis=0) for w in windows])

    # Determine number of components
    pca_full = PCA().fit(window_means)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = min(n_components, window_means.shape[1])

    # Fit PCA with selected components
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(window_means)
    reconstructed = pca.inverse_transform(transformed)

    # Reconstruction error per window (MSE across sensors)
    recon_errors = np.mean((window_means - reconstructed) ** 2, axis=1)

    # Threshold: mean + outlier_std * std
    mean_err = np.mean(recon_errors)
    std_err = np.std(recon_errors)
    threshold = mean_err + outlier_std * std_err

    is_outlier = recon_errors > threshold
    n_removed = int(np.sum(is_outlier))

    logger.info(
        "PCA cleaning: %d / %d windows flagged (n_components=%d, threshold=%.4f).",
        n_removed, len(windows), n_components, threshold,
    )

    return PCACleaningResult(
        is_outlier=is_outlier,
        reconstruction_errors=recon_errors,
        n_removed=n_removed,
        n_total=len(windows),
        n_components=n_components,
        explained_variance_ratio=pca.explained_variance_ratio_,
        threshold=threshold,
    )


def apply_pca_cleaning(
    windows: list[np.ndarray],
    result: PCACleaningResult,
) -> list[np.ndarray]:
    """Remove windows flagged by PCA-based cleaning."""
    clean = [w for w, is_out in zip(windows, result.is_outlier) if not is_out]
    logger.info(
        "PCA cleaning: removed %d windows. %d remaining.",
        result.n_removed, len(clean),
    )
    return clean
