"""Operating regime segmentation using KMeans clustering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Transient"}


@dataclass
class RegimeResult:
    """Result of regime segmentation."""

    labels: np.ndarray
    regime_names: dict[int, str]
    cluster_centers: np.ndarray
    n_clusters: int
    feature_columns: list[str]
    window_counts: dict[str, int]


def segment_regimes(
    window_features: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> RegimeResult:
    """Segment operating windows into regimes using KMeans on load-indicator features.

    Args:
        window_features: DataFrame with one row per window, containing feature_columns.
        feature_columns: Columns for clustering (e.g., speed, power, discharge_pressure).
        n_clusters: Number of regimes (default 3: Low/Medium/High).
        random_state: Random seed.

    Returns:
        RegimeResult with ordered labels (0=lowest load, n-1=highest load).
    """
    X = window_features[feature_columns].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = kmeans.fit_predict(X)

    # Order clusters by ascending mean of first feature so 0=Low, n-1=High
    center_means = kmeans.cluster_centers_[:, 0]
    order = np.argsort(center_means)
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[l] for l in raw_labels])

    regime_names = {i: REGIME_LABELS.get(i, f"Regime_{i}") for i in range(n_clusters)}
    window_counts = {
        regime_names[i]: int(np.sum(labels == i)) for i in range(n_clusters)
    }

    logger.info("Regime segmentation: %s", window_counts)

    return RegimeResult(
        labels=labels,
        regime_names=regime_names,
        cluster_centers=kmeans.cluster_centers_[order],
        n_clusters=n_clusters,
        feature_columns=feature_columns,
        window_counts=window_counts,
    )


def compute_window_features(
    windows: list[np.ndarray],
    sensor_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute per-window mean features for regime segmentation.

    Args:
        windows: List of (window_size, n_sensors) arrays.
        sensor_columns: Full list of sensor column names matching array column order.
        feature_columns: Subset of sensor_columns to use for regime features.

    Returns:
        DataFrame with one row per window, columns = feature_columns.
    """
    col_indices = [sensor_columns.index(c) for c in feature_columns]
    means = [w[:, col_indices].mean(axis=0) for w in windows]
    return pd.DataFrame(means, columns=feature_columns)


def segment_regimes_unsupervised(
    windows: list[np.ndarray],
    n_clusters: int = 3,
    n_components: int = 3,
    random_state: int = 42,
) -> RegimeResult:
    """Fallback regime segmentation when explicit load signals are unavailable.

    Uses PCA on per-window means of the full sensor set, then clusters on top components.
    """
    window_means = np.array([w.mean(axis=0) for w in windows])

    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(window_means)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = kmeans.fit_predict(reduced)

    center_means = kmeans.cluster_centers_[:, 0]
    order = np.argsort(center_means)
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[l] for l in raw_labels])

    regime_names = {i: REGIME_LABELS.get(i, f"Regime_{i}") for i in range(n_clusters)}
    window_counts = {
        regime_names[i]: int(np.sum(labels == i)) for i in range(n_clusters)
    }

    logger.info("Unsupervised regime segmentation (PCA+KMeans): %s", window_counts)

    return RegimeResult(
        labels=labels,
        regime_names=regime_names,
        cluster_centers=kmeans.cluster_centers_[order],
        n_clusters=n_clusters,
        feature_columns=[f"PC{i+1}" for i in range(n_components)],
        window_counts=window_counts,
    )
