"""Ray Data dataset for window-level batching."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import ray.data

logger = logging.getLogger(__name__)


def split_windows(
    windows: list[np.ndarray],
    train_pct: float = 0.85,
    val_pct: float = 0.10,
    regime_labels: Optional[np.ndarray | list[int]] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split windows into train/val/test at window boundaries.

    No row-level splitting -- prevents data leakage between adjacent timesteps.

    If `regime_labels` is given (one label per window, same order as `windows`),
    the split is stratified per regime: each regime's windows are sliced
    contiguously -- preserving chronological order and the no-leakage property --
    before the per-regime slices are concatenated. This matters because
    `run_cleaning_pipeline` returns windows ordered by regime (all of one regime,
    then all of the next), so a single contiguous cut over the full list would
    put validation/test entirely inside whichever regime(s) fall at the tail
    instead of sampling every regime. Omit `regime_labels` only when the windows
    are not regime-ordered (e.g. a single-regime unit, or pre-split test data).
    """
    if regime_labels is None:
        n = len(windows)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        train = windows[:n_train]
        val = windows[n_train:n_train + n_val]
        test = windows[n_train + n_val:]
    else:
        regime_labels = np.asarray(regime_labels)
        train, val, test = [], [], []
        for regime in np.unique(regime_labels):
            regime_windows = [w for w, r in zip(windows, regime_labels) if r == regime]
            n = len(regime_windows)
            n_train = int(n * train_pct)
            n_val = int(n * val_pct)
            train.extend(regime_windows[:n_train])
            val.extend(regime_windows[n_train:n_train + n_val])
            test.extend(regime_windows[n_train + n_val:])

    logger.info("Split: %d train, %d val, %d test windows.", len(train), len(val), len(test))
    return train, val, test


def windows_to_ray_dataset(windows: list[np.ndarray]) -> ray.data.Dataset:
    """Convert a list of window arrays into a Ray Dataset.

    Each item in the dataset is one window (120 rows x N sensors) stored as
    a dict with key 'window' containing the flattened array and 'shape' for reshaping.
    """
    items = [{"window": w.astype(np.float32).tobytes(), "rows": w.shape[0], "cols": w.shape[1]} for w in windows]
    return ray.data.from_items(items)


def windows_to_numpy(windows: list[np.ndarray]) -> np.ndarray:
    """Stack windows into a single array of shape (n_windows, window_size, n_sensors).

    Simpler alternative to Ray Dataset when data fits in memory.
    """
    return np.array(windows, dtype=np.float32)
