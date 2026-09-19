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


def split_windows_with_ids(
    windows: list[np.ndarray],
    window_ids: np.ndarray | list[int],
    train_pct: float = 0.85,
    val_pct: float = 0.10,
    regime_labels: Optional[np.ndarray | list[int]] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], list[int], list[int]]:
    """Same split as `split_windows`, but also partitions a parallel
    `window_ids` array so callers can record which windows landed in
    train/val/test. `window_ids` must match `windows` in length and order."""
    window_ids = list(window_ids)
    if len(window_ids) != len(windows):
        raise ValueError(f"window_ids has {len(window_ids)} entries but windows has {len(windows)}")

    if regime_labels is None:
        n = len(windows)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        train = windows[:n_train]
        val = windows[n_train:n_train + n_val]
        test = windows[n_train + n_val:]
        train_ids = window_ids[:n_train]
        val_ids = window_ids[n_train:n_train + n_val]
        test_ids = window_ids[n_train + n_val:]
    else:
        regime_labels = np.asarray(regime_labels)
        train, val, test = [], [], []
        train_ids, val_ids, test_ids = [], [], []
        for regime in np.unique(regime_labels):
            regime_windows = [w for w, r in zip(windows, regime_labels) if r == regime]
            regime_ids = [i for i, r in zip(window_ids, regime_labels) if r == regime]
            n = len(regime_windows)
            n_train = int(n * train_pct)
            n_val = int(n * val_pct)
            train.extend(regime_windows[:n_train])
            val.extend(regime_windows[n_train:n_train + n_val])
            test.extend(regime_windows[n_train + n_val:])
            train_ids.extend(regime_ids[:n_train])
            val_ids.extend(regime_ids[n_train:n_train + n_val])
            test_ids.extend(regime_ids[n_train + n_val:])

    logger.info(
        "Split (with ids): %d train, %d val, %d test windows.", len(train), len(val), len(test),
    )
    return train, val, test, train_ids, val_ids, test_ids


def split_indices(
    n: int,
    train_pct: float = 0.85,
    val_pct: float = 0.10,
    regime_labels: Optional[np.ndarray | list[int]] = None,
) -> tuple[list[int], list[int], list[int]]:
    """Same stratified-by-regime, chronological-within-regime split as
    split_windows_with_ids, but returns position indices (0..n-1) so a
    caller can apply the identical split to several parallel arrays at
    once. `regime_labels` is any per-window grouping label, not
    necessarily an operating regime (main.py also uses calendar-month
    labels here via training.split_stratify_by)."""
    if regime_labels is None:
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        return list(range(n_train)), list(range(n_train, n_train + n_val)), list(range(n_train + n_val, n))

    regime_labels = np.asarray(regime_labels)
    train_idx, val_idx, test_idx = [], [], []
    for regime in np.unique(regime_labels):
        idx_this_regime = [i for i in range(n) if regime_labels[i] == regime]
        n_r = len(idx_this_regime)
        n_train = int(n_r * train_pct)
        n_val = int(n_r * val_pct)
        train_idx.extend(idx_this_regime[:n_train])
        val_idx.extend(idx_this_regime[n_train:n_train + n_val])
        test_idx.extend(idx_this_regime[n_train + n_val:])

    return train_idx, val_idx, test_idx


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
