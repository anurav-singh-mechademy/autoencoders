"""Tests for dataset splitting and conversion."""

import numpy as np
import pytest

from autoencoder.training.dataset import split_windows, split_indices, windows_to_numpy


class TestSplitWindows:
    def test_split_sizes(self, synthetic_windows):
        train, val, test = split_windows(synthetic_windows)
        total = len(train) + len(val) + len(test)
        assert total == len(synthetic_windows)

    def test_split_proportions(self):
        windows = [np.random.randn(120, 20) for _ in range(100)]
        train, val, test = split_windows(windows, train_pct=0.85, val_pct=0.10)
        assert len(train) == 85
        assert len(val) == 10
        assert len(test) == 5

    def test_no_overlap(self, synthetic_windows):
        train, val, test = split_windows(synthetic_windows)
        # Windows are sequential slices so no overlap by construction
        assert len(train) + len(val) + len(test) == len(synthetic_windows)

    def test_stratifies_by_regime(self):
        # windows are grouped by regime (as run_cleaning_pipeline returns them),
        # not interleaved by time -- a plain contiguous split would put val/test
        # entirely inside the tail regime.
        regime_labels = [0] * 60 + [1] * 30 + [2] * 10
        windows = [np.full((120, 5), label) for label in regime_labels]

        train, val, test = split_windows(windows, train_pct=0.85, val_pct=0.10, regime_labels=regime_labels)

        train_regimes = {int(w[0, 0]) for w in train}
        val_regimes = {int(w[0, 0]) for w in val}
        test_regimes = {int(w[0, 0]) for w in test}

        assert train_regimes == {0, 1, 2}
        assert val_regimes == {0, 1, 2}
        assert test_regimes == {0, 1, 2}

    def test_without_regime_labels_matches_plain_split(self, synthetic_windows):
        train, val, test = split_windows(synthetic_windows, regime_labels=None)
        train2, val2, test2 = split_windows(synthetic_windows)
        assert len(train) == len(train2) and len(val) == len(val2) and len(test) == len(test2)
        assert all(np.array_equal(a, b) for a, b in zip(train, train2))


class TestSplitIndices:
    """split_indices must match split_windows_with_ids' partitioning exactly
    (it's the same logic, factored out to return positions instead of
    slicing a single array) so a caller can apply one split to several
    parallel arrays -- e.g. windows, raw-file window ids, and regime labels
    -- before they diverge (main.py's step_clean splits before outlier
    -removal cleaning for exactly this reason)."""

    def test_sizes(self):
        train_idx, val_idx, test_idx = split_indices(100, train_pct=0.85, val_pct=0.10)
        assert len(train_idx) == 85
        assert len(val_idx) == 10
        assert len(test_idx) == 5

    def test_no_overlap_and_covers_everything(self):
        train_idx, val_idx, test_idx = split_indices(100, train_pct=0.85, val_pct=0.10)
        all_idx = train_idx + val_idx + test_idx
        assert sorted(all_idx) == list(range(100))

    def test_stratifies_by_regime(self):
        regime_labels = [0] * 60 + [1] * 30 + [2] * 10
        train_idx, val_idx, test_idx = split_indices(100, train_pct=0.85, val_pct=0.10, regime_labels=regime_labels)
        regime_labels = np.asarray(regime_labels)
        assert set(regime_labels[train_idx]) == {0, 1, 2}
        assert set(regime_labels[val_idx]) == {0, 1, 2}
        assert set(regime_labels[test_idx]) == {0, 1, 2}

    def test_matches_split_windows_with_ids(self):
        from autoencoder.training.dataset import split_windows_with_ids
        regime_labels = [0] * 60 + [1] * 30 + [2] * 10
        windows = [np.full((5, 2), i) for i in range(100)]
        window_ids = list(range(1000, 1100))

        train_idx, val_idx, test_idx = split_indices(100, train_pct=0.85, val_pct=0.10, regime_labels=regime_labels)
        train_w, val_w, test_w, train_ids, val_ids, test_ids = split_windows_with_ids(
            windows, window_ids, train_pct=0.85, val_pct=0.10, regime_labels=regime_labels,
        )

        assert [window_ids[i] for i in train_idx] == train_ids
        assert [window_ids[i] for i in val_idx] == val_ids
        assert [window_ids[i] for i in test_idx] == test_ids


class TestWindowsToNumpy:
    def test_shape(self, synthetic_windows):
        arr = windows_to_numpy(synthetic_windows)
        assert arr.shape == (10, 120, 20)
        assert arr.dtype == np.float32
