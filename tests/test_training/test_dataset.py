"""Tests for dataset splitting and conversion."""

import numpy as np
import pytest

from autoencoder.training.dataset import split_windows, windows_to_numpy


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


class TestWindowsToNumpy:
    def test_shape(self, synthetic_windows):
        arr = windows_to_numpy(synthetic_windows)
        assert arr.shape == (10, 120, 20)
        assert arr.dtype == np.float32
