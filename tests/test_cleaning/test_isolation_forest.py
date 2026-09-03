"""Tests for Isolation Forest cleaning."""

import numpy as np
import pytest

from autoencoder.cleaning.isolation_forest import (
    apply_isolation_forest,
    clean_isolation_forest,
)


class TestCleanIsolationForest:
    def test_basic_output(self, synthetic_windows):
        result = clean_isolation_forest(synthetic_windows, contamination=0.1)
        assert len(result.is_outlier) == len(synthetic_windows)
        assert len(result.scores) == len(synthetic_windows)
        assert result.n_total == len(synthetic_windows)

    def test_detects_outliers(self, windows_with_outliers):
        result = clean_isolation_forest(windows_with_outliers, contamination=0.2)
        assert result.n_removed > 0

    def test_outlier_windows_flagged(self, windows_with_outliers):
        result = clean_isolation_forest(windows_with_outliers, contamination=0.2)
        # Windows 2 and 7 are injected outliers
        flagged_indices = set(np.where(result.is_outlier)[0])
        assert 2 in flagged_indices or 7 in flagged_indices


class TestApplyIsolationForest:
    def test_removes_correct_count(self, windows_with_outliers):
        result = clean_isolation_forest(windows_with_outliers, contamination=0.2)
        clean = apply_isolation_forest(windows_with_outliers, result)
        assert len(clean) == result.n_total - result.n_removed
