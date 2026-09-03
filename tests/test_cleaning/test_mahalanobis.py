"""Tests for Mahalanobis distance cleaning."""

import numpy as np
import pytest

from autoencoder.cleaning.mahalanobis import apply_mahalanobis_cleaning, clean_mahalanobis


class TestCleanMahalanobis:
    def test_basic_output(self, synthetic_windows):
        result = clean_mahalanobis(synthetic_windows, chi2_percentile=97.5)
        assert len(result.is_outlier) == len(synthetic_windows)
        assert len(result.distances) == len(synthetic_windows)
        assert result.threshold > 0

    def test_detects_outliers(self):
        """Use a larger dataset where outliers are clearly separable."""
        rng = np.random.default_rng(42)
        # 50 normal windows + 2 outlier windows
        windows = [rng.normal(50.0, 2.0, (120, 10)) for _ in range(50)]
        windows.append(rng.normal(200.0, 2.0, (120, 10)))  # extreme outlier
        windows.append(rng.normal(-100.0, 2.0, (120, 10)))  # extreme outlier
        result = clean_mahalanobis(windows, chi2_percentile=97.5)
        assert result.n_removed >= 2

    def test_distances_positive(self, synthetic_windows):
        result = clean_mahalanobis(synthetic_windows)
        assert np.all(result.distances >= 0)


class TestApplyMahalanobisCleaning:
    def test_correct_count(self, windows_with_outliers):
        result = clean_mahalanobis(windows_with_outliers, chi2_percentile=95.0)
        clean = apply_mahalanobis_cleaning(windows_with_outliers, result)
        assert len(clean) == result.n_total - result.n_removed
