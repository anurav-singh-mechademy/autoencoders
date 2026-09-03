"""Tests for PCA-based cleaning."""

import numpy as np
import pytest

from autoencoder.cleaning.pca import apply_pca_cleaning, clean_pca


class TestCleanPCA:
    def test_basic_output(self, synthetic_windows):
        result = clean_pca(synthetic_windows, variance_threshold=0.95, outlier_std=3.0)
        assert len(result.is_outlier) == len(synthetic_windows)
        assert len(result.reconstruction_errors) == len(synthetic_windows)
        assert result.n_components > 0

    def test_detects_outliers(self, windows_with_outliers):
        result = clean_pca(windows_with_outliers, variance_threshold=0.95, outlier_std=2.0)
        assert result.n_removed > 0

    def test_explained_variance(self, synthetic_windows):
        result = clean_pca(synthetic_windows, variance_threshold=0.95)
        assert np.sum(result.explained_variance_ratio) > 0


class TestApplyPCACleaning:
    def test_correct_count(self, windows_with_outliers):
        result = clean_pca(windows_with_outliers, outlier_std=2.0)
        clean = apply_pca_cleaning(windows_with_outliers, result)
        assert len(clean) == result.n_total - result.n_removed
