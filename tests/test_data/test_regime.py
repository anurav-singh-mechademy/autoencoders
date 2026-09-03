"""Tests for regime segmentation module."""

import numpy as np
import pandas as pd
import pytest

from autoencoder.data.regime import (
    compute_window_features,
    segment_regimes,
    segment_regimes_unsupervised,
)


class TestComputeWindowFeatures:
    def test_output_shape(self, synthetic_windows, sensor_columns, regime_feature_columns):
        features = compute_window_features(
            synthetic_windows, sensor_columns, regime_feature_columns,
        )
        assert features.shape == (len(synthetic_windows), len(regime_feature_columns))
        assert list(features.columns) == regime_feature_columns


class TestSegmentRegimes:
    def test_label_count(self, synthetic_windows, sensor_columns, regime_feature_columns):
        features = compute_window_features(
            synthetic_windows, sensor_columns, regime_feature_columns,
        )
        result = segment_regimes(features, regime_feature_columns, n_clusters=3)
        assert len(result.labels) == len(synthetic_windows)
        assert set(result.labels).issubset({0, 1, 2})

    def test_regime_names(self, synthetic_windows, sensor_columns, regime_feature_columns):
        features = compute_window_features(
            synthetic_windows, sensor_columns, regime_feature_columns,
        )
        result = segment_regimes(features, regime_feature_columns, n_clusters=3)
        assert result.regime_names == {0: "Low", 1: "Medium", 2: "High"}

    def test_window_counts_sum(self, synthetic_windows, sensor_columns, regime_feature_columns):
        features = compute_window_features(
            synthetic_windows, sensor_columns, regime_feature_columns,
        )
        result = segment_regimes(features, regime_feature_columns, n_clusters=3)
        total = sum(result.window_counts.values())
        assert total == len(synthetic_windows)


class TestSegmentRegimesUnsupervised:
    def test_basic(self, synthetic_windows):
        result = segment_regimes_unsupervised(synthetic_windows, n_clusters=2)
        assert len(result.labels) == len(synthetic_windows)
        assert result.n_clusters == 2
        assert sum(result.window_counts.values()) == len(synthetic_windows)
