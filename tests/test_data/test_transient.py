"""Tests for transient window detection."""

import numpy as np
import pytest

from autoencoder.data.transient import detect_transient_windows, remove_transient_windows


class TestDetectTransientWindows:
    def test_no_transients(self, synthetic_windows):
        """All synthetic windows have low speed variation => none transient."""
        result = detect_transient_windows(
            synthetic_windows,
            speed_col_index=0,
            operating_range=100.0,
            threshold_pct=50.0,  # very high threshold
        )
        assert result.n_transient == 0
        assert result.n_total == len(synthetic_windows)
        assert len(result.speed_changes) == len(synthetic_windows)

    def test_with_transients(self):
        """Inject a window with large speed variation."""
        rng = np.random.default_rng(42)
        normal_window = rng.normal(50.0, 2.0, (120, 10))
        transient_window = np.copy(normal_window)
        transient_window[:60, 0] = 10.0   # low speed first half
        transient_window[60:, 0] = 90.0   # high speed second half

        windows = [normal_window, transient_window, normal_window]
        result = detect_transient_windows(
            windows, speed_col_index=0, operating_range=100.0, threshold_pct=15.0,
        )
        assert result.n_transient == 1
        assert result.is_transient[1] is True or result.is_transient[1] == True

    def test_threshold_sensitivity(self, synthetic_windows):
        """Very low threshold flags everything as transient."""
        result = detect_transient_windows(
            synthetic_windows,
            speed_col_index=0,
            operating_range=1.0,   # tiny range
            threshold_pct=0.001,   # tiny threshold
        )
        assert result.n_transient == len(synthetic_windows)


class TestRemoveTransientWindows:
    def test_remove(self):
        rng = np.random.default_rng(42)
        windows = [rng.normal(50.0, 2.0, (120, 10)) for _ in range(5)]
        result = detect_transient_windows(
            windows, speed_col_index=0, operating_range=100.0, threshold_pct=50.0,
        )
        clean = remove_transient_windows(windows, result)
        assert len(clean) == len(windows)  # none removed with high threshold
