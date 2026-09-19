"""Tests for cleaning validation module."""

from pathlib import Path

import numpy as np
import pytest

from autoencoder.cleaning.validation import validate_cleaning


class TestValidateCleaning:
    def test_basic_report(self, synthetic_windows, sensor_columns, tmp_path):
        # Use first 8 as "cleaned", all 10 as "original"
        original = synthetic_windows
        cleaned = synthetic_windows[:8]
        per_step = {"isolation_forest": 1, "pca": 1, "mahalanobis": 0}

        report = validate_cleaning(
            original_windows=original,
            cleaned_windows=cleaned,
            per_step_removals=per_step,
            sensor_columns=sensor_columns,
            output_dir=str(tmp_path),
            n_sensor_pairs=3,
        )
        assert report.total_original == 10
        assert report.total_after_cleaning == 8
        assert 15.0 < report.removal_pct < 25.0
        assert report.excessive_removal is False
        assert len(report.plot_paths) > 0

    def test_excessive_removal_warning(self, synthetic_windows, sensor_columns, tmp_path):
        original = synthetic_windows
        cleaned = synthetic_windows[:3]  # 70% removed
        per_step = {"isolation_forest": 4, "pca": 2, "mahalanobis": 1}

        report = validate_cleaning(
            original_windows=original,
            cleaned_windows=cleaned,
            per_step_removals=per_step,
            sensor_columns=sensor_columns,
            output_dir=str(tmp_path),
            max_removal_pct=40.0,
        )
        assert report.excessive_removal is True

    def test_sensor_name_with_slash_does_not_break_scatter_plot_path(self, synthetic_windows, sensor_columns, tmp_path):
        # Real historian tag naming convention (e.g. "5LI-5071D/PV") -- a "/"
        # embedded in a sensor name must not be read as a path separator when
        # building the scatter-plot filename (regression: this used to raise
        # FileNotFoundError trying to save into a non-existent subdirectory).
        slashy_columns = list(sensor_columns)
        slashy_columns[0] = "5LI-5071D/PV"
        report = validate_cleaning(
            original_windows=synthetic_windows,
            cleaned_windows=synthetic_windows[:8],
            per_step_removals={"isolation_forest": 1, "pca": 1, "mahalanobis": 0},
            sensor_columns=slashy_columns,
            output_dir=str(tmp_path),
            n_sensor_pairs=3,
        )
        assert len(report.plot_paths) > 0
        for path in report.plot_paths:
            assert Path(path).exists()
            assert "/" not in Path(path).name
