"""Tests for cleaning report generation."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoencoder.reporting.cleaning_report import generate_cleaning_report


class TestGenerateCleaningReport:
    def test_generates_html(self, tmp_path):
        # Mock pipeline result
        mock_result = MagicMock()
        mock_result.original_count = 100
        mock_result.cleaned_count = 85
        mock_result.step_logs = [
            MagicMock(step_name="isolation_forest", regime="Low", windows_before=50, windows_after=45, windows_removed=5),
            MagicMock(step_name="pca", regime="Low", windows_before=45, windows_after=42, windows_removed=3),
        ]
        mock_result.per_step_removals = {"isolation_forest": 10, "pca": 5, "mahalanobis": 0}
        mock_result.per_regime_results = {"Low": {"original": 50, "cleaned": 42}, "High": {"original": 50, "cleaned": 43}}
        mock_result.validation_report = None

        output_path = tmp_path / "report.html"
        result_path = generate_cleaning_report(
            pipeline_result=mock_result,
            sensor_columns=["s1", "s2", "s3"],
            equipment_id="TURBINE-001",
            output_path=output_path,
        )

        assert Path(result_path).exists()
        content = Path(result_path).read_text()
        assert "TURBINE-001" in content
        assert "isolation_forest" in content
        assert "85" in content

    def test_with_validation_plots(self, tmp_path):
        mock_result = MagicMock()
        mock_result.original_count = 100
        mock_result.cleaned_count = 90
        mock_result.step_logs = []
        mock_result.per_step_removals = {}
        mock_result.per_regime_results = {}
        mock_vr = MagicMock()
        mock_vr.excessive_removal = True
        mock_vr.plot_paths = ["/tmp/plot1.png", "/tmp/plot2.png"]
        mock_result.validation_report = mock_vr

        output_path = tmp_path / "report2.html"
        result_path = generate_cleaning_report(
            pipeline_result=mock_result,
            sensor_columns=["s1"],
            equipment_id="PUMP-002",
            output_path=output_path,
        )

        content = Path(result_path).read_text()
        assert "WARNING" in content
        assert "plot1.png" in content
