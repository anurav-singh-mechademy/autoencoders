"""Tests for per-sensor diagnosis."""

import torch
import numpy as np
import pytest

from autoencoder.inference.diagnosis import diagnose_window


class TestDiagnoseWindow:
    def test_basic_output(self):
        x = torch.randn(120, 20)
        x_hat = torch.randn(120, 20)
        result = diagnose_window(x, x_hat)
        assert "sensor_errors" in result
        assert "top_contributors" in result
        assert "total_error" in result
        assert result["sensor_errors"].shape == (20,)
        assert result["total_error"] > 0

    def test_with_sensor_names(self):
        x = torch.randn(120, 5)
        x_hat = torch.randn(120, 5)
        names = ["temp", "speed", "power", "pressure", "flow"]
        result = diagnose_window(x, x_hat, sensor_names=names, top_k=3)
        assert len(result["top_contributors"]) == 3
        assert all(c["name"] in names for c in result["top_contributors"])

    def test_zero_error(self):
        x = torch.randn(120, 10)
        result = diagnose_window(x, x, top_k=3)
        assert result["total_error"] == pytest.approx(0.0, abs=1e-6)

    def test_contributions_have_pct(self):
        x = torch.randn(120, 20)
        x_hat = x + torch.randn(120, 20) * 0.1
        result = diagnose_window(x, x_hat, top_k=5)
        for c in result["top_contributors"]:
            assert "contribution_pct" in c
            assert c["contribution_pct"] >= 0


class TestDiagnoseWithBaselines:
    def test_error_ratios_computed(self):
        x = torch.randn(120, 10)
        x_hat = x + torch.randn(120, 10) * 0.5
        baselines = np.ones(10) * 0.1
        result = diagnose_window(x, x_hat, sensor_baselines=baselines)
        assert "error_ratios" in result
        assert result["error_ratios"].shape == (10,)
        # Ratios should equal sensor_errors / baselines
        expected = result["sensor_errors"] / baselines
        np.testing.assert_allclose(result["error_ratios"], expected, rtol=1e-5)

    def test_sensor_flags_binary(self):
        x = torch.randn(120, 10)
        x_hat = x + torch.randn(120, 10) * 0.5
        baselines = np.ones(10) * 0.1
        result = diagnose_window(x, x_hat, sensor_baselines=baselines)
        assert "sensor_flags" in result
        assert set(np.unique(result["sensor_flags"])).issubset({0, 1})

    def test_pct_flagged_calculation(self):
        x = torch.zeros(120, 10)
        # Make 3 sensors have high error
        x_hat = x.clone()
        x_hat[:, 0:3] = 10.0
        baselines = np.ones(10) * 0.01
        result = diagnose_window(x, x_hat, sensor_baselines=baselines, flag_threshold=1.0)
        # At least 3/10 = 30% flagged
        assert result["pct_flagged_sensors"] >= 30.0
        assert result["sensors_anomalous"] is True

    def test_no_baselines_skips_ratios(self):
        x = torch.randn(120, 10)
        x_hat = torch.randn(120, 10)
        result = diagnose_window(x, x_hat)
        assert "error_ratios" not in result
        assert "sensor_flags" not in result
        assert "pct_flagged_sensors" not in result
        assert "sensors_anomalous" not in result

    def test_zero_baseline_safe(self):
        """Ensure zero baselines don't cause division errors."""
        x = torch.randn(120, 5)
        x_hat = torch.randn(120, 5)
        baselines = np.array([0.0, 0.1, 0.0, 0.1, 0.1])
        result = diagnose_window(x, x_hat, sensor_baselines=baselines)
        assert not np.any(np.isnan(result["error_ratios"]))
        assert not np.any(np.isinf(result["error_ratios"]))
