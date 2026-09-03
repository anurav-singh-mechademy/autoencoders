"""Tests for retrain trigger detection."""

import numpy as np
import pytest

from autoencoder.monitoring.retrain_triggers import check_retrain_needed


class TestCheckRetrainNeeded:
    def test_no_retrain_stable(self):
        rng = np.random.default_rng(42)
        daily = [1.0] * 50
        baseline = rng.normal(0, 1, 200)
        recent = rng.normal(0, 1, 200)
        result = check_retrain_needed(daily, baseline, recent)
        assert not result["retrain_recommended"]
        assert result["reasons"] == []

    def test_drift_triggers_retrain(self):
        rng = np.random.default_rng(42)
        daily = [1.0] * 50
        baseline = rng.normal(0, 1, 200)
        recent = rng.normal(5, 1, 200)  # significant shift
        result = check_retrain_needed(daily, baseline, recent)
        assert result["retrain_recommended"]
        assert any("KS-test" in r for r in result["reasons"])

    def test_high_false_alarm_rate(self):
        rng = np.random.default_rng(42)
        daily = [1.0] * 50
        baseline = rng.normal(0, 1, 200)
        # Recent errors are all above P90 of baseline
        p90 = float(np.percentile(baseline, 90))
        recent = np.full(200, p90 + 1.0)
        result = check_retrain_needed(daily, baseline, recent)
        assert result["retrain_recommended"]
        assert result["false_alarm_rate"]["triggered"]

    def test_output_structure(self):
        daily = [1.0] * 10
        baseline = np.random.rand(100)
        recent = np.random.rand(100)
        result = check_retrain_needed(daily, baseline, recent)
        assert "retrain_recommended" in result
        assert "reasons" in result
        assert "trend" in result
        assert "drift" in result
        assert "false_alarm_rate" in result
