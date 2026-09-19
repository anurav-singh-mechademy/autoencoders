"""Tests for threshold computation — percentile and elbow methods."""

import numpy as np
import pytest
from scipy import stats

from autoencoder.alerting.thresholds import (
    compute_thresholds,
    compute_thresholds_elbow,
    _compute_anomaly_curve,
    _find_elbow,
    _find_stable_threshold,
)


class TestComputeThresholds:
    def test_basic(self):
        errors = np.random.rand(100)
        t = compute_thresholds(errors)
        assert t["green_yellow"] <= t["yellow_red"]
        assert t["n_samples"] == 100
        assert t["green_yellow_percentile"] == 90
        assert t["yellow_red_percentile"] == 99

    def test_ordered(self):
        errors = np.arange(1000, dtype=float)
        t = compute_thresholds(errors)
        assert t["green_yellow"] < t["yellow_red"]
        assert t["min"] == 0.0
        assert t["max"] == 999.0

    def test_custom_percentiles(self):
        errors = np.arange(100, dtype=float)
        t = compute_thresholds(errors, green_yellow_percentile=50, yellow_red_percentile=95, method="percentile")
        assert t["green_yellow_percentile"] == 50
        assert t["yellow_red_percentile"] == 95
        assert t["green_yellow"] == pytest.approx(49.5, abs=1.0)

    def test_constant_errors(self):
        errors = np.ones(50) * 5.0
        t = compute_thresholds(errors)
        assert t["green_yellow"] == pytest.approx(5.0)
        assert t["yellow_red"] == pytest.approx(5.0)
        assert t["std"] == pytest.approx(0.0)

    def test_method_label_defaults_to_robust(self):
        errors = np.random.rand(50)
        t = compute_thresholds(errors)
        assert t["method"] == "robust"
        assert t["spread"] == "mad"

    def test_method_label_percentile(self):
        errors = np.random.rand(50)
        t = compute_thresholds(errors, method="percentile")
        assert t["method"] == "percentile"
        assert t["spread"] is None

    def test_unknown_method_raises(self):
        errors = np.random.rand(50)
        with pytest.raises(ValueError):
            compute_thresholds(errors, method="not_a_real_method")


class TestComputeThresholdsRobust:
    def test_ordered(self):
        rng = np.random.default_rng(0)
        errors = rng.lognormal(mean=-3.0, sigma=0.5, size=500)
        t = compute_thresholds(errors, method="robust")
        assert t["green_yellow"] < t["yellow_red"]

    def test_recovers_true_percentiles_on_lognormal_data(self):
        """The whole method is built on log(score) being ~normal -- so on
        data that's genuinely log-normal, it should land close to the true
        population percentiles (computed analytically, not from the sample)."""
        rng = np.random.default_rng(1)
        mu, sigma = -2.0, 0.4
        errors = rng.lognormal(mean=mu, sigma=sigma, size=20000)
        t = compute_thresholds(errors, green_yellow_percentile=90, yellow_red_percentile=99, method="robust")

        true_p90 = float(np.exp(mu + stats.norm.ppf(0.90) * sigma))
        true_p99 = float(np.exp(mu + stats.norm.ppf(0.99) * sigma))
        assert t["green_yellow"] == pytest.approx(true_p90, rel=0.1)
        assert t["yellow_red"] == pytest.approx(true_p99, rel=0.1)

    def test_mad_and_iqr_agree_on_lognormal_data(self):
        rng = np.random.default_rng(2)
        errors = rng.lognormal(mean=-2.5, sigma=0.6, size=5000)
        t_mad = compute_thresholds(errors, method="robust", spread="mad")
        t_iqr = compute_thresholds(errors, method="robust", spread="iqr")
        assert t_mad["green_yellow"] == pytest.approx(t_iqr["green_yellow"], rel=0.15)
        assert t_mad["yellow_red"] == pytest.approx(t_iqr["yellow_red"], rel=0.15)

    def test_constant_errors_both_thresholds_equal_the_constant(self):
        errors = np.ones(50) * 5.0
        t = compute_thresholds(errors, method="robust")
        assert t["green_yellow"] == pytest.approx(5.0)
        assert t["yellow_red"] == pytest.approx(5.0)

    def test_zero_errors_do_not_crash(self):
        errors = np.zeros(20)
        t = compute_thresholds(errors, method="robust")
        assert t["green_yellow"] >= 0
        assert t["yellow_red"] >= 0

    def test_unknown_spread_raises(self):
        errors = np.random.rand(50) + 0.1
        with pytest.raises(ValueError):
            compute_thresholds(errors, method="robust", spread="not_a_real_spread")


class TestComputeThresholdsGPDTail:
    """Mirrors the real failure mode found in production: a well-behaved
    bulk plus a small, genuine cluster of severe-error windows that
    upstream cleaning didn't remove. 'robust' (median+MAD) is blind to that
    cluster and underestimates yellow_red; 'gpd_tail' should track the true
    tail far more closely."""

    @staticmethod
    def _mixture(rng, n_bulk=2000, n_severe=30, bulk_scale=1.0, severe_scale=50.0):
        bulk = rng.lognormal(mean=-1.0, sigma=0.3, size=n_bulk) * bulk_scale
        severe = rng.lognormal(mean=2.0, sigma=0.5, size=n_severe) * severe_scale
        return np.concatenate([bulk, severe])

    def test_green_yellow_matches_plain_percentile(self):
        rng = np.random.default_rng(0)
        errors = self._mixture(rng)
        t = compute_thresholds(errors, method="gpd_tail")
        assert t["green_yellow"] == pytest.approx(np.percentile(errors, 90))

    def test_tracks_true_tail_better_than_robust(self):
        rng = np.random.default_rng(1)
        errors = self._mixture(rng, n_bulk=5000, n_severe=100)
        true_p99 = np.percentile(errors, 99)

        t_robust = compute_thresholds(errors, method="robust")
        t_gpd = compute_thresholds(errors, method="gpd_tail")

        # robust, blind to the severe cluster, badly underestimates P99;
        # gpd_tail, fit from the actual exceedances, lands far closer to it.
        assert t_robust["yellow_red"] < true_p99 / 2
        assert abs(t_gpd["yellow_red"] - true_p99) < abs(t_robust["yellow_red"] - true_p99)

    def test_ordered(self):
        rng = np.random.default_rng(2)
        errors = self._mixture(rng)
        t = compute_thresholds(errors, method="gpd_tail")
        assert t["green_yellow"] < t["yellow_red"]

    def test_falls_back_to_robust_with_too_few_exceedances(self, caplog):
        rng = np.random.default_rng(3)
        errors = rng.lognormal(mean=-1.0, sigma=0.3, size=50)  # only ~5 points above P90
        t_gpd = compute_thresholds(errors, method="gpd_tail", min_gpd_exceedances=20)
        log_errors = np.log(np.maximum(errors, np.finfo(np.float64).tiny))
        from autoencoder.alerting.thresholds import _robust_log_threshold
        expected_fallback = float(np.exp(_robust_log_threshold(log_errors, 99, "mad")))
        assert t_gpd["yellow_red"] == pytest.approx(expected_fallback)
        assert "falling back to 'robust'" in caplog.text

    def test_method_label(self):
        rng = np.random.default_rng(4)
        errors = self._mixture(rng)
        t = compute_thresholds(errors, method="gpd_tail")
        assert t["method"] == "gpd_tail"
        assert t["spread"] == "mad"


class TestComputeAnomolyCurve:
    def test_decreasing(self):
        """Anomaly % should decrease as threshold increases."""
        rng = np.random.default_rng(42)
        # 50 windows, 10 sensors, ratios between 0 and 100
        ratios = [rng.uniform(0, 100, size=10) for _ in range(50)]
        thresholds = np.arange(1, 101, 5, dtype=float)
        curve = _compute_anomaly_curve(ratios, thresholds, sensor_anomaly_pct=10.0)
        # Curve should be non-increasing (monotone decreasing or flat)
        for i in range(len(curve) - 1):
            assert curve[i] >= curve[i + 1]

    def test_all_below_threshold(self):
        """If all ratios are low, anomaly % should be 0 at high thresholds."""
        ratios = [np.ones(10) * 0.5 for _ in range(20)]
        thresholds = np.array([1.0, 5.0, 10.0])
        curve = _compute_anomaly_curve(ratios, thresholds)
        assert curve[-1] == 0.0

    def test_all_above_threshold(self):
        """If all ratios are high, anomaly % should be 100 at low thresholds."""
        ratios = [np.ones(10) * 1000.0 for _ in range(20)]
        thresholds = np.array([1.0, 5.0, 10.0])
        curve = _compute_anomaly_curve(ratios, thresholds)
        assert curve[0] == 100.0


class TestFindElbow:
    def test_obvious_elbow(self):
        """Curve that drops steeply then flattens — elbow should be in the transition."""
        thresholds = np.arange(1, 21, dtype=float)
        # Steep drop from 100 to 10 in first 5 points, then flat
        curve = np.concatenate([
            np.linspace(100, 10, 5),
            np.ones(15) * 10,
        ])
        elbow = _find_elbow(thresholds, curve)
        # Elbow should be in the first few thresholds (steep-to-flat transition)
        assert 1.0 <= elbow <= 10.0

    def test_flat_curve(self):
        """Flat curve — elbow defaults to first threshold."""
        thresholds = np.arange(1, 11, dtype=float)
        curve = np.ones(10) * 50.0
        elbow = _find_elbow(thresholds, curve)
        assert elbow == 1.0


class TestFindStableThreshold:
    def test_finds_stable_region(self):
        thresholds = np.arange(1, 21, dtype=float)
        curve = np.concatenate([
            np.linspace(80, 10, 10),
            np.ones(10) * 10,
        ])
        stable = _find_stable_threshold(thresholds, curve, stability_window=5, max_change=0.5)
        # Stable region starts at index 10 (threshold=11)
        assert stable >= 10.0

    def test_no_stable_region(self):
        thresholds = np.arange(1, 11, dtype=float)
        curve = np.linspace(100, 10, 10)  # Always changing
        stable = _find_stable_threshold(thresholds, curve, stability_window=5, max_change=0.1)
        # Falls back to last threshold
        assert stable == 10.0


class TestComputeThresholdsElbow:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        ratios = [rng.uniform(0, 50, size=10) for _ in range(30)]
        result = compute_thresholds_elbow(ratios, sweep_start=1, sweep_end=100, sweep_step=5)
        assert "flag_threshold" in result
        assert "stable_threshold" in result
        assert "anomaly_curve" in result
        assert result["method"] == "elbow"

    def test_flag_threshold_in_range(self):
        rng = np.random.default_rng(42)
        ratios = [rng.uniform(0, 50, size=10) for _ in range(30)]
        result = compute_thresholds_elbow(ratios, sweep_start=1, sweep_end=100, sweep_step=5)
        assert 1.0 <= result["flag_threshold"] <= 100.0
        assert 1.0 <= result["stable_threshold"] <= 100.0

    def test_anomaly_curve_is_dict(self):
        rng = np.random.default_rng(42)
        ratios = [rng.uniform(0, 10, size=5) for _ in range(20)]
        result = compute_thresholds_elbow(ratios, sweep_start=1, sweep_end=50, sweep_step=5)
        assert isinstance(result["anomaly_curve"], dict)
        assert len(result["anomaly_curve"]) > 0
