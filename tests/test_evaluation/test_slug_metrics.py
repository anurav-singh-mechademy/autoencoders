"""Tests for per-slug detection metrics (Steps 2, 3, 5, 6)."""

import numpy as np
import pytest

from autoencoder.evaluation import slug_metrics as sm


class TestEpisodeStats:
    def test_never_active(self):
        series = np.zeros(5, dtype=bool)
        stats = sm.episode_stats(series, list(range(5)), list(range(1, 6)))
        assert stats["n_episodes"] == 0
        assert stats["n_windows_active"] == 0
        assert stats["median_duration"] is None

    def test_single_episode_duration(self):
        series = np.array([False, True, True, False])
        stats = sm.episode_stats(series, [0, 1, 2, 3], [1, 2, 3, 4])
        assert stats["n_episodes"] == 1
        assert stats["n_windows_active"] == 2
        assert stats["median_duration"] == 2  # end(3) - start(1)


class TestPointwiseMetrics:
    def test_perfect_detection(self):
        binary = np.array([False, True, True, False])
        zones = np.array(["green", "red", "red", "green"])
        scores = np.array([0.1, 0.9, 0.9, 0.1])
        m = sm.pointwise_metrics(binary, zones, scores)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["n_windows_active"] == 2
        assert m["n_windows_total"] == 4

    def test_usable_mask_excludes_windows(self):
        binary = np.array([True, True, True])
        zones = np.array(["red", "red", "red"])
        scores = np.array([0.9, np.nan, 0.9])
        usable = np.array([True, False, True])
        m = sm.pointwise_metrics(binary, zones, scores, usable=usable)
        assert m["n_windows_total"] == 2

    def test_pr_auc_nan_when_single_class(self):
        binary = np.zeros(5, dtype=bool)  # slug never active -- no positive class
        zones = np.array(["green"] * 5)
        scores = np.random.rand(5)
        m = sm.pointwise_metrics(binary, zones, scores)
        assert np.isnan(m["pr_auc"])


class TestSeverityCorrelation:
    def test_zero_variance_reports_degeneracy(self):
        binary = np.array([True, True, True, False])
        max_event_level = np.array([3.0, 3.0, 3.0, np.nan])
        n_active_nodes = np.array([1.0, 1.0, 1.0, 0.0])  # also degenerate/constant while active
        scores = np.array([0.5, 0.6, 0.7, 0.1])
        result = sm.severity_correlation(binary, max_event_level, n_active_nodes, scores)
        assert result["status"].startswith("not computable")

    def test_falls_back_to_n_active_nodes(self):
        binary = np.array([True, True, True, True])
        max_event_level = np.array([3.0, 3.0, 3.0, 3.0])  # degenerate
        n_active_nodes = np.array([1.0, 2.0, 3.0, 4.0])  # varies
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        result = sm.severity_correlation(binary, max_event_level, n_active_nodes, scores)
        assert result["status"] == "ok"
        assert result["field_used"] == "n_active_nodes"
        assert result["spearman_rho"] == pytest.approx(1.0)

    def test_prefers_max_event_level_when_usable(self):
        binary = np.array([True, True, True])
        max_event_level = np.array([1.0, 2.0, 3.0])
        n_active_nodes = np.array([5.0, 5.0, 5.0])
        scores = np.array([0.1, 0.2, 0.3])
        result = sm.severity_correlation(binary, max_event_level, n_active_nodes, scores)
        assert result["field_used"] == "max_event_level"


class TestConcurrencyCorrelation:
    """Unlike severity_correlation, this always targets n_active_nodes --
    reported independently, not as a fallback that only appears when
    max_event_level is degenerate."""

    def test_perfect_correlation(self):
        binary = np.array([True, True, True, True])
        n_active_nodes = np.array([1.0, 2.0, 3.0, 4.0])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        result = sm.concurrency_correlation(binary, n_active_nodes, scores)
        assert result["status"] == "ok"
        assert result["field_used"] == "n_active_nodes"
        assert result["spearman_rho"] == pytest.approx(1.0)

    def test_reported_even_when_severity_would_have_been_usable(self):
        # Contrast with severity_correlation: even though max_event_level
        # (not passed here at all) might be perfectly fine, concurrency_correlation
        # still reports n_active_nodes' own correlation on its own terms.
        binary = np.array([True, True, True])
        n_active_nodes = np.array([1.0, 1.0, 2.0])
        scores = np.array([0.2, 0.1, 0.9])
        result = sm.concurrency_correlation(binary, n_active_nodes, scores)
        assert result["status"] == "ok"

    def test_zero_variance_reports_degeneracy(self):
        binary = np.array([True, True, True, False])
        n_active_nodes = np.array([2.0, 2.0, 2.0, 5.0])  # constant while active
        scores = np.array([0.5, 0.6, 0.7, 0.1])
        result = sm.concurrency_correlation(binary, n_active_nodes, scores)
        assert result["status"].startswith("not computable")

    def test_only_active_windows_considered(self):
        binary = np.array([True, True, False, False])
        n_active_nodes = np.array([1.0, 2.0, 99.0, 99.0])
        scores = np.array([0.1, 0.2, 100.0, 100.0])
        result = sm.concurrency_correlation(binary, n_active_nodes, scores)
        assert result["n_windows"] == 2


class TestSensorAttributionMetrics:
    def test_perfect_ranking_gives_max_ap(self):
        # Both gt sensors ranked 1st and 2nd -- best possible AP (1.0).
        ranked = [["a", "b", "c", "d"]]
        gt = [frozenset({"a", "b"})]
        result = sm.sensor_attribution_metrics(ranked, gt)
        assert result["mean_average_precision"] == pytest.approx(1.0)
        # gt sensors at rank positions 0 and 1 (of n=4, max index 3) -> mean position 0.5, /3
        assert result["mean_gt_rank_percentile"] == pytest.approx(0.5 / 3)

    def test_gt_ranked_last_gives_low_ap(self):
        ranked = [["a", "b", "c", "d"]]
        gt = [frozenset({"d"})]
        result = sm.sensor_attribution_metrics(ranked, gt)
        assert result["mean_average_precision"] == pytest.approx(1 / 4)
        assert result["mean_gt_rank_percentile"] == pytest.approx(1.0)

    def test_extra_unranked_candidates_dont_tank_the_score(self):
        # gt includes a sensor the model never puts near the top of THIS
        # window's ranking (still present, just lower) -- unlike Jaccard,
        # a single well-ranked hit should not be crushed by an unranked one.
        ranked = [["a", "b", "c", "d", "e"]]
        gt = [frozenset({"a", "e"})]  # a is rank 0 (great), e is rank 4 (bad)
        result = sm.sensor_attribution_metrics(ranked, gt)
        # AP = (1/1 + 2/5) / 2 = 0.7 -- rewarded for the good hit, not zeroed
        # out by the bad one the way Jaccard-over-top-k would be.
        assert result["mean_average_precision"] == pytest.approx(0.7)

    def test_empty_gt_windows_skipped(self):
        ranked = [["a", "b"], ["a", "b"]]
        gt = [frozenset(), frozenset({"b"})]
        result = sm.sensor_attribution_metrics(ranked, gt)
        assert result["n_windows"] == 1
        assert result["n_skipped_empty_gt"] == 1

    def test_gt_sensor_outside_ranked_universe_skipped(self):
        # gt sensor never appears in this window's ranked list at all (e.g.
        # not one of the model's trained sensor_columns) -- can't be scored.
        ranked = [["a", "b"]]
        gt = [frozenset({"z"})]
        result = sm.sensor_attribution_metrics(ranked, gt)
        assert result["n_windows"] == 0
        assert result["n_skipped_empty_gt"] == 1

    def test_no_windows(self):
        result = sm.sensor_attribution_metrics([], [])
        assert result["n_windows"] == 0
        assert result["mean_average_precision"] is None

    def test_concurrency_split(self):
        ranked = [["a", "b"], ["a", "b"], ["a", "b"]]
        gt = [frozenset({"a"})] * 3
        concurrency = [1, 1, 5]
        result = sm.sensor_attribution_by_concurrency(ranked, gt, concurrency, concurrency_threshold=1)
        assert result["low_concurrency"]["n_windows"] == 2
        assert result["high_concurrency"]["n_windows"] == 1
