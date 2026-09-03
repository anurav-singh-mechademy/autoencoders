"""Tests for the cleaning pipeline orchestrator, in particular the `steps` param."""

import numpy as np
import pytest

from autoencoder.data.cleaning import run_cleaning_pipeline
from autoencoder.data.regime import RegimeResult


@pytest.fixture
def windows_with_outliers(n_sensors):
    """50 healthy windows + a few injected outliers -- large enough that IF, PCA,
    and Mahalanobis each still have >=10 windows left to run on afterwards."""
    rng = np.random.default_rng(99)
    windows = [rng.normal(loc=50.0, scale=5.0, size=(120, n_sensors)) for _ in range(50)]
    windows[2] = rng.normal(loc=200.0, scale=50.0, size=(120, n_sensors))
    windows[7] = rng.normal(loc=-100.0, scale=50.0, size=(120, n_sensors))
    return windows


@pytest.fixture
def single_regime_result(windows_with_outliers):
    n = len(windows_with_outliers)
    return RegimeResult(
        labels=np.zeros(n, dtype=int),
        regime_names={0: "All"},
        cluster_centers=np.zeros((1, 1)),
        n_clusters=1,
        feature_columns=[],
        window_counts={"All": n},
    )


class TestSteps:
    def test_default_runs_all_three(self, windows_with_outliers, single_regime_result, sensor_columns):
        result = run_cleaning_pipeline(
            windows=windows_with_outliers,
            regime_result=single_regime_result,
            sensor_columns=sensor_columns,
        )
        ran = {log.step_name for log in result.step_logs}
        assert ran == {"isolation_forest", "pca", "mahalanobis"}

    def test_empty_steps_skips_cleaning(self, windows_with_outliers, single_regime_result, sensor_columns):
        result = run_cleaning_pipeline(
            windows=windows_with_outliers,
            regime_result=single_regime_result,
            sensor_columns=sensor_columns,
            steps=[],
        )
        assert result.cleaned_count == result.original_count
        assert result.step_logs == []

    def test_single_step_only_runs_that_step(self, windows_with_outliers, single_regime_result, sensor_columns):
        result = run_cleaning_pipeline(
            windows=windows_with_outliers,
            regime_result=single_regime_result,
            sensor_columns=sensor_columns,
            steps=["pca"],
        )
        ran = {log.step_name for log in result.step_logs}
        assert ran == {"pca"}
        assert result.per_step_removals["isolation_forest"] == 0
        assert result.per_step_removals["mahalanobis"] == 0

    def test_reordered_steps_run_in_given_order(self, windows_with_outliers, single_regime_result, sensor_columns):
        result = run_cleaning_pipeline(
            windows=windows_with_outliers,
            regime_result=single_regime_result,
            sensor_columns=sensor_columns,
            steps=["mahalanobis", "isolation_forest"],
        )
        order = [log.step_name for log in result.step_logs]
        assert order == ["mahalanobis", "isolation_forest"]

    def test_unknown_step_raises(self, windows_with_outliers, single_regime_result, sensor_columns):
        with pytest.raises(ValueError):
            run_cleaning_pipeline(
                windows=windows_with_outliers,
                regime_result=single_regime_result,
                sensor_columns=sensor_columns,
                steps=["not_a_real_step"],
            )


class TestRejectedWindows:
    def test_rejected_count_matches_removed_count(self, windows_with_outliers, single_regime_result, sensor_columns):
        result = run_cleaning_pipeline(
            windows=windows_with_outliers,
            regime_result=single_regime_result,
            sensor_columns=sensor_columns,
        )
        assert len(result.rejected_windows) == result.original_count - result.cleaned_count

    def test_rejected_windows_are_not_in_cleaned_windows(self, windows_with_outliers, single_regime_result, sensor_columns):
        result = run_cleaning_pipeline(
            windows=windows_with_outliers,
            regime_result=single_regime_result,
            sensor_columns=sensor_columns,
        )
        cleaned_ids = {id(w) for w in result.cleaned_windows}
        rejected_ids = {id(w) for w in result.rejected_windows}
        assert cleaned_ids.isdisjoint(rejected_ids)

    def test_no_rejections_when_steps_empty(self, windows_with_outliers, single_regime_result, sensor_columns):
        result = run_cleaning_pipeline(
            windows=windows_with_outliers,
            regime_result=single_regime_result,
            sensor_columns=sensor_columns,
            steps=[],
        )
        assert result.rejected_windows == []

    def test_injected_outliers_are_among_rejected(self, windows_with_outliers, single_regime_result, sensor_columns):
        # windows[2] and windows[7] are injected outliers (see fixture) -- at
        # least one of the two extreme injected windows should be caught by
        # at least one of the three cleaning stages.
        result = run_cleaning_pipeline(
            windows=windows_with_outliers,
            regime_result=single_regime_result,
            sensor_columns=sensor_columns,
        )
        rejected_ids = {id(w) for w in result.rejected_windows}
        injected_ids = {id(windows_with_outliers[2]), id(windows_with_outliers[7])}
        assert rejected_ids & injected_ids
