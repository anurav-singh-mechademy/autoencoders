"""Detect distribution drift in reconstruction errors over time."""

from __future__ import annotations

import numpy as np
from scipy import stats


def rolling_error_trend(
    daily_errors: list[float],
    window_days: int = 7,
) -> dict:
    """Compute rolling mean of daily error scores to detect upward trends.

    Args:
        daily_errors: List of daily average anomaly scores, oldest first.
        window_days: Rolling window size in days.

    Returns:
        Dict with rolling_mean array, trend_slope, and is_increasing flag.
    """
    arr = np.array(daily_errors, dtype=float)

    if len(arr) < window_days:
        return {
            "rolling_mean": arr.tolist(),
            "trend_slope": 0.0,
            "is_increasing": False,
            "n_days": len(arr),
        }

    # Rolling mean
    kernel = np.ones(window_days) / window_days
    rolling = np.convolve(arr, kernel, mode="valid")

    # Linear regression on rolling mean to get trend slope
    x = np.arange(len(rolling))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, rolling)

    return {
        "rolling_mean": rolling.tolist(),
        "trend_slope": float(slope),
        "is_increasing": slope > 0 and p_value < 0.05,
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "n_days": len(arr),
    }


def ks_test_drift(
    baseline_errors: np.ndarray,
    recent_errors: np.ndarray,
    significance: float = 0.01,
) -> dict:
    """Two-sample Kolmogorov-Smirnov test to detect distribution shift.

    Args:
        baseline_errors: Error distribution from training or initial deployment.
        recent_errors: Recent error distribution to compare.
        significance: P-value threshold for significance.

    Returns:
        Dict with statistic, p_value, is_drifted flag.
    """
    statistic, p_value = stats.ks_2samp(baseline_errors, recent_errors)

    return {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "is_drifted": p_value < significance,
        "significance": significance,
        "baseline_n": len(baseline_errors),
        "recent_n": len(recent_errors),
    }


def per_sensor_drift(
    baseline_errors: np.ndarray,
    recent_errors: np.ndarray,
    significance: float = 0.01,
) -> list[dict]:
    """Run KS-test per sensor to identify which sensors have drifted.

    Args:
        baseline_errors: Shape (n_windows, n_sensors) from training.
        recent_errors: Shape (n_windows, n_sensors) from recent inference.
        significance: P-value threshold.

    Returns:
        List of dicts per sensor, sorted by KS statistic descending.
    """
    n_sensors = baseline_errors.shape[1]
    results = []

    for i in range(n_sensors):
        r = ks_test_drift(baseline_errors[:, i], recent_errors[:, i], significance)
        r["sensor_index"] = i
        results.append(r)

    results.sort(key=lambda x: x["ks_statistic"], reverse=True)
    return results
