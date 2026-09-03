"""Detect conditions that suggest the model should be retrained."""

from __future__ import annotations

import numpy as np

from autoencoder.monitoring.drift import rolling_error_trend, ks_test_drift


def check_retrain_needed(
    daily_errors: list[float],
    baseline_errors: np.ndarray,
    recent_errors: np.ndarray,
    drift_weeks_threshold: int = 6,
    false_alarm_rate_multiplier: float = 2.0,
    baseline_false_alarm_rate: float | None = None,
    ks_significance: float = 0.01,
) -> dict:
    """Check multiple conditions to determine if retraining is needed.

    Conditions checked:
        1. Sustained upward trend in errors over drift_weeks_threshold weeks
        2. KS-test detects distribution shift
        3. False alarm rate exceeds baseline × multiplier

    Args:
        daily_errors: Daily average anomaly scores, oldest first.
        baseline_errors: Training error distribution (1D array).
        recent_errors: Recent error distribution (1D array).
        drift_weeks_threshold: Weeks of increasing trend to trigger.
        false_alarm_rate_multiplier: How many times baseline rate triggers retrain.
        baseline_false_alarm_rate: Expected false alarm rate (e.g., 0.05).
            If None, estimated as fraction of baseline above P90.
        ks_significance: P-value for KS test.

    Returns:
        Dict with retrain_recommended, reasons list, and per-check details.
    """
    reasons = []

    # Check 1: Sustained trend
    trend = rolling_error_trend(daily_errors, window_days=7)
    min_days = drift_weeks_threshold * 7
    trend_triggered = trend["is_increasing"] and trend["n_days"] >= min_days
    if trend_triggered:
        reasons.append(f"sustained upward trend over {trend['n_days']} days (slope={trend['trend_slope']:.4f})")

    # Check 2: KS-test drift
    drift = ks_test_drift(baseline_errors, recent_errors, significance=ks_significance)
    if drift["is_drifted"]:
        reasons.append(f"KS-test detects distribution shift (statistic={drift['ks_statistic']:.4f}, p={drift['p_value']:.4e})")

    # Check 3: False alarm rate
    if baseline_false_alarm_rate is None:
        p90 = float(np.percentile(baseline_errors, 90))
        baseline_false_alarm_rate = float(np.mean(baseline_errors > p90))

    current_false_alarm_rate = float(np.mean(recent_errors > float(np.percentile(baseline_errors, 90))))
    fa_triggered = current_false_alarm_rate > baseline_false_alarm_rate * false_alarm_rate_multiplier
    if fa_triggered:
        reasons.append(
            f"false alarm rate {current_false_alarm_rate:.2%} exceeds "
            f"{false_alarm_rate_multiplier}x baseline ({baseline_false_alarm_rate:.2%})"
        )

    return {
        "retrain_recommended": len(reasons) > 0,
        "reasons": reasons,
        "trend": trend,
        "drift": drift,
        "false_alarm_rate": {
            "current": current_false_alarm_rate,
            "baseline": baseline_false_alarm_rate,
            "multiplier": false_alarm_rate_multiplier,
            "triggered": fa_triggered,
        },
    }
