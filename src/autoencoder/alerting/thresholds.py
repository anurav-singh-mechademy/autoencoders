"""Compute anomaly thresholds from training error distribution.

Three methods available:
    1. Robust (default) — median + z*scale on log-transformed scores, where
       scale is a normal-consistent estimator (MAD or IQR) of spread and z is
       the standard-normal quantile for the target percentile. See
       `compute_thresholds` for the derivation.
    2. Percentile — plain empirical P90/P99 of the training error distribution.
       Simple, but sensitive to noise in exactly the tail it's measuring: with
       only a few hundred/thousand calibration windows, the specific values
       sitting at or near the 90th/99th percentile are themselves a small,
       high-variance sample.
    3. Elbow method (secondary) — sweep ratio thresholds on test data, find
       the knee where anomaly % stabilises. Adapted from old autoencoder code.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# MAD is a consistent estimator of sigma for a normal distribution once scaled
# by this constant (median(|x - median(x)|) * 1.4826 ~= sigma).
_MAD_TO_SIGMA = 1.4826
# IQR ~= 1.349*sigma for a normal distribution.
_IQR_TO_SIGMA = 1.349
# z-score of the 75th percentile (Q3) under a standard normal distribution.
_Q3_Z = 0.6745


def _log_transform(training_errors: np.ndarray) -> np.ndarray:
    """Log-transform scores for robust threshold calibration.

    Reconstruction-error scores are non-negative and right-skewed (a long
    tail of high-error windows), not symmetric -- the median/MAD and
    median/IQR estimators below assume a roughly symmetric, bell-shaped
    distribution, which log(score) approximates far better than score itself.
    Non-positive values (possible in principle for a near-perfect
    reconstruction) are clipped to a small positive epsilon rather than
    dropped, so a handful of degenerate scores don't shrink the sample.
    """
    eps = np.finfo(np.float64).tiny
    return np.log(np.maximum(training_errors.astype(np.float64), eps))


def _robust_log_threshold(log_errors: np.ndarray, percentile: float, spread: str) -> float:
    """One robust threshold in log-space, targeting the same percentile a
    plain `np.percentile` call would, but estimated from the distribution's
    bulk (median + spread) rather than from the noisy tail itself.

    For a normal distribution, percentile p sits at z(p) standard deviations
    from the mean/median, where z(p) is the standard-normal quantile
    (`scipy.stats.norm.ppf`) -- this is exactly what makes P90 ~= 1.28*sigma
    and P99 ~= 2.33*sigma "universal" constants, not tied to this dataset.
    MAD and IQR are both classical robust estimators of sigma; scaling them
    by 1.4826 / 1.349 (their normal-distribution consistency constants) lets
    the same z(p) be plugged in directly. The IQR form is derived by
    re-expressing `median + z*sigma` from Q3 (= median + 0.6745*sigma):
    `Q3 + k*IQR = median + (0.6745 + k*1.349)*sigma`, so matching to `z`
    requires `k = (z - 0.6745) / 1.349`.

    This is a starting point, not a finished calibration -- it assumes the
    log-transformed score is roughly normal in its bulk, which should be
    checked against real data (e.g. via compute_thresholds' percentile
    method on the same data) and adjusted if the two disagree substantially.
    """
    z = float(stats.norm.ppf(percentile / 100.0))
    median = float(np.median(log_errors))

    if spread == "mad":
        mad = float(np.median(np.abs(log_errors - median)))
        return median + z * _MAD_TO_SIGMA * mad
    elif spread == "iqr":
        q1, q3 = np.percentile(log_errors, [25, 75])
        iqr = float(q3 - q1)
        k = (z - _Q3_Z) / _IQR_TO_SIGMA
        return float(q3) + k * iqr
    else:
        raise ValueError(f"Unknown spread estimator: {spread!r}. Use 'mad' or 'iqr'.")


def compute_thresholds(
    training_errors: np.ndarray,
    green_yellow_percentile: float = 90,
    yellow_red_percentile: float = 99,
    method: str = "robust",
    spread: str = "mad",
) -> dict:
    """Compute Green/Yellow and Yellow/Red thresholds from training errors.

    Args:
        training_errors: Array of window-level anomaly scores from training data.
        green_yellow_percentile: Percentile for Green→Yellow boundary (default P90).
        yellow_red_percentile: Percentile for Yellow→Red boundary (default P99).
        method: "robust" (default) — median + z*scale on log-transformed
            scores, where scale is a normal-consistent MAD/IQR estimate of
            spread (see `_robust_log_threshold`). Less sensitive to noise in
            the extreme tail than reading the percentile off directly, since
            it's estimated from the bulk of the distribution instead.
            "percentile" — plain `np.percentile` on the raw scores.
        spread: "mad" (default) or "iqr" -- which robust spread estimator to
            use when method="robust". Ignored for method="percentile".

    Returns:
        Dict with 'green_yellow', 'yellow_red', 'mean', 'std', 'min', 'max'.
    """
    if method == "percentile":
        green_yellow = float(np.percentile(training_errors, green_yellow_percentile))
        yellow_red = float(np.percentile(training_errors, yellow_red_percentile))
    elif method == "robust":
        log_errors = _log_transform(np.asarray(training_errors))
        green_yellow = float(np.exp(_robust_log_threshold(log_errors, green_yellow_percentile, spread)))
        yellow_red = float(np.exp(_robust_log_threshold(log_errors, yellow_red_percentile, spread)))
    else:
        raise ValueError(f"Unknown threshold method: {method!r}. Use 'robust' or 'percentile'.")

    return {
        "green_yellow": green_yellow,
        "yellow_red": yellow_red,
        "mean": float(np.mean(training_errors)),
        "std": float(np.std(training_errors)),
        "min": float(np.min(training_errors)),
        "max": float(np.max(training_errors)),
        "green_yellow_percentile": green_yellow_percentile,
        "yellow_red_percentile": yellow_red_percentile,
        "n_samples": len(training_errors),
        "method": method,
        "spread": spread if method == "robust" else None,
    }


# ---------------------------------------------------------------------------
# Elbow method — secondary / fallback threshold calibration
# ---------------------------------------------------------------------------

def _compute_anomaly_curve(
    error_ratios_per_window: list[np.ndarray],
    thresholds: np.ndarray,
    sensor_anomaly_pct: float = 10.0,
) -> np.ndarray:
    """For each candidate threshold, compute % of windows flagged as anomalous.

    A window is anomalous if >sensor_anomaly_pct% of its sensors have
    error_ratio > threshold.

    Args:
        error_ratios_per_window: List of arrays, each shape (n_sensors,).
        thresholds: 1-D array of candidate threshold values to sweep.
        sensor_anomaly_pct: % of sensors that must exceed threshold for
            the window to be flagged.

    Returns:
        Array of shape (len(thresholds),) with anomaly % at each threshold.
    """
    n_windows = len(error_ratios_per_window)
    n_sensors = error_ratios_per_window[0].shape[0]
    min_flagged = sensor_anomaly_pct / 100.0 * n_sensors

    curve = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        count = 0
        for ratios in error_ratios_per_window:
            if (ratios > t).sum() > min_flagged:
                count += 1
        curve[i] = count / n_windows * 100.0
    return curve


def _find_elbow(thresholds: np.ndarray, curve: np.ndarray) -> float:
    """Find the elbow/knee point where the anomaly curve flattens.

    Uses the maximum-distance-to-line method (no external dependency):
    draw a line from the first point to the last point of the curve,
    the elbow is the point with maximum perpendicular distance from
    that line.

    Args:
        thresholds: 1-D array of candidate thresholds (x-axis).
        curve: 1-D array of anomaly % at each threshold (y-axis).

    Returns:
        The threshold value at the elbow point.
    """
    # Normalise to [0, 1] range for fair distance calculation
    x = (thresholds - thresholds[0]) / (thresholds[-1] - thresholds[0] + 1e-10)
    y = (curve - curve[-1]) / (curve[0] - curve[-1] + 1e-10)

    # Line from first to last point
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    line_len = np.sqrt(dx**2 + dy**2)

    if line_len < 1e-10:
        return float(thresholds[0])

    # Perpendicular distance from each point to the line
    distances = np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]) / line_len

    elbow_idx = int(np.argmax(distances))
    return float(thresholds[elbow_idx])


def _find_stable_threshold(
    thresholds: np.ndarray,
    curve: np.ndarray,
    stability_window: int = 5,
    max_change: float = 0.5,
) -> float:
    """Find the first threshold where the anomaly curve stays flat.

    Args:
        thresholds: 1-D array of candidate thresholds.
        curve: 1-D array of anomaly %.
        stability_window: Number of consecutive points that must be stable.
        max_change: Maximum allowed % change within the window.

    Returns:
        The threshold at the start of the first stable region.
    """
    for i in range(len(curve) - stability_window):
        window = curve[i : i + stability_window]
        if max(window) - min(window) < max_change:
            return float(thresholds[i])
    return float(thresholds[-1])


def compute_thresholds_elbow(
    error_ratios_per_window: list[np.ndarray],
    sweep_start: float = 1.0,
    sweep_end: float = 500.0,
    sweep_step: float = 5.0,
    sensor_anomaly_pct: float = 10.0,
    stability_window: int = 5,
    stability_max_change: float = 0.5,
) -> dict:
    """Compute sensor flag threshold using the elbow method.

    Sweeps candidate thresholds, computes what % of windows would be
    flagged anomalous at each threshold, then finds the elbow/knee point
    where the curve flattens.

    This is a secondary/fallback method. The primary method is percentile-based.

    How it works (from old autoencoder code):
        1. For each window, we already have error_ratios = sensor_error / baseline
        2. Sweep threshold values (e.g. 1 to 500)
        3. At each threshold: flag sensors where ratio > threshold,
           flag window as anomalous if >10% of sensors are flagged
        4. Compute anomaly % across all windows at each threshold
        5. This gives a decreasing curve (high threshold → fewer anomalies)
        6. Find the elbow (max distance from line) = optimal threshold

    Args:
        error_ratios_per_window: List of per-sensor error ratio arrays from
            test/validation windows. Each array shape (n_sensors,).
        sweep_start: Start of threshold sweep range.
        sweep_end: End of threshold sweep range.
        sweep_step: Step size for sweep.
        sensor_anomaly_pct: % of sensors that must exceed threshold to flag window.
        stability_window: Consecutive stable points needed.
        stability_max_change: Max % change considered "stable".

    Returns:
        Dict with:
            flag_threshold: recommended threshold (elbow point)
            stable_threshold: threshold where curve fully flattens
            anomaly_curve: dict of {threshold: anomaly_%}
            method: "elbow"
    """
    thresholds = np.arange(sweep_start, sweep_end + sweep_step, sweep_step)
    curve = _compute_anomaly_curve(
        error_ratios_per_window, thresholds, sensor_anomaly_pct,
    )

    elbow = _find_elbow(thresholds, curve)
    stable = _find_stable_threshold(thresholds, curve, stability_window, stability_max_change)

    # Build sparse curve dict for logging / inspection
    anomaly_curve = {float(t): float(c) for t, c in zip(thresholds, curve)}

    logger.info(
        "Elbow method: elbow=%.1f (anomaly=%.1f%%), stable=%.1f (anomaly=%.1f%%)",
        elbow,
        anomaly_curve.get(elbow, -1),
        stable,
        anomaly_curve.get(stable, -1),
    )

    return {
        "flag_threshold": elbow,
        "stable_threshold": stable,
        "anomaly_curve": anomaly_curve,
        "sensor_anomaly_pct": sensor_anomaly_pct,
        "method": "elbow",
    }
