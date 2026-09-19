"""Sensor dropout filtering -- excludes windows with too many null sensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensorDropoutResult:
    """Result of sensor dropout detection."""

    is_dropped: np.ndarray        # Boolean mask -- True = window excluded
    n_dropped: int
    n_total: int
    bad_sensor_pct: np.ndarray    # Per-window fraction (%) of null sensors
    threshold_used: float


def detect_sensor_dropout_windows(
    windows: list[np.ndarray],
    threshold_pct: float = 5.0,
) -> SensorDropoutResult:
    """Flag windows where too many sensors are null.

    A sensor is considered null within a window if any of its values in the
    window are NaN. A window is excluded if the fraction of null sensors
    exceeds threshold_pct (default 5%, per spec).

    Deliberately NOT flagging "stuck" (flat/frozen) sensors here: a sensor
    that's genuinely deadband-/compression-logged (a real, common historian
    behaviour, not a fault) still carries valid signal for every OTHER
    sensor in the window, so excluding the whole window for it would throw
    away good data for a reason that has nothing to do with data quality.
    Chronically-dead sensors are instead handled once, at the feature level,
    by autoencoder.data.preprocessing.remove_null_or_stuck_columns -- a
    sensor still present at this point is one the pipeline has decided to
    keep and train on as-is, flat stretches included.

    Args:
        windows: List of (window_size, n_sensors) arrays.
        threshold_pct: Percentage of sensors above which a window is excluded.

    Returns:
        SensorDropoutResult with boolean mask and diagnostics.
    """
    n_total = len(windows)
    bad_sensor_pct = np.zeros(n_total)

    for i, w in enumerate(windows):
        n_sensors = w.shape[1]
        is_null = np.isnan(w).any(axis=0)
        n_bad = int(np.sum(is_null))
        bad_sensor_pct[i] = (n_bad / n_sensors) * 100.0

    is_dropped = bad_sensor_pct > threshold_pct
    n_dropped = int(np.sum(is_dropped))

    logger.info(
        "Sensor dropout filter: %d / %d windows excluded (>%.1f%% null sensors).",
        n_dropped, n_total, threshold_pct,
    )

    return SensorDropoutResult(
        is_dropped=is_dropped,
        n_dropped=n_dropped,
        n_total=n_total,
        bad_sensor_pct=bad_sensor_pct,
        threshold_used=threshold_pct,
    )


def remove_sensor_dropout_windows(
    windows: list[np.ndarray],
    result: SensorDropoutResult,
) -> list[np.ndarray]:
    """Remove windows flagged for excessive null sensors."""
    clean = [w for w, dropped in zip(windows, result.is_dropped) if not dropped]
    logger.info(
        "Removed %d sensor-dropout windows. %d remaining.",
        result.n_dropped, len(clean),
    )
    return clean
