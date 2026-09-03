"""Transient window detection -- startup, shutdown, and load-ramp removal."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransientResult:
    """Result of transient window detection."""

    is_transient: np.ndarray       # Boolean mask -- True = transient
    n_transient: int
    n_total: int
    speed_changes: np.ndarray      # Per-window speed change values
    threshold_used: float


def detect_transient_windows(
    windows: list[np.ndarray],
    speed_col_index: int,
    operating_range: float,
    threshold_pct: float = 15.0,
) -> TransientResult:
    """Detect transient windows based on speed variation within each window.

    A window is transient if:
        (max(speed) - min(speed)) > threshold_pct% of operating_range

    Args:
        windows: List of (window_size, n_sensors) arrays.
        speed_col_index: Column index of the speed sensor.
        operating_range: Full operating range of speed (max - min from full dataset).
        threshold_pct: Percentage of operating range above which a window is transient.

    Returns:
        TransientResult with boolean mask and diagnostics.
    """
    threshold = operating_range * (threshold_pct / 100.0)
    speed_changes = np.array([
        np.ptp(w[:, speed_col_index]) for w in windows
    ])
    is_transient = speed_changes > threshold

    n_transient = int(np.sum(is_transient))
    logger.info(
        "Transient detection: %d / %d windows marked (threshold=%.2f).",
        n_transient, len(windows), threshold,
    )

    return TransientResult(
        is_transient=is_transient,
        n_transient=n_transient,
        n_total=len(windows),
        speed_changes=speed_changes,
        threshold_used=threshold,
    )


def remove_transient_windows(
    windows: list[np.ndarray],
    transient_result: TransientResult,
) -> list[np.ndarray]:
    """Remove transient windows, returning only non-transient ones."""
    clean = [
        w for w, is_t in zip(windows, transient_result.is_transient) if not is_t
    ]
    logger.info(
        "Removed %d transient windows. %d remaining.",
        transient_result.n_transient, len(clean),
    )
    return clean
