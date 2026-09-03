"""Robust scaling and window construction for sensor data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

WINDOW_ROWS = 120  # 30 minutes at 15-second intervals


@dataclass
class ColumnFilterResult:
    """Result of low-variance / constant column removal."""

    kept_columns: list[str]
    removed_columns: list[str]
    variances: dict[str, float]


def remove_low_variance_columns(
    df: pd.DataFrame,
    sensor_columns: list[str],
    variance_threshold: float = 1e-5,
) -> ColumnFilterResult:
    """Remove sensor columns with variance below a threshold.

    Constant or near-constant sensors (e.g. boolean run-status, stuck
    sensors) add noise to the autoencoder without useful signal. This
    step drops them before training.

    Args:
        df: Input DataFrame.
        sensor_columns: List of sensor column names to evaluate.
        variance_threshold: Columns with variance <= this are removed.

    Returns:
        ColumnFilterResult with kept/removed lists and per-column variances.
    """
    variances = df[sensor_columns].var(numeric_only=True)

    low_var = variances[variances <= variance_threshold].index.tolist()
    kept = [c for c in sensor_columns if c not in low_var]

    logger.info(
        "Low-variance filter: removed %d / %d columns (threshold=%.1e). "
        "Removed: %s",
        len(low_var), len(sensor_columns), variance_threshold, low_var,
    )

    return ColumnFilterResult(
        kept_columns=kept,
        removed_columns=low_var,
        variances={str(k): float(v) for k, v in variances.items()},
    )


@dataclass
class ScalerResult:
    """Container for a fitted scaler and its metadata."""

    scaler: RobustScaler
    sensor_columns: list[str]
    n_sensors: int
    medians: np.ndarray
    iqrs: np.ndarray


def fit_robust_scaler(
    df: pd.DataFrame,
    sensor_columns: list[str],
) -> ScalerResult:
    """Fit a RobustScaler on the training data.

    Formula: X_scaled = (X - median) / IQR
    The scaler is fit ONLY on training data and reused at inference. Never refit on live data.
    """
    scaler = RobustScaler()
    scaler.fit(df[sensor_columns].values)

    return ScalerResult(
        scaler=scaler,
        sensor_columns=sensor_columns,
        n_sensors=len(sensor_columns),
        medians=scaler.center_,
        iqrs=scaler.scale_,
    )


def apply_scaling(
    df: pd.DataFrame,
    scaler_result: ScalerResult,
) -> pd.DataFrame:
    """Apply a fitted RobustScaler to a DataFrame.

    Returns a new DataFrame with scaled sensor values and non-sensor columns preserved.
    """
    scaled_values = scaler_result.scaler.transform(
        df[scaler_result.sensor_columns].values
    )
    result = df.copy()
    result[scaler_result.sensor_columns] = scaled_values
    return result


def construct_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_ROWS,
    timestamp_column: str = "timestamp",
    sensor_columns: Optional[list[str]] = None,
    drop_incomplete: bool = True,
) -> list[np.ndarray]:
    """Segment a time-sorted DataFrame into fixed-size windows.

    Args:
        df: Input DataFrame, must be sorted by timestamp.
        window_size: Number of rows per window (default 120).
        timestamp_column: Name of the timestamp column.
        sensor_columns: Sensor columns to include. If None, uses all numeric columns.
        drop_incomplete: If True, discard the last window if it has fewer than window_size rows.

    Returns:
        List of numpy arrays, each of shape (window_size, n_sensors).
    """
    if sensor_columns is None:
        sensor_columns = [
            c for c in df.columns
            if c != timestamp_column and pd.api.types.is_numeric_dtype(df[c])
        ]

    values = df[sensor_columns].values
    n_rows = len(values)
    n_windows = n_rows // window_size

    windows = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        windows.append(values[start:end])

    if not drop_incomplete and n_rows % window_size > 0:
        windows.append(values[n_windows * window_size:])

    logger.info(
        "Constructed %d complete windows of size %d from %d rows.",
        len(windows), window_size, n_rows,
    )
    return windows


def construct_windows_with_metadata(
    df: pd.DataFrame,
    window_size: int = WINDOW_ROWS,
    timestamp_column: str = "timestamp",
    sensor_columns: Optional[list[str]] = None,
) -> list[dict]:
    """Construct windows with associated metadata (start/end timestamps, index range).

    Returns:
        List of dicts: 'data' (ndarray), 'start_time', 'end_time',
        'start_idx', 'end_idx', 'window_id'.
    """
    if sensor_columns is None:
        sensor_columns = [
            c for c in df.columns
            if c != timestamp_column and pd.api.types.is_numeric_dtype(df[c])
        ]

    n_rows = len(df)
    n_windows = n_rows // window_size
    windows = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        chunk = df.iloc[start:end]

        windows.append({
            "data": chunk[sensor_columns].values,
            "start_time": chunk[timestamp_column].iloc[0] if timestamp_column in chunk.columns else None,
            "end_time": chunk[timestamp_column].iloc[-1] if timestamp_column in chunk.columns else None,
            "start_idx": start,
            "end_idx": end,
            "window_id": i,
        })

    logger.info("Constructed %d windows with metadata.", len(windows))
    return windows
