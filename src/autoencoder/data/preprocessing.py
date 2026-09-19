"""Robust scaling and window construction for sensor data."""

from __future__ import annotations

import logging
import warnings
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
class NullOrStuckFilterResult:
    """Result of the per-window null-or-stuck column filter."""

    kept_columns: list[str]
    removed_columns: list[str]
    null_pct: dict[str, float]   # % of fixed-size windows each column was null in
    stuck_pct: dict[str, float]  # % of fixed-size windows each column was flat in


def remove_null_or_stuck_columns(
    df: pd.DataFrame,
    sensor_columns: list[str],
    window_size: int = WINDOW_ROWS,
    max_null_pct: float = 50.0,
    max_stuck_pct: float = 100.0,
    stuck_atol: float = 1e-9,
) -> NullOrStuckFilterResult:
    """Remove sensor columns that are null in more than max_null_pct, or
    flat ("stuck") in more than max_stuck_pct, of the file's fixed-size
    windows.

    remove_low_variance_columns() can't catch a sensor that's null/flat MOST
    of the time but has a few genuine excursions spread across a long
    series -- those are enough to give it healthy-looking whole-file
    variance, even though it's dead at the window granularity the model
    actually trains on.

    Null and stuck are scored and thresholded SEPARATELY, deliberately: a
    null sensor is missing data, unambiguously a data-quality problem: a
    flat/stuck sensor is frequently just deadband-/compression-logging, or a
    real steady operating state -- genuine signal, not a fault. The default
    max_stuck_pct=100.0 means a window-fraction can never exceed it (100% is
    the ceiling, and this checks "greater than"), so by default this
    function drops a column ONLY for being null-dominated, never merely for
    being stuck -- a sensor that's flat in every single window still trains
    fine (the model just learns to reconstruct a constant). A truly
    always-constant-since-the-dawn-of-time sensor is caught anyway, by
    remove_low_variance_columns()'s independent whole-file check. Lower
    max_stuck_pct explicitly if you do want to cull frequently-stuck sensors.

    Args:
        df: Input DataFrame (raw, unscaled -- run this before fit_robust_scaler
            so a column dropped here never influences the scaler).
        sensor_columns: List of sensor column names to evaluate.
        window_size: Rows per window -- must match construct_windows(_with_metadata)'s
            window_size for "per window" to mean the same thing downstream.
        max_null_pct: Columns null in more than this % of windows are removed.
        max_stuck_pct: Columns flat in more than this % of windows are removed.
        stuck_atol: Absolute tolerance for "flat" within a window -- matches
            sensor_dropout's historical default (sensor_dropout itself no
            longer checks "stuck" at all -- see its module docstring).

    Returns:
        NullOrStuckFilterResult with kept/removed lists and per-column %.
    """
    values = df[sensor_columns].to_numpy(dtype=float)
    n_rows = len(values)
    n_windows = n_rows // window_size

    null_counts = np.zeros(len(sensor_columns))
    stuck_counts = np.zeros(len(sensor_columns))
    for i in range(n_windows):
        w = values[i * window_size:(i + 1) * window_size]
        is_null = np.isnan(w).any(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spread = np.nanmax(w, axis=0) - np.nanmin(w, axis=0)
        is_stuck = np.nan_to_num(spread, nan=0.0) <= stuck_atol
        null_counts += is_null
        stuck_counts += is_stuck

    if n_windows:
        null_pct_arr = null_counts / n_windows * 100
        stuck_pct_arr = stuck_counts / n_windows * 100
    else:
        null_pct_arr = np.zeros(len(sensor_columns))
        stuck_pct_arr = np.zeros(len(sensor_columns))
    null_pct = {c: float(p) for c, p in zip(sensor_columns, null_pct_arr)}
    stuck_pct = {c: float(p) for c, p in zip(sensor_columns, stuck_pct_arr)}

    removed = [
        c for c in sensor_columns
        if null_pct[c] > max_null_pct or stuck_pct[c] > max_stuck_pct
    ]
    kept = [c for c in sensor_columns if c not in removed]

    logger.info(
        "Null-or-stuck filter: removed %d / %d columns (max_null_pct=%.0f%%, max_stuck_pct=%.0f%% "
        "of %d windows). Removed: %s",
        len(removed), len(sensor_columns), max_null_pct, max_stuck_pct, n_windows, removed,
    )

    return NullOrStuckFilterResult(
        kept_columns=kept,
        removed_columns=removed,
        null_pct=null_pct,
        stuck_pct=stuck_pct,
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
    tail_compression_scale: Optional[float] = None,
) -> pd.DataFrame:
    """Apply a fitted RobustScaler to a DataFrame.

    Args:
        tail_compression_scale: If set (call it c), apply the generalized
            inverse hyperbolic sine transform c * asinh(z / c) to the
            RobustScaler output z, where z = (x - median) / IQR.

            RobustScaler divides by the interquartile range, which is
            near-zero for tags that sit in a narrow band almost all the
            time (deadband-logged sensors, gas-composition fractions,
            analyzer-fault readings). A glitch reading on such a tag --
            confirmed on real historian data as a correlated event, tens of
            tags spiking together at one timestamp, not independent per-tag
            noise -- then rescales to a z-score in the thousands or more,
            which swamps the MSE loss and prevents the autoencoder from
            learning anything from the other (well-behaved) sensors.

            asinh(x) behaves like x for |x| << c (so normal-range readings,
            and genuine large-but-real excursions up to roughly c, pass
            through close to unchanged) and like sign(x)*c*log(2|x|/c) for
            |x| >> c (so a 2e9 z-score compresses to roughly 20*log(4e9) =
            ~440, not the ~20 a hard clip would give, and stays strictly
            larger than a milder 100-sigma glitch -- relative severity, and
            the gradient, are preserved instead of saturating to a flat
            zero-gradient plateau). This is the generalized/inverse
            hyperbolic sine (IHS) transform (Johnson 1949; Burbidge, Magee
            & Robb 1988) -- a standard variance-stabilizing alternative to
            log() for data that (unlike a log-transform target) can be zero
            or negative, used for exactly this "narrow normal band + rare
            extreme excursion" distribution shape in econometrics and
            high-dynamic-range instrument data (e.g. flow cytometry).
            Unlike dropping the offending rows/windows, every reading stays
            in the series -- important for a monitoring pipeline where a
            timestamp can't simply be discarded from a live equipment feed.

            c sets where the transform bends from linear to logarithmic;
            pick it around the largest *genuine* excursion the sensor set
            shows (see configs/40_data.yaml for how that was estimated).
            None (default) preserves the original untransformed behaviour.

    Returns a new DataFrame with scaled sensor values and non-sensor columns preserved.
    """
    scaled_values = scaler_result.scaler.transform(
        df[scaler_result.sensor_columns].values
    )
    scaled_values = apply_tail_compression(scaled_values, tail_compression_scale)
    result = df.copy()
    result[scaler_result.sensor_columns] = scaled_values
    return result


def apply_tail_compression(
    scaled_values: np.ndarray,
    tail_compression_scale: Optional[float],
) -> np.ndarray:
    """c*asinh(z/c) tail compression on already-RobustScaler-transformed values.

    Split out of apply_scaling() so any other caller that scales data outside
    a DataFrame (e.g. a single raw inference window, already run through
    scaler.transform()) can apply the identical transform a model was
    calibrated with -- see apply_scaling()'s docstring for the full
    rationale. A no-op when tail_compression_scale is None.
    """
    if tail_compression_scale is None:
        return scaled_values
    c = tail_compression_scale
    return c * np.arcsinh(scaled_values / c)


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


def compute_event_touched_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_ROWS,
    active_nodes_column: str = "n_active_nodes",
) -> np.ndarray:
    """Boolean array, one entry per window, True if that window was touched
    by >=1 labeled ground-truth event node.

    Uses the SAME windowing convention as construct_windows_with_metadata
    (window i covers rows [i*window_size, (i+1)*window_size), trailing
    incomplete window dropped), so the returned array can be indexed
    directly by the window_id construct_windows_with_metadata assigns.

    Args:
        df: A DataFrame carrying the ground-truth bookkeeping column (e.g. a
            *_combined_with_events.parquet file's own n_active_nodes column
            -- excluded from sensor_columns via data.exclude_columns, but
            still present in the DataFrame itself; apply_scaling() preserves
            non-sensor columns, so this works on either the raw or scaled df).
        active_nodes_column: Name of the per-row active-node-count column.

    Raises:
        KeyError if active_nodes_column isn't in df -- silently treating
        "no ground truth available" as "nothing is event-touched" would let
        every window through unfiltered with no warning.
    """
    if active_nodes_column not in df.columns:
        raise KeyError(
            f"'{active_nodes_column}' not found in the input data -- can't determine which "
            "windows are event-touched without it."
        )
    values = df[active_nodes_column].to_numpy()
    n_windows = len(values) // window_size
    trimmed = values[: n_windows * window_size].reshape(n_windows, window_size)
    return (trimmed > 0).any(axis=1)
