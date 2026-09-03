"""Data ingestion -- load sensor data from local files, S3, or DataFrames."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(path: str | Path, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Load sensor data from CSV or Parquet (local path or S3 URI).

    Auto-detects format from extension. Parses timestamp column and sorts by it.
    """
    path_str = str(path)
    logger.info("Loading data from: %s", path_str)

    if path_str.endswith(".parquet"):
        df = pd.read_parquet(path_str)
    else:
        df = pd.read_csv(path_str)

    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(timestamp_column).reset_index(drop=True)

    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))
    return df


def detect_sensor_columns(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    exclude_columns: Optional[list[str]] = None,
) -> list[str]:
    """Auto-detect sensor columns -- all numeric columns except timestamp and exclusions."""
    exclude = {timestamp_column}
    if exclude_columns:
        exclude.update(exclude_columns)

    sensor_cols = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]
    logger.info("Detected %d sensor columns.", len(sensor_cols))
    return sorted(sensor_cols)


def validate_dataframe(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    expected_sensors: Optional[list[str]] = None,
) -> list[str]:
    """Validate a DataFrame. Returns list of error strings (empty = valid)."""
    errors: list[str] = []

    if timestamp_column not in df.columns:
        errors.append(f"Missing timestamp column: '{timestamp_column}'")

    if expected_sensors:
        missing = set(expected_sensors) - set(df.columns)
        if missing:
            errors.append(f"Missing sensor columns: {sorted(missing)}")

    sensor_cols = expected_sensors or [
        c for c in df.columns if c != timestamp_column
    ]
    non_numeric = [
        c for c in sensor_cols
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        errors.append(f"Non-numeric sensor columns: {non_numeric}")

    return errors
