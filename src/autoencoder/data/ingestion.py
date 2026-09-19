"""Data ingestion -- load sensor data from local files, S3, or DataFrames."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def read_parquet_robust(path: str | Path) -> pd.DataFrame:
    """Read a parquet file, working around a pyarrow limitation where a wide
    table containing large list<...> columns can fail to convert to pandas
    in one shot with:

        ArrowNotImplementedError: Nested data conversions not implemented
        for chunked array outputs

    Observed on a real export with list<string> columns (concurrent
    rule-slug/sensor unions per row) that grew large enough per-row (up to
    ~1,884 concurrent entries on one equipment) to force pyarrow to store
    the column across multiple internal chunks -- converting a *table*
    containing such a column hits this unimplemented path, even though
    converting each chunk of that column *alone* works fine. Plain
    pd.read_parquet() -- used for every other file in this pipeline --
    never hits this because none of them carry list-typed columns this
    large.

    Reads scalar/non-nested columns the normal (fast) way, and any list-typed
    columns separately, chunk by chunk, then reassembles one DataFrame.
    Behaves identically to pd.read_parquet(path) when there are no list
    columns or none large enough to trigger the bug.
    """
    schema = pq.read_schema(path)
    list_columns = [f.name for f in schema if pa.types.is_list(f.type) or pa.types.is_large_list(f.type)]
    scalar_columns = [name for name in schema.names if name not in list_columns]

    df = pd.read_parquet(path, columns=scalar_columns) if scalar_columns else pd.DataFrame(index=range(pq.ParquetFile(path).metadata.num_rows))

    for col in list_columns:
        table = pq.read_table(path, columns=[col])
        chunked = table.column(col)
        values = []
        for chunk in chunked.chunks:
            values.extend(chunk.to_pylist())
        df[col] = values

    return df


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
