#!/usr/bin/env python
"""Prepare a raw real-world sensor export for the autoencoder pipeline.

Real historian exports (unlike the synthetic generator) have missing data:
sensors added/retired mid-series, comms dropouts, decommissioned tags. Nothing
downstream (RobustScaler, variance filter, windowing) tolerates NaNs reaching
the model -- a single NaN row poisons the whole training run (NaN loss).

This script:
    1. Drops columns whose null fraction exceeds --max-null-pct (too sparse to
       learn a meaningful normal baseline from).
    2. Forward-fills remaining gaps, then back-fills any leading gap, so every
       kept column is fully populated.
    3. Writes a cleaned CSV ready for main.py's normal --data argument.

Imputed/flat stretches are intentionally left for the existing sensor-dropout
filter (main.py step_clean, threshold_pct in configs/*.yaml) to catch --
forward-filled runs read as "stuck" and windows dominated by them are excluded
from training the same way they would be for any other frozen sensor.

Usage:
    python scripts/prepare_real_data.py --input data/dvn_data_1.csv \\
        --output data/dvn_data_1_clean.csv --timestamp-column datetime
"""

import argparse
import logging

import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def prepare_real_data(
    df: pd.DataFrame,
    timestamp_column: str = "datetime",
    max_null_pct: float = 50.0,
) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c != timestamp_column]

    # Historian exports can contain literal +/-inf (e.g. a derived ratio tag
    # whose reference value hit zero) -- pandas' isnull() doesn't count inf
    # as missing, so it sails through null-pct filtering and ffill/bfill
    # untouched, then crashes RobustScaler.fit downstream with "infinity or
    # a value too large for dtype". Treat it as missing, same as NaN, before
    # anything else runs.
    numeric_cols = df[sensor_cols].select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        inf_mask = np.isinf(df[numeric_cols].to_numpy())
        n_inf = int(inf_mask.sum())
        if n_inf:
            df = df.copy()
            df[numeric_cols] = df[numeric_cols].mask(pd.DataFrame(inf_mask, columns=numeric_cols, index=df.index), np.nan)
            affected = [c for c, bad in zip(numeric_cols, inf_mask.any(axis=0)) if bad]
            logger.warning(
                "Replaced %d +/-inf value(s) across %d column(s) with NaN before null/fill handling: %s",
                n_inf, len(affected), affected,
            )

    null_pct = df[sensor_cols].isnull().mean() * 100

    dropped = null_pct[null_pct > max_null_pct].index.tolist()
    kept = [c for c in sensor_cols if c not in dropped]
    logger.info(
        "Dropping %d / %d columns with >%.0f%% missing data.",
        len(dropped), len(sensor_cols), max_null_pct,
    )

    out = df[[timestamp_column] + kept].copy()
    remaining_nulls_before = int(out[kept].isnull().sum().sum())
    out[kept] = out[kept].ffill().bfill()
    remaining_nulls_after = int(out[kept].isnull().sum().sum())

    logger.info(
        "Filled %d missing values via forward/back-fill on %d kept columns "
        "(%d values still null afterwards -- would mean an entirely-empty "
        "kept column).",
        remaining_nulls_before - remaining_nulls_after, len(kept), remaining_nulls_after,
    )
    if remaining_nulls_after:
        still_null = out[kept].columns[out[kept].isnull().any()].tolist()
        logger.warning("Columns still containing nulls after fill: %s", still_null)

    return out


def main():
    parser = argparse.ArgumentParser(description="Clean a raw real sensor export for the pipeline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timestamp-column", default="datetime")
    parser.add_argument("--max-null-pct", type=float, default=50.0,
                         help="Drop columns with more than this %% missing (default 50).")
    args = parser.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    df[args.timestamp_column] = pd.to_datetime(df[args.timestamp_column])
    df = df.sort_values(args.timestamp_column).reset_index(drop=True)

    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), args.input)

    cleaned = prepare_real_data(df, timestamp_column=args.timestamp_column, max_null_pct=args.max_null_pct)
    if args.output.endswith(".parquet"):
        cleaned.to_parquet(args.output, index=False)
    else:
        cleaned.to_csv(args.output, index=False)
    logger.info("Wrote %d rows, %d columns to %s", len(cleaned), len(cleaned.columns), args.output)


if __name__ == "__main__":
    main()
