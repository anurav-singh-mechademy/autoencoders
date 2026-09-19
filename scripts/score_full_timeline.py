#!/usr/bin/env python
"""CLI: Score the windows of one train/val/test split across a dataset's
full timeline.

Ground-truth evaluation must only ever be run against the model's own
held-out TEST split (see scripts/evaluate_against_labels.py) -- scoring
train/val windows and calling that "evaluation" is optimistic in the same
way training-error calibration is. This script constructs windows straight
from the raw file (not from the cleaned/regime-reordered training array),
then filters down to exactly the window_ids belonging to the requested
split (--split, default "test") per artefacts/split_window_ids.json (see
main.py's step_train). It saves a window-scores table keyed by
window_id/window_start/window_end so it can be joined against ground truth
resampled the same way (see autoencoder.evaluation.ground_truth) -- both are
a deterministic function of (file, window_size), so a plain integer
window_id join is safe.

Usage:
    python scripts/score_full_timeline.py \\
        --data data/5K501M/5K501M_combined_with_events.parquet \\
        --model-dir output_5K501M/artefacts \\
        --split-ids output_5K501M/artefacts/split_window_ids.json \\
        --output output_5K501M/evaluation/window_scores.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging
from autoencoder.data.ingestion import read_parquet_robust
from autoencoder.data.preprocessing import construct_windows_with_metadata
from autoencoder.artefacts.serialisation import load_artefacts
from autoencoder.inference.pipeline import infer_window
from autoencoder.alerting.zones import classify_zone

setup_logging()
logger = logging.getLogger(__name__)


def _load_timestamped_data(path: str, timestamp_column: str) -> pd.DataFrame:
    """Load a parquet/CSV file whose timestamp may be the index (as in
    *_equipment_combined.parquet) or a plain column."""
    if path.endswith(".parquet"):
        df = read_parquet_robust(path)
    else:
        df = pd.read_csv(path)

    if timestamp_column not in df.columns:
        df = df.reset_index()
        if timestamp_column not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": timestamp_column})

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(timestamp_column).reset_index(drop=True)
    return df


def score_windows(
    windows: list[dict],
    model,
    scaler,
    sensor_baselines,
    sensor_columns: list[str],
    thresholds: dict,
    tail_compression_scale: float | None,
    max_null_pct: float = 5.0,
    max_consecutive_nulls: int = 3,
    max_null_dominant_sensor_pct: float = 30.0,
    flag_threshold: float = 3.0,
    anomaly_sensor_pct: float = 10.0,
    top_k: int | None = None,
) -> list[dict]:
    records = []
    for w in windows:
        result = infer_window(
            window=w["data"],
            model=model,
            scaler=scaler,
            sensor_names=sensor_columns,
            top_k=top_k,
            sensor_baselines=sensor_baselines,
            flag_threshold=flag_threshold,
            anomaly_sensor_pct=anomaly_sensor_pct,
            max_null_pct=max_null_pct,
            max_consecutive_nulls=max_consecutive_nulls,
            max_null_dominant_sensor_pct=max_null_dominant_sensor_pct,
            already_scaled=False,
            tail_compression_scale=tail_compression_scale,
        )
        zone = classify_zone(result.window_score, thresholds) if result.usable else None
        top_names = [c["name"] for c in result.top_contributors] if result.usable else []
        masked_names = [sensor_columns[i] for i in result.masked_sensors] if result.masked_sensors else []
        # Full per-sensor ranking (best/highest-error first), not just the
        # truncated top_k -- rank-aware sensor-attribution evaluation (see
        # evaluate_against_labels.py) needs the whole ranking, since ground
        # -truth sensors are a candidate pool that a truncated top-k can't
        # be fairly scored against. sensor_errors is already computed for
        # every sensor regardless of top_k, so this costs nothing extra.
        ranked_names = (
            [sensor_columns[i] for i in np.argsort(-result.sensor_errors)] if result.usable else []
        )

        records.append({
            "window_id": w["window_id"],
            "window_start": w["start_time"],
            "window_end": w["end_time"],
            "anomaly_score": result.window_score,
            "zone": zone,
            "usable": result.usable,
            "top_sensors": top_names,
            "ranked_sensors": ranked_names,
            "masked_sensors": masked_names,
            "quality_flags": ",".join(result.quality_flags),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Score one train/val/test split's windows across a dataset's full timeline.")
    parser.add_argument("--data", required=True, help="Path to sensor data (CSV/Parquet), e.g. a *_combined_with_events.parquet file")
    parser.add_argument("--model-dir", required=True, help="Path to trained model artefacts directory")
    parser.add_argument(
        "--split-ids", required=True,
        help="Path to artefacts/split_window_ids.json (see main.py's step_train) -- restricts scoring to "
             "the requested --split's raw-file window ids instead of the whole file.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                         help="Which split to score. Ground-truth evaluation should always use 'test'.")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--window-rows", type=int, default=120)
    parser.add_argument("--output", required=True, help="Output path for the window-scores table (.parquet or .csv)")
    parser.add_argument("--sensor-flag-threshold", type=float, default=3.0)
    parser.add_argument("--anomaly-sensor-pct", type=float, default=10.0)
    parser.add_argument("--max-null-pct", type=float, default=5.0)
    parser.add_argument("--max-consecutive-nulls", type=int, default=3)
    parser.add_argument("--max-null-dominant-sensor-pct", type=float, default=30.0)
    args = parser.parse_args()

    model, scaler, thresholds, metadata, _, sensor_baselines = load_artefacts(args.model_dir)
    sensor_columns = metadata["sensor_columns"]
    tail_compression_scale = metadata.get("tail_compression_scale")
    if tail_compression_scale is None:
        logger.warning(
            "No tail_compression_scale in %s/training_metadata.json -- scores will be "
            "wrong if this model was trained with preprocessing.tail_compression_scale set.",
            args.model_dir,
        )

    df = _load_timestamped_data(args.data, args.timestamp_column)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), args.data)

    windows = construct_windows_with_metadata(
        df, window_size=args.window_rows, timestamp_column=args.timestamp_column, sensor_columns=sensor_columns,
    )
    logger.info("Constructed %d windows (window_rows=%d)", len(windows), args.window_rows)

    with open(args.split_ids) as f:
        split_window_ids = json.load(f)
    if args.split not in split_window_ids:
        raise ValueError(f"{args.split_ids} has no '{args.split}' split (keys: {list(split_window_ids.keys())})")
    keep_ids = set(int(i) for i in split_window_ids[args.split])
    windows = [w for w in windows if w["window_id"] in keep_ids]
    logger.info("Restricted to '%s' split: %d windows (%d requested ids)", args.split, len(windows), len(keep_ids))
    if len(windows) != len(keep_ids):
        logger.warning(
            "%d requested '%s' window id(s) were not found by constructing windows from --data -- "
            "--data, --window-rows, and --timestamp-column must match what produced --split-ids.",
            len(keep_ids) - len(windows), args.split,
        )

    top_k = min(10, max(1, len(sensor_columns) // 5))
    records = score_windows(
        windows, model, scaler, sensor_baselines, sensor_columns, thresholds, tail_compression_scale,
        max_null_pct=args.max_null_pct,
        max_consecutive_nulls=args.max_consecutive_nulls,
        max_null_dominant_sensor_pct=args.max_null_dominant_sensor_pct,
        flag_threshold=args.sensor_flag_threshold,
        anomaly_sensor_pct=args.anomaly_sensor_pct,
        top_k=top_k,
    )

    out_df = pd.DataFrame(records)
    n_usable = int(out_df["usable"].sum())
    logger.info("Scored %d windows: %d usable, %d unusable", len(out_df), n_usable, len(out_df) - n_usable)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.output.endswith(".parquet"):
        out_df.to_parquet(args.output, index=False)
    else:
        out_df = out_df.copy()
        out_df["top_sensors"] = out_df["top_sensors"].map(",".join)
        out_df["ranked_sensors"] = out_df["ranked_sensors"].map(",".join)
        out_df["masked_sensors"] = out_df["masked_sensors"].map(",".join)
        out_df.to_csv(args.output, index=False)
    logger.info("Saved window scores to %s", args.output)


if __name__ == "__main__":
    main()
