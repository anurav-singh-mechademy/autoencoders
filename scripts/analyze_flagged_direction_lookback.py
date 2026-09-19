#!/usr/bin/env python
"""CLI: Among FLAGGED (yellow/red) windows, compare each window's top
implicated sensor(s) against their own mean over a configurable lookback
period beforehand, and report true/false-positive rate for rising vs.
falling groups. Deltas are in RobustScaler units (divided by scaler.scale_)
so they're comparable to the rest of this analysis without an explicit
scaler.transform() call on raw slices.

Usage:
    python scripts/analyze_flagged_direction_lookback.py \\
        --output-dir output_5P921A_regime_fix \\
        --data data/5P921A/5P921A_combined_with_events.parquet \\
        --event-labels data/5P921A/5P921A_event_labels_long.parquet \\
        --lookback-hours 3
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging
from autoencoder.data.ingestion import read_parquet_robust
from autoencoder.evaluation.ground_truth import load_window_ground_truth

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--event-labels", required=True)
    parser.add_argument("--window-rows", type=int, default=120)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--top-k", type=int, default=3, help="How many of each window's top_sensors to average the lookback delta over.")
    parser.add_argument("--lookback-hours", type=float, default=3.0, help="How far back (hours) to look for the 'before' baseline, immediately preceding the flagged window.")
    parser.add_argument("--row-interval-seconds", type=float, default=15.0, help="Native row sampling interval, to convert --lookback-hours to a row count.")
    args = parser.parse_args()

    with open(f"{args.output_dir}/artefacts/training_metadata.json") as f:
        metadata = json.load(f)
    sensor_columns = metadata["sensor_columns"]
    sensor_index = {name: i for i, name in enumerate(sensor_columns)}
    scaler = joblib.load(f"{args.output_dir}/artefacts/scaler.pkl")
    scale = scaler.scale_

    df = read_parquet_robust(args.data)
    if args.timestamp_column not in df.columns:
        df = df.reset_index()
    df[args.timestamp_column] = pd.to_datetime(df[args.timestamp_column])
    df = df.sort_values(args.timestamp_column).reset_index(drop=True)

    lookback_rows = max(1, round(args.lookback_hours * 3600 / args.row_interval_seconds))
    logger.info("Lookback of %.1fh -> %d rows at %.0fs/row", args.lookback_hours, lookback_rows, args.row_interval_seconds)

    window_rows = args.window_rows
    ws = pd.read_parquet(f"{args.output_dir}/evaluation/window_scores.parquet")
    test_window_ids = np.load(f"{args.output_dir}/test_window_ids.npy")
    ws = ws.set_index("window_id").loc[test_window_ids].reset_index()

    gt = load_window_ground_truth(args.data, args.event_labels, window_size=window_rows)
    gt = gt.filter_to_window_ids(test_window_ids)
    touched = gt.is_anomaly.astype(bool)

    flagged_mask = ws["usable"].to_numpy() & (ws["zone"] != "green").to_numpy()
    logger.info("%d / %d test windows flagged (yellow/red)", flagged_mask.sum(), len(ws))

    sensor_arrays = {name: df[name].to_numpy(dtype=float) for name in sensor_columns}

    deltas = np.full(len(ws), np.nan)
    for i in np.where(flagged_mask)[0]:
        wid = int(ws["window_id"].iloc[i])
        curr_start, curr_end = wid * window_rows, (wid + 1) * window_rows
        lb_start, lb_end = max(0, curr_start - lookback_rows), curr_start
        if lb_end <= lb_start:
            continue
        top_names = list(ws["top_sensors"].iloc[i])[: args.top_k]
        per_sensor_deltas = []
        for name in top_names:
            if name not in sensor_index:
                continue
            arr = sensor_arrays[name]
            curr_vals = arr[curr_start:curr_end]
            lb_vals = arr[lb_start:lb_end]
            curr_mean = np.nanmean(curr_vals) if np.any(~np.isnan(curr_vals)) else np.nan
            lb_mean = np.nanmean(lb_vals) if np.any(~np.isnan(lb_vals)) else np.nan
            if np.isnan(curr_mean) or np.isnan(lb_mean):
                continue
            s = scale[sensor_index[name]]
            if s == 0:
                continue
            per_sensor_deltas.append((curr_mean - lb_mean) / s)
        if per_sensor_deltas:
            deltas[i] = float(np.mean(per_sensor_deltas))

    print()
    print("=" * 92)
    print(f"FLAGGED-WINDOW DIRECTIONALITY (LOOKBACK) -- {args.output_dir}")
    print(f"(delta = mean[window] - mean[preceding {args.lookback_hours:g}h], in RobustScaler-IQR units,")
    print(f" averaged over each flagged window's top-{args.top_k} implicated sensor(s))")
    print("=" * 92)

    valid = flagged_mask & ~np.isnan(deltas)
    rising = valid & (deltas > 0)
    falling = valid & (deltas < 0)

    print(f"{'group':30s} {'n_flagged':>10s} {'n_TP':>6s} {'n_FP':>6s} {'precision':>10s} {'FP_rate':>8s}")
    print("-" * 92)
    for label, m in [
        ("rising (delta>0, higher than before)", rising),
        ("falling (delta<0, lower than before)", falling),
    ]:
        n = int(m.sum())
        if n == 0:
            print(f"{label:30s} {0:>10d}      --     --         --       --")
            continue
        n_tp = int(touched[m].sum())
        n_fp = n - n_tp
        precision = n_tp / n
        print(f"{label:38s} {n:>10d} {n_tp:>6d} {n_fp:>6d} {precision:>10.3f} {1 - precision:>8.3f}")

    n_all = int(valid.sum())
    n_tp_all = int(touched[valid].sum())
    print("-" * 92)
    print(f"{'all flagged (with valid delta)':38s} {n_all:>10d} {n_tp_all:>6d} {n_all - n_tp_all:>6d} {n_tp_all / n_all:>10.3f} {1 - n_tp_all / n_all:>8.3f}")
    n_novalid = int(flagged_mask.sum() - valid.sum())
    if n_novalid:
        print(f"({n_novalid} flagged windows excluded -- no lookback data or all-NaN sensor(s))")
    print("=" * 92)


if __name__ == "__main__":
    main()
