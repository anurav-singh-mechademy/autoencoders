#!/usr/bin/env python
"""CLI: Among FLAGGED (yellow/red) windows, split by whether the window's
top-implicated sensor(s) trend up or down within the window (mean of last
third of rows minus mean of first third, over the top-K sensors), and
report true/false-positive rate per direction.

Usage:
    python scripts/analyze_flagged_direction.py \\
        --output-dir output_5P921A_regime_fix \\
        --data data/5P921A/5P921A_combined_with_events.parquet \\
        --event-labels data/5P921A/5P921A_event_labels_long.parquet
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging
from autoencoder.evaluation.ground_truth import load_window_ground_truth

setup_logging()
logger = logging.getLogger(__name__)


def sensor_trend(values: np.ndarray) -> float:
    """Signed trend across a window: mean(last third) - mean(first third).
    Robust to single noisy samples, unlike a bare endpoint diff or a full
    least-squares slope (which a mid-window spike can dominate)."""
    n = len(values)
    third = max(1, n // 3)
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return float("nan")
    first = values[:third]
    last = values[-third:]
    first = first[~np.isnan(first)]
    last = last[~np.isnan(last)]
    if len(first) == 0 or len(last) == 0:
        return float("nan")
    return float(last.mean() - first.mean())


def window_top_sensor_trend(window_scaled: np.ndarray, top_sensor_idxs: list[int]) -> float:
    """Mean signed trend across a window's top-implicated sensors (already
    scaled, shape (rows, n_sensors))."""
    trends = [sensor_trend(window_scaled[:, i]) for i in top_sensor_idxs if i is not None]
    trends = [t for t in trends if not np.isnan(t)]
    if not trends:
        return float("nan")
    return float(np.mean(trends))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--event-labels", required=True)
    parser.add_argument("--window-rows", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=3, help="How many of each window's top_sensors to average the trend over.")
    args = parser.parse_args()

    with open(f"{args.output_dir}/artefacts/training_metadata.json") as f:
        metadata = json.load(f)
    sensor_columns = metadata["sensor_columns"]
    sensor_index = {name: i for i, name in enumerate(sensor_columns)}

    test_windows = np.load(f"{args.output_dir}/test_windows.npy")
    test_window_ids = np.load(f"{args.output_dir}/test_window_ids.npy")
    ws = pd.read_parquet(f"{args.output_dir}/evaluation/window_scores.parquet")
    ws = ws.set_index("window_id").loc[test_window_ids].reset_index()

    gt = load_window_ground_truth(args.data, args.event_labels, window_size=args.window_rows)
    gt = gt.filter_to_window_ids(test_window_ids)
    touched = gt.is_anomaly.astype(bool)

    flagged_mask = ws["usable"].to_numpy() & (ws["zone"] != "green").to_numpy()
    logger.info("%d / %d test windows flagged (yellow/red)", flagged_mask.sum(), len(ws))

    trends = np.full(len(ws), np.nan)
    for i in np.where(flagged_mask)[0]:
        top_names = list(ws["top_sensors"].iloc[i])[: args.top_k]
        idxs = [sensor_index[n] for n in top_names if n in sensor_index]
        trends[i] = window_top_sensor_trend(test_windows[i], idxs)

    print()
    print("=" * 90)
    print(f"FLAGGED-WINDOW DIRECTIONALITY -- {args.output_dir}")
    print(f"(trend = mean[last third] - mean[first third] of scaled values, averaged over each")
    print(f" window's top-{args.top_k} implicated sensor(s); computed only for flagged windows)")
    print("=" * 90)

    valid = flagged_mask & ~np.isnan(trends)
    rising = valid & (trends > 0)
    falling = valid & (trends < 0)
    flat = valid & (trends == 0)

    print(f"{'group':30s} {'n_flagged':>10s} {'n_TP':>6s} {'n_FP':>6s} {'precision':>10s} {'FP_rate':>8s}")
    print("-" * 90)
    for label, m in [
        ("rising (trend>0)", rising),
        ("falling (trend<0)", falling),
        ("flat (trend==0)", flat),
    ]:
        n = int(m.sum())
        if n == 0:
            print(f"{label:30s} {0:>10d}      --     --         --       --")
            continue
        n_tp = int(touched[m].sum())
        n_fp = n - n_tp
        precision = n_tp / n
        print(f"{label:30s} {n:>10d} {n_tp:>6d} {n_fp:>6d} {precision:>10.3f} {1 - precision:>8.3f}")

    n_all = int(valid.sum())
    n_tp_all = int(touched[valid].sum())
    print("-" * 90)
    print(f"{'all flagged (with valid trend)':30s} {n_all:>10d} {n_tp_all:>6d} {n_all - n_tp_all:>6d} {n_tp_all / n_all:>10.3f} {1 - n_tp_all / n_all:>8.3f}")
    n_novalid = int(flagged_mask.sum() - valid.sum())
    if n_novalid:
        print(f"({n_novalid} flagged windows excluded -- top sensor(s) all-NaN in this window)")
    print("=" * 90)
    print(
        "Read: if 'falling' has a much higher FP_rate (lower precision) than\n"
        "'rising', that's a direct, quantitative confirmation that the model's\n"
        "false positives concentrate on dipping sensors while its true positives\n"
        "concentrate on rising ones -- exactly the visual pattern being tested."
    )


if __name__ == "__main__":
    main()
