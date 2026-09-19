#!/usr/bin/env python
"""CLI: Recode every flagged (yellow/red) window of the chosen modality
(--suppress negative/positive, via analyze_reconstruction_error_sign.py's
row-tied modality) to "green", then recompute precision/recall/PR-AUC over
the whole test set and compare to the unmodified baseline. PR-AUC is
threshold-independent so it's expected to be identical either way.

Usage:
    python scripts/simulate_modality_suppression.py \\
        --output-dir output_5P921A \\
        --data data/5P921A/5P921A_combined_with_events.parquet \\
        --event-labels data/5P921A/5P921A_event_labels_long.parquet
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch

from autoencoder.logging_config import setup_logging
from autoencoder.artefacts.serialisation import load_artefacts
from autoencoder.evaluation.ground_truth import load_window_ground_truth
from autoencoder.evaluation.slug_metrics import pointwise_metrics

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--event-labels", required=True)
    parser.add_argument("--window-rows", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=3, help="How many top-implicated sensors the modality signal averages over.")
    parser.add_argument("--suppress", choices=["negative", "positive"], default="negative",
                         help="Which modality's flagged windows to recode to 'green'.")
    args = parser.parse_args()

    with open(f"{args.output_dir}/artefacts/training_metadata.json") as f:
        metadata = json.load(f)
    sensor_columns = metadata["sensor_columns"]
    sensor_index = {name: i for i, name in enumerate(sensor_columns)}

    model, scaler, thresholds, metadata2, training_errors, sensor_baselines = load_artefacts(f"{args.output_dir}/artefacts")

    test_windows = np.load(f"{args.output_dir}/test_windows.npy")
    test_window_ids = np.load(f"{args.output_dir}/test_window_ids.npy")
    ws = pd.read_parquet(f"{args.output_dir}/evaluation/window_scores.parquet")
    ws = ws.set_index("window_id").loc[test_window_ids].reset_index()

    gt = load_window_ground_truth(args.data, args.event_labels, window_size=args.window_rows)
    gt = gt.filter_to_window_ids(test_window_ids)
    touched = np.asarray(gt.is_anomaly).astype(bool)

    x = torch.tensor(test_windows, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        x_hat = model(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    r = (x - x_hat).numpy()
    row_sq_error = (r ** 2).mean(axis=2)  # (n, rows) -- mirrors per_row_mse (all sensors)

    signed_error_row_tied = np.full(len(ws), np.nan)
    for i, top_names in enumerate(ws["top_sensors"]):
        idxs = [sensor_index[n] for n in list(top_names)[: args.top_k] if n in sensor_index]
        if idxs:
            target = np.percentile(row_sq_error[i], 95)
            row_idx = int(np.argmin(np.abs(row_sq_error[i] - target)))
            signed_error_row_tied[i] = float(r[i, row_idx, idxs].mean())

    usable = ws["usable"].to_numpy()
    zone = ws["zone"].to_numpy(dtype=object)
    score = ws["anomaly_score"].to_numpy()
    flagged = usable & (zone != "green")

    valid_flagged = flagged & ~np.isnan(signed_error_row_tied)
    to_suppress = valid_flagged & (signed_error_row_tied < 0) if args.suppress == "negative" else valid_flagged & (signed_error_row_tied > 0)

    adjusted_zone = zone.copy()
    adjusted_zone[to_suppress] = "green"

    baseline = pointwise_metrics(touched, zone, score, usable=usable)
    adjusted = pointwise_metrics(touched, adjusted_zone, score, usable=usable)

    n_suppressed = int(to_suppress.sum())
    n_suppressed_was_tp = int(touched[to_suppress].sum())
    n_suppressed_was_fp = n_suppressed - n_suppressed_was_tp

    print()
    print("=" * 92)
    print(f"MODALITY SUPPRESSION SIMULATION -- {args.output_dir}")
    print(f"(suppress = recode {args.suppress}-modality [top-{args.top_k}] flagged windows to 'green')")
    print("=" * 92)
    print(f"Windows suppressed: {n_suppressed}  (of which {n_suppressed_was_tp} were true positives, {n_suppressed_was_fp} were false positives)")
    print()
    print(f"{'metric':12s} {'baseline':>10s} {'suppressed':>12s} {'delta':>10s}")
    print("-" * 92)
    for key, name in [("pr_auc", "PR-AUC"), ("precision", "precision"), ("recall", "recall"), ("f1", "f1")]:
        b, a = baseline[key], adjusted[key]
        delta = a - b if not (np.isnan(a) or np.isnan(b)) else float("nan")
        print(f"{name:12s} {b:>10.3f} {a:>12.3f} {delta:>+10.3f}")
    print("-" * 92)
    print(f"{'tp':12s} {baseline['tp']:>10d} {adjusted['tp']:>12d}")
    print(f"{'fp':12s} {baseline['fp']:>10d} {adjusted['fp']:>12d}")
    print(f"{'fn':12s} {baseline['fn']:>10d} {adjusted['fn']:>12d}")
    print("=" * 92)
    print(
        "Read: PR-AUC is expected to be IDENTICAL (it's computed purely from the\n"
        "continuous anomaly_score's ranking, never from zone/threshold -- this\n"
        "operation only recodes zone). If precision goes up and recall goes down,\n"
        "the negative-modality flags were a mix that included some real events;\n"
        "if recall is unchanged, none of the suppressed windows were real events\n"
        "and this is a pure win (fewer false alarms, nothing lost)."
    )


if __name__ == "__main__":
    main()
