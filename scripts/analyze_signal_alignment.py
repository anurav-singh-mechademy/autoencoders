#!/usr/bin/env python
"""CLI: Correlate anomaly_score and ground truth against two model-free
statistical reference metrics computed on the same scaled test_windows.npy
the autoencoder sees -- volatility (mean std of row-to-row diffs) and
magnitude (mean absolute scaled value).

Usage:
    python scripts/analyze_signal_alignment.py \\
        --output-dir output_5P921A_regime_fix \\
        --data data/5P921A/5P921A_combined_with_events.parquet \\
        --event-labels data/5P921A/5P921A_event_labels_long.parquet
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from autoencoder.logging_config import setup_logging
from autoencoder.evaluation.ground_truth import load_window_ground_truth

setup_logging()
logger = logging.getLogger(__name__)


def compute_reference_metrics(test_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (volatility, magnitude), both purely statistical -- no
    model involved. test_windows: shape (n_windows, window_rows, n_sensors),
    already scaled the same way the autoencoder was trained on.
    """
    diffs = np.diff(test_windows, axis=1)                  # (n, rows-1, sensors)
    volatility = diffs.std(axis=1).mean(axis=1)             # (n,)
    magnitude = np.abs(test_windows).mean(axis=(1, 2))       # (n,)
    return volatility, magnitude


def spearman_report(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> dict:
    rho, p = spearmanr(a, b)
    return {"a": name_a, "b": name_b, "rho": float(rho), "p_value": float(p), "n": len(a)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="e.g. output_5P921A_regime_fix (needs test_windows.npy, test_window_ids.npy, evaluation/window_scores.parquet)")
    parser.add_argument("--data", required=True, help="*_combined_with_events.parquet")
    parser.add_argument("--event-labels", required=True, help="*_event_labels_long.parquet")
    parser.add_argument("--window-rows", type=int, default=120)
    args = parser.parse_args()

    test_windows = np.load(f"{args.output_dir}/test_windows.npy")
    test_window_ids = np.load(f"{args.output_dir}/test_window_ids.npy")
    ws = pd.read_parquet(f"{args.output_dir}/evaluation/window_scores.parquet")

    # Align window_scores (which may be a different order/subset if scored
    # separately) to test_windows.npy's own order via window_id.
    score_by_id = dict(zip(ws["window_id"], ws["anomaly_score"]))
    usable_by_id = dict(zip(ws["window_id"], ws["usable"]))
    missing = [int(w) for w in test_window_ids if w not in score_by_id]
    if missing:
        raise ValueError(f"{len(missing)} test_window_ids not found in window_scores.parquet (e.g. {missing[:5]})")

    anomaly_score = np.array([score_by_id[w] for w in test_window_ids])
    usable = np.array([bool(usable_by_id[w]) for w in test_window_ids])

    volatility, magnitude = compute_reference_metrics(test_windows)

    gt = load_window_ground_truth(args.data, args.event_labels, window_size=args.window_rows)
    gt = gt.filter_to_window_ids(test_window_ids)
    touched = gt.is_anomaly.astype(float)

    mask = usable
    logger.info("Using %d / %d usable test windows", mask.sum(), len(mask))

    a, v, m, t = anomaly_score[mask], volatility[mask], magnitude[mask], touched[mask]

    rows = [
        spearman_report("anomaly_score", a, "volatility", v),
        spearman_report("anomaly_score", a, "magnitude", m),
        spearman_report("anomaly_score", a, "ground_truth(touched)", t),
        spearman_report("ground_truth(touched)", t, "volatility", v),
        spearman_report("ground_truth(touched)", t, "magnitude", m),
        spearman_report("volatility", v, "magnitude", m),
    ]

    print()
    print("=" * 78)
    print(f"SIGNAL-ALIGNMENT CHECK -- {args.output_dir}")
    print("=" * 78)
    print(f"{'a':24s} vs {'b':24s} {'rho':>8s} {'p':>10s} {'n':>6s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['a']:24s} vs {r['b']:24s} {r['rho']:>8.3f} {r['p_value']:>10.2e} {r['n']:>6d}")
    print("=" * 78)
    print(
        "Read: if anomaly_score-vs-volatility/magnitude is high while\n"
        "ground_truth-vs-volatility/magnitude is low, the model IS tracking a\n"
        "coherent statistical signal property that the labels largely don't\n"
        "reflect -- consistent with a ground-truth-alignment explanation. If\n"
        "anomaly_score-vs-volatility/magnitude is ALSO low, the model's score\n"
        "isn't well explained by simple statistical movement either."
    )


if __name__ == "__main__":
    main()
