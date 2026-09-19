#!/usr/bin/env python
"""CLI: Correlate anomaly_score and ground truth against two SIGNED
directional metrics -- within-window drift and level vs. training
baseline -- and report a rising/falling (or above/below-baseline)
precision-recall-correlation breakdown for each.

Usage:
    python scripts/analyze_directional_alignment.py \\
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


def compute_reference_metrics(
    test_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-window (volatility, magnitude, drift, level) on scaled
    test_windows (shape n_windows x window_rows x n_sensors). drift = signed
    within-window trend; level = signed mean vs. the RobustScaler baseline (0).
    """
    diffs = np.diff(test_windows, axis=1)               # (n, rows-1, sensors)
    volatility = diffs.std(axis=1).mean(axis=1)          # (n,)
    magnitude = np.abs(test_windows).mean(axis=(1, 2))   # (n,)
    drift = diffs.mean(axis=(1, 2))                      # (n,) signed
    level = test_windows.mean(axis=(1, 2))               # (n,) signed
    return volatility, magnitude, drift, level


def spearman_report(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> dict:
    rho, p = spearmanr(a, b)
    return {"a": name_a, "b": name_b, "rho": float(rho), "p_value": float(p), "n": len(a)}


def directional_breakdown(
    anomaly_score: np.ndarray, touched: np.ndarray, flagged: np.ndarray, signed_metric: np.ndarray, labels: tuple[str, str]
) -> list[dict]:
    rows = []
    for label, mask in [
        (labels[0], signed_metric > 0),
        (labels[1], signed_metric < 0),
    ]:
        n = int(mask.sum())
        if n == 0:
            rows.append({"direction": label, "n": 0})
            continue
        a, t, f = anomaly_score[mask], touched[mask], flagged[mask]
        if n > 2 and t.std() > 0 and a.std() > 0:
            rho, p = spearmanr(a, t)
        else:
            rho, p = float("nan"), float("nan")
        n_touched = int(t.sum())
        n_flagged = int(f.sum())
        recall = float(f[t.astype(bool)].mean()) if n_touched else float("nan")
        precision = float(t[f.astype(bool)].mean()) if n_flagged else float("nan")
        rows.append(
            {
                "direction": label,
                "n": n,
                "n_touched": n_touched,
                "n_flagged": n_flagged,
                "score_vs_gt_rho": rho,
                "score_vs_gt_p": p,
                "recall": recall,
                "precision": precision,
            }
        )
    return rows


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

    score_by_id = dict(zip(ws["window_id"], ws["anomaly_score"]))
    usable_by_id = dict(zip(ws["window_id"], ws["usable"]))
    zone_by_id = dict(zip(ws["window_id"], ws["zone"]))
    missing = [int(w) for w in test_window_ids if w not in score_by_id]
    if missing:
        raise ValueError(f"{len(missing)} test_window_ids not found in window_scores.parquet (e.g. {missing[:5]})")

    anomaly_score = np.array([score_by_id[w] for w in test_window_ids])
    usable = np.array([bool(usable_by_id[w]) for w in test_window_ids])
    flagged = np.array([zone_by_id[w] != "green" for w in test_window_ids])

    volatility, magnitude, drift, level = compute_reference_metrics(test_windows)

    gt = load_window_ground_truth(args.data, args.event_labels, window_size=args.window_rows)
    gt = gt.filter_to_window_ids(test_window_ids)
    touched = gt.is_anomaly.astype(float)

    mask = usable
    logger.info("Using %d / %d usable test windows", mask.sum(), len(mask))

    a, v, m, d, lv, t, f = (
        anomaly_score[mask],
        volatility[mask],
        magnitude[mask],
        drift[mask],
        level[mask],
        touched[mask],
        flagged[mask],
    )

    overall = [
        spearman_report("anomaly_score", a, "drift(signed,within-window)", d),
        spearman_report("ground_truth(touched)", t, "drift(signed,within-window)", d),
        spearman_report("anomaly_score", a, "level(signed,vs baseline)", lv),
        spearman_report("ground_truth(touched)", t, "level(signed,vs baseline)", lv),
    ]

    print()
    print("=" * 92)
    print(f"DIRECTIONAL ALIGNMENT CHECK -- {args.output_dir}")
    print("=" * 92)
    print(f"{'a':24s} vs {'b':32s} {'rho':>8s} {'p':>10s} {'n':>6s}")
    print("-" * 92)
    for r in overall:
        print(f"{r['a']:24s} vs {r['b']:32s} {r['rho']:>8.3f} {r['p_value']:>10.2e} {r['n']:>6d}")

    for metric_name, metric_vals, labels in [
        ("drift (within-window trend)", d, ("rising (drift>0, 'spiking')", "falling (drift<0, 'crashing')")),
        ("level (vs. training baseline)", lv, ("above baseline (level>0, 'spiking')", "below baseline (level<0, 'crashed low')")),
    ]:
        print()
        print(f"-- split by {metric_name} --")
        print(f"n {labels[0].split(' (')[0]}: {int((metric_vals > 0).sum())}   n {labels[1].split(' (')[0]}: {int((metric_vals < 0).sum())}")
        print("-" * 92)
        print(f"{'direction':40s} {'n':>5s} {'n_touch':>8s} {'n_flag':>7s} {'score_vs_gt_rho':>16s} {'recall':>8s} {'precision':>10s}")
        print("-" * 92)
        for row in directional_breakdown(a, t, f, metric_vals, labels):
            if row["n"] == 0:
                print(f"{row['direction']:40s}     0  (no windows)")
                continue
            print(
                f"{row['direction']:40s} {row['n']:>5d} {row['n_touched']:>8d} {row['n_flagged']:>7d} "
                f"{row['score_vs_gt_rho']:>16.3f} {row['recall']:>8.3f} {row['precision']:>10.3f}"
            )
    print("=" * 92)
    print(
        "Read: 'drift' is the signed mean step-to-step change WITHIN a window\n"
        "(instantaneous trend -- tends to average toward zero over 120 rows unless\n"
        "a sharp ramp happens inside one window). 'level' is the signed mean value\n"
        "of the whole window relative to the RobustScaler training median (0) --\n"
        "positive = window sits above normal operation, negative = window sits\n"
        "below it. If recall/precision/score-vs-GT correlation is much weaker in\n"
        "the 'below baseline' / 'falling' group, that's quantitative confirmation\n"
        "that agreement concentrates on upward excursions and breaks down on\n"
        "downward ones."
    )


if __name__ == "__main__":
    main()
