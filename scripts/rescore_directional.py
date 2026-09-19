#!/usr/bin/env python
"""CLI: Recompute the window anomaly score with an asymmetric (directional)
weight on the reconstruction residual's sign, without retraining -- reuses
the trained model on the existing test_windows.npy. Sweeps alpha in
`weight = 1 + alpha * dir_sign * sign(r)` (dir_sign = +1 for
--direction=high, -1 for low; alpha=0 reproduces the original symmetric
score) and reports Spearman(score, ground_truth) overall and split by
signal level (above/below the training baseline).

Usage:
    python scripts/rescore_directional.py \\
        --output-dir output_5P921A_regime_fix \\
        --data data/5P921A/5P921A_combined_with_events.parquet \\
        --event-labels data/5P921A/5P921A_event_labels_long.parquet \\
        --alphas 0,0.25,0.5,0.75 \\
        --direction both
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from autoencoder.logging_config import setup_logging
from autoencoder.artefacts.serialisation import load_artefacts
from autoencoder.evaluation.ground_truth import load_window_ground_truth

setup_logging()
logger = logging.getLogger(__name__)


def directional_window_scores(
    x: torch.Tensor, x_hat: torch.Tensor, alpha: float, dir_sign: int
) -> np.ndarray:
    """Per-window directional score: P95 of per-row error, with an asymmetric weight on the signed residual."""
    r = (x - x_hat).numpy()
    weight = 1.0 + alpha * dir_sign * np.sign(r)
    weighted_sq = weight * (r ** 2)
    per_row = weighted_sq.mean(axis=2)          # (n_windows, rows) -- mean over sensors
    return np.percentile(per_row, 95, axis=1)   # (n_windows,) -- P95 over rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--event-labels", required=True)
    parser.add_argument("--window-rows", type=int, default=120)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75", help="Comma-separated alpha values to sweep (0 = original symmetric score).")
    parser.add_argument("--direction", choices=["high", "low", "both"], default="both",
                         help="'high' upweights actual>reconstruction (spiking) errors, 'low' upweights actual<reconstruction (crashing) errors.")
    args = parser.parse_args()

    model, scaler, thresholds, metadata, training_errors, sensor_baselines = load_artefacts(f"{args.output_dir}/artefacts")

    test_windows = np.load(f"{args.output_dir}/test_windows.npy")
    test_window_ids = np.load(f"{args.output_dir}/test_window_ids.npy")
    ws = pd.read_parquet(f"{args.output_dir}/evaluation/window_scores.parquet")

    score_by_id = dict(zip(ws["window_id"], ws["anomaly_score"]))
    usable_by_id = dict(zip(ws["window_id"], ws["usable"]))
    missing = [int(w) for w in test_window_ids if w not in score_by_id]
    if missing:
        raise ValueError(f"{len(missing)} test_window_ids not found in window_scores.parquet (e.g. {missing[:5]})")
    original_score = np.array([score_by_id[w] for w in test_window_ids])
    usable = np.array([bool(usable_by_id[w]) for w in test_window_ids])

    gt = load_window_ground_truth(args.data, args.event_labels, window_size=args.window_rows)
    gt = gt.filter_to_window_ids(test_window_ids)
    touched = gt.is_anomaly.astype(float)

    level = test_windows.mean(axis=(1, 2))  # signed, vs. RobustScaler training median (0)

    x = torch.tensor(test_windows, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        n = x.shape[0]
        x_hat = torch.empty_like(x)
        batch = 512
        for i in range(0, n, batch):
            x_hat[i : i + batch] = model(x[i : i + batch].reshape(-1, x.shape[-1])).reshape(x[i : i + batch].shape)

    mask = usable
    logger.info("Using %d / %d usable test windows", mask.sum(), len(mask))

    alphas = [float(a) for a in args.alphas.split(",")]
    directions = ["high", "low"] if args.direction == "both" else [args.direction]

    print()
    print("=" * 100)
    print(f"DIRECTIONAL RESCORING -- {args.output_dir}")
    print("=" * 100)

    baseline_rho, baseline_p = spearmanr(original_score[mask], touched[mask])
    print(f"Original symmetric anomaly_score vs ground_truth: rho={baseline_rho:.3f} (p={baseline_p:.2e}, n={int(mask.sum())})")
    print()

    above = mask & (level > 0)
    below = mask & (level < 0)
    n_above_touched = int(touched[above].sum())
    n_below_touched = int(touched[below].sum())
    if n_above_touched > 1:
        above_rho_orig, _ = spearmanr(original_score[above], touched[above])
    else:
        above_rho_orig = float("nan")
    if n_below_touched > 1:
        below_rho_orig, _ = spearmanr(original_score[below], touched[below])
    else:
        below_rho_orig = float("nan")
    print(f"  original, above-baseline (n={int(above.sum())}, touched={n_above_touched}): rho={above_rho_orig:.3f}")
    print(f"  original, below-baseline (n={int(below.sum())}, touched={n_below_touched}): rho={below_rho_orig:.3f}")
    print()
    print(f"{'direction':10s} {'alpha':>6s} {'overall_rho':>12s} {'above_rho':>10s} {'below_rho':>10s}")
    print("-" * 100)

    for direction in directions:
        dir_sign = 1 if direction == "high" else -1
        for alpha in alphas:
            scores = directional_window_scores(x, x_hat, alpha, dir_sign)
            overall_rho, _ = spearmanr(scores[mask], touched[mask])
            above_rho = spearmanr(scores[above], touched[above])[0] if n_above_touched > 1 else float("nan")
            below_rho = spearmanr(scores[below], touched[below])[0] if n_below_touched > 1 else float("nan")
            print(f"{direction:10s} {alpha:>6.2f} {overall_rho:>12.3f} {above_rho:>10.3f} {below_rho:>10.3f}")

    print("=" * 100)
    print(
        "Read: alpha=0 must match the 'Original' rho line above (sanity check that\n"
        "this reduces to the real pipeline's score when the weighting is off).\n"
        "direction=high upweights actual>reconstruction (spiking) residuals;\n"
        "direction=low upweights actual<reconstruction (crashing) residuals. If a\n"
        "direction+alpha improves the *_rho column that was weak in the original\n"
        "breakdown without hurting the other, that's evidence the fault signal is\n"
        "concentrated in one residual sign and a directional loss could help."
    )


if __name__ == "__main__":
    main()
