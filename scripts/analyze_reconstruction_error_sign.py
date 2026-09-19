#!/usr/bin/env python
"""CLI: Retain the SIGN of the reconstruction error per test window
(discarded elsewhere by squaring, see src/autoencoder/model/loss.py) as
three variants -- signed_error_all (mean over all sensors/rows),
signed_error_top (mean over top-k implicated sensors, all rows), and
signed_error_row_tied (top-k sensors, at the single row closest to the
window's own 95th-percentile row error -- the same row window_anomaly_score()
is built from) -- then report precision/PR-AUC/recall split by sign for
each. --save persists the augmented window_scores.parquet.

Usage:
    python scripts/analyze_reconstruction_error_sign.py \\
        --output-dir output_5P921A_regime_fix \\
        --data data/5P921A/5P921A_combined_with_events.parquet \\
        --event-labels data/5P921A/5P921A_event_labels_long.parquet \\
        --save output_5P921A_regime_fix/evaluation/window_scores_signed.parquet
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from autoencoder.logging_config import setup_logging
from autoencoder.artefacts.serialisation import load_artefacts
from autoencoder.evaluation.ground_truth import load_window_ground_truth
from autoencoder.evaluation.slug_metrics import pointwise_metrics

setup_logging()
logger = logging.getLogger(__name__)


def compute_signed_errors(
    x: torch.Tensor, x_hat: torch.Tensor, ws: pd.DataFrame, sensor_index: dict[str, int], top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """(signed_error_all, signed_error_top), one value per window."""
    r = (x - x_hat).numpy()  # (n, rows, sensors)
    signed_error_all = r.mean(axis=(1, 2))

    signed_error_top = np.full(len(ws), np.nan)
    for i, top_names in enumerate(ws["top_sensors"]):
        idxs = [sensor_index[n] for n in list(top_names)[:top_k] if n in sensor_index]
        if idxs:
            signed_error_top[i] = float(r[i][:, idxs].mean())
    return signed_error_all, signed_error_top


def compute_row_tied_modality(
    x: torch.Tensor, x_hat: torch.Tensor, ws: pd.DataFrame, sensor_index: dict[str, int], top_k: int
) -> np.ndarray:
    """Signed residual at row* (closest to the window's 95th-percentile row
    error, i.e. the row window_anomaly_score() is built from), averaged
    over the top-k implicated sensors."""
    r = (x - x_hat).numpy()                    # (n, rows, sensors), signed
    row_sq_error = (r ** 2).mean(axis=2)         # (n, rows) -- mirrors per_row_mse (all sensors)

    modality = np.full(len(ws), np.nan)
    for i, top_names in enumerate(ws["top_sensors"]):
        idxs = [sensor_index[n] for n in list(top_names)[:top_k] if n in sensor_index]
        if not idxs:
            continue
        target = np.percentile(row_sq_error[i], 95)
        row_idx = int(np.argmin(np.abs(row_sq_error[i] - target)))
        modality[i] = float(r[i, row_idx, idxs].mean())
    return modality


def report_group(label: str, mask: np.ndarray, touched: np.ndarray) -> None:
    n = int(mask.sum())
    if n == 0:
        print(f"{label:34s} {0:>10d}      --     --         --       --")
        return
    n_tp = int(touched[mask].sum())
    n_fp = n - n_tp
    precision = n_tp / n
    print(f"{label:34s} {n:>10d} {n_tp:>6d} {n_fp:>6d} {precision:>10.3f} {1 - precision:>8.3f}")


def report_modality_metrics(label: str, modality_mask: np.ndarray, touched: np.ndarray, zones: np.ndarray, scores: np.ndarray) -> None:
    """pointwise_metrics computed over one modality's full window population
    (flagged and unflagged), so recall/PR-AUC have a valid denominator."""
    m = pointwise_metrics(touched, zones, scores, usable=modality_mask)
    n_flagged = m["tp"] + m["fp"]
    print(
        f"{label:34s} {n_flagged:>10d} {m['tp']:>6d} {m['fp']:>6d} "
        f"{m['pr_auc']:>8.3f} {m['precision']:>10.3f} {m['recall']:>8.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--event-labels", required=True)
    parser.add_argument("--window-rows", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=3, help="How many top-implicated sensors signed_error_top averages over.")
    parser.add_argument("--save", default=None, help="Optional path to write window_scores.parquet augmented with signed_error_all / signed_error_top.")
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

    signed_error_all, signed_error_top = compute_signed_errors(x, x_hat, ws, sensor_index, args.top_k)
    signed_error_row_tied = compute_row_tied_modality(x, x_hat, ws, sensor_index, args.top_k)
    ws["reconstruction_error_signed_all"] = signed_error_all
    ws["reconstruction_error_signed_top"] = signed_error_top
    ws["reconstruction_error_signed_row_tied"] = signed_error_row_tied

    usable = ws["usable"].to_numpy()
    flagged = usable & (ws["zone"] != "green").to_numpy()

    print()
    print("=" * 92)
    print(f"RECONSTRUCTION ERROR SIGN -- {args.output_dir}")
    print("=" * 92)

    for metric_name, metric in [
        ("signed_error_all (mean over ALL sensors)", signed_error_all),
        (f"signed_error_top (mean over top-{args.top_k} implicated sensors, ALL rows)", signed_error_top),
        (f"signed_error_row_tied (top-{args.top_k} sensors, AT the score-driving row)", signed_error_row_tied),
    ]:
        m_usable = usable & ~np.isnan(metric)
        rho, p = spearmanr(metric[m_usable], touched[m_usable]) if m_usable.sum() > 2 else (float("nan"), float("nan"))
        print(f"\n-- {metric_name} --")
        print(f"Spearman(metric, ground_truth) over all usable windows: rho={rho:.3f} (p={p:.2e}, n={int(m_usable.sum())})")

        m_flagged = flagged & ~np.isnan(metric)
        pos_flagged = m_flagged & (metric > 0)
        neg_flagged = m_flagged & (metric < 0)
        print(f"\nAmong FLAGGED (yellow/red) windows only:")
        print(f"{'group':34s} {'n_flagged':>10s} {'n_TP':>6s} {'n_FP':>6s} {'precision':>10s} {'FP_rate':>8s}")
        print("-" * 92)
        report_group("positive error (actual > recon)", pos_flagged, touched)
        report_group("negative error (actual < recon)", neg_flagged, touched)
        report_group("all flagged (valid metric)", m_flagged, touched)

        pos_modality = usable & ~np.isnan(metric) & (metric > 0)
        neg_modality = usable & ~np.isnan(metric) & (metric < 0)
        zones = ws["zone"].to_numpy()
        scores = ws["anomaly_score"].to_numpy()
        print(f"\nBy modality (each modality's own full window population -- flagged AND unflagged):")
        print(f"{'modality':34s} {'n_flagged':>10s} {'n_TP':>6s} {'n_FP':>6s} {'pr_auc':>8s} {'precision':>10s} {'recall':>8s}")
        print("-" * 92)
        report_modality_metrics("positive error (actual > recon)", pos_modality, touched, zones, scores)
        report_modality_metrics("negative error (actual < recon)", neg_modality, touched, zones, scores)

    if args.save:
        ws.to_parquet(args.save)
        logger.info("Saved augmented window scores (with signed error columns) to %s", args.save)

    print("=" * 92)


if __name__ == "__main__":
    main()
