#!/usr/bin/env python
"""Compares the existing row-tied modality (unweighted mean sign) against
two alternative constructions -- A: error-weighted signed average across
top-k sensors at row*; B: majority-of-signs consensus (positive/negative/
mixed) -- reporting PR-AUC/precision/recall and the positive-vs-negative
gap for each, per equipment.

Usage:
    python scripts/investigate_modality_constructions.py
"""

from __future__ import annotations

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

EQUIPMENT = {
    "5P921A": ("data/5P921A/5P921A_combined_with_events.parquet", "data/5P921A/5P921A_event_labels_long.parquet"),
    "5ST901A": ("data/5ST901A/5ST901A_combined_with_events.parquet", "data/5ST901A/5ST901A_event_labels_long.parquet"),
    "5ST901B": ("data/5ST901B/5ST901B_combined_with_events.parquet", "data/5ST901B/5ST901B_event_labels_long.parquet"),
    "5K512B": ("data/5K512B/5K512B_combined_with_events.parquet", "data/5K512B/5K512B_event_labels_long.parquet"),
}
TOP_K = 3
WINDOW_ROWS = 120


def row_star_indices(r: np.ndarray) -> np.ndarray:
    """row* per window: closest to the 95th-percentile row error (same as compute_row_tied_modality)."""
    row_sq_error = (r ** 2).mean(axis=2)  # (n, rows)
    target = np.percentile(row_sq_error, 95, axis=1)  # (n,)
    return np.argmin(np.abs(row_sq_error - target[:, None]), axis=1)  # (n,)


def compute_all_modalities(r: np.ndarray, ws: pd.DataFrame, sensor_index: dict[str, int], top_k: int):
    """(baseline, weighted_A, consensus_B) arrays, one value per window."""
    n = len(ws)
    row_idx = row_star_indices(r)

    baseline = np.full(n, np.nan)
    weighted_a = np.full(n, np.nan)
    consensus_b = np.array(["nan"] * n, dtype=object)

    for i, top_names in enumerate(ws["top_sensors"]):
        idxs = [sensor_index[nm] for nm in list(top_names)[:top_k] if nm in sensor_index]
        if not idxs:
            continue
        resid = r[i, row_idx[i], idxs]  # signed residuals of the top-k sensors at row*

        baseline[i] = float(resid.mean())

        sq = resid ** 2
        if sq.sum() > 0:
            weighted_a[i] = float((sq * resid).sum() / sq.sum())
        else:
            weighted_a[i] = 0.0

        n_pos = int((resid > 0).sum())
        n_neg = int((resid < 0).sum())
        k = len(resid)
        if n_pos > k / 2:
            consensus_b[i] = "positive"
        elif n_neg > k / 2:
            consensus_b[i] = "negative"
        else:
            consensus_b[i] = "mixed"

    return baseline, weighted_a, consensus_b


def metrics_row(group_mask: np.ndarray, touched: np.ndarray, zones: np.ndarray, scores: np.ndarray) -> dict:
    n = int(group_mask.sum())
    if n == 0:
        return {"n": 0, "pr_auc": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "f1": float("nan"),
                "n_windows_active": 0, "n_windows_total": 0}
    m = pointwise_metrics(touched, zones, scores, usable=group_mask)
    m["n"] = n
    return m


def gap(pos: dict, neg: dict, key: str) -> float:
    a, b = pos.get(key), neg.get(key)
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return float("nan")
    return a - b


def run_equipment(eq: str, data_path: str, event_labels_path: str) -> dict:
    output_dir = f"output_{eq}_regime_fix"
    with open(f"{output_dir}/artefacts/training_metadata.json") as f:
        metadata = json.load(f)
    sensor_columns = metadata["sensor_columns"]
    sensor_index = {name: i for i, name in enumerate(sensor_columns)}

    model, scaler, thresholds, metadata2, training_errors, sensor_baselines = load_artefacts(f"{output_dir}/artefacts")

    test_windows = np.load(f"{output_dir}/test_windows.npy")
    test_window_ids = np.load(f"{output_dir}/test_window_ids.npy")
    ws = pd.read_parquet(f"{output_dir}/evaluation/window_scores.parquet")
    ws = ws.set_index("window_id").loc[test_window_ids].reset_index()

    gt = load_window_ground_truth(data_path, event_labels_path, window_size=WINDOW_ROWS)
    gt = gt.filter_to_window_ids(test_window_ids)
    touched = np.asarray(gt.is_anomaly).astype(bool)

    x = torch.tensor(test_windows, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        x_hat = model(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    r = (x - x_hat).numpy()

    baseline, weighted_a, consensus_b = compute_all_modalities(r, ws, sensor_index, TOP_K)

    usable = ws["usable"].to_numpy()
    zones = ws["zone"].to_numpy()
    scores = ws["anomaly_score"].to_numpy()

    results = {}

    base_valid = usable & ~np.isnan(baseline)
    results["baseline_positive"] = metrics_row(base_valid & (baseline > 0), touched, zones, scores)
    results["baseline_negative"] = metrics_row(base_valid & (baseline < 0), touched, zones, scores)

    a_valid = usable & ~np.isnan(weighted_a)
    results["A_positive"] = metrics_row(a_valid & (weighted_a > 0), touched, zones, scores)
    results["A_negative"] = metrics_row(a_valid & (weighted_a < 0), touched, zones, scores)

    b_valid = usable & (consensus_b != "nan")
    results["B_positive"] = metrics_row(b_valid & (consensus_b == "positive"), touched, zones, scores)
    results["B_negative"] = metrics_row(b_valid & (consensus_b == "negative"), touched, zones, scores)
    results["B_mixed"] = metrics_row(b_valid & (consensus_b == "mixed"), touched, zones, scores)

    return results


def main():
    all_results = {}
    for eq, (data_path, event_labels_path) in EQUIPMENT.items():
        print(f"\n{'=' * 100}\n{eq}\n{'=' * 100}")
        try:
            results = run_equipment(eq, data_path, event_labels_path)
        except Exception as exc:  # keep going across equipment
            logger.exception("Failed on %s: %s", eq, exc)
            continue
        all_results[eq] = results

        header = f"{'group':22s} {'n':>6s} {'pr_auc':>8s} {'precision':>10s} {'recall':>8s}"
        print(header)
        print("-" * len(header))
        for key, m in results.items():
            print(f"{key:22s} {m['n']:>6d} {m['pr_auc']:>8.3f} {m['precision']:>10.3f} {m['recall']:>8.3f}")

        print("\nPOS-vs-NEG gaps (positive minus negative):")
        for label, pos_k, neg_k in [
            ("baseline", "baseline_positive", "baseline_negative"),
            ("A (weighted)", "A_positive", "A_negative"),
            ("B (consensus)", "B_positive", "B_negative"),
        ]:
            pos, neg = results[pos_k], results[neg_k]
            g_pr = gap(pos, neg, "pr_auc")
            g_prec = gap(pos, neg, "precision")
            print(f"  {label:16s} pr_auc_gap={g_pr:+.3f}   precision_gap={g_prec:+.3f}")

    print(f"\n{'=' * 100}\nSUMMARY: pr_auc gap (positive - negative) by construction\n{'=' * 100}")
    print(f"{'equipment':10s} {'baseline':>10s} {'A_weighted':>12s} {'B_consensus':>12s}")
    for eq, results in all_results.items():
        g_base = gap(results["baseline_positive"], results["baseline_negative"], "pr_auc")
        g_a = gap(results["A_positive"], results["A_negative"], "pr_auc")
        g_b = gap(results["B_positive"], results["B_negative"], "pr_auc")
        print(f"{eq:10s} {g_base:>10.3f} {g_a:>12.3f} {g_b:>12.3f}")


if __name__ == "__main__":
    main()
