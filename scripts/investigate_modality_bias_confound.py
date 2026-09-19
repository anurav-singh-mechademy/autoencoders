#!/usr/bin/env python
"""Compares the row-tied reconstruction-error modality split (see
analyze_reconstruction_error_sign.py) between validation (known-normal) and
test windows, per equipment -- a similarly skewed split on validation data
would mean the modality signal is largely a baked-in model bias rather than
anomaly-driven. Recomputes top-k sensor ranking directly per window since
there's no window_scores.parquet for the validation split.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from autoencoder.artefacts.serialisation import load_artefacts

EQUIPMENT = ["5P921A", "5ST901A", "5ST901B", "5K512B"]
TOP_K = 3


def per_sensor_mse(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """(n, sensors) -- mean squared error per sensor, over rows, per window."""
    return ((x - x_hat) ** 2).mean(axis=1)


def row_mse(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """(n, rows) -- mean squared error per row, over sensors, per window."""
    return ((x - x_hat) ** 2).mean(axis=2)


def compute_modality(windows: np.ndarray, model, top_k: int = TOP_K) -> np.ndarray:
    """Row-tied, top-k-sensor signed modality per window (mirrors
    compute_row_tied_modality, but derives its own top-k ranking per window)."""
    x = torch.tensor(windows, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        x_hat = model(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    r = (x - x_hat).numpy()
    x_np = x.numpy()
    x_hat_np = x_hat.numpy()

    sensor_mse = per_sensor_mse(x_np, x_hat_np)          # (n, sensors)
    row_sq_error = row_mse(x_np, x_hat_np)                # (n, rows)

    n = windows.shape[0]
    modality = np.full(n, np.nan)
    for i in range(n):
        top_idxs = np.argsort(-sensor_mse[i])[:top_k]
        target = np.percentile(row_sq_error[i], 95)
        row_idx = int(np.argmin(np.abs(row_sq_error[i] - target)))
        modality[i] = float(r[i, row_idx, top_idxs].mean())
    return modality


def split_rate(modality: np.ndarray) -> tuple[int, float, float]:
    valid = modality[~np.isnan(modality)]
    n = len(valid)
    pos = (valid > 0).sum()
    neg = (valid < 0).sum()
    return n, pos / n if n else float("nan"), neg / n if n else float("nan")


def main():
    rows = []
    for eq in EQUIPMENT:
        out_dir = f"output_{eq}_regime_fix"
        model, scaler, thresholds, metadata, training_errors, sensor_baselines = load_artefacts(f"{out_dir}/artefacts")

        val_windows = np.load(f"{out_dir}/val_windows.npy")
        test_windows = np.load(f"{out_dir}/test_windows.npy")

        val_modality = compute_modality(val_windows, model)
        test_modality = compute_modality(test_windows, model)

        n_val, val_pos, val_neg = split_rate(val_modality)
        n_test, test_pos, test_neg = split_rate(test_modality)

        rows.append({
            "equipment": eq,
            "n_val": n_val,
            "val_pos_pct": 100 * val_pos,
            "val_neg_pct": 100 * val_neg,
            "n_test": n_test,
            "test_pos_pct": 100 * test_pos,
            "test_neg_pct": 100 * test_neg,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:6.1f}")
    print(df.to_string(index=False))
    df.to_csv("scripts/_modality_bias_confound_results.csv", index=False)


if __name__ == "__main__":
    main()
