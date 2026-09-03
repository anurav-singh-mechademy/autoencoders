#!/usr/bin/env python
"""Generate FastSHAP explanations for windows and sanity-check them.

Loads a trained model + FastSHAP explainer from --model-dir, computes
per-sensor Shapley attribution for every window in --data, and verifies the
efficiency property every correctly-trained explainer must satisfy:

    sum_i shapley_i(x) ≈ v(full) - v(empty)

i.e. the attributed values must add up to exactly the gap between the
window's real anomaly score and the score of a fully-masked ("nothing
known") input. A large gap means the explainer is undertrained (more epochs
/ mask samples) rather than a config problem, since the projection in
autoencoder.explain.fastshap.normalize_efficiency enforces this exactly at
every forward pass -- so gaps close to zero are expected by construction and
a persistently large gap points at a real training issue.

This is a direct generate-and-inspect tool; for accuracy against known
injected fault sensors (ground truth), use scripts/evaluate_detection.py
with a model trained via `--explain`.

Usage:
    python scripts/explain.py \\
        --data output/cleaned_windows.npy \\
        --model-dir output/artefacts \\
        --output output/explanations
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from autoencoder.logging_config import setup_logging
from autoencoder.artefacts.serialisation import load_artefacts, load_explainer
from autoencoder.model.loss import per_row_mse
from autoencoder.explain.explain_window import explain_window_fastshap
from autoencoder.explain.value_function import empty_value

setup_logging()
logger = logging.getLogger(__name__)


def explain_windows(windows: np.ndarray, model, explainer, sensor_columns: list[str], top_k: int) -> list[dict]:
    """Compute per-window Shapley attribution + the efficiency-property gap."""
    device = torch.device("cpu")
    n_sensors = windows.shape[2]
    v_empty = empty_value(model, n_sensors, device)

    records = []
    for i in range(len(windows)):
        x = torch.tensor(windows[i], dtype=torch.float32, device=device)
        shapley = explain_window_fastshap(x, model, explainer, v_empty=v_empty)

        with torch.no_grad():
            v_full = per_row_mse(x, model(x)).mean().item()
        efficiency_gap = float(abs(shapley.sum() - (v_full - v_empty)))

        ranked = np.argsort(-np.abs(shapley))[:top_k]
        records.append({
            "window_id": i,
            "window_score_mean_row_mse": v_full,
            "efficiency_gap": efficiency_gap,
            "top_sensors": [
                {"name": sensor_columns[idx], "shapley_value": float(shapley[idx])}
                for idx in ranked
            ],
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate + sanity-check FastSHAP explanations.")
    parser.add_argument("--data", required=True, help="Cleaned/scaled windows (.npy, shape (n, window_size, n_sensors))")
    parser.add_argument("--model-dir", required=True, help="Trained model artefacts dir (must include explainer_weights.pt)")
    parser.add_argument("--output", required=True, help="Output directory for per-window explanations")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K sensors to report per window")
    args = parser.parse_args()

    model, _scaler, _thresholds, metadata, _training_errors, _sensor_baselines = load_artefacts(args.model_dir)
    explainer = load_explainer(args.model_dir)
    if explainer is None:
        raise SystemExit(
            f"No FastSHAP explainer found in {args.model_dir} -- "
            "train one first with `python main.py ... --explain` (see module docstring)."
        )

    sensor_columns = metadata.get("sensor_columns", [f"sensor_{i}" for i in range(metadata["n_sensors"])])
    windows = np.load(args.data)
    logger.info("Loaded %d windows, %d sensors", len(windows), windows.shape[2])

    records = explain_windows(windows, model, explainer, sensor_columns, args.top_k)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "explanations.json", "w") as f:
        json.dump(records, f, indent=2)

    max_gap = max(r["efficiency_gap"] for r in records)
    mean_gap = float(np.mean([r["efficiency_gap"] for r in records]))

    logger.info("Wrote %d explanations to %s", len(records), output_dir / "explanations.json")
    print(f"\nExplained {len(records)} windows.")
    print(f"Efficiency-property gap: mean={mean_gap:.6f}  max={max_gap:.6f}")
    print("(should be ~0 by construction -- a persistently large gap means the explainer needs more training)")


if __name__ == "__main__":
    main()
