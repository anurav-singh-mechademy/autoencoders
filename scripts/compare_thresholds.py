#!/usr/bin/env python
"""Compare Green/Yellow/Red zone breakdowns across threshold calibration methods.

Recomputes thresholds from a trained model's saved calibration error
distribution under each method/spread combination, then reclassifies an
existing set of window scores (from `scripts/evaluate_detection.py`'s
window_results.csv) under each -- no retraining or re-inference needed,
since thresholding and zone classification are pure functions of already-
computed scores.

Usage:
    python scripts/compare_thresholds.py \\
        --model-dir output/artefacts \\
        --window-results output/evaluation/window_results.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from autoencoder.alerting.thresholds import compute_thresholds
from autoencoder.alerting.zones import classify_batch

METHODS = [
    ("percentile", None),
    ("robust", "mad"),
    ("robust", "iqr"),
]


def main():
    parser = argparse.ArgumentParser(description="Compare zone breakdowns across threshold methods.")
    parser.add_argument("--model-dir", required=True, help="Artefact dir with training_error_distribution.npy")
    parser.add_argument("--window-results", required=True, help="window_results.csv from evaluate_detection.py")
    parser.add_argument("--green-yellow", type=float, default=90, help="Green/Yellow percentile")
    parser.add_argument("--yellow-red", type=float, default=99, help="Yellow/Red percentile")
    args = parser.parse_args()

    calibration_errors = np.load(f"{args.model_dir}/training_error_distribution.npy")
    df = pd.read_csv(args.window_results)
    usable = df[df["usable"]]
    scores = usable["score"].values
    print(f"Calibration errors: n={len(calibration_errors)}")
    print(f"Scored windows: n={len(scores)} (usable of {len(df)} total)\n")

    rows = []
    for method, spread in METHODS:
        kwargs = {"method": method}
        if spread is not None:
            kwargs["spread"] = spread
        t = compute_thresholds(
            calibration_errors,
            green_yellow_percentile=args.green_yellow,
            yellow_red_percentile=args.yellow_red,
            **kwargs,
        )
        zones = classify_batch(scores, t)
        counts = {z: zones.count(z) for z in ("green", "yellow", "red")}
        n = len(zones)
        label = f"{method}" + (f"-{spread}" if spread else "")
        rows.append({
            "method": label,
            "green_yellow_threshold": t["green_yellow"],
            "yellow_red_threshold": t["yellow_red"],
            "green": counts["green"],
            "green_pct": counts["green"] / n * 100,
            "yellow": counts["yellow"],
            "yellow_pct": counts["yellow"] / n * 100,
            "red": counts["red"],
            "red_pct": counts["red"] / n * 100,
        })

    report = pd.DataFrame(rows).set_index("method")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(report[["green_yellow_threshold", "yellow_red_threshold"]])
    print()
    print(report[["green", "green_pct", "yellow", "yellow_pct", "red", "red_pct"]].round(1))


if __name__ == "__main__":
    main()
