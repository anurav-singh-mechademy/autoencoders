#!/usr/bin/env python
"""CLI: Run inference on a single window or batch of windows.

Usage:
    python scripts/infer.py --model model_artefacts/ --window window.npy
    python scripts/infer.py --model model_artefacts/ --window data.csv --sensor-columns "temp,speed,power"
"""

import argparse
import json
import logging

import numpy as np
import yaml

from autoencoder.artefacts.serialisation import load_artefacts
from autoencoder.inference.pipeline import infer_window
from autoencoder.alerting.zones import classify_zone
from autoencoder.alerting.persistence import apply_persistence

from autoencoder.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run autoencoder inference.")
    parser.add_argument("--model", required=True, help="Path to model artefacts directory.")
    parser.add_argument("--window", required=True, help="Path to window data (.npy or .csv).")
    parser.add_argument("--sensor-columns", default=None, help="Comma-separated sensor column names (for CSV).")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top contributing sensors to show.")
    parser.add_argument("--zone-history", default=None, help="Comma-separated previous zones for persistence (e.g., 'green,yellow,red').")
    parser.add_argument("--config", default=None, help="YAML config file -- reads inference.missing_data thresholds if given, else uses infer_window's built-in defaults.")
    args = parser.parse_args()

    missing_data_cfg = {}
    if args.config:
        with open(args.config) as f:
            missing_data_cfg = (yaml.safe_load(f).get("inference", {}) or {}).get("missing_data", {}) or {}

    # Load model artefacts
    logger.info("Loading model from %s", args.model)
    model, scaler, thresholds, metadata, training_errors, sensor_baselines = load_artefacts(args.model)
    n_sensors = metadata["n_sensors"]
    logger.info("Model loaded: %d sensors, latent_dim=%d", n_sensors, metadata["latent_dim"])

    # Load window
    window_path = args.window
    if window_path.endswith(".npy"):
        window = np.load(window_path)
    elif window_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(window_path)
        if args.sensor_columns:
            cols = [c.strip() for c in args.sensor_columns.split(",")]
        else:
            cols = [c for c in df.columns if c != "timestamp"]
        window = df[cols].values
    else:
        logger.error("Unsupported file format: %s. Use .npy or .csv", window_path)
        return

    logger.info("Window shape: %s", window.shape)

    # Sensor names
    sensor_names = [f"sensor_{i}" for i in range(n_sensors)]

    # Run inference
    result = infer_window(
        window=window,
        model=model,
        scaler=scaler,
        sensor_names=sensor_names,
        top_k=args.top_k,
        max_null_pct=missing_data_cfg.get("max_null_pct_per_sensor", 5.0),
        max_consecutive_nulls=missing_data_cfg.get("max_consecutive_nulls_ffill", 3),
        max_null_dominant_sensor_pct=missing_data_cfg.get("max_null_dominant_sensor_pct", 30.0),
        tail_compression_scale=metadata.get("tail_compression_scale"),
    )

    if not result.usable:
        print(f"\nWindow NOT usable for inference.")
        print(f"Quality flags: {result.quality_flags}")
        return

    if result.masked_sensors:
        masked_names = [sensor_names[i] for i in result.masked_sensors]
        print(f"\n{len(result.masked_sensors)} sensor(s) excluded from scoring (null-dominant): {masked_names}")

    # Classify zone
    zone = classify_zone(result.window_score, thresholds)

    # Persistence (if history provided)
    persistence_result = None
    if args.zone_history:
        history = [z.strip() for z in args.zone_history.split(",")]
        history.append(zone)
        persistence_result = apply_persistence(history)

    # Output
    print(f"\n{'='*50}")
    print(f"  Inference Result")
    print(f"{'='*50}")
    print(f"  Window score:  {result.window_score:.6f}")
    print(f"  Zone:          {zone.upper()}")
    print(f"  Green/Yellow:  {thresholds['green_yellow']:.6f}")
    print(f"  Yellow/Red:    {thresholds['yellow_red']:.6f}")

    if persistence_result:
        print(f"\n  Persistence alert: {persistence_result['alert_level'].upper()}")
        print(f"  Reason: {persistence_result['reason']}")

    print(f"\n  Top Contributing Sensors:")
    for c in result.top_contributors:
        print(f"    {c['name']:20s}  error={c['error']:.6f}  ({c['contribution_pct']:.1f}%)")

    if result.quality_flags:
        print(f"\n  Quality flags: {result.quality_flags}")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
