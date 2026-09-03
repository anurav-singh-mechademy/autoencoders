#!/usr/bin/env python
"""CLI: Run the data cleaning pipeline and generate a report.

Usage:
    python scripts/run_cleaning.py --data sensor_data.csv --output cleaned_output/
    python scripts/run_cleaning.py --data data.parquet --output out/ --config configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from autoencoder.data.ingestion import load_data, detect_sensor_columns
from autoencoder.data.preprocessing import fit_robust_scaler, apply_scaling, construct_windows
from autoencoder.data.regime import compute_window_features, segment_regimes
from autoencoder.data.cleaning import run_cleaning_pipeline
from autoencoder.reporting.cleaning_report import generate_cleaning_report

from autoencoder.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the data cleaning pipeline.")
    parser.add_argument("--data", required=True, help="Path to sensor data file (CSV or parquet).")
    parser.add_argument("--output", required=True, help="Output directory for cleaned data and report.")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--equipment-id", default="equipment_01", help="Equipment identifier.")
    parser.add_argument("--regime-features", nargs="+", default=None, help="Columns for regime segmentation.")
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    cleaning_cfg = config.get("cleaning", {})
    regime_cfg = config.get("regime", {})

    # Load data
    logger.info("Loading data from %s", args.data)
    df = load_data(args.data)
    sensor_columns = detect_sensor_columns(df)
    logger.info("Found %d sensor columns, %d rows", len(sensor_columns), len(df))

    # Scale
    scaler_result = fit_robust_scaler(df, sensor_columns)
    scaled_df = apply_scaling(df, scaler_result)

    # Construct windows
    windows = construct_windows(scaled_df, sensor_columns=sensor_columns)
    logger.info("Constructed %d windows", len(windows))

    if len(windows) < 10:
        logger.error("Too few windows (%d) for cleaning. Need at least 10.", len(windows))
        sys.exit(1)

    # Regime segmentation
    regime_features = args.regime_features or regime_cfg.get("features", sensor_columns[:3])
    n_clusters = regime_cfg.get("n_clusters", 3)
    window_features = compute_window_features(windows, sensor_columns, regime_features)
    regime_result = segment_regimes(window_features, regime_features, n_clusters=n_clusters)
    logger.info("Segmented into %d regimes: %s", regime_result.n_clusters, regime_result.window_counts)

    # Run cleaning
    output_dir = Path(args.output)
    pipeline_result = run_cleaning_pipeline(
        windows=windows,
        regime_result=regime_result,
        sensor_columns=sensor_columns,
        output_dir=output_dir / "validation",
        if_contamination=cleaning_cfg.get("isolation_forest", {}).get("contamination", 0.05),
        pca_variance_threshold=cleaning_cfg.get("pca", {}).get("variance_threshold", 0.95),
        pca_outlier_std=cleaning_cfg.get("pca", {}).get("outlier_std", 3.0),
        md_chi2_percentile=cleaning_cfg.get("mahalanobis", {}).get("chi2_percentile", 97.5),
        steps=cleaning_cfg.get("order", ["isolation_forest", "pca", "mahalanobis"]),
    )

    logger.info(
        "Cleaning: %d -> %d windows (%.1f%% removed)",
        pipeline_result.original_count,
        pipeline_result.cleaned_count,
        (1 - pipeline_result.cleaned_count / pipeline_result.original_count) * 100,
    )

    # Generate report
    report_path = generate_cleaning_report(
        pipeline_result=pipeline_result,
        sensor_columns=sensor_columns,
        equipment_id=args.equipment_id,
        output_path=str(output_dir / "cleaning_report.html"),
    )
    logger.info("Report saved to %s", report_path)

    # Save cleaned windows and their regime labels (cleaned_windows.npy is
    # ordered by regime, not by time -- the labels let split_windows stratify
    # train/val/test across regimes instead of cutting along regime boundaries).
    import numpy as np
    cleaned_arr = np.array(pipeline_result.cleaned_windows)
    np.save(output_dir / "cleaned_windows.npy", cleaned_arr)
    np.save(output_dir / "cleaned_regime_labels.npy", np.array(pipeline_result.cleaned_regime_labels))
    logger.info("Saved %d cleaned windows to %s", len(pipeline_result.cleaned_windows), output_dir)


if __name__ == "__main__":
    main()
