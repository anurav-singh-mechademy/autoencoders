#!/usr/bin/env python
"""CLI: Train the autoencoder model.

Usage:
    python scripts/train.py --data cleaned_windows.npy --output model_artefacts/ --n-sensors 20
    python scripts/train.py --data cleaned/ --output model/ --config configs/default.yaml
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import yaml

from autoencoder.model.architecture import get_latent_dim
from autoencoder.model.loss import per_row_mse, window_anomaly_score
from autoencoder.training.dataset import split_windows, windows_to_numpy
from autoencoder.training.trainer import TrainConfig, train_model
from autoencoder.alerting.thresholds import compute_thresholds
from autoencoder.artefacts.serialisation import save_artefacts

import torch

from autoencoder.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train the autoencoder model.")
    parser.add_argument("--data", required=True, help="Path to cleaned windows (.npy file).")
    parser.add_argument("--output", required=True, help="Output directory for model artefacts.")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--n-sensors", type=int, default=None, help="Number of sensors (auto-detected if omitted).")
    parser.add_argument("--scaler", default=None, help="Path to fitted scaler (.pkl).")
    parser.add_argument(
        "--regime-labels", default=None,
        help="Path to cleaned_regime_labels.npy (auto-detected next to --data if omitted).",
    )
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    thresh_cfg = config.get("thresholds", {})

    # Load data
    logger.info("Loading windows from %s", args.data)
    data = np.load(args.data)
    n_sensors = args.n_sensors or data.shape[2]
    logger.info("Loaded %d windows, %d sensors", len(data), n_sensors)

    # Load regime labels so the split is stratified per regime -- cleaned_windows.npy
    # is ordered by regime (all-Low, then all-Medium, ...), so splitting without
    # these labels would put validation/test entirely inside whichever regime(s)
    # land at the tail of the array.
    regime_labels_path = args.regime_labels
    if regime_labels_path is None:
        candidate = Path(args.data).parent / "cleaned_regime_labels.npy"
        if candidate.exists():
            regime_labels_path = candidate
    regime_labels = None
    if regime_labels_path:
        regime_labels = np.load(regime_labels_path)
        logger.info("Loaded regime labels from %s", regime_labels_path)
    else:
        logger.warning(
            "No regime labels found -- splitting windows without regime "
            "stratification. If %s is regime-ordered, validation/test may be "
            "biased toward a single regime.", args.data,
        )

    # Split
    windows_list = [data[i] for i in range(len(data))]
    train_pct = train_cfg.get("splits", {}).get("train", 0.85)
    val_pct = train_cfg.get("splits", {}).get("val", 0.10)
    train_w, val_w, test_w = split_windows(
        windows_list, train_pct=train_pct, val_pct=val_pct, regime_labels=regime_labels,
    )
    logger.info("Split: %d train, %d val, %d test", len(train_w), len(val_w), len(test_w))

    train_arr = windows_to_numpy(train_w)
    val_arr = windows_to_numpy(val_w)
    test_arr = windows_to_numpy(test_w) if test_w else None

    # Build config
    latent_dim = get_latent_dim(n_sensors)
    tc = TrainConfig(
        n_sensors=n_sensors,
        latent_dim=latent_dim,
        dropout=config.get("model", {}).get("dropout", 0.2),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        max_epochs=train_cfg.get("max_epochs", 150),
        patience=train_cfg.get("early_stopping_patience", 15),
        lr_factor=train_cfg.get("lr_scheduler", {}).get("factor", 0.5),
        lr_patience=train_cfg.get("lr_scheduler", {}).get("patience", 10),
    )

    # Train
    logger.info("Training: latent_dim=%d, lr=%.4f, max_epochs=%d", latent_dim, tc.lr, tc.max_epochs)
    model, history = train_model(train_arr, val_arr, tc)

    # Compute calibration errors for thresholds (train or val, see main.step_train
    # for why val is preferred -- train-error calibration is optimistic).
    device = torch.device("cpu")
    calibration_source = thresh_cfg.get("calibration_source", "train")
    calibration_arr = val_arr if calibration_source == "val" else train_arr
    model.eval()
    training_errors = []
    with torch.no_grad():
        for i in range(len(calibration_arr)):
            x = torch.tensor(calibration_arr[i], dtype=torch.float32, device=device)
            x_hat = model(x)
            row_errs = per_row_mse(x, x_hat)
            score = window_anomaly_score(row_errs)
            training_errors.append(score)

    training_errors = np.array(training_errors)

    # Compute thresholds
    thresholds = compute_thresholds(
        training_errors,
        green_yellow_percentile=thresh_cfg.get("green_yellow", 90),
        yellow_red_percentile=thresh_cfg.get("yellow_red", 99),
        method=thresh_cfg.get("method", "robust"),
        spread=thresh_cfg.get("spread", "mad"),
    )
    logger.info("Thresholds: green/yellow=%.4f, yellow/red=%.4f", thresholds["green_yellow"], thresholds["yellow_red"])

    # Metadata
    metadata = {
        "n_sensors": n_sensors,
        "latent_dim": latent_dim,
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "n_train_windows": len(train_w),
        "n_val_windows": len(val_w),
        "n_test_windows": len(test_w) if test_w else 0,
    }

    # Load scaler if provided
    scaler = None
    if args.scaler:
        import joblib
        scaler = joblib.load(args.scaler)
        logger.info("Loaded scaler from %s", args.scaler)

    # Save artefacts
    output_dir = Path(args.output)
    save_artefacts(
        output_dir=output_dir,
        model=model,
        scaler=scaler,
        thresholds=thresholds,
        metadata=metadata,
        training_errors=training_errors,
    )
    logger.info("All artefacts saved to %s", output_dir)

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
