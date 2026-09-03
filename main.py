#!/usr/bin/env python
"""End-to-end autoencoder pipeline: clean → train → infer → alert.

Usage:
    # Full pipeline (clean + train + infer on test windows)
    python main.py --data sensor_data.csv --output output/

    # With config
    python main.py --data data.parquet --output output/ --config configs/default.yaml

    # Skip cleaning (use pre-cleaned windows)
    python main.py --data cleaned_windows.npy --output output/ --skip-cleaning

    # Inference only (with existing model)
    python main.py --data new_data.csv --output output/ --model-dir output/artefacts/ --infer-only

Steps:
    1. Load & validate data
    2. Remove low-variance / constant columns
    3. Fit RobustScaler on training data
    4. Construct 30-min windows (120 rows each)
    5. Segment operating regimes (KMeans)
    6. Clean: Isolation Forest → PCA → Mahalanobis (per regime)
    7. Generate cleaning report
    8. Split: 85% train / 10% val / 5% test
    9. Train autoencoder with early stopping
    10. Compute thresholds (P90/P99) and sensor baselines
    11. Save all artefacts
    12. Run inference on test windows → classify zones → apply persistence
    13. Print summary
"""

import argparse
import json
import logging
import sys
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np
import torch
import yaml

from autoencoder.data.ingestion import load_data, detect_sensor_columns, validate_dataframe
from autoencoder.data.preprocessing import (
    remove_low_variance_columns,
    fit_robust_scaler,
    apply_scaling,
    construct_windows,
)
from autoencoder.data.regime import compute_window_features, segment_regimes, segment_regimes_unsupervised
from autoencoder.data.sensor_dropout import detect_sensor_dropout_windows, remove_sensor_dropout_windows
from autoencoder.data.cleaning import run_cleaning_pipeline
from autoencoder.reporting.cleaning_report import generate_cleaning_report

from autoencoder.model.architecture import get_latent_dim
from autoencoder.model.loss import per_row_mse, window_anomaly_score
from autoencoder.training.dataset import split_windows, windows_to_numpy
from autoencoder.training.trainer import TrainConfig, train_model, compute_sensor_baselines
from autoencoder.training.hyperparams import grid_search

from autoencoder.alerting.thresholds import compute_thresholds
from autoencoder.alerting.zones import classify_zone, classify_batch
from autoencoder.alerting.persistence import apply_persistence
from autoencoder.artefacts.serialisation import (
    save_artefacts, load_artefacts, save_explainer, load_explainer,
)

from autoencoder.explain.fastshap import FastSHAPConfig, train_fastshap_explainer

from autoencoder.inference.pipeline import infer_window

from autoencoder.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("main")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(config_path: str | None) -> dict:
    if config_path:
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def compute_training_errors(model, windows_arr, device) -> np.ndarray:
    """Compute window-level anomaly scores on a set of windows."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(len(windows_arr)):
            x = torch.tensor(windows_arr[i], dtype=torch.float32, device=device)
            x_hat = model(x)
            row_errs = per_row_mse(x, x_hat)
            scores.append(window_anomaly_score(row_errs))
    return np.array(scores)


def print_banner(text: str):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


# ── Step 1: Clean ────────────────────────────────────────────────────────────

def step_clean(data_path: str, output_dir: Path, config: dict, equipment_id: str):
    """Load data → preprocess → clean → save cleaned windows + report."""
    cleaning_cfg = config.get("cleaning", {})
    regime_cfg = config.get("regime", {})
    ts_col = config.get("data", {}).get("timestamp_column", "timestamp")

    print_banner("Step 1: Data Loading & Cleaning")

    # Load
    df = load_data(data_path, timestamp_column=ts_col)
    errors = validate_dataframe(df, timestamp_column=ts_col)
    if errors:
        logger.error("Validation errors: %s", errors)
        sys.exit(1)

    sensor_columns = detect_sensor_columns(df, timestamp_column=ts_col)
    logger.info("Loaded %d rows, %d sensor columns", len(df), len(sensor_columns))

    # Remove low-variance columns
    var_threshold = config.get("preprocessing", {}).get("variance_threshold", 1e-5)
    filter_result = remove_low_variance_columns(df, sensor_columns, variance_threshold=var_threshold)
    sensor_columns = filter_result.kept_columns
    if filter_result.removed_columns:
        logger.info("Removed low-variance columns: %s", filter_result.removed_columns)
    logger.info("Kept %d sensor columns after filtering", len(sensor_columns))

    # Scale -- fit ONLY on the train portion of the data to avoid leaking
    # val/test statistics into the scaler. Rows are still time-ordered here
    # (before windowing/cleaning), so a chronological head slice sized to the
    # configured train split approximates "train" at this stage. The final
    # train/val/test partition (used for model training) is recomputed later,
    # per regime, on the cleaned windows -- this early slice exists solely to
    # bound what the scaler is allowed to see.
    train_pct = config.get("training", {}).get("splits", {}).get("train", 0.85)
    n_train_rows = int(len(df) * train_pct)
    scaler_result = fit_robust_scaler(df.iloc[:n_train_rows], sensor_columns)
    scaled_df = apply_scaling(df, scaler_result)
    logger.info(
        "Fit RobustScaler on first %d / %d rows (train_pct=%.2f).",
        n_train_rows, len(df), train_pct,
    )

    # Save scaler for later use
    import joblib
    artefacts_dir = output_dir / "artefacts"
    artefacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_result.scaler, artefacts_dir / "scaler.pkl")

    # Windows
    windows = construct_windows(scaled_df, sensor_columns=sensor_columns)
    logger.info("Constructed %d windows", len(windows))

    if len(windows) < 10:
        logger.error("Too few windows (%d). Need at least 10.", len(windows))
        sys.exit(1)

    # Regime segmentation -- cluster on configured load-indicator features
    # (speed/power/discharge-pressure). If none of those are configured or
    # present in this equipment's sensor set, fall back to PCA on the full
    # sensor set rather than guessing arbitrary columns.
    configured_features = regime_cfg.get("features", [])
    regime_features = [f for f in configured_features if f in sensor_columns]

    n_clusters = regime_cfg.get("n_clusters", 3)
    if regime_features:
        window_features = compute_window_features(windows, sensor_columns, regime_features)
        regime_result = segment_regimes(window_features, regime_features, n_clusters=n_clusters)
    else:
        logger.warning(
            "No configured load-indicator features (%s) found in sensor columns -- "
            "falling back to PCA-based regime segmentation.",
            configured_features,
        )
        regime_result = segment_regimes_unsupervised(windows, n_clusters=n_clusters)
    logger.info("Regimes: %s", regime_result.window_counts)

    # Sensor dropout filter: exclude windows where >5% of sensors are null/stuck.
    # Must run after regime segmentation but before per-regime cleaning, so the
    # regime labels are filtered in lockstep with the windows they describe.
    dropout_cfg = config.get("preprocessing", {}).get("sensor_dropout", {})
    dropout_result = detect_sensor_dropout_windows(
        windows, threshold_pct=dropout_cfg.get("threshold_pct", 5.0),
    )
    if dropout_result.n_dropped:
        keep_mask = ~dropout_result.is_dropped
        windows = remove_sensor_dropout_windows(windows, dropout_result)
        filtered_labels = regime_result.labels[keep_mask]
        regime_result = dataclass_replace(
            regime_result,
            labels=filtered_labels,
            window_counts={
                regime_result.regime_names[i]: int(np.sum(filtered_labels == i))
                for i in range(regime_result.n_clusters)
            },
        )
        logger.info("Regimes after sensor dropout filter: %s", regime_result.window_counts)

    # Clean
    clean_dir = output_dir / "cleaning"
    clean_dir.mkdir(parents=True, exist_ok=True)

    pipeline_result = run_cleaning_pipeline(
        windows=windows,
        regime_result=regime_result,
        sensor_columns=sensor_columns,
        output_dir=clean_dir / "validation",
        if_contamination=cleaning_cfg.get("isolation_forest", {}).get("contamination", 0.05),
        pca_variance_threshold=cleaning_cfg.get("pca", {}).get("variance_threshold", 0.95),
        pca_outlier_std=cleaning_cfg.get("pca", {}).get("outlier_std", 3.0),
        md_chi2_percentile=cleaning_cfg.get("mahalanobis", {}).get("chi2_percentile", 97.5),
        steps=cleaning_cfg.get("order", ["isolation_forest", "pca", "mahalanobis"]),
    )

    removed_pct = (1 - pipeline_result.cleaned_count / pipeline_result.original_count) * 100
    logger.info(
        "Cleaning: %d → %d windows (%.1f%% removed)",
        pipeline_result.original_count, pipeline_result.cleaned_count, removed_pct,
    )

    # Report
    report_path = generate_cleaning_report(
        pipeline_result=pipeline_result,
        sensor_columns=sensor_columns,
        equipment_id=equipment_id,
        output_path=str(clean_dir / "cleaning_report.html"),
    )
    logger.info("Cleaning report: %s", report_path)

    # Save cleaned windows and their regime labels (the array is ordered by
    # regime, not by time -- step_train needs the labels to stratify the
    # train/val/test split across regimes instead of cutting along regime
    # boundaries).
    cleaned_arr = np.array(pipeline_result.cleaned_windows)
    regime_labels = np.array(pipeline_result.cleaned_regime_labels)
    np.save(output_dir / "cleaned_windows.npy", cleaned_arr)
    np.save(output_dir / "cleaned_regime_labels.npy", regime_labels)
    with open(output_dir / "sensor_columns.json", "w") as f:
        json.dump(sensor_columns, f)

    # Windows the cleaning pipeline rejected (isolation_forest/pca/mahalanobis
    # outliers) -- excluded from the autoencoder's own training set as designed,
    # but saved separately since they're the only windows with real sensor-level
    # deviation. Used by step_train to give the FastSHAP explainer (when enabled)
    # something to learn row-specific attribution from; see explain/fastshap.py.
    rejected_arr = np.array(pipeline_result.rejected_windows) if pipeline_result.rejected_windows else None
    if rejected_arr is not None:
        np.save(output_dir / "rejected_windows.npy", rejected_arr)
        logger.info("Saved %d cleaning-rejected windows for explainer training", len(rejected_arr))

    return cleaned_arr, sensor_columns, scaler_result, regime_labels, rejected_arr


# ── Step 2: Train ────────────────────────────────────────────────────────────

def step_train(cleaned_arr: np.ndarray, sensor_columns: list[str],
               scaler, output_dir: Path, config: dict, regime_labels=None, rejected_arr=None):
    """Split → train → compute thresholds + baselines → save artefacts."""
    train_cfg = config.get("training", {})
    thresh_cfg = config.get("thresholds", {})

    print_banner("Step 2: Model Training")

    n_sensors = cleaned_arr.shape[2]
    logger.info("Training on %d windows, %d sensors", len(cleaned_arr), n_sensors)

    if regime_labels is None:
        logger.warning(
            "No regime labels available -- splitting windows without regime "
            "stratification. If cleaned_arr is regime-ordered, validation/test "
            "may be biased toward a single regime."
        )

    # Split
    windows_list = [cleaned_arr[i] for i in range(len(cleaned_arr))]
    train_pct = train_cfg.get("splits", {}).get("train", 0.85)
    val_pct = train_cfg.get("splits", {}).get("val", 0.10)
    train_w, val_w, test_w = split_windows(
        windows_list, train_pct=train_pct, val_pct=val_pct, regime_labels=regime_labels,
    )
    logger.info("Split: %d train, %d val, %d test", len(train_w), len(val_w), len(test_w))

    train_arr = windows_to_numpy(train_w)
    val_arr = windows_to_numpy(val_w)
    test_arr = windows_to_numpy(test_w) if test_w else None

    # Config
    latent_dim = config.get("model", {}).get("latent_dim") or get_latent_dim(n_sensors)
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
        overfit_ratio_threshold=train_cfg.get("overfit_ratio_threshold", 1.5),
        seed=train_cfg.get("seed"),
    )

    logger.info("Architecture: %d → 128 → 64 → 32 → %d (latent)", n_sensors, latent_dim)

    # Train
    model, history = train_model(train_arr, val_arr, tc)
    final_ratio = history["val_train_ratio"][-1]
    logger.info(
        "Training done: %d epochs, final train=%.6f, val=%.6f, val/train=%.2f",
        len(history["train_loss"]),
        history["train_loss"][-1],
        history["val_loss"][-1],
        final_ratio,
    )
    if final_ratio > tc.overfit_ratio_threshold:
        logger.warning(
            "Final val/train ratio %.2f exceeds overfit threshold %.2f.",
            final_ratio, tc.overfit_ratio_threshold,
        )

    device = torch.device("cpu")

    # Calibration errors for thresholds. Calibrating on the model's own training
    # errors is optimistic (those are exactly the windows it was fit to
    # reconstruct well) -- val errors give an unbiased read of the normal-error
    # distribution the P90/P99 cutoffs are supposed to characterise. Ablation
    # study (see output_ablation_full_seeded/) confirmed val-calibration raises
    # Red F1 (0.884->0.916) and any-flag precision (0.336->0.529) at negligible
    # recall cost, consistently across 3 seeds.
    calibration_source = thresh_cfg.get("calibration_source", "train")
    calibration_arr = val_arr if calibration_source == "val" else train_arr
    training_errors = compute_training_errors(model, calibration_arr, device)

    # Thresholds
    thresholds = compute_thresholds(
        training_errors,
        green_yellow_percentile=thresh_cfg.get("green_yellow", 90),
        yellow_red_percentile=thresh_cfg.get("yellow_red", 99),
        method=thresh_cfg.get("method", "robust"),
        spread=thresh_cfg.get("spread", "mad"),
    )
    logger.info(
        "Thresholds (calibrated on %s errors): green/yellow=%.6f, yellow/red=%.6f",
        calibration_source, thresholds["green_yellow"], thresholds["yellow_red"],
    )

    # Sensor baselines from test split
    sensor_baselines = None
    if test_arr is not None and len(test_arr) > 0:
        sensor_baselines = compute_sensor_baselines(model, test_arr, device)
        logger.info("Sensor baselines computed: mean=%.6f", sensor_baselines.mean())

    # Optional: train a FastSHAP explainer for amortized per-sensor Shapley
    # attribution. Additive to the pipeline -- disabled by default so
    # existing runs/configs are unaffected; see src/autoencoder/explain/.
    explain_cfg = config.get("explainability", {})
    explainer = None
    explainer_history = None
    if explain_cfg.get("enabled", False):
        print_banner("Step 2b: FastSHAP Explainer Training")
        fastshap_cfg = FastSHAPConfig(
            n_sensors=n_sensors,
            hidden_dim=explain_cfg.get("hidden_dim", 128),
            dropout=explain_cfg.get("dropout", 0.1),
            lr=explain_cfg.get("learning_rate", 1e-3),
            weight_decay=explain_cfg.get("weight_decay", 0.0),
            max_epochs=explain_cfg.get("max_epochs", 100),
            patience=explain_cfg.get("early_stopping_patience", 10),
            lr_factor=explain_cfg.get("lr_scheduler", {}).get("factor", 0.5),
            lr_patience=explain_cfg.get("lr_scheduler", {}).get("patience", 5),
            n_mask_samples=explain_cfg.get("n_mask_samples", 32),
            batch_size=explain_cfg.get("batch_size", 256),
            seed=explain_cfg.get("seed"),
        )
        # Rejected (isolation_forest/pca/mahalanobis-flagged) windows are folded
        # into the explainer's own training set only -- the autoencoder's train_arr
        # above is untouched. Without them, every row FastSHAP ever trains on sits
        # close to its own median, so masking any sensor changes the error by a
        # similar amount regardless of which row it is; the explainer then has no
        # incentive to learn row-specific attribution and collapses to a nearly
        # input-independent ranking (confirmed empirically -- see conversation/
        # commit history). Rejected windows are real examples of a sensor sitting
        # far from its baseline, giving the network something to actually learn
        # attribution *from*. Validation stays on the clean val_arr split so
        # early stopping still reflects normal-operation behaviour.
        if rejected_arr is not None and len(rejected_arr) > 0:
            explainer_train_arr = np.concatenate([train_arr, rejected_arr], axis=0)
            logger.info(
                "FastSHAP training set: %d clean + %d rejected = %d windows",
                len(train_arr), len(rejected_arr), len(explainer_train_arr),
            )
        else:
            explainer_train_arr = train_arr
            logger.warning(
                "No rejected windows available for FastSHAP training -- explainer "
                "will only see near-median normal rows and may fail to learn "
                "row-specific attribution (see explain/fastshap.py)."
            )
        explainer, explainer_history = train_fastshap_explainer(model, explainer_train_arr, val_arr, fastshap_cfg)
        logger.info(
            "FastSHAP training done: %d epochs, final train=%.6f, val=%.6f",
            len(explainer_history["train_loss"]),
            explainer_history["train_loss"][-1],
            explainer_history["val_loss"][-1],
        )

    # Metadata
    metadata = {
        "n_sensors": n_sensors,
        "latent_dim": latent_dim,
        "sensor_columns": sensor_columns,
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "n_train_windows": len(train_w),
        "n_val_windows": len(val_w),
        "n_test_windows": len(test_w) if test_w else 0,
        "threshold_calibration_source": calibration_source,
    }

    # Save artefacts
    artefacts_dir = output_dir / "artefacts"
    save_artefacts(
        output_dir=artefacts_dir,
        model=model,
        scaler=scaler,
        thresholds=thresholds,
        metadata=metadata,
        training_errors=training_errors,
        sensor_baselines=sensor_baselines,
    )

    # Save training history
    with open(artefacts_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    if explainer is not None:
        save_explainer(artefacts_dir, explainer)
        with open(artefacts_dir / "explainer_training_history.json", "w") as f:
            json.dump(explainer_history, f, indent=2)

    logger.info("All artefacts saved to %s", artefacts_dir)

    return model, thresholds, sensor_baselines, test_arr, sensor_columns, explainer


# ── Step 3: Inference on test windows ────────────────────────────────────────

def step_infer(model, scaler, thresholds, sensor_baselines,
               test_windows, sensor_columns, config, explainer=None):
    """Run inference on test windows, classify zones, apply persistence."""
    inference_cfg = config.get("inference", {})
    alerting_cfg = config.get("alerting", {}).get("persistence", {})

    print_banner("Step 3: Inference on Test Windows")

    if test_windows is None or len(test_windows) == 0:
        logger.warning("No test windows available for inference demo.")
        return

    flag_threshold = inference_cfg.get("sensor_flag_threshold", 3.0)
    anomaly_sensor_pct = inference_cfg.get("anomaly_sensor_pct", 10.0)

    # Integrated Gradients is training-free (unlike FastSHAP), so it's picked
    # purely at inference time via config -- and takes priority over any
    # loaded/trained FastSHAP explainer if both happen to be configured,
    # since infer_window treats the two as mutually exclusive.
    attribution_method = inference_cfg.get("attribution_method", "heuristic")
    use_ig = attribution_method == "integrated_gradients"
    ig_cfg = inference_cfg.get("integrated_gradients", {})
    if use_ig:
        explainer = None

    sensor_names = sensor_columns if sensor_columns else [f"sensor_{i}" for i in range(test_windows.shape[2])]

    # Run inference on each test window
    zones = []
    results = []
    for i in range(len(test_windows)):
        result = infer_window(
            window=test_windows[i],
            model=model,
            scaler=scaler,
            sensor_names=sensor_names,
            top_k=5,
            sensor_baselines=sensor_baselines,
            flag_threshold=flag_threshold,
            anomaly_sensor_pct=anomaly_sensor_pct,
            already_scaled=True,
            explainer=explainer,
            use_integrated_gradients=use_ig,
            ig_n_steps=ig_cfg.get("n_steps", 50),
            ig_baseline=ig_cfg.get("baseline", 0.0),
        )
        results.append(result)

        if result.usable:
            zone = classify_zone(result.window_score, thresholds)
        else:
            zone = "green"  # unusable windows don't trigger alerts
        zones.append(zone)

    # Apply persistence
    persistence = apply_persistence(
        zones,
        red_to_yellow_alert=alerting_cfg.get("red_to_yellow_alert", 2),
        red_to_red_alert=alerting_cfg.get("red_to_red_alert", 3),
        yellow_consecutive=alerting_cfg.get("yellow_consecutive", 4),
        mixed_window_hours=alerting_cfg.get("mixed_window_hours"),
    )

    # Summary
    zone_counts = {"green": 0, "yellow": 0, "red": 0}
    for z in zones:
        zone_counts[z] += 1

    scores = [r.window_score for r in results if r.usable]
    flagged_count = sum(
        1 for r in results
        if r.usable and r.sensors_anomalous is True
    )

    print(f"\n  Test windows:     {len(test_windows)}")
    print(f"  Usable:           {sum(1 for r in results if r.usable)}")
    print(f"  Zone breakdown:   Green={zone_counts['green']}  Yellow={zone_counts['yellow']}  Red={zone_counts['red']}")
    print(f"  Sensor-anomalous: {flagged_count} windows (>10% sensors flagged)")
    if scores:
        print(f"  Score range:      {min(scores):.6f} — {max(scores):.6f}")
        print(f"  Mean score:       {np.mean(scores):.6f}")
    print(f"  Persistence:      {persistence['alert_level'].upper()} — {persistence['reason']}")

    # Show top contributors from the worst window
    if scores:
        worst_idx = int(np.argmax(scores))
        worst = results[worst_idx]
        print(f"\n  Worst window (#{worst_idx}, score={worst.window_score:.6f}, zone={zones[worst_idx]}):")
        print(f"  Attribution method: {worst.attribution_method}")
        for c in worst.top_contributors[:5]:
            ratio_str = ""
            if worst.sensor_flags is not None:
                idx = c["index"]
                flag = "FLAGGED" if worst.sensor_flags[idx] else "ok"
                ratio_str = f"  ratio={worst.sensor_flags[idx]:.1f}x  [{flag}]"
            attr_str = f"  attribution={c['attribution_value']:.6f}" if "attribution_value" in c else ""
            print(f"    {c['name']:25s}  error={c['error']:.6f}  ({c['contribution_pct']:.1f}%){ratio_str}{attr_str}")


# ── Step 4: Inference-only mode ──────────────────────────────────────────────

def step_infer_only(data_path: str, model_dir: str, output_dir: Path, config: dict):
    """Load model + new data → infer → alert."""
    print_banner("Inference-Only Mode")
    ts_col = config.get("data", {}).get("timestamp_column", "timestamp")

    # Load artefacts
    model, scaler, thresholds, metadata, training_errors, sensor_baselines = load_artefacts(model_dir)
    explainer = load_explainer(model_dir)
    n_sensors = metadata["n_sensors"]
    sensor_columns = metadata.get("sensor_columns", [f"sensor_{i}" for i in range(n_sensors)])

    # Load data
    if data_path.endswith(".npy"):
        windows_arr = np.load(data_path)
        test_windows = windows_arr
    else:
        df = load_data(data_path, timestamp_column=ts_col)
        sensor_cols = detect_sensor_columns(df, timestamp_column=ts_col)
        # Use only columns the model was trained on
        sensor_cols = [c for c in sensor_cols if c in sensor_columns]
        scaled = scaler.transform(df[sensor_cols].values)
        # Construct windows manually
        window_size = 120
        n_windows = len(scaled) // window_size
        test_windows = np.array([
            scaled[i * window_size:(i + 1) * window_size]
            for i in range(n_windows)
        ])

    logger.info("Loaded %d windows for inference", len(test_windows))

    step_infer(model, scaler, thresholds, sensor_baselines,
               test_windows, sensor_columns, config, explainer=explainer)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autoencoder Equipment Health Monitoring — End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to sensor data (CSV, Parquet, or .npy)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument("--equipment-id", default="equipment_01", help="Equipment identifier")
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip cleaning (data is pre-cleaned .npy)")
    parser.add_argument("--infer-only", action="store_true", help="Inference only (requires --model-dir)")
    parser.add_argument("--model-dir", default=None, help="Path to existing model artefacts (for --infer-only)")
    parser.add_argument(
        "--explain", action="store_true",
        help="Train a FastSHAP explainer for per-sensor Shapley attribution alongside the "
             "model (overrides the config's explainability.enabled). Ignored with "
             "--infer-only, which always loads an explainer from --model-dir if one is there.",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    if args.explain:
        config.setdefault("explainability", {})["enabled"] = True
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_banner("Autoencoder Equipment Health Monitoring")
    print(f"  Data:       {args.data}")
    print(f"  Output:     {args.output}")
    print(f"  Config:     {args.config}")
    print(f"  Equipment:  {args.equipment_id}")

    # ── Inference-only mode ──
    if args.infer_only:
        if not args.model_dir:
            logger.error("--model-dir required for --infer-only mode")
            sys.exit(1)
        step_infer_only(args.data, args.model_dir, output_dir, config)
        print_banner("Done")
        return

    # ── Full pipeline ──
    if args.skip_cleaning:
        # Load pre-cleaned windows
        print_banner("Step 1: Loading Pre-Cleaned Windows")
        cleaned_arr = np.load(args.data)
        n_sensors = cleaned_arr.shape[2]
        sensor_columns = [f"sensor_{i}" for i in range(n_sensors)]
        # Try to load the real sensor names saved alongside a prior cleaning run.
        # Falls back to generic sensor_i names (above) if unavailable -- but that
        # silently breaks sensor-level diagnosis (wrong names reported to the
        # client), so this sidecar should always be present for --skip-cleaning
        # runs whose source .npy came from step_clean below.
        sensor_columns_path = Path(args.data).parent / "sensor_columns.json"
        if sensor_columns_path.exists():
            with open(sensor_columns_path) as f:
                sensor_columns = json.load(f)
            logger.info("Loaded sensor names from %s", sensor_columns_path)
        else:
            logger.warning(
                "No sensor_columns.json found alongside %s -- sensor-level diagnosis "
                "will report generic names (sensor_0..sensor_%d) instead of real tags.",
                args.data, n_sensors - 1,
            )
        # Try to load existing scaler
        scaler = None
        scaler_path = output_dir / "artefacts" / "scaler.pkl"
        if scaler_path.exists():
            import joblib
            scaler = joblib.load(scaler_path)
            logger.info("Loaded existing scaler from %s", scaler_path)
        # Try to load regime labels saved alongside a prior cleaning run
        regime_labels = None
        regime_labels_path = Path(args.data).parent / "cleaned_regime_labels.npy"
        if regime_labels_path.exists():
            regime_labels = np.load(regime_labels_path)
            logger.info("Loaded regime labels from %s", regime_labels_path)
        # Try to load cleaning-rejected windows saved alongside a prior cleaning run
        rejected_arr = None
        rejected_path = Path(args.data).parent / "rejected_windows.npy"
        if rejected_path.exists():
            rejected_arr = np.load(rejected_path)
            logger.info("Loaded %d rejected windows from %s", len(rejected_arr), rejected_path)
    else:
        cleaned_arr, sensor_columns, scaler_result, regime_labels, rejected_arr = step_clean(
            args.data, output_dir, config, args.equipment_id,
        )
        scaler = scaler_result.scaler

    # Train
    model, thresholds, sensor_baselines, test_arr, sensor_columns, explainer = step_train(
        cleaned_arr, sensor_columns, scaler, output_dir, config,
        regime_labels=regime_labels, rejected_arr=rejected_arr,
    )

    # Infer on test windows
    step_infer(model, scaler, thresholds, sensor_baselines,
               test_arr, sensor_columns, config, explainer=explainer)

    print_banner("Pipeline Complete")
    print(f"  Artefacts:  {output_dir / 'artefacts'}")
    print(f"  Cleaning:   {output_dir / 'cleaning'}")
    print(f"  Windows:    {output_dir / 'cleaned_windows.npy'}")
    print()


if __name__ == "__main__":
    main()
