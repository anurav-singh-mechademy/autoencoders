#!/usr/bin/env python
"""End-to-end autoencoder pipeline: clean → train → infer → alert.

Usage:
    # Full pipeline (clean + train + infer on test windows)
    python main.py --data sensor_data.csv --output output/

    # With config
    python main.py --data data.parquet --output output/ --config configs/default.yaml

    # Skip cleaning (use pre-split windows from a prior run)
    python main.py --data output/train_windows.npy --output output/ --skip-cleaning

    # Inference only (with existing model)
    python main.py --data new_data.csv --output output/ --model-dir output/artefacts/ --infer-only

Steps:
    1. Load & validate data
    2. Remove low-variance / constant columns, and null-dominated columns
    3. Fit RobustScaler on training data
    4. Construct 30-min windows (120 rows each)
    5. Segment operating regimes (KMeans)
    6. Null-based sensor-dropout filter (applies to all windows)
    7. Split: train / val / test (stratified by regime), BEFORE outlier
       -removal cleaning -- so real anomalies aren't stripped out of
       contention for the test split (see step_clean's docstring)
    8. Clean train + val ONLY: Isolation Forest → PCA → Mahalanobis (per
       regime). Test is left untouched -- whatever really happened.
    9. Generate cleaning reports (train, val)
    10. Train autoencoder with early stopping
    11. Compute thresholds (P90/P99) and sensor baselines
    12. Save all artefacts
    13. Run inference on test windows → classify zones → apply persistence
    14. Print summary
"""

import argparse
import copy
import json
import logging
import sys
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from autoencoder.data.ingestion import load_data, detect_sensor_columns, validate_dataframe
from autoencoder.data.preprocessing import (
    remove_low_variance_columns,
    remove_null_or_stuck_columns,
    fit_robust_scaler,
    apply_scaling,
    construct_windows_with_metadata,
    compute_event_touched_windows,
)
from autoencoder.data.regime import compute_window_features, segment_regimes, segment_regimes_unsupervised
from autoencoder.data.sensor_dropout import detect_sensor_dropout_windows, remove_sensor_dropout_windows
from autoencoder.data.cleaning import run_cleaning_pipeline
from autoencoder.reporting.cleaning_report import generate_cleaning_report
from autoencoder.evaluation.ground_truth import compute_split_node_coverage, restrict_to_event_sensors

from autoencoder.model.architecture import pick_latent_dim, compute_hidden_widths
from autoencoder.model.loss import per_row_mse, window_anomaly_score
from autoencoder.training.dataset import split_indices, windows_to_numpy
from autoencoder.training.trainer import TrainConfig, train_model, compute_sensor_baselines
from autoencoder.training.hyperparams import grid_search

from autoencoder.alerting.thresholds import compute_thresholds
from autoencoder.alerting.zones import classify_zone, classify_batch
from autoencoder.alerting.persistence import apply_persistence
from autoencoder.artefacts.serialisation import save_artefacts, load_artefacts

from autoencoder.inference.pipeline import infer_window
from autoencoder.inference.missing_data import assess_window_quality

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

def step_clean(data_path: str, output_dir: Path, config: dict, equipment_id: str, event_labels_path: str | None = None):
    """Load data → preprocess → clean → save cleaned windows + report."""
    cleaning_cfg = config.get("cleaning", {})
    regime_cfg = config.get("regime", {})
    ts_col = config.get("data", {}).get("timestamp_column", "timestamp")
    window_rows = config.get("data", {}).get("window_rows", 120)

    print_banner("Step 1: Data Loading & Cleaning")

    # Load
    df = load_data(data_path, timestamp_column=ts_col)

    # Guard against literal +/-inf in raw sensor readings (a real historian
    # glitch on some equipment) -- RobustScaler.fit() aborts on any infinite
    # value, so replace with NaN and let the usual null-handling filters below
    # deal with it (a no-op for columns that never had an infinite value).
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[numeric_cols].to_numpy()).sum())
    if n_inf:
        inf_cols = [c for c in numeric_cols if np.isinf(df[c].to_numpy()).any()]
        logger.warning(
            "Found %d infinite value(s) in raw data across %d column(s) (%s) -- replacing with NaN.",
            n_inf, len(inf_cols), inf_cols,
        )
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # exclude_columns: non-sensor metadata (e.g. ground-truth bookkeeping columns) to drop before sensor detection.
    exclude_columns = config.get("data", {}).get("exclude_columns", [])
    sensor_columns = detect_sensor_columns(df, timestamp_column=ts_col, exclude_columns=exclude_columns)
    logger.info("Loaded %d rows, %d sensor columns", len(df), len(sensor_columns))

    errors = validate_dataframe(df, timestamp_column=ts_col, expected_sensors=sensor_columns)
    if errors:
        logger.error("Validation errors: %s", errors)
        sys.exit(1)

    # Remove low-variance columns
    var_threshold = config.get("preprocessing", {}).get("variance_threshold", 1e-5)
    filter_result = remove_low_variance_columns(df, sensor_columns, variance_threshold=var_threshold)
    sensor_columns = filter_result.kept_columns
    if filter_result.removed_columns:
        logger.info("Removed low-variance columns: %s", filter_result.removed_columns)
    logger.info("Kept %d sensor columns after filtering", len(sensor_columns))

    # Remove columns null-dominated in most windows (whole-file variance alone
    # can miss this). Stuck/flat columns are kept by default (max_stuck_pct=100)
    # since a flat sensor is often real signal, not a data-quality problem.
    max_null_pct = config.get("preprocessing", {}).get("max_null_pct", 50.0)
    max_stuck_pct = config.get("preprocessing", {}).get("max_stuck_pct", 100.0)
    null_stuck_result = remove_null_or_stuck_columns(
        df, sensor_columns, window_size=window_rows, max_null_pct=max_null_pct, max_stuck_pct=max_stuck_pct,
    )
    sensor_columns = null_stuck_result.kept_columns
    if null_stuck_result.removed_columns:
        logger.info(
            "Removed null-or-stuck columns (max_null_pct=%.0f%%, max_stuck_pct=%.0f%% of windows): %s",
            max_null_pct, max_stuck_pct, null_stuck_result.removed_columns,
        )
    logger.info("Kept %d sensor columns after filtering", len(sensor_columns))

    # Restrict to sensors implicated by labeled events (opt-in; see
    # restrict_to_event_sensors()'s docstring). restrict_to_event_sensors_stage
    # picks WHEN: "pre_split" narrows sensor_columns right here; "post_split"
    # defers narrowing until after regime segmentation and the split below, so
    # regime detection still sees the full sensor set instead of only the
    # fault-indicator sensors (which would otherwise confound "regime" with "anomaly").
    restrict_to_event_sensors_flag = config.get("preprocessing", {}).get("restrict_to_event_sensors", False)
    restrict_stage = config.get("preprocessing", {}).get("restrict_to_event_sensors_stage", "pre_split")
    if restrict_stage not in ("pre_split", "post_split"):
        raise ValueError(
            f"Unknown preprocessing.restrict_to_event_sensors_stage={restrict_stage!r} -- "
            "expected 'pre_split' or 'post_split'."
        )
    deferred_event_sensor_result = None
    if restrict_to_event_sensors_flag:
        if not event_labels_path:
            logger.error(
                "preprocessing.restrict_to_event_sensors is enabled but no --event-labels path "
                "was given. Aborting."
            )
            sys.exit(1)
        event_sensor_result = restrict_to_event_sensors(
            sensor_columns, event_labels_path, equipment_tag=equipment_id,
        )
        if restrict_stage == "pre_split":
            sensor_columns = event_sensor_result.kept_columns
        else:
            deferred_event_sensor_result = event_sensor_result
            logger.info(
                "Event-sensor restriction deferred to post-split (restrict_to_event_sensors_stage="
                "'post_split'): regime segmentation and the split below will see all %d sensor "
                "column(s); train/val/test will be sliced down to %d afterwards.",
                len(sensor_columns), len(event_sensor_result.kept_columns),
            )

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
    tail_compression_scale = config.get("preprocessing", {}).get("tail_compression_scale")
    scaled_df = apply_scaling(df, scaler_result, tail_compression_scale=tail_compression_scale)
    logger.info(
        "Fit RobustScaler on first %d / %d rows (train_pct=%.2f).",
        n_train_rows, len(df), train_pct,
    )
    if tail_compression_scale is not None:
        logger.info(
            "Applied tail compression c*asinh(z/c), c=%.1f, to scaled sensor values.",
            tail_compression_scale,
        )

    # Save scaler for later use
    import joblib
    artefacts_dir = output_dir / "artefacts"
    artefacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_result.scaler, artefacts_dir / "scaler.pkl")

    # Constructed with metadata so each window keeps its raw-file window_id,
    # carried through filtering/cleaning so step_train can save exactly which
    # raw windows ended up in train/val/test.
    windows_meta = construct_windows_with_metadata(
        scaled_df, window_size=window_rows, sensor_columns=sensor_columns, timestamp_column=ts_col,
    )
    windows = [w["data"] for w in windows_meta]
    window_ids = np.array([w["window_id"] for w in windows_meta])
    # Carried in lockstep with window_ids; only used by training.split_stratify_by="month".
    window_start_times = np.array([w["start_time"] for w in windows_meta])
    logger.info("Constructed %d windows", len(windows))

    if len(windows) < 10:
        logger.error("Too few windows (%d). Need at least 10.", len(windows))
        sys.exit(1)

    # Exclude windows where >5% of sensors are null, before regime segmentation
    # (whose PCA fallback can't tolerate any NaN).
    dropout_cfg = config.get("preprocessing", {}).get("sensor_dropout", {})
    dropout_result = detect_sensor_dropout_windows(
        windows, threshold_pct=dropout_cfg.get("threshold_pct", 5.0),
    )
    if dropout_result.n_dropped:
        keep_mask = ~dropout_result.is_dropped
        windows = remove_sensor_dropout_windows(windows, dropout_result)
        window_ids = window_ids[keep_mask]
        window_start_times = window_start_times[keep_mask]
        logger.info(
            "Dropped %d / %d windows for excess null sensors (threshold=%.1f%%).",
            dropout_result.n_dropped, len(keep_mask), dropout_cfg.get("threshold_pct", 5.0),
        )

    # Eliminate any remaining NaN cells before they reach regime segmentation,
    # per-regime cleaning, or training -- none of those tolerate NaN, and the
    # filters above only guarantee a window has *few* null sensors, not that
    # it has *zero* null cells (a sensor under the dropout threshold can still
    # have a short raw gap inside this specific window). Reuses the same
    # bounded-forward-fill + null-dominant-sensor-masking logic already used
    # at inference time (autoencoder.inference.missing_data), so a sensor's
    # treatment here is consistent with how it's treated when the trained
    # model later scores live windows. Masked sensors are zeroed -- the
    # scaled-space median, since windows here are already scaled -- matching
    # infer_window's exact convention (src/autoencoder/inference/pipeline.py).
    missing_data_cfg = config.get("inference", {}).get("missing_data", {})
    max_null_pct = missing_data_cfg.get("max_null_pct_per_sensor", 5.0)
    max_consecutive_nulls = missing_data_cfg.get("max_consecutive_nulls_ffill", 3)
    max_null_dominant_sensor_pct = missing_data_cfg.get("max_null_dominant_sensor_pct", 30.0)

    filled_windows = []
    keep_quality_mask = np.ones(len(windows), dtype=bool)
    n_filled_windows = 0
    n_masked_sensor_instances = 0
    for i, w in enumerate(windows):
        quality = assess_window_quality(
            w,
            max_null_pct_per_sensor=max_null_pct,
            max_consecutive_nulls=max_consecutive_nulls,
            max_null_dominant_sensor_pct=max_null_dominant_sensor_pct,
        )
        if not quality["usable"]:
            keep_quality_mask[i] = False
            continue
        w_clean = quality["filled_window"]
        if quality["masked_sensors"]:
            w_clean = np.array(w_clean, dtype=w_clean.dtype, copy=True)
            w_clean[:, quality["masked_sensors"]] = 0.0
            n_masked_sensor_instances += len(quality["masked_sensors"])
        if quality["fill_info"]["total_filled"] or quality["masked_sensors"]:
            n_filled_windows += 1
        filled_windows.append(w_clean)

    n_quality_dropped = int((~keep_quality_mask).sum())
    if n_quality_dropped:
        window_ids = window_ids[keep_quality_mask]
        window_start_times = window_start_times[keep_quality_mask]
        logger.info(
            "Dropped %d additional window(s) for excess null-dominant sensors (>%.0f%% of sensors).",
            n_quality_dropped, max_null_dominant_sensor_pct,
        )
    windows = filled_windows
    logger.info(
        "Forward-filled/masked residual nulls in %d windows (%d sensor-instances masked to 0).",
        n_filled_windows, n_masked_sensor_instances,
    )

    if len(windows) < 10:
        logger.error("Too few windows (%d) after null handling. Need at least 10.", len(windows))
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

    # ── Split BEFORE outlier-removal cleaning ──
    # Isolation Forest / PCA / Mahalanobis (below) exist to strip windows
    # that look statistically anomalous, so the model trains on a clean
    # "normal" distribution -- but a real labeled fault event looks exactly
    # like a statistical outlier too. Running that cleaning before the split
    # would remove real anomalies from contention for the test split
    # entirely (verified: one equipment's test split kept ending up with
    # zero labeled events because 97% of its real event-touching windows
    # were being cleaned away before the split ever saw them). Splitting
    # first and cleaning only train/val fixes this: test is guaranteed to
    # be exactly "whatever the equipment actually did" in its held-out
    # window range, anomalies included, which is the entire point of a test
    # split for a fault-detection model. Test still went through the null
    # -based sensor-dropout filter above (a scorability concern, not an
    # anomalousness one), and scaling (a train-only-fit transform applied
    # uniformly) -- both of those are unrelated to outlier-removal cleaning.
    train_pct = config.get("training", {}).get("splits", {}).get("train", 0.70)
    val_pct = config.get("training", {}).get("splits", {}).get("val", 0.15)

    # training.split_stratify_by picks the grouping label split_indices stratifies by:
    # "regime" (default) or "month" (so every split gets a proportional share of every month).
    split_stratify_by = config.get("training", {}).get("split_stratify_by", "regime")
    if split_stratify_by == "month":
        split_strata = np.array([pd.Timestamp(t).strftime("%Y-%m") for t in window_start_times])
    elif split_stratify_by == "regime":
        split_strata = regime_result.labels
    else:
        raise ValueError(
            f"Unknown training.split_stratify_by={split_stratify_by!r} -- expected 'regime' or 'month'."
        )

    train_idx, val_idx, test_idx = split_indices(
        len(windows), train_pct=train_pct, val_pct=val_pct, regime_labels=split_strata,
    )
    logger.info(
        "Split (pre-cleaning): %d train, %d val, %d test windows.",
        len(train_idx), len(val_idx), len(test_idx),
    )

    def _subset_windows_and_ids(idx: list[int]):
        return [windows[i] for i in idx], window_ids[idx] if idx else np.array([], dtype=window_ids.dtype)

    def _regime_result_for(idx: list[int]):
        labels_subset = regime_result.labels[idx] if idx else np.array([], dtype=int)
        return dataclass_replace(
            regime_result,
            labels=labels_subset,
            window_counts={
                regime_result.regime_names[i]: int(np.sum(labels_subset == i))
                for i in range(regime_result.n_clusters)
            },
        )

    train_windows, train_raw_ids = _subset_windows_and_ids(train_idx)
    val_windows, val_raw_ids = _subset_windows_and_ids(val_idx)
    test_windows, test_ids = _subset_windows_and_ids(test_idx)

    # Apply the event-sensor restriction deferred from above (restrict_stage="post_split"):
    # slice train/val/test down to the event-implicated sensors and use that as sensor_columns downstream.
    if deferred_event_sensor_result is not None:
        n_full = len(sensor_columns)
        keep_idx = [sensor_columns.index(c) for c in deferred_event_sensor_result.kept_columns]
        train_windows = [w[:, keep_idx] for w in train_windows]
        val_windows = [w[:, keep_idx] for w in val_windows]
        test_windows = [w[:, keep_idx] for w in test_windows]
        sensor_columns = deferred_event_sensor_result.kept_columns

        # Slice the scaler's fitted center_/scale_ down to the kept columns so it
        # matches inference's input, instead of erroring on a feature-count mismatch.
        restricted_scaler = copy.deepcopy(scaler_result.scaler)
        restricted_scaler.center_ = restricted_scaler.center_[keep_idx]
        restricted_scaler.scale_ = restricted_scaler.scale_[keep_idx]
        restricted_scaler.n_features_in_ = len(keep_idx)
        scaler_result = dataclass_replace(
            scaler_result,
            scaler=restricted_scaler,
            sensor_columns=sensor_columns,
            n_sensors=len(sensor_columns),
            medians=restricted_scaler.center_,
            iqrs=restricted_scaler.scale_,
        )
        joblib.dump(scaler_result.scaler, artefacts_dir / "scaler.pkl")

        logger.info(
            "Applied post-split event-sensor restriction: sensor_columns reduced from %d to %d "
            "(regime segmentation and the split above used all %d). Scaler re-sliced to match.",
            n_full, len(sensor_columns), n_full,
        )

    # Optional: drop train/val windows touched by a labeled event node (test is left untouched -- it needs real anomalies to evaluate against).
    if config.get("preprocessing", {}).get("exclude_event_touched_windows", False):
        if "n_active_nodes" not in df.columns:
            logger.error(
                "preprocessing.exclude_event_touched_windows is enabled but the input data has "
                "no 'n_active_nodes' column -- can't tell which windows are event-touched. "
                "Aborting."
            )
            sys.exit(1)
        event_touched = compute_event_touched_windows(df, window_size=window_rows)

        def _drop_event_touched(subset_windows: list, subset_ids: np.ndarray):
            keep = np.array([not event_touched[int(wid)] for wid in subset_ids], dtype=bool)
            kept_windows = [w for w, k in zip(subset_windows, keep) if k]
            kept_ids = subset_ids[keep] if len(subset_ids) else subset_ids
            return kept_windows, kept_ids, len(subset_windows) - len(kept_windows)

        train_windows, train_raw_ids, n_train_dropped = _drop_event_touched(train_windows, train_raw_ids)
        val_windows, val_raw_ids, n_val_dropped = _drop_event_touched(val_windows, val_raw_ids)
        logger.info(
            "Excluded event-touched windows: train %d dropped (%d remain), val %d dropped "
            "(%d remain). Test left untouched (%d windows).",
            n_train_dropped, len(train_windows), n_val_dropped, len(val_windows), len(test_windows),
        )

    # Clean -- only train and val (test is deliberately left untouched, see above).
    clean_dir = output_dir / "cleaning"
    clean_dir.mkdir(parents=True, exist_ok=True)

    cleaning_kwargs = dict(
        sensor_columns=sensor_columns,
        if_contamination=cleaning_cfg.get("isolation_forest", {}).get("contamination", 0.05),
        pca_variance_threshold=cleaning_cfg.get("pca", {}).get("variance_threshold", 0.95),
        pca_outlier_std=cleaning_cfg.get("pca", {}).get("outlier_std", 3.0),
        md_chi2_percentile=cleaning_cfg.get("mahalanobis", {}).get("chi2_percentile", 97.5),
        steps=cleaning_cfg.get("order", ["isolation_forest", "pca", "mahalanobis"]),
    )
    train_pipeline_result = run_cleaning_pipeline(
        windows=train_windows, regime_result=_regime_result_for(train_idx),
        window_ids=train_raw_ids.tolist(), output_dir=clean_dir / "train_validation", **cleaning_kwargs,
    )
    val_pipeline_result = run_cleaning_pipeline(
        windows=val_windows, regime_result=_regime_result_for(val_idx),
        window_ids=val_raw_ids.tolist(), output_dir=clean_dir / "val_validation", **cleaning_kwargs,
    )

    for split_name, result in [("train", train_pipeline_result), ("val", val_pipeline_result)]:
        removed_pct = (1 - result.cleaned_count / result.original_count) * 100 if result.original_count else 0.0
        logger.info(
            "Cleaning (%s): %d → %d windows (%.1f%% removed)",
            split_name, result.original_count, result.cleaned_count, removed_pct,
        )

    # Report -- one per cleaned split.
    for split_name, result in [("train", train_pipeline_result), ("val", val_pipeline_result)]:
        report_path = generate_cleaning_report(
            pipeline_result=result,
            sensor_columns=sensor_columns,
            equipment_id=f"{equipment_id}_{split_name}",
            output_path=str(clean_dir / f"{split_name}_cleaning_report.html"),
        )
        logger.info("Cleaning report (%s): %s", split_name, report_path)

    train_arr = np.array(train_pipeline_result.cleaned_windows)
    train_ids = np.array(train_pipeline_result.cleaned_window_ids)
    val_arr = np.array(val_pipeline_result.cleaned_windows)
    val_ids = np.array(val_pipeline_result.cleaned_window_ids)
    test_arr = windows_to_numpy(test_windows) if test_windows else np.zeros((0, window_rows, len(sensor_columns)), dtype=np.float32)
    test_ids = np.array(test_ids)

    np.save(output_dir / "train_windows.npy", train_arr)
    np.save(output_dir / "train_window_ids.npy", train_ids)
    np.save(output_dir / "val_windows.npy", val_arr)
    np.save(output_dir / "val_window_ids.npy", val_ids)
    np.save(output_dir / "test_windows.npy", test_arr)
    np.save(output_dir / "test_window_ids.npy", test_ids)
    with open(output_dir / "sensor_columns.json", "w") as f:
        json.dump(sensor_columns, f)

    return train_arr, val_arr, test_arr, sensor_columns, scaler_result, train_ids, val_ids, test_ids


# ── Step 2: Train ────────────────────────────────────────────────────────────

def step_train(train_arr: np.ndarray, val_arr: np.ndarray, test_arr: np.ndarray, sensor_columns: list[str],
               scaler, output_dir: Path, config: dict, train_ids=None, val_ids=None, test_ids=None,
               data_path: str | None = None, event_labels_path: str | None = None):
    """Train → compute thresholds + baselines → save artefacts.

    train/val/test are expected to already be split (see step_clean) --
    outlier-removal cleaning was applied only to train/val there; test is
    deliberately left as whatever the equipment actually did in its held
    -out window range, so real anomalies survive into it.
    """
    train_cfg = config.get("training", {})
    thresh_cfg = config.get("thresholds", {})

    print_banner("Step 2: Model Training")

    n_sensors = train_arr.shape[2]
    logger.info("Training on %d windows, %d sensors", len(train_arr), n_sensors)
    logger.info("Split: %d train, %d val, %d test", len(train_arr), len(val_arr), len(test_arr))

    if train_ids is None:
        train_ids = np.arange(len(train_arr))
    if val_ids is None:
        val_ids = np.arange(len(train_arr), len(train_arr) + len(val_arr))
    if test_ids is None:
        test_ids = np.arange(len(train_arr) + len(val_arr), len(train_arr) + len(val_arr) + len(test_arr))

    # ── Gate 1: training split must have a minimum amount of data ──
    # A model calibrated on too short a history hasn't seen enough of the
    # equipment's real operating envelope (seasonal load swings, periodic
    # maintenance cycles, etc.) to have a trustworthy notion of "normal".
    window_rows = config.get("data", {}).get("window_rows", 120)
    frequency_seconds = config.get("data", {}).get("frequency_seconds", 15)
    min_train_days = train_cfg.get("min_train_days", 182.5)  # ~6 months
    train_days = len(train_arr) * window_rows * frequency_seconds / 86400
    if train_days < min_train_days:
        logger.error(
            "Training split has only %.1f days of data (%d windows) -- below the required "
            "minimum of %.1f days (~6 months). Aborting before training.",
            train_days, len(train_arr), min_train_days,
        )
        sys.exit(1)
    logger.info("Training split has %.1f days of data (>= required %.1f).", train_days, min_train_days)

    # ── Gate 2: test split must contain at least one labeled event node ──
    # A test split with zero labeled events can't evaluate detection quality
    # at all -- every metric downstream (precision/recall/PR-AUC) would be
    # degenerate, and "0 false alarms" would look identical to "model works
    # great" and to "there was nothing to detect in the first place".
    if event_labels_path:
        coverage = compute_split_node_coverage(
            event_labels_path, data_path,
            {"train": train_ids, "val": val_ids, "test": test_ids},
            window_size=window_rows,
        )
        test_nodes = coverage["test"]["n_active_nodes_full"] + coverage["test"]["n_active_nodes_partial"]
        if test_nodes < 1:
            logger.error(
                "Test split has 0 active event nodes (checked against %s) -- can't evaluate "
                "detection quality against an empty test split. Aborting before training.",
                event_labels_path,
            )
            sys.exit(1)
        logger.info("Test split active-node check passed: %d node(s) present.", test_nodes)
    else:
        logger.warning(
            "No event_labels_path given -- skipping the test-split active-node gate. Pass "
            "--event-labels to enforce it."
        )

    # Config
    model_cfg = config.get("model", {})
    latent_dim = model_cfg.get("latent_dim")
    if latent_dim is None:
        latent_dim = pick_latent_dim(
            train_arr.reshape(-1, n_sensors),
            variance_threshold=model_cfg.get("latent_variance_threshold", 0.95),
            floor=model_cfg.get("latent_dim_floor", 4),
        )
        logger.info(
            "Picked latent_dim=%d via PCA (variance_threshold=%.2f, floor=%d)",
            latent_dim, model_cfg.get("latent_variance_threshold", 0.95), model_cfg.get("latent_dim_floor", 4),
        )

    max_hidden_layers = model_cfg.get("max_hidden_layers", 3)
    hidden_widths = compute_hidden_widths(n_sensors, latent_dim, max_hidden_layers)
    arch_str = " → ".join(str(w) for w in [n_sensors, *hidden_widths, latent_dim])
    logger.info("Architecture: %s (latent)", arch_str)

    # Early-stopping patience scales modestly with depth and always stays
    # above lr_patience -- patience <= lr_patience would let early stopping
    # fire before ReduceLROnPlateau ever gets a chance to act, silently
    # disabling it regardless of depth (verified: this was happening at
    # patience=5, lr_patience=10 for every config before this fix). The
    # per-layer scaling itself is now deliberately small: giving a 7-layer
    # network 45 epochs of patience (vs. depth-capped-at-3's much smaller
    # value) bought nothing -- it got stuck in a stable bad plateau at
    # epoch 1 and never improved across 46 epochs and two LR reductions.
    # Depth is now capped at max_hidden_layers instead (see
    # compute_hidden_widths), so a large patience_per_hidden_layer is no
    # longer doing useful work covering for unbounded depth.
    lr_patience = train_cfg.get("lr_scheduler", {}).get("patience", 10)
    patience_per_layer = train_cfg.get("patience_per_hidden_layer", 2)
    patience = train_cfg.get("early_stopping_patience")
    if patience is None:
        patience = lr_patience + patience_per_layer * (len(hidden_widths) + 1)
        logger.info(
            "Picked early_stopping_patience=%d (lr_patience=%d + %d/hidden-layer x %d layers)",
            patience, lr_patience, patience_per_layer, len(hidden_widths) + 1,
        )

    tc = TrainConfig(
        n_sensors=n_sensors,
        latent_dim=latent_dim,
        max_hidden_layers=max_hidden_layers,
        dropout=config.get("model", {}).get("dropout", 0.2),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        max_epochs=train_cfg.get("max_epochs", 150),
        patience=patience,
        lr_factor=train_cfg.get("lr_scheduler", {}).get("factor", 0.5),
        lr_patience=lr_patience,
        overfit_ratio_threshold=train_cfg.get("overfit_ratio_threshold", 1.5),
        seed=train_cfg.get("seed"),
    )

    # Train
    model, history = train_model(train_arr, val_arr, tc)
    # Report the RESTORED (best-epoch) model's numbers, not the last epoch
    # trained -- with early stopping, the last epoch is `patience` epochs past
    # the best point and is discarded; history["train_loss"][-1]/["val_loss"][-1]
    # describe a checkpoint that was never kept.
    best_ratio = history["best_val_train_ratio"]
    logger.info(
        "Training done: %d epochs run, best epoch %d: train=%.6f, val=%.6f, val/train=%.2f",
        len(history["train_loss"]),
        history["best_epoch"],
        history["best_train_loss"],
        history["best_val_loss"],
        best_ratio,
    )
    if len(history["train_loss"]) > history["best_epoch"]:
        logger.info(
            "  (last epoch trained, %d, before early stopping: train=%.6f, val=%.6f, val/train=%.2f -- "
            "discarded, shown for reference only)",
            len(history["train_loss"]), history["train_loss"][-1], history["val_loss"][-1],
            history["val_train_ratio"][-1],
        )
    if best_ratio > tc.overfit_ratio_threshold:
        logger.warning(
            "Best-epoch val/train ratio %.2f exceeds overfit threshold %.2f.",
            best_ratio, tc.overfit_ratio_threshold,
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

    # Sensor baselines from the VAL split -- NOT test. compute_sensor_baselines
    # wants per-sensor mean reconstruction error under NORMAL operation (used
    # for ratio-based flagging at inference); val is still outlier-cleaned
    # (see step_clean), while test now deliberately contains real anomalies,
    # which would inflate the baseline and make flagging less sensitive.
    sensor_baselines = None
    if val_arr is not None and len(val_arr) > 0:
        sensor_baselines = compute_sensor_baselines(model, val_arr, device)
        logger.info("Sensor baselines computed: mean=%.6f", sensor_baselines.mean())

    # Metadata
    metadata = {
        "n_sensors": n_sensors,
        "latent_dim": latent_dim,
        "max_hidden_layers": max_hidden_layers,
        "sensor_columns": sensor_columns,
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": history["best_epoch"],
        "final_train_loss": history["best_train_loss"],
        "final_val_loss": history["best_val_loss"],
        "last_epoch_train_loss": history["train_loss"][-1],
        "last_epoch_val_loss": history["val_loss"][-1],
        "n_train_windows": len(train_arr),
        "n_val_windows": len(val_arr),
        "n_test_windows": len(test_arr),
        "threshold_calibration_source": calibration_source,
        "tail_compression_scale": config.get("preprocessing", {}).get("tail_compression_scale"),
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

    # Raw-file window ids per split -- lets scripts/score_full_timeline.py
    # restrict scoring to the test split only, and scripts/evaluate_against_labels.py
    # report ground-truth node coverage per split (see compute_split_node_coverage).
    split_window_ids = {
        "window_size": config.get("data", {}).get("window_rows", 120),
        "train": [int(i) for i in train_ids],
        "val": [int(i) for i in val_ids],
        "test": [int(i) for i in test_ids],
    }
    with open(artefacts_dir / "split_window_ids.json", "w") as f:
        json.dump(split_window_ids, f, indent=2)
    logger.info("Saved %s", artefacts_dir / "split_window_ids.json")

    logger.info("All artefacts saved to %s", artefacts_dir)

    return model, thresholds, sensor_baselines, test_arr, sensor_columns


# ── Step 3: Inference on test windows ────────────────────────────────────────

def step_infer(model, scaler, thresholds, sensor_baselines,
               test_windows, sensor_columns, config):
    """Run inference on test windows, classify zones, apply persistence."""
    inference_cfg = config.get("inference", {})
    alerting_cfg = config.get("alerting", {}).get("persistence", {})

    print_banner("Step 3: Inference on Test Windows")

    if test_windows is None or len(test_windows) == 0:
        logger.warning("No test windows available for inference demo.")
        return

    flag_threshold = inference_cfg.get("sensor_flag_threshold", 3.0)
    anomaly_sensor_pct = inference_cfg.get("anomaly_sensor_pct", 10.0)
    missing_data_cfg = inference_cfg.get("missing_data", {})
    max_null_pct = missing_data_cfg.get("max_null_pct_per_sensor", 5.0)
    max_consecutive_nulls = missing_data_cfg.get("max_consecutive_nulls_ffill", 3)
    max_null_dominant_sensor_pct = missing_data_cfg.get("max_null_dominant_sensor_pct", 30.0)

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
            max_null_pct=max_null_pct,
            max_consecutive_nulls=max_consecutive_nulls,
            max_null_dominant_sensor_pct=max_null_dominant_sensor_pct,
            already_scaled=True,
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
        for c in worst.top_contributors[:5]:
            ratio_str = ""
            if worst.sensor_flags is not None:
                idx = c["index"]
                flag = "FLAGGED" if worst.sensor_flags[idx] else "ok"
                ratio_str = f"  ratio={worst.sensor_flags[idx]:.1f}x  [{flag}]"
            print(f"    {c['name']:25s}  error={c['error']:.6f}  ({c['contribution_pct']:.1f}%){ratio_str}")


# ── Step 4: Inference-only mode ──────────────────────────────────────────────

def step_infer_only(data_path: str, model_dir: str, output_dir: Path, config: dict):
    """Load model + new data → infer → alert."""
    print_banner("Inference-Only Mode")
    ts_col = config.get("data", {}).get("timestamp_column", "timestamp")

    # Load artefacts
    model, scaler, thresholds, metadata, training_errors, sensor_baselines = load_artefacts(model_dir)
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
               test_windows, sensor_columns, config)


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
        "--event-labels", default=None,
        help="Path to a *_event_labels_long.parquet file matching --data. When given, training "
             "aborts if the test split ends up with zero labeled event nodes (see "
             "autoencoder.evaluation.ground_truth.compute_split_node_coverage). Omit to skip this "
             "check, e.g. for equipment with no ground truth at all.",
    )

    args = parser.parse_args()
    config = load_config(args.config)
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
        # Load pre-split train/val/test windows (see step_clean) -- pass
        # --data as the train_windows.npy path; val/test/ids/sensor names
        # are derived as siblings in the same directory.
        print_banner("Step 1: Loading Pre-Split Windows")
        data_dir = Path(args.data).parent

        def _load_split(name: str) -> np.ndarray:
            path = data_dir / f"{name}_windows.npy"
            return np.load(path) if path.exists() else np.zeros((0, 120, 0), dtype=np.float32)

        def _load_ids(name: str, arr: np.ndarray) -> np.ndarray:
            path = data_dir / f"{name}_window_ids.npy"
            if path.exists():
                return np.load(path)
            logger.warning(
                "No %s found -- split_window_ids.json will use array positions instead of "
                "raw-file window ids, so it won't align with ground truth built from the "
                "original file.", path,
            )
            return np.arange(len(arr))

        train_arr = np.load(args.data)
        val_arr = _load_split("val")
        test_arr = _load_split("test")
        train_ids = _load_ids("train", train_arr)
        val_ids = _load_ids("val", val_arr)
        test_ids = _load_ids("test", test_arr)

        n_sensors = train_arr.shape[2]
        sensor_columns = [f"sensor_{i}" for i in range(n_sensors)]
        # Try to load the real sensor names saved alongside a prior cleaning run.
        # Falls back to generic sensor_i names (above) if unavailable -- but that
        # silently breaks sensor-level diagnosis (wrong names reported to the
        # client), so this sidecar should always be present for --skip-cleaning
        # runs whose source .npy came from step_clean below.
        sensor_columns_path = data_dir / "sensor_columns.json"
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
    else:
        train_arr, val_arr, test_arr, sensor_columns, scaler_result, train_ids, val_ids, test_ids = step_clean(
            args.data, output_dir, config, args.equipment_id, event_labels_path=args.event_labels,
        )
        scaler = scaler_result.scaler

    # Train
    model, thresholds, sensor_baselines, test_arr, sensor_columns = step_train(
        train_arr, val_arr, test_arr, sensor_columns, scaler, output_dir, config,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        data_path=args.data, event_labels_path=args.event_labels,
    )

    # Infer on test windows
    step_infer(model, scaler, thresholds, sensor_baselines,
               test_arr, sensor_columns, config)

    print_banner("Pipeline Complete")
    print(f"  Artefacts:  {output_dir / 'artefacts'}")
    print(f"  Cleaning:   {output_dir / 'cleaning'}")
    print(f"  Windows:    {output_dir / 'train_windows.npy'}, {output_dir / 'val_windows.npy'}, {output_dir / 'test_windows.npy'}")
    print()


if __name__ == "__main__":
    main()
