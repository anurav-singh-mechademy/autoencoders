"""Inference pipeline: scale → forward pass → score → diagnose a single 30-min window."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import per_row_mse, per_sensor_mse, window_anomaly_score
from autoencoder.inference.missing_data import assess_window_quality
from autoencoder.inference.diagnosis import diagnose_window
from autoencoder.explain.fastshap import FastSHAPExplainer
from autoencoder.explain.explain_window import explain_window_fastshap
from autoencoder.explain.integrated_gradients import integrated_gradients_window

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of running inference on a single window."""

    window_score: float
    row_errors: np.ndarray          # shape (120,)
    sensor_errors: np.ndarray       # shape (n_sensors,)
    reconstruction: np.ndarray      # shape (120, n_sensors)
    top_contributors: list[dict]
    quality_flags: list[str]
    usable: bool
    # Ratio-based fields (present when sensor_baselines provided)
    sensor_flags: np.ndarray | None = None
    pct_flagged_sensors: float | None = None
    sensors_anomalous: bool | None = None
    # Present when an explainer or Integrated Gradients is requested (see attribution_method)
    attribution_values: np.ndarray | None = None
    attribution_method: str = "heuristic"


def is_equipment_running(
    window: np.ndarray,
    sensor_index: int,
    threshold: float,
    min_pct: float = 50.0,
) -> bool:
    """Check if equipment is running based on a key sensor (e.g. motor current).

    Args:
        window: Raw sensor data, shape (n_rows, n_sensors).
        sensor_index: Column index of the running indicator sensor.
        threshold: Minimum value for the sensor to be considered "running".
        min_pct: Minimum % of rows above threshold to consider running.

    Returns:
        True if equipment is running.
    """
    values = window[:, sensor_index]
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return False
    pct_above = (valid > threshold).sum() / len(valid) * 100
    return pct_above >= min_pct


def _not_usable_result(window: np.ndarray, flags: list[str]) -> InferenceResult:
    """Build an InferenceResult for unusable windows."""
    return InferenceResult(
        window_score=float("nan"),
        row_errors=np.full(window.shape[0], float("nan")),
        sensor_errors=np.full(window.shape[1], float("nan")),
        reconstruction=np.full_like(window, float("nan")),
        top_contributors=[],
        quality_flags=flags,
        usable=False,
    )


def infer_window(
    window: np.ndarray,
    model: Autoencoder,
    scaler,
    sensor_names: list[str] | None = None,
    top_k: int | None = None,
    max_null_pct: float = 5.0,
    max_consecutive_nulls: int = 3,
    device: torch.device | None = None,
    sensor_baselines: np.ndarray | None = None,
    flag_threshold: float = 3.0,
    anomaly_sensor_pct: float = 10.0,
    running_sensor_index: int | None = None,
    running_threshold: float = 0.0,
    already_scaled: bool = False,
    explainer: FastSHAPExplainer | None = None,
    use_integrated_gradients: bool = False,
    ig_n_steps: int = 50,
    ig_baseline: float = 0.0,
) -> InferenceResult:
    """Run the full inference pipeline on a single window.

    Steps:
        0. Check if equipment is running (optional)
        1. Assess missing data and forward-fill if possible
        2. Apply scaler (already fitted on training data)
        3. Forward pass through model
        4. Compute per-row and per-sensor errors
        5. Compute window anomaly score (P95 of row errors)
        6. Diagnose top contributing sensors + ratio-based flags

    Args:
        window: Sensor data, shape (n_rows, n_sensors). Raw (unscaled) by
            default -- see `already_scaled`.
        model: Trained autoencoder in eval mode.
        scaler: Fitted RobustScaler (from training).
        sensor_names: Optional sensor name list.
        top_k: Top K sensors for diagnosis.
        max_null_pct: Max null % per sensor before rejecting.
        max_consecutive_nulls: Max consecutive nulls for forward-fill.
        device: Torch device.
        sensor_baselines: Per-sensor mean MSE from test split for error ratios.
        flag_threshold: Error ratio above which a sensor is flagged.
        anomaly_sensor_pct: % of flagged sensors to declare sensors_anomalous.
        running_sensor_index: Column index for running-equipment check (None = skip).
        running_threshold: Min value for the running sensor.
        already_scaled: Set True if `window` has already been through `scaler`
            (e.g. it came from a cleaned/scaled training split). Skips step 2
            to avoid applying the RobustScaler twice, which distorts every
            reconstruction error. Production callers pass raw windows and
            leave this False so the scaler is applied exactly once, here.
        explainer: Optional trained FastSHAP explainer (see
            src/autoencoder/explain/). When given, top_contributors are
            ranked by Shapley attribution instead of the raw per-sensor MSE
            heuristic. Omit to keep the existing heuristic behaviour.
            Mutually exclusive with `use_integrated_gradients`.
        use_integrated_gradients: When True, rank top_contributors by
            Integrated Gradients attribution (see
            src/autoencoder/explain/integrated_gradients.py) -- a
            training-free alternative to FastSHAP: one Riemann-sum
            integral over the frozen model's gradient, no explainer network
            to train. Mutually exclusive with `explainer`.
        ig_n_steps: Riemann-sum steps for Integrated Gradients (only used
            when `use_integrated_gradients=True`).
        ig_baseline: Baseline value for Integrated Gradients (only used
            when `use_integrated_gradients=True`).

    Returns:
        InferenceResult with score, errors, diagnosis, quality info.
    """
    if explainer is not None and use_integrated_gradients:
        raise ValueError(
            "infer_window: pass either `explainer` (FastSHAP) or "
            "`use_integrated_gradients=True`, not both."
        )

    if device is None:
        device = torch.device("cpu")

    # Step 0: Equipment running check
    if running_sensor_index is not None:
        if not is_equipment_running(window, running_sensor_index, running_threshold):
            return _not_usable_result(window, ["equipment_not_running"])

    # Step 1: Missing data assessment
    quality = assess_window_quality(
        window,
        max_null_pct_per_sensor=max_null_pct,
        max_consecutive_nulls=max_consecutive_nulls,
    )

    if not quality["usable"]:
        return _not_usable_result(window, quality["quality_flags"])

    clean_window = quality["filled_window"]

    # Step 2: Scale
    scaled = clean_window if already_scaled else scaler.transform(clean_window)

    # Step 3: Forward pass
    model.eval()
    x = torch.tensor(scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        x_hat = model(x)

    # Step 4: Errors
    row_errors = per_row_mse(x, x_hat).cpu().numpy()
    sensor_errors = per_sensor_mse(x, x_hat).cpu().numpy()

    # Step 5: Window score
    score = window_anomaly_score(row_errors)

    # Step 6: Diagnosis with optional attribution (FastSHAP or Integrated
    # Gradients) + ratio-based analysis
    attribution_values = None
    attribution_method = "heuristic"
    if explainer is not None:
        attribution_values = explain_window_fastshap(x, model, explainer)
        attribution_method = "fastshap"
    elif use_integrated_gradients:
        attribution_values = integrated_gradients_window(x, model, baseline=ig_baseline, n_steps=ig_n_steps)
        attribution_method = "integrated_gradients"

    diag = diagnose_window(
        x, x_hat,
        sensor_names=sensor_names,
        top_k=top_k,
        sensor_baselines=sensor_baselines,
        flag_threshold=flag_threshold,
        anomaly_sensor_pct=anomaly_sensor_pct,
        attribution_values=attribution_values,
        attribution_method=attribution_method,
    )

    return InferenceResult(
        window_score=score,
        row_errors=row_errors,
        sensor_errors=sensor_errors,
        reconstruction=x_hat.cpu().numpy(),
        top_contributors=diag["top_contributors"],
        quality_flags=quality["quality_flags"],
        usable=True,
        sensor_flags=diag.get("sensor_flags"),
        pct_flagged_sensors=diag.get("pct_flagged_sensors"),
        sensors_anomalous=diag.get("sensors_anomalous"),
        attribution_values=diag.get("attribution_values"),
        attribution_method=diag.get("attribution_method", "heuristic"),
    )
