"""Inference pipeline: scale → forward pass → score → diagnose a single 30-min window."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import per_row_mse, per_sensor_mse, window_anomaly_score
from autoencoder.data.preprocessing import apply_tail_compression
from autoencoder.inference.missing_data import assess_window_quality
from autoencoder.inference.diagnosis import diagnose_window

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
    error_ratios: np.ndarray | None = None  # continuous error/baseline ratio, shape (n_sensors,)
    sensor_flags: np.ndarray | None = None  # binary (error_ratios > flag_threshold), shape (n_sensors,)
    pct_flagged_sensors: float | None = None
    sensors_anomalous: bool | None = None
    # Null-dominant sensor indices excluded from scoring/diagnosis this window
    # (see assess_window_quality) -- empty when none were masked.
    masked_sensors: list[int] = field(default_factory=list)


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


def _not_usable_result(window: np.ndarray, flags: list[str], masked_sensors: list[int] | None = None) -> InferenceResult:
    """Build an InferenceResult for unusable windows."""
    return InferenceResult(
        window_score=float("nan"),
        row_errors=np.full(window.shape[0], float("nan")),
        sensor_errors=np.full(window.shape[1], float("nan")),
        reconstruction=np.full_like(window, float("nan")),
        top_contributors=[],
        quality_flags=flags,
        usable=False,
        masked_sensors=masked_sensors or [],
    )


def infer_window(
    window: np.ndarray,
    model: Autoencoder,
    scaler,
    sensor_names: list[str] | None = None,
    top_k: int | None = None,
    max_null_pct: float = 5.0,
    max_consecutive_nulls: int = 3,
    max_null_dominant_sensor_pct: float = 30.0,
    device: torch.device | None = None,
    sensor_baselines: np.ndarray | None = None,
    flag_threshold: float = 3.0,
    anomaly_sensor_pct: float = 10.0,
    running_sensor_index: int | None = None,
    running_threshold: float = 0.0,
    already_scaled: bool = False,
    tail_compression_scale: float | None = None,
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
        max_null_pct: Max raw null % per sensor before it's excluded ("masked")
            from scoring/diagnosis in this window, even if its gaps were
            individually fillable.
        max_consecutive_nulls: Max consecutive nulls for forward-fill.
        max_null_dominant_sensor_pct: Max fraction (%) of all sensors allowed
            to be masked before the whole window is rejected instead.
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
        tail_compression_scale: If the model was trained with
            preprocessing.tail_compression_scale set (see
            src/autoencoder/data/preprocessing.py's apply_scaling()), pass
            the same value here so raw windows get the identical c*asinh(z/c)
            transform the thresholds were calibrated under -- omitting it
            silently compares compressed-calibration thresholds against
            uncompressed scores. Ignored when already_scaled=True (the input
            is assumed to already include compression, if any).

    Returns:
        InferenceResult with score, errors, diagnosis, quality info.
    """
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
        max_null_dominant_sensor_pct=max_null_dominant_sensor_pct,
    )

    if not quality["usable"]:
        return _not_usable_result(window, quality["quality_flags"], quality["masked_sensors"])

    clean_window = quality["filled_window"]
    masked_sensors = quality["masked_sensors"]

    if masked_sensors:
        masked_names = [sensor_names[i] if sensor_names else f"sensor_{i}" for i in masked_sensors]
        logger.info(
            "Window has %d null-dominant sensor(s) -- excluded from inference, not "
            "used to score or diagnose this window: %s",
            len(masked_sensors), masked_names,
        )

    # Step 2: Scale
    if already_scaled:
        scaled = clean_window
    else:
        scaled = scaler.transform(clean_window)
        scaled = apply_tail_compression(scaled, tail_compression_scale)

    # Null-dominant sensors may still carry unfillable NaNs (a long enough
    # gap) at this point -- passing those into the model would poison every
    # output via the Linear layers, not just that column. Neutralize them at
    # the RobustScaler's median (== 0 in scaled space, invariant under any
    # asinh tail compression too, so this holds regardless of preprocessing
    # config) instead of the model ever seeing them.
    if masked_sensors:
        scaled = np.array(scaled, dtype=np.float32, copy=True)
        scaled[:, masked_sensors] = 0.0

    # Step 3: Forward pass
    model.eval()
    x = torch.tensor(scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        x_hat = model(x)

    # Step 4: Errors. Masked sensors are excluded from the row-level errors
    # that feed the window score -- their reconstruction reflects an
    # artificial neutral input, not real signal, and would otherwise dilute
    # or inflate every row's error regardless of the equipment's actual state.
    if masked_sensors:
        keep = np.ones(x.shape[1], dtype=bool)
        keep[masked_sensors] = False
        row_errors = per_row_mse(x[:, keep], x_hat[:, keep]).cpu().numpy()
    else:
        row_errors = per_row_mse(x, x_hat).cpu().numpy()
    sensor_errors = per_sensor_mse(x, x_hat).cpu().numpy()

    # Step 5: Window score
    score = window_anomaly_score(row_errors)

    # Step 6: Diagnosis -- per-sensor MSE-share ranking + ratio-based analysis
    diag = diagnose_window(
        x, x_hat,
        sensor_names=sensor_names,
        top_k=top_k,
        sensor_baselines=sensor_baselines,
        flag_threshold=flag_threshold,
        anomaly_sensor_pct=anomaly_sensor_pct,
        masked_sensors=masked_sensors,
    )

    return InferenceResult(
        window_score=score,
        row_errors=row_errors,
        sensor_errors=sensor_errors,
        reconstruction=x_hat.cpu().numpy(),
        top_contributors=diag["top_contributors"],
        quality_flags=quality["quality_flags"],
        masked_sensors=masked_sensors,
        usable=True,
        error_ratios=diag.get("error_ratios"),
        sensor_flags=diag.get("sensor_flags"),
        pct_flagged_sensors=diag.get("pct_flagged_sensors"),
        sensors_anomalous=diag.get("sensors_anomalous"),
    )
