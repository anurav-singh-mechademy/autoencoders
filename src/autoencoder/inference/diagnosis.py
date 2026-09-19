"""Per-sensor diagnosis: identify which sensors drive the anomaly score."""

from __future__ import annotations

import numpy as np
import torch

from autoencoder.model.loss import per_sensor_mse, sensor_contributions


def diagnose_window(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    sensor_names: list[str] | None = None,
    top_k: int | None = None,
    sensor_baselines: np.ndarray | None = None,
    flag_threshold: float = 3.0,
    anomaly_sensor_pct: float = 10.0,
    masked_sensors: list[int] | None = None,
) -> dict:
    """Diagnose which sensors are contributing most to reconstruction error.

    Ranks sensors by their share of total per-sensor MSE reconstruction error.

    Args:
        x: Original window, shape (n_rows, n_sensors).
        x_hat: Reconstructed window, same shape.
        sensor_names: Optional list of sensor names for labelling.
        top_k: Number of top contributors to return.
        sensor_baselines: Per-sensor mean MSE from test split, shape (n_sensors,).
            When provided, computes normalized error ratios and binary flags.
        flag_threshold: Error ratio above which a sensor is flagged anomalous.
        anomaly_sensor_pct: % of flagged sensors to declare whole window sensor-anomalous.
        masked_sensors: Indices excluded from scoring (see
            assess_window_quality) -- their reconstruction reflects a
            neutral imputed input, not real signal, so they're zeroed out
            of total_error/contribution ranking and forced un-flagged
            rather than let them dominate or spuriously trip either.

    Returns:
        Dict with:
            sensor_errors: per-sensor MSE array
            top_contributors: list of (index, name, error, pct)
            total_error: scalar total
        When sensor_baselines provided, also includes:
            error_ratios: per-sensor error / baseline ratio
            sensor_flags: binary 0/1 array (1 = flagged)
            pct_flagged_sensors: % of sensors flagged
            sensors_anomalous: bool — True if pct_flagged > anomaly_sensor_pct
    """
    sensor_errors = per_sensor_mse(x, x_hat).detach().cpu().numpy()

    # Masked (null-dominant) sensors reconstruct an artificial neutral input,
    # not real signal -- zero their error out before ranking so they can't be
    # reported as a top contributor or skew total_error.
    ranking_errors = sensor_errors
    if masked_sensors:
        ranking_errors = sensor_errors.copy()
        ranking_errors[masked_sensors] = 0.0

    total_error = float(ranking_errors.sum())

    contributions = sensor_contributions(ranking_errors, top_k=top_k)

    top_list = []
    for idx, pct in contributions:
        name = sensor_names[idx] if sensor_names else f"sensor_{idx}"
        top_list.append({
            "index": idx,
            "name": name,
            "error": float(sensor_errors[idx]),
            "contribution_pct": float(pct),
        })

    result = {
        "sensor_errors": sensor_errors,
        "top_contributors": top_list,
        "total_error": total_error,
    }

    if sensor_baselines is not None:
        safe_baselines = np.where(sensor_baselines > 0, sensor_baselines, 1e-10)
        error_ratios = sensor_errors / safe_baselines
        sensor_flags = (error_ratios > flag_threshold).astype(int)

        # Masked sensors compare a neutral imputed value against a baseline
        # computed from real readings -- that ratio is meaningless, so force
        # them un-flagged and drop them from the flagged-% denominator too
        # rather than let them spuriously trip (or mask) sensors_anomalous.
        n_scored = len(sensor_flags)
        if masked_sensors:
            sensor_flags = sensor_flags.copy()
            sensor_flags[masked_sensors] = 0
            n_scored = len(sensor_flags) - len(masked_sensors)
        pct_flagged = float(sensor_flags.sum() / n_scored * 100) if n_scored else 0.0

        result["error_ratios"] = error_ratios
        result["sensor_flags"] = sensor_flags
        result["pct_flagged_sensors"] = pct_flagged
        result["sensors_anomalous"] = pct_flagged > anomaly_sensor_pct

    return result
