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
    attribution_values: np.ndarray | None = None,
    attribution_method: str = "heuristic",
) -> dict:
    """Diagnose which sensors are contributing most to reconstruction error.

    Args:
        x: Original window, shape (n_rows, n_sensors).
        x_hat: Reconstructed window, same shape.
        sensor_names: Optional list of sensor names for labelling.
        top_k: Number of top contributors to return.
        sensor_baselines: Per-sensor mean MSE from test split, shape (n_sensors,).
            When provided, computes normalized error ratios and binary flags.
        flag_threshold: Error ratio above which a sensor is flagged anomalous.
        anomaly_sensor_pct: % of flagged sensors to declare whole window sensor-anomalous.
        attribution_values: Optional per-sensor attribution from an
            axiomatic method (FastSHAP or Integrated Gradients, see
            src/autoencoder/explain/), shape (n_sensors,). When provided,
            top_contributors are ranked by |attribution_value| instead of
            raw reconstruction-error share -- this captures cross-sensor
            interaction effects (the AE's bottleneck mixes every sensor
            together) that the raw MSE split misses. sensor_errors/
            total_error are always computed from the actual reconstruction
            regardless.
        attribution_method: Label for what produced `attribution_values`
            (e.g. "fastshap", "integrated_gradients"). Ignored when
            `attribution_values` is None -- the result always reports
            "heuristic" in that case.

    Returns:
        Dict with:
            sensor_errors: per-sensor MSE array
            top_contributors: list of (index, name, error, pct[, attribution_value])
            total_error: scalar total
            attribution_method: "heuristic" or whatever was passed in
        When attribution_values provided, also includes:
            attribution_values: the input array, passed through for convenience
        When sensor_baselines provided, also includes:
            error_ratios: per-sensor error / baseline ratio
            sensor_flags: binary 0/1 array (1 = flagged)
            pct_flagged_sensors: % of sensors flagged
            sensors_anomalous: bool — True if pct_flagged > anomaly_sensor_pct
    """
    sensor_errors = per_sensor_mse(x, x_hat).detach().cpu().numpy()
    total_error = float(sensor_errors.sum())

    if attribution_values is not None:
        contributions = sensor_contributions(np.abs(attribution_values), top_k=top_k)
        method = attribution_method
    else:
        contributions = sensor_contributions(sensor_errors, top_k=top_k)
        method = "heuristic"

    top_list = []
    for idx, pct in contributions:
        name = sensor_names[idx] if sensor_names else f"sensor_{idx}"
        entry = {
            "index": idx,
            "name": name,
            "error": float(sensor_errors[idx]),
            "contribution_pct": float(pct),
        }
        if attribution_values is not None:
            entry["attribution_value"] = float(attribution_values[idx])
        top_list.append(entry)

    result = {
        "sensor_errors": sensor_errors,
        "top_contributors": top_list,
        "total_error": total_error,
        "attribution_method": method,
    }
    if attribution_values is not None:
        result["attribution_values"] = attribution_values

    if sensor_baselines is not None:
        safe_baselines = np.where(sensor_baselines > 0, sensor_baselines, 1e-10)
        error_ratios = sensor_errors / safe_baselines
        sensor_flags = (error_ratios > flag_threshold).astype(int)
        pct_flagged = float(sensor_flags.sum() / len(sensor_flags) * 100)

        result["error_ratios"] = error_ratios
        result["sensor_flags"] = sensor_flags
        result["pct_flagged_sensors"] = pct_flagged
        result["sensors_anomalous"] = pct_flagged > anomaly_sensor_pct

    return result
