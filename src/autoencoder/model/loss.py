"""Loss functions and error computation for the autoencoder."""

from __future__ import annotations

import torch
import numpy as np


def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean squared error between input and reconstruction."""
    return torch.mean((x - x_hat) ** 2)


def per_row_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """MSE per row (per 15-second reading). Shape: (n_rows,)."""
    return torch.mean((x - x_hat) ** 2, dim=1)


def per_sensor_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """MSE per sensor across all rows. Shape: (n_sensors,)."""
    return torch.mean((x - x_hat) ** 2, dim=0)


def window_anomaly_score(row_errors: torch.Tensor | np.ndarray) -> float:
    """Aggregate 120 per-row errors into a single window score using 95th percentile."""
    if isinstance(row_errors, torch.Tensor):
        row_errors = row_errors.detach().cpu().numpy()
    return float(np.percentile(row_errors, 95))


def sensor_contributions(sensor_errors: np.ndarray, top_k: int | None = None) -> list[tuple[int, float]]:
    """Rank sensors by their contribution to the total anomaly score.

    Args:
        sensor_errors: Array of shape (n_sensors,) with mean error per sensor.
        top_k: Number of top sensors to return. Default: min(10, n_sensors // 5).

    Returns:
        List of (sensor_index, contribution_pct) sorted descending.
    """
    total = sensor_errors.sum()
    if total == 0:
        return []

    n_sensors = len(sensor_errors)
    if top_k is None:
        top_k = min(10, max(1, n_sensors // 5))

    pcts = (sensor_errors / total) * 100
    ranked = sorted(enumerate(pcts), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
