"""Row/window-level Shapley attribution via a trained FastSHAP explainer."""

from __future__ import annotations

import numpy as np
import torch

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import per_row_mse
from autoencoder.explain.fastshap import FastSHAPExplainer, normalize_efficiency
from autoencoder.explain.value_function import empty_value


def explain_rows_fastshap(
    x: torch.Tensor,
    model: Autoencoder,
    explainer: FastSHAPExplainer,
    v_empty: float | None = None,
    baseline: float = 0.0,
) -> np.ndarray:
    """Per-row Shapley value estimates, shape (n_rows, n_sensors).

    A single forward pass through the explainer per row -- no masking or
    resampling at explanation time, which is the whole point of FastSHAP:
    amortize the (expensive, sampling-based) attribution cost into training,
    so explaining a live window is as cheap as scoring it.
    """
    model.eval()
    explainer.eval()
    device = x.device
    n_sensors = x.shape[-1]

    if v_empty is None:
        v_empty = empty_value(model, n_sensors, device, baseline=baseline)

    with torch.no_grad():
        v_full = per_row_mse(x, model(x))
        raw_phi = explainer(x)
        phi = normalize_efficiency(raw_phi, v_full, v_empty)

    return phi.cpu().numpy()


def explain_window_fastshap(
    x: torch.Tensor,
    model: Autoencoder,
    explainer: FastSHAPExplainer,
    v_empty: float | None = None,
    baseline: float = 0.0,
) -> np.ndarray:
    """Window-level Shapley attribution: mean of per-row values, shape (n_sensors,).

    Averaging mirrors `per_sensor_mse`'s own reduction (mean over the
    window's rows), so the two stay on a comparable per-sensor footing.
    """
    row_phi = explain_rows_fastshap(x, model, explainer, v_empty=v_empty, baseline=baseline)
    return row_phi.mean(axis=0)
