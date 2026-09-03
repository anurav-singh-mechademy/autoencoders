"""Integrated Gradients: axiomatic, training-free attribution for the
autoencoder's reconstruction-error score.

Unlike FastSHAP, this needs no separate trained explainer network -- it's a
direct calculus computation on the already-trained, frozen autoencoder:
integrate the score's gradient along a straight-line path from a baseline
(0.0, the RobustScaler's median -- matching the masking baseline used
elsewhere, see explain/value_function.py) to the actual input, then scale by
(input - baseline). Like Shapley values, satisfies a completeness axiom
(attributions sum exactly to score(x) - score(baseline)), but the sum is
computed deterministically via a Riemann-sum integral rather than estimated
from randomly sampled feature subsets -- avoiding the combinatorial sample-
complexity problem that FastSHAP ran into on this dataset's high sensor
count (see explain/fastshap.py's docstring for that history).

Reference: Sundararajan, Taly & Yan, "Axiomatic Attribution for Deep
Networks" (ICML 2017).
"""

from __future__ import annotations

import numpy as np
import torch

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import per_row_mse


def _row_score(model: Autoencoder, x: torch.Tensor) -> torch.Tensor:
    """Scalar-per-row anomaly score being explained: reconstruction MSE."""
    return per_row_mse(x, model(x))


def integrated_gradients(
    x: torch.Tensor,
    model: Autoencoder,
    baseline: float = 0.0,
    n_steps: int = 50,
) -> np.ndarray:
    """Per-row Integrated Gradients attribution, shape (n_rows, n_sensors).

    Args:
        x: Input rows, shape (n_rows, n_sensors), scaled.
        model: Trained, frozen autoencoder.
        baseline: Reference value every sensor is interpolated from.
        n_steps: Riemann-sum steps along the baseline->input path. More
            steps means a more accurate integral at linearly more compute
            (each step is one forward+backward pass through the tiny AE);
            50 is the standard default from the original paper.

    Returns:
        Array of shape (n_rows, n_sensors). Summed along axis=1, each row's
        attributions approximate score(x) - score(baseline) for that row
        (the completeness axiom) -- exact as n_steps -> infinity, approximate
        at finite n_steps via the Riemann sum.
    """
    model.eval()
    device = x.device
    baseline_x = torch.full_like(x, baseline)
    diff = x - baseline_x

    total_grads = torch.zeros_like(x)
    # Midpoint-ish rule: skip alpha=0 (the baseline itself, where every row
    # is identical and score is by definition score(baseline)).
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)[1:]

    for alpha in alphas:
        x_alpha = (baseline_x + alpha * diff).clone().requires_grad_(True)
        # Sum over rows before differentiating: rows are scored independently
        # (no cross-row terms in a feedforward per-row model), so this is
        # equivalent to -- but far cheaper than -- backpropagating each row
        # separately, and gives exactly the per-row gradient for each row.
        score = _row_score(model, x_alpha).sum()
        grad, = torch.autograd.grad(score, x_alpha)
        total_grads += grad

    avg_grads = total_grads / n_steps
    attributions = diff * avg_grads
    return attributions.detach().cpu().numpy()


def integrated_gradients_window(
    x: torch.Tensor,
    model: Autoencoder,
    baseline: float = 0.0,
    n_steps: int = 50,
) -> np.ndarray:
    """Window-level attribution: mean of per-row Integrated Gradients, shape (n_sensors,).

    Averaging mirrors `per_sensor_mse`'s own reduction (mean over the
    window's rows), keeping this on the same per-sensor footing as the
    existing heuristic and FastSHAP's window-level attribution.
    """
    return integrated_gradients(x, model, baseline=baseline, n_steps=n_steps).mean(axis=0)
