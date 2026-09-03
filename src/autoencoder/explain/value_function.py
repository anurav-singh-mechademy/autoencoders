"""Value function v(S) for FastSHAP: masked reconstruction error under a frozen autoencoder.

v(S) is defined as the reconstruction error the autoencoder produces when
only the sensors in subset S are given their real values and every other
sensor is replaced by a fixed baseline (0.0 -- the RobustScaler median, so
"masked" coincides with the scaled data's own center). This is a
deterministic single-reference value function (as opposed to marginalizing
over the training distribution), chosen because it reuses the scaling
convention already in place everywhere else in this codebase and needs no
extra sampling machinery to evaluate.
"""

from __future__ import annotations

import torch

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import per_row_mse
from autoencoder.explain.masking import apply_mask


@torch.no_grad()
def masked_row_error(
    model: Autoencoder,
    x: torch.Tensor,
    mask: torch.Tensor,
    baseline: float = 0.0,
) -> torch.Tensor:
    """v(S) for each row: reconstruction error of `model` fed x with mask==0
    features replaced by `baseline`. x, mask: shape (batch, n_sensors)."""
    model.eval()
    x_masked = apply_mask(x, mask, baseline=baseline)
    x_hat = model(x_masked)
    return per_row_mse(x_masked, x_hat)


@torch.no_grad()
def empty_value(
    model: Autoencoder,
    n_sensors: int,
    device: torch.device | None = None,
    baseline: float = 0.0,
) -> float:
    """v(∅): reconstruction error when every feature is masked to baseline.

    Independent of x by construction -- a fully-masked row is always the
    same baseline-filled vector regardless of what x was -- so this is one
    scalar shared across every row the explainer ever scores, computed once
    from the frozen model.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    x0 = torch.full((1, n_sensors), baseline, dtype=torch.float32, device=device)
    x_hat = model(x0)
    return per_row_mse(x0, x_hat).item()
