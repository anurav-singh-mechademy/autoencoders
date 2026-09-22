"""PyTorch feedforward autoencoder for equipment health monitoring."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


def pick_latent_dim(
    data: np.ndarray,
    variance_threshold: float = 0.95,
    floor: int = 4,
) -> int:
    """Pick a latent dimension from the data's own intrinsic dimensionality.

    Fits PCA on `data` (rows x sensors, already scaled the same way the
    autoencoder will see it) and returns the number of components needed to
    explain `variance_threshold` of the total variance -- a data-driven
    estimate of how many independent modes of variation "normal" behaviour
    actually has, instead of an arbitrary fraction of the sensor count.
    Floored at `floor` so a degenerate PCA result (e.g. heavily correlated
    or near-duplicate sensors collapsing to 1-2 components) doesn't leave
    the autoencoder with barely a bottleneck to learn through.

    Args:
        data: Shape (n_rows, n_sensors), already scaled.
        variance_threshold: Fraction of total variance the chosen
            components must explain (e.g. 0.95).
        floor: Minimum latent dimension regardless of PCA's answer.

    Returns:
        The picked latent dimension.
    """
    n_components = min(data.shape[0], data.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(data)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_needed = int(np.searchsorted(cumulative, variance_threshold)) + 1
    n_needed = min(n_needed, n_components)
    return max(floor, n_needed)


def compute_hidden_widths(n_sensors: int, latent_dim: int, max_hidden_layers: int = 3) -> list[int]:
    """Encoder/decoder hidden-layer widths: halve n_sensors repeatedly until
    reaching latent_dim, capped at max_hidden_layers.

    The funnel's depth is proportional to how much bigger n_sensors is than
    the chosen latent_dim, rather than a fixed number of hardcoded layer
    sizes -- a bottleneck that takes many halvings to reach gets many hidden
    layers; one already close to n_sensors gets none (a direct
    Linear(n_sensors, latent_dim) encoder/decoder). But depth is capped at
    max_hidden_layers regardless of how large that gap is: past the cap, the
    final layer jumps directly to latent_dim instead of continuing to halve.

    This cap exists because unbounded depth is not free: a 476-sensor,
    latent_dim=4 case that halved all the way down produced a 7-hidden-layer
    stack that got stuck in a stable, unmoving bad plateau immediately after
    epoch 1 (46 epochs run, two learning-rate reductions, no improvement) --
    a plain Adam optimizer with no warmup struggles with a stack this deep
    regardless of patience or LR scheduling. A shallower network with a
    larger final compression step (verified: cap=3 leaves equipment with a
    small n_sensors/latent_dim gap, e.g. 72->4 or 19->4, completely
    unaffected -- they were already at or under depth 3) is far more
    reliably trainable in practice.
    """
    widths = []
    w = n_sensors
    while len(widths) < max_hidden_layers:
        w //= 2
        if w <= latent_dim:
            break
        widths.append(w)
    return widths


class Autoencoder(nn.Module):
    """Symmetric feedforward autoencoder.

    Hidden-layer widths and depth come from compute_hidden_widths (see its
    docstring): n_sensors is halved repeatedly until reaching latent_dim, so
    the funnel stays proportional to the actual gap between input size and
    bottleneck size at any n_sensors, rather than a fixed 3-layer pyramid.

    Every hidden layer gets LayerNorm + ReLU + Dropout; the final
    encoder/decoder layers are plain Linear (no activation) since both the
    latent code and reconstructed sensor values can be any range.

    LayerNorm (not BatchNorm) is used because each gradient-update batch is a
    single 30-minute window: its 120 rows are highly autocorrelated
    steady-state samples, so a batch's own mean/variance is a poor, noisy
    stand-in for the sensor's true global spread (in practice ~2 orders of
    magnitude smaller than the global variance). BatchNorm would normalise by
    that tiny per-window variance, injecting unstable scaling into every
    step. LayerNorm normalises across the sensors within a single row
    instead, so it has no dependence on how windows are batched.
    """

    def __init__(
        self,
        n_sensors: int,
        latent_dim: int,
        dropout: float = 0.2,
        max_hidden_layers: int = 3,
        hidden_widths: list[int] | None = None,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.latent_dim = latent_dim
        self.max_hidden_layers = max_hidden_layers
        # hidden_widths overrides the halving funnel (e.g. [n_sensors, n_sensors // 2]) -- a funnel that halves
        # straight down to the bottleneck under-fits relative to PCA with the same latent size (protocol_v2, 2026-09-21).
        widths = list(hidden_widths) if hidden_widths is not None else compute_hidden_widths(n_sensors, latent_dim, max_hidden_layers)
        self.hidden_widths = widths

        encoder_layers: list[nn.Module] = []
        prev = n_sensors
        for w in widths:
            encoder_layers += [nn.Linear(prev, w), nn.LayerNorm(w), nn.ReLU(), nn.Dropout(dropout)]
            prev = w
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for w in reversed(widths):
            decoder_layers += [nn.Linear(prev, w), nn.LayerNorm(w), nn.ReLU(), nn.Dropout(dropout)]
            prev = w
        decoder_layers.append(nn.Linear(prev, n_sensors))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
