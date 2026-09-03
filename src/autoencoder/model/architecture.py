"""PyTorch feedforward autoencoder for equipment health monitoring."""

from __future__ import annotations

import torch
import torch.nn as nn


def get_latent_dim(n_sensors: int) -> int:
    """Compute recommended latent dimension: max(8, n_sensors // 8)."""
    return max(8, n_sensors // 8)


class Autoencoder(nn.Module):
    """Symmetric feedforward autoencoder.

    Architecture (from tech spec):
        Input(N) -> 128 -> 64 -> 32 -> L (latent) -> 32 -> 64 -> 128 -> N

    LayerNorm on encoder input layers, dropout on FC-1 and FC-2.
    Linear output (no activation) since sensor values can be any range.

    LayerNorm (not BatchNorm) is used because each gradient-update batch is a
    single 30-minute window: its 120 rows are highly autocorrelated
    steady-state samples, so a batch's own mean/variance is a poor,
    noisy stand-in for the sensor's true global spread (in practice ~2
    orders of magnitude smaller than the global variance). BatchNorm would
    normalise by that tiny per-window variance, injecting unstable scaling
    into every step. LayerNorm normalises across the 158 sensors within a
    single row instead, so it has no dependence on how windows are batched.
    """

    def __init__(self, n_sensors: int, latent_dim: int | None = None, dropout: float = 0.2):
        super().__init__()
        self.n_sensors = n_sensors
        self.latent_dim = latent_dim or get_latent_dim(n_sensors)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_sensors, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_sensors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
