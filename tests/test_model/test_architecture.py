"""Tests for autoencoder architecture."""

import torch
import pytest

from autoencoder.model.architecture import Autoencoder, get_latent_dim


class TestGetLatentDim:
    def test_small_sensor_count(self):
        assert get_latent_dim(20) == 8  # 20//8=2, max(8,2)=8

    def test_large_sensor_count(self):
        assert get_latent_dim(120) == 15  # 120//8=15

    def test_minimum(self):
        assert get_latent_dim(5) == 8


class TestAutoencoder:
    def test_output_shape(self):
        model = Autoencoder(n_sensors=50, latent_dim=12)
        x = torch.randn(120, 50)
        out = model(x)
        assert out.shape == (120, 50)

    def test_encode_shape(self):
        model = Autoencoder(n_sensors=50, latent_dim=12)
        x = torch.randn(120, 50)
        latent = model.encode(x)
        assert latent.shape == (120, 12)

    def test_default_latent_dim(self):
        model = Autoencoder(n_sensors=80)
        assert model.latent_dim == 10  # max(8, 80//8)

    def test_reconstruction_improves(self):
        """After a few training steps, reconstruction error should decrease."""
        model = Autoencoder(n_sensors=20, latent_dim=8)
        x = torch.randn(120, 20)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        initial_loss = torch.mean((x - model(x)) ** 2).item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = torch.mean((x - model(x)) ** 2)
            loss.backward()
            optimizer.step()

        final_loss = torch.mean((x - model(x)) ** 2).item()
        assert final_loss < initial_loss

    def test_eval_mode_no_dropout(self):
        model = Autoencoder(n_sensors=20, latent_dim=8, dropout=0.5)
        x = torch.randn(120, 20)
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        # In eval mode, outputs should be deterministic
        assert torch.allclose(out1, out2)
