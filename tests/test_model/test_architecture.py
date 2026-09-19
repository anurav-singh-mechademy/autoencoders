"""Tests for autoencoder architecture."""

import numpy as np
import torch
import pytest

from autoencoder.model.architecture import Autoencoder, compute_hidden_widths, pick_latent_dim


class TestComputeHiddenWidths:
    def test_halves_until_reaching_latent_dim(self):
        # 72 -> 36 -> 18 -> 9 -> (9//2=4 <= 4, stop)
        assert compute_hidden_widths(72, 4) == [36, 18, 9]

    def test_stops_one_step_earlier_for_larger_latent_dim(self):
        # Same 72 sensors, but latent_dim=8 means 9 still qualifies (9 > 8)
        # while the next halving (4) would not -- same widths list as
        # latent_dim=4 above, since 9 is > both.
        assert compute_hidden_widths(72, 8) == [36, 18, 9]

    def test_fewer_layers_for_larger_latent_dim(self):
        # latent_dim=20: 36 > 20, but 36//2=18 <= 20, so only one hidden layer.
        assert compute_hidden_widths(72, 20) == [36]

    def test_no_hidden_layers_when_latent_dim_close_to_n_sensors(self):
        # 10 // 2 = 5 <= 8 immediately -- direct Linear(n_sensors, latent_dim).
        assert compute_hidden_widths(10, 8) == []

    def test_small_sensor_count(self):
        assert compute_hidden_widths(20, 4) == [10, 5]

    def test_cap_limits_depth_for_large_gap(self):
        # 476 -> 4 would normally take 6 halvings (238,119,59,29,14,7);
        # capped at 3, it stops after 3 and the final Linear layer jumps
        # directly from 59 to latent_dim=4 instead of continuing to halve.
        assert compute_hidden_widths(476, 4, max_hidden_layers=3) == [238, 119, 59]

    def test_cap_does_not_affect_already_shallow_cases(self):
        # A gap that would naturally stay within the cap is unaffected.
        assert compute_hidden_widths(72, 4, max_hidden_layers=3) == [36, 18, 9]
        assert compute_hidden_widths(19, 4, max_hidden_layers=3) == [9]

    def test_default_cap_is_three(self):
        assert compute_hidden_widths(476, 4) == compute_hidden_widths(476, 4, max_hidden_layers=3)


class TestPickLatentDim:
    def test_recovers_true_rank_of_noiseless_low_rank_data(self):
        # Data constructed as an exact linear mixing of 3 independent
        # gaussian factors into 10 sensor columns has rank 3 -- PCA should
        # need exactly 3 components to explain 95% (in fact ~100%) of
        # variance, regardless of the 10-sensor surface width.
        rng = np.random.default_rng(0)
        latent = rng.normal(size=(2000, 3))
        mixing = rng.normal(size=(3, 10))
        data = latent @ mixing
        assert pick_latent_dim(data, variance_threshold=0.95, floor=1) == 3

    def test_floor_applies_to_degenerate_low_rank_data(self):
        # Rank-1 data floored up to a higher minimum.
        rng = np.random.default_rng(1)
        latent = rng.normal(size=(2000, 1))
        mixing = rng.normal(size=(1, 10))
        data = latent @ mixing
        assert pick_latent_dim(data, variance_threshold=0.95, floor=4) == 4

    def test_never_exceeds_n_sensors(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(50, 6))  # full-rank noise, more rows than cols
        picked = pick_latent_dim(data, variance_threshold=0.999, floor=1)
        assert picked <= 6


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

    def test_hidden_widths_match_compute_hidden_widths(self):
        """Regression test: encoder/decoder hidden widths used to be fixed
        at 128/64/32 regardless of n_sensors -- they must follow
        compute_hidden_widths(n_sensors, latent_dim) exactly."""
        model = Autoencoder(n_sensors=476, latent_dim=32)
        linear_layers = [m for m in model.encoder if isinstance(m, torch.nn.Linear)]
        widths = [layer.out_features for layer in linear_layers]
        assert widths == [*compute_hidden_widths(476, 32), 32]

    def test_no_hidden_layers_when_latent_dim_close_to_n_sensors(self):
        model = Autoencoder(n_sensors=10, latent_dim=8)
        linear_layers = [m for m in model.encoder if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 1  # direct Linear(10, 8), no hidden layers
        x = torch.randn(5, 10)
        assert model(x).shape == (5, 10)

    def test_max_hidden_layers_caps_depth(self):
        model = Autoencoder(n_sensors=476, latent_dim=4, max_hidden_layers=3)
        linear_layers = [m for m in model.encoder if isinstance(m, torch.nn.Linear)]
        widths = [layer.out_features for layer in linear_layers]
        assert widths == [238, 119, 59, 4]  # capped at 3 hidden layers, not 6
        x = torch.randn(5, 476)
        assert model(x).shape == (5, 476)

    def test_max_hidden_layers_defaults_to_three(self):
        default_model = Autoencoder(n_sensors=476, latent_dim=4)
        capped_model = Autoencoder(n_sensors=476, latent_dim=4, max_hidden_layers=3)
        default_widths = [m.out_features for m in default_model.encoder if isinstance(m, torch.nn.Linear)]
        capped_widths = [m.out_features for m in capped_model.encoder if isinstance(m, torch.nn.Linear)]
        assert default_widths == capped_widths

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
