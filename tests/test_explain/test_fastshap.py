"""Tests for the FastSHAP explainer network and training loop."""

import numpy as np
import torch
import pytest

from autoencoder.model.architecture import Autoencoder
from autoencoder.explain.fastshap import (
    FastSHAPExplainer,
    FastSHAPConfig,
    normalize_efficiency,
    train_fastshap_explainer,
)


class TestNormalizeEfficiency:
    def test_efficiency_constraint_satisfied(self):
        raw_phi = torch.randn(10, 6)
        v_full = torch.randn(10)
        v_empty = 0.5
        phi = normalize_efficiency(raw_phi, v_full, v_empty)
        totals = phi.sum(dim=-1)
        expected = v_full - v_empty
        torch.testing.assert_close(totals, expected)


class TestFastSHAPExplainer:
    def test_output_shape(self):
        explainer = FastSHAPExplainer(n_sensors=12, hidden_dim=16)
        x = torch.randn(5, 12)
        out = explainer(x)
        assert out.shape == (5, 12)


@pytest.fixture
def small_data():
    rng = np.random.default_rng(42)
    train = rng.normal(0, 1, (4, 30, 8)).astype(np.float32)
    val = rng.normal(0, 1, (2, 30, 8)).astype(np.float32)
    return train, val


class TestTrainFastshapExplainer:
    def test_returns_trained_explainer(self, small_data):
        train_arr, val_arr = small_data
        model = Autoencoder(n_sensors=8, latent_dim=4)
        model.eval()
        config = FastSHAPConfig(
            n_sensors=8, hidden_dim=16, max_epochs=3, patience=3,
            n_mask_samples=8, batch_size=32, seed=0,
        )
        explainer, history = train_fastshap_explainer(model, train_arr, val_arr, config)
        assert isinstance(explainer, FastSHAPExplainer)
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0

    def test_odd_mask_samples_rejected(self, small_data):
        train_arr, val_arr = small_data
        model = Autoencoder(n_sensors=8, latent_dim=4)
        config = FastSHAPConfig(n_sensors=8, n_mask_samples=7)
        with pytest.raises(ValueError):
            train_fastshap_explainer(model, train_arr, val_arr, config)

    def test_frozen_model_unchanged(self, small_data):
        train_arr, val_arr = small_data
        model = Autoencoder(n_sensors=8, latent_dim=4)
        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        config = FastSHAPConfig(
            n_sensors=8, hidden_dim=16, max_epochs=2, patience=2,
            n_mask_samples=8, batch_size=32, seed=0,
        )
        train_fastshap_explainer(model, train_arr, val_arr, config)
        after = model.state_dict()
        for k in before:
            torch.testing.assert_close(before[k], after[k])

    def test_seeded_runs_are_reproducible(self, small_data):
        train_arr, val_arr = small_data
        config = FastSHAPConfig(
            n_sensors=8, hidden_dim=16, max_epochs=3, patience=3,
            n_mask_samples=8, batch_size=32, seed=123,
        )
        model1 = Autoencoder(n_sensors=8, latent_dim=4)
        model1.eval()
        explainer1, history1 = train_fastshap_explainer(model1, train_arr, val_arr, config)

        model2 = Autoencoder(n_sensors=8, latent_dim=4)
        model2.load_state_dict(model1.state_dict())
        model2.eval()
        explainer2, history2 = train_fastshap_explainer(model2, train_arr, val_arr, config)

        assert history1["train_loss"] == history2["train_loss"]
