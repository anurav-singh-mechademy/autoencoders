"""Tests for the training loop."""

import numpy as np
import torch
import pytest

from autoencoder.model.architecture import Autoencoder
from autoencoder.training.trainer import train_one_epoch, validate, train_model, TrainConfig


@pytest.fixture
def small_train_data():
    """5 windows of shape (120, 20) for fast tests."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (5, 120, 20)).astype(np.float32)


@pytest.fixture
def small_val_data():
    rng = np.random.default_rng(99)
    return rng.normal(0, 1, (2, 120, 20)).astype(np.float32)


class TestTrainOneEpoch:
    def test_returns_loss(self, small_train_data):
        model = Autoencoder(n_sensors=20, latent_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = train_one_epoch(model, small_train_data, optimizer, torch.device("cpu"))
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self, small_train_data):
        model = Autoencoder(n_sensors=20, latent_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss1 = train_one_epoch(model, small_train_data, optimizer, torch.device("cpu"))
        for _ in range(5):
            train_one_epoch(model, small_train_data, optimizer, torch.device("cpu"))
        loss2 = train_one_epoch(model, small_train_data, optimizer, torch.device("cpu"))
        assert loss2 < loss1


class TestValidate:
    def test_returns_loss(self, small_val_data):
        model = Autoencoder(n_sensors=20, latent_dim=8)
        loss = validate(model, small_val_data, torch.device("cpu"))
        assert isinstance(loss, float)
        assert loss > 0


class TestTrainModel:
    def test_end_to_end(self, small_train_data, small_val_data):
        config = TrainConfig(n_sensors=20, latent_dim=8, max_epochs=10, patience=5)
        model, history = train_model(small_train_data, small_val_data, config)
        assert isinstance(model, Autoencoder)
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0
        # Training should have reduced loss
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_seeded_runs_are_reproducible(self, small_train_data, small_val_data):
        config = TrainConfig(n_sensors=20, latent_dim=8, max_epochs=5, patience=5, seed=123)
        model1, history1 = train_model(small_train_data, small_val_data, config)
        model2, history2 = train_model(small_train_data, small_val_data, config)
        assert history1["train_loss"] == history2["train_loss"]
        for p1, p2 in zip(model1.state_dict().values(), model2.state_dict().values()):
            assert torch.equal(p1, p2)

    def test_unseeded_runs_may_differ(self, small_train_data, small_val_data):
        config = TrainConfig(n_sensors=20, latent_dim=8, max_epochs=5, patience=5, seed=None)
        torch.manual_seed(1)
        _, history1 = train_model(small_train_data, small_val_data, config)
        torch.manual_seed(2)
        _, history2 = train_model(small_train_data, small_val_data, config)
        assert history1["train_loss"] != history2["train_loss"]
