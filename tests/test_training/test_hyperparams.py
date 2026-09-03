"""Tests for hyperparameter search."""

import numpy as np
import pytest

from autoencoder.training.hyperparams import grid_search, SearchResult


class TestGridSearch:
    @pytest.fixture
    def small_data(self):
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, (5, 120, 20)).astype(np.float32)
        val = rng.normal(0, 1, (2, 120, 20)).astype(np.float32)
        return train, val

    def test_basic_search(self, small_data):
        train, val = small_data
        result = grid_search(
            train, val,
            n_sensors=20,
            latent_dims=[8],
            learning_rates=[1e-3],
            dropouts=[0.2],
            max_epochs=5,
            patience=3,
        )
        assert isinstance(result, SearchResult)
        assert result.best_config is not None
        assert result.best_val_loss > 0
        assert len(result.all_results) == 1

    def test_multiple_combos(self, small_data):
        train, val = small_data
        result = grid_search(
            train, val,
            n_sensors=20,
            latent_dims=[8, 12],
            learning_rates=[1e-3],
            dropouts=[0.2],
            max_epochs=3,
            patience=2,
        )
        assert len(result.all_results) == 2
        # Best should have the lowest val loss
        best_from_results = min(r["final_val_loss"] for r in result.all_results)
        assert result.best_val_loss == pytest.approx(best_from_results, abs=0.01)
