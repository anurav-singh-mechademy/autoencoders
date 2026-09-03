"""Hyperparameter search: grid search over key parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product

import numpy as np

from autoencoder.training.trainer import TrainConfig, train_model

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a hyperparameter search."""

    best_config: TrainConfig
    best_val_loss: float
    all_results: list[dict]


def grid_search(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    n_sensors: int,
    latent_dims: list[int] | None = None,
    learning_rates: list[float] | None = None,
    dropouts: list[float] | None = None,
    max_epochs: int = 50,
    patience: int = 10,
) -> SearchResult:
    """Run grid search over hyperparameter combinations.

    Args:
        train_windows: Training data, shape (n_train, 120, n_sensors).
        val_windows: Validation data, shape (n_val, 120, n_sensors).
        n_sensors: Number of sensor columns.
        latent_dims: Latent dimensions to try. Default: [8, 12, 16].
        learning_rates: LRs to try. Default: [1e-3, 5e-4].
        dropouts: Dropout rates to try. Default: [0.1, 0.2].
        max_epochs: Max epochs per trial (shorter for search).
        patience: Early stopping patience per trial.

    Returns:
        SearchResult with best config and all trial results.
    """
    if latent_dims is None:
        latent_dims = [8, 12, 16]
    if learning_rates is None:
        learning_rates = [1e-3, 5e-4]
    if dropouts is None:
        dropouts = [0.1, 0.2]

    all_results = []
    best_val_loss = float("inf")
    best_config = None

    combos = list(product(latent_dims, learning_rates, dropouts))
    logger.info("Grid search: %d combinations", len(combos))

    for i, (ld, lr, do) in enumerate(combos):
        config = TrainConfig(
            n_sensors=n_sensors,
            latent_dim=ld,
            lr=lr,
            dropout=do,
            max_epochs=max_epochs,
            patience=patience,
        )

        logger.info("Trial %d/%d: latent=%d lr=%.4f dropout=%.2f", i + 1, len(combos), ld, lr, do)

        model, history = train_model(train_windows, val_windows, config)
        final_val = history["val_loss"][-1]

        trial = {
            "latent_dim": ld,
            "learning_rate": lr,
            "dropout": do,
            "final_val_loss": final_val,
            "best_val_loss": min(history["val_loss"]),
            "epochs_trained": len(history["val_loss"]),
        }
        all_results.append(trial)

        if final_val < best_val_loss:
            best_val_loss = final_val
            best_config = config

        logger.info("  -> val_loss=%.6f (best so far: %.6f)", final_val, best_val_loss)

    return SearchResult(
        best_config=best_config,
        best_val_loss=best_val_loss,
        all_results=all_results,
    )
