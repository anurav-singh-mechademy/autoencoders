"""Training loop using Ray Train with window-level gradient accumulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import mse_loss, per_sensor_mse

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    n_sensors: int
    latent_dim: int         # picked via architecture.pick_latent_dim() (PCA-based) by the caller
    max_hidden_layers: int = 3  # caps encoder/decoder depth -- see architecture.compute_hidden_widths
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 150
    patience: int = 15
    lr_factor: float = 0.5
    lr_patience: int = 10
    overfit_ratio_threshold: float = 1.5
    batch_size: int = 256
    seed: int | None = None


def train_one_epoch(
    model: Autoencoder,
    windows: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Train for one epoch with shuffled row-level minibatches.

    Deviates from the tech spec's "one window = one gradient update" recipe
    (Sec 5.2): windows are stored contiguously by regime, so a literal
    window-per-step, window-order-per-epoch loop feeds long unshuffled runs
    of single-regime, highly autocorrelated rows into each step -- verified
    empirically to prevent convergence on this data regardless of model
    normalisation (BatchNorm vs LayerNorm made no difference; both plateaued
    far above a trivial constant-prediction baseline). Shuffling rows across
    all windows before batching fixes this and converges normally.
    """
    model.train()
    rows = windows.reshape(-1, windows.shape[-1])
    n_rows = rows.shape[0]
    perm = np.random.permutation(n_rows)

    total_loss = 0.0
    n_batches = 0
    for start in range(0, n_rows, batch_size):
        idx = perm[start:start + batch_size]
        batch = torch.tensor(rows[idx], dtype=torch.float32, device=device)
        optimizer.zero_grad()
        x_hat = model(batch)
        loss = mse_loss(batch, x_hat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(
    model: Autoencoder,
    windows: np.ndarray,
    device: torch.device,
) -> float:
    """Compute validation loss (no gradient updates)."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in range(len(windows)):
            window = torch.tensor(windows[i], dtype=torch.float32, device=device)
            x_hat = model(window)
            loss = mse_loss(window, x_hat)
            total_loss += loss.item()

    return total_loss / len(windows)


def train_model(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    config: TrainConfig,
) -> tuple[Autoencoder, dict]:
    """Train the autoencoder locally (single process).

    Args:
        train_windows: Shape (n_train, 120, n_sensors).
        val_windows: Shape (n_val, 120, n_sensors).
        config: Training hyperparameters.

    Returns:
        (trained_model, training_history_dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.seed is not None:
        # Covers weight init (below) and every subsequent call to
        # np.random.permutation in train_one_epoch's row shuffle, plus
        # dropout masks -- all draw from these same global RNGs.
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    model = Autoencoder(
        n_sensors=config.n_sensors,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
        max_hidden_layers=config.max_hidden_layers,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=config.lr_factor, patience=config.lr_patience)

    history = {"train_loss": [], "val_loss": [], "val_train_ratio": [], "lr": []}
    best_val_loss = float("inf")
    best_train_loss = None
    best_epoch = None
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(config.max_epochs):
        train_loss = train_one_epoch(model, train_windows, optimizer, device, batch_size=config.batch_size)
        val_loss = validate(model, val_windows, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        val_train_ratio = val_loss / train_loss if train_loss > 0 else float("inf")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_train_ratio"].append(val_train_ratio)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %d: train=%.6f val=%.6f val/train=%.2f lr=%.6f",
                epoch + 1, train_loss, val_loss, val_train_ratio, current_lr,
            )

        if val_train_ratio > config.overfit_ratio_threshold:
            logger.warning(
                "Epoch %d: val/train ratio %.2f exceeds overfit threshold %.2f -- possible overfitting.",
                epoch + 1, val_train_ratio, config.overfit_ratio_threshold,
            )

        if epochs_without_improvement >= config.patience:
            logger.info("Early stopping at epoch %d.", epoch + 1)
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # The returned model's weights are the best_epoch checkpoint, not
    # whatever epoch training happened to be on when patience ran out --
    # history["train_loss"][-1]/["val_loss"][-1] describe an epoch that was
    # discarded. Callers reporting "the trained model's" loss/overfit ratio
    # should use these, not the last entries of the per-epoch lists above.
    history["best_epoch"] = best_epoch
    history["best_train_loss"] = best_train_loss
    history["best_val_loss"] = best_val_loss
    history["best_val_train_ratio"] = (
        best_val_loss / best_train_loss if best_train_loss else float("inf")
    )

    logger.info("Training complete. Best val loss: %.6f", best_val_loss)
    return model, history


def compute_sensor_baselines(
    model: Autoencoder,
    test_windows: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute per-sensor mean MSE on test windows as baseline reference.

    This baseline is used at inference time to compute error ratios:
    ratio = live_sensor_error / baseline_sensor_error, giving a normalized
    "how much worse than normal" score per sensor.

    Args:
        model: Trained autoencoder (will be set to eval mode).
        test_windows: Shape (n_windows, 120, n_sensors).
        device: Torch device.

    Returns:
        Array of shape (n_sensors,) with mean per-sensor MSE across test windows.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    all_sensor_errors = []

    with torch.no_grad():
        for i in range(len(test_windows)):
            x = torch.tensor(test_windows[i], dtype=torch.float32, device=device)
            x_hat = model(x)
            sensor_err = per_sensor_mse(x, x_hat).cpu().numpy()
            all_sensor_errors.append(sensor_err)

    baselines = np.mean(all_sensor_errors, axis=0)
    logger.info("Computed sensor baselines: mean=%.6f, shape=%s", baselines.mean(), baselines.shape)
    return baselines


def _ray_train_fn(config: dict):
    """Ray Train worker function."""
    train_windows = config["train_windows"]
    val_windows = config["val_windows"]
    train_config = config["train_config"]

    device = ray.train.torch.get_device()

    model = Autoencoder(
        n_sensors=train_config.n_sensors,
        latent_dim=train_config.latent_dim,
        dropout=train_config.dropout,
        max_hidden_layers=train_config.max_hidden_layers,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=train_config.lr_factor, patience=train_config.lr_patience)

    best_val_loss = float("inf")

    for epoch in range(train_config.max_epochs):
        train_loss = train_one_epoch(model, train_windows, optimizer, device, batch_size=train_config.batch_size)
        val_loss = validate(model, val_windows, device)
        scheduler.step(val_loss)

        val_train_ratio = val_loss / train_loss if train_loss > 0 else float("inf")
        ray.train.report({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_train_ratio": val_train_ratio,
            "epoch": epoch + 1,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss


def train_model_distributed(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    config: TrainConfig,
    num_workers: int = 1,
    use_gpu: bool = False,
) -> ray.train.Result:
    """Train using Ray Train for distributed/scalable execution.

    Args:
        train_windows: Shape (n_train, 120, n_sensors).
        val_windows: Shape (n_val, 120, n_sensors).
        config: Training hyperparameters.
        num_workers: Number of Ray Train workers.
        use_gpu: Whether to use GPU workers.

    Returns:
        ray.train.Result with metrics and checkpoint.
    """
    trainer = TorchTrainer(
        train_loop_per_worker=_ray_train_fn,
        train_loop_config={
            "train_windows": train_windows,
            "val_windows": val_windows,
            "train_config": config,
        },
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    )

    result = trainer.fit()
    logger.info("Ray Train complete. Best result: %s", result.metrics)
    return result
