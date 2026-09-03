"""FastSHAP: an amortized explainer network for the autoencoder's anomaly score.

The heuristic in `model.loss.sensor_contributions` ranks sensors by their raw
share of total reconstruction error. Because the autoencoder's bottleneck
mixes every sensor together, sensor i's reconstruction error also depends on
what happened to sensors j != i -- the raw-error split misses that
interaction entirely. FastSHAP (Jethani, Sudarshan, Covert, Lee & Ranganath,
"FastSHAP: Real-Time Shapley Value Estimation", ICLR 2022) fixes this by
training a small explainer network to predict genuine Shapley values -- fit
once against a sampling-based value function, then usable at inference time
with a single forward pass (no per-window sampling).

Training is unsupervised with respect to the *task* (it never sees labels):
it only needs the frozen autoencoder and unlabelled training rows, so it
slots into this pipeline the same way the autoencoder itself does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import per_row_mse
from autoencoder.explain.masking import sample_masks, shapley_kernel_size_probs, apply_mask
from autoencoder.explain.value_function import empty_value

logger = logging.getLogger(__name__)


@dataclass
class FastSHAPConfig:
    """FastSHAP explainer hyperparameters."""

    n_sensors: int
    hidden_dim: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0    # unlike the AE's, L2 reg here fights the input-sensitivity
                                  # FastSHAP needs -- verified to shrink first-layer weights
                                  # toward numerical zero even without the LayerNorm collapse
    max_epochs: int = 100
    patience: int = 10
    lr_factor: float = 0.5
    lr_patience: int = 5
    n_mask_samples: int = 32     # masks per batch per training step -- must be even (paired sampling)
    batch_size: int = 256        # rows per gradient update
    baseline: float = 0.0        # masked-feature fill value (RobustScaler median)
    seed: int | None = None


class FastSHAPExplainer(nn.Module):
    """phi(x) -> raw per-sensor Shapley value estimates, shape (n_sensors,).

    Deliberately does NOT use LayerNorm the way the autoencoder's own
    encoder block does. LayerNorm renormalizes its input to unit variance
    regardless of the scale of the weights feeding it, which removes most of
    the gradient pressure that would otherwise stop an under-used layer's
    weights from shrinking -- empirically, training this network with
    LayerNorm immediately after the first Linear layer reliably collapsed
    that layer's weights to ~1e-38 (float32 underflow), making the whole
    network's output literally independent of the input regardless of
    hyperparameters (confirmed: same collapse with and without weight
    decay). A plain Linear+ReLU+Dropout stack does not exhibit this failure.
    """

    def __init__(self, n_sensors: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_sensors = n_sensors
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_sensors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_efficiency(raw_phi: torch.Tensor, v_full: torch.Tensor, v_empty: float) -> torch.Tensor:
    """Project raw phi onto the efficiency constraint sum_i phi_i = v(full) - v(empty).

    raw_phi: (batch, n_sensors). v_full: (batch,). This is the additive
    correction from the FastSHAP paper (Sec 3.3): it distributes the gap
    between the network's unconstrained sum and the required total equally
    across features, giving an exact efficiency guarantee at every forward
    pass instead of relying on the network to learn it unconstrained.
    """
    n_sensors = raw_phi.shape[-1]
    target = v_full - v_empty
    correction = (target - raw_phi.sum(dim=-1)) / n_sensors
    return raw_phi + correction.unsqueeze(-1)


def _iter_batches(n_rows: int, batch_size: int, rng: np.random.Generator):
    perm = rng.permutation(n_rows)
    for start in range(0, n_rows, batch_size):
        yield perm[start:start + batch_size]


def _fastshap_loss(
    explainer: FastSHAPExplainer,
    model: Autoencoder,
    x: torch.Tensor,
    v_empty: float,
    n_mask_samples: int,
    baseline: float,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    """FastSHAP loss for a batch of rows `x`, shape (batch, n_sensors).

    Samples `n_mask_samples` paired subsets *independently for every row* in
    the batch, computes the true masked value function under the frozen
    autoencoder, predicts the same quantity from the efficiency-projected
    explainer output, and returns the Shapley-kernel-weighted MSE between the
    two.

    Independent per-row masks are essential, not an optimization detail: a
    single mask set shared across the whole batch lets the network satisfy
    the loss by predicting phi_i ~ (v_full - v_empty) / n_sensors for every
    sensor -- a uniform split by subset size that ignores which sensors are
    actually in S -- since every row sees the same (S, v(S)) pair and the
    average is a strong minimizer of that degenerate objective. Per-row
    masks remove the shortcut: the network can only fit v(S) across many
    different S per row by actually learning which sensors drive that row's
    score.
    """
    n_sensors = x.shape[-1]
    batch_size = x.shape[0]

    with torch.no_grad():
        v_full = per_row_mse(x, model(x))  # (batch,)

    raw_phi = explainer(x)  # (batch, n_sensors) -- only tensor requiring grad
    phi = normalize_efficiency(raw_phi, v_full, v_empty)  # (batch, n_sensors)

    total_masks = batch_size * n_mask_samples
    masks_np = sample_masks(n_sensors, total_masks, rng, paired=True).reshape(batch_size, n_mask_samples, n_sensors)
    sizes = masks_np.sum(axis=-1)  # (batch, K)
    size_probs = shapley_kernel_size_probs(n_sensors)
    # Kernel weight pi(S) ~ (n-1) / (k(n-k)); self-normalized over each row's
    # own K samples so the loss scale doesn't depend on n_sensors or K.
    kernel_w = size_probs[sizes.astype(int) - 1]  # (batch, K)
    kernel_w = kernel_w / kernel_w.sum(axis=1, keepdims=True)

    masks = torch.tensor(masks_np, dtype=torch.float32, device=device)  # (batch, K, n_sensors)
    weights = torch.tensor(kernel_w, dtype=torch.float32, device=device)  # (batch, K)

    x_rep = x.unsqueeze(1).expand(batch_size, n_mask_samples, n_sensors).reshape(-1, n_sensors)
    masks_rep = masks.reshape(-1, n_sensors)

    with torch.no_grad():
        x_masked = apply_mask(x_rep, masks_rep, baseline=baseline)
        v_s = per_row_mse(x_masked, model(x_masked)).reshape(batch_size, n_mask_samples)

    phi_exp = phi.unsqueeze(1).expand(batch_size, n_mask_samples, n_sensors)
    pred_v_s = v_empty + (masks * phi_exp).sum(dim=-1)  # (batch, K)

    sq_err = (pred_v_s - v_s) ** 2
    return (sq_err * weights).sum(dim=1).mean()


def train_fastshap_explainer(
    model: Autoencoder,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    config: FastSHAPConfig,
) -> tuple[FastSHAPExplainer, dict]:
    """Train a FastSHAP explainer to amortize per-row Shapley attribution for
    `model`'s reconstruction-error score.

    `model` is frozen throughout -- only the explainer's weights receive
    gradients. Rows are pooled across windows and shuffled, mirroring
    `training.trainer.train_one_epoch`'s row-shuffling rationale (windows are
    highly autocorrelated single-regime runs; shuffling rows is what makes
    training converge on this data).

    Args:
        model: Trained, frozen autoencoder whose score is being explained.
        train_windows: Shape (n_train, window_size, n_sensors), scaled.
        val_windows: Shape (n_val, window_size, n_sensors), scaled.
        config: FastSHAP hyperparameters.

    Returns:
        (trained_explainer, training_history_dict)
    """
    if config.n_mask_samples % 2 != 0:
        raise ValueError("FastSHAPConfig.n_mask_samples must be even (paired sampling)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.seed is not None:
        torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    explainer = FastSHAPExplainer(config.n_sensors, config.hidden_dim, config.dropout).to(device)
    optimizer = Adam(explainer.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=config.lr_factor, patience=config.lr_patience)

    v_empty = empty_value(model, config.n_sensors, device, baseline=config.baseline)
    logger.info("FastSHAP v(empty) = %.6f", v_empty)

    train_rows = train_windows.reshape(-1, config.n_sensors)
    val_rows = val_windows.reshape(-1, config.n_sensors)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(config.max_epochs):
        explainer.train()
        total_loss = 0.0
        n_batches = 0
        for idx in _iter_batches(len(train_rows), config.batch_size, rng):
            x = torch.tensor(train_rows[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = _fastshap_loss(
                explainer, model, x, v_empty, config.n_mask_samples, config.baseline, rng, device,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches

        explainer.eval()
        with torch.no_grad():
            val_losses = []
            for idx in _iter_batches(len(val_rows), config.batch_size, rng):
                x = torch.tensor(val_rows[idx], dtype=torch.float32, device=device)
                val_losses.append(_fastshap_loss(
                    explainer, model, x, v_empty, config.n_mask_samples, config.baseline, rng, device,
                ).item())
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if (epoch + 1) % 10 == 0:
            logger.info(
                "FastSHAP epoch %d: train=%.6f val=%.6f lr=%.6f",
                epoch + 1, train_loss, val_loss, current_lr,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in explainer.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            logger.info("FastSHAP early stopping at epoch %d.", epoch + 1)
            break

    if best_state:
        explainer.load_state_dict(best_state)
    explainer.to(device)
    explainer.eval()

    logger.info("FastSHAP training complete. Best val loss: %.6f", best_val_loss)
    return explainer, history
