"""Subset masking utilities for Shapley-value estimation (FastSHAP)."""

from __future__ import annotations

import numpy as np
import torch


def shapley_kernel_size_probs(n_sensors: int) -> np.ndarray:
    """P(|S|=k) for k=1..n_sensors-1, proportional to (n-1) / (k * (n-k)).

    This is the subset-size distribution implied by the Shapley kernel
    (Lundberg & Lee, 2017): sampling a size from this distribution and then a
    uniform-random subset of that size is equivalent to sampling directly
    from the (otherwise unnormalizable) Shapley kernel weighting over all
    2^n_sensors subsets.
    """
    if n_sensors < 2:
        raise ValueError("shapley_kernel_size_probs requires n_sensors >= 2")
    k = np.arange(1, n_sensors)
    weights = (n_sensors - 1) / (k * (n_sensors - k))
    return weights / weights.sum()


def sample_masks(
    n_sensors: int,
    n_samples: int,
    rng: np.random.Generator,
    paired: bool = True,
) -> np.ndarray:
    """Sample `n_samples` binary masks (1=keep, 0=masked-out), shape (n_samples, n_sensors).

    Sizes are drawn from the Shapley kernel size distribution; each mask is a
    uniform-random subset of its sampled size. With `paired=True` (the
    default), every drawn mask is immediately followed by its complement --
    the standard FastSHAP variance-reduction trick -- so `n_samples` must be
    even in that case.
    """
    if paired and n_samples % 2 != 0:
        raise ValueError("n_samples must be even when paired=True")

    size_probs = shapley_kernel_size_probs(n_sensors)
    sizes_all = np.arange(1, n_sensors)
    n_draws = n_samples // 2 if paired else n_samples
    sizes = rng.choice(sizes_all, size=n_draws, p=size_probs)

    # Vectorized uniform-random subset of each drawn size: give every sensor
    # an i.i.d. random score per draw, find each row's `size`-th smallest
    # score by sorting once, and keep everything at or below that threshold
    # -- equivalent in distribution to sampling `size` indices without
    # replacement (ties among continuous random scores are ~never exact).
    # This one-sort-per-batch approach replaces a Python-level loop of one
    # rng.choice call per draw, which dominates wall-clock time once n_draws
    # reaches into the thousands per training step -- as it does once
    # FastSHAP samples masks independently per row instead of once per batch
    # (see `fastshap._fastshap_loss`). An earlier rank-via-double-argsort
    # attempt was actually slower than the loop it replaced; a single sort
    # plus a per-row threshold comparison is the one that wins.
    scores = rng.random((n_draws, n_sensors))
    sorted_scores = np.sort(scores, axis=1)
    threshold = sorted_scores[np.arange(n_draws), sizes - 1]
    keep = (scores <= threshold[:, None]).astype(np.float32)

    if not paired:
        return keep

    masks = np.empty((n_samples, n_sensors), dtype=np.float32)
    masks[0::2] = keep
    masks[1::2] = 1.0 - keep
    return masks


def apply_mask(x: torch.Tensor, mask: torch.Tensor, baseline: float = 0.0) -> torch.Tensor:
    """Replace masked-out (mask==0) features with `baseline`. x, mask: same shape."""
    return mask * x + (1.0 - mask) * baseline
