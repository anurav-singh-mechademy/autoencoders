"""Tests for FastSHAP subset masking utilities."""

import numpy as np
import torch
import pytest

from autoencoder.explain.masking import shapley_kernel_size_probs, sample_masks, apply_mask


class TestShapleyKernelSizeProbs:
    def test_sums_to_one(self):
        probs = shapley_kernel_size_probs(10)
        assert probs.sum() == pytest.approx(1.0)

    def test_symmetric_for_small_n(self):
        probs = shapley_kernel_size_probs(6)
        np.testing.assert_allclose(probs, probs[::-1], rtol=1e-6)

    def test_requires_at_least_two_sensors(self):
        with pytest.raises(ValueError):
            shapley_kernel_size_probs(1)


class TestSampleMasks:
    def test_shape(self):
        rng = np.random.default_rng(0)
        masks = sample_masks(20, 16, rng)
        assert masks.shape == (16, 20)

    def test_binary(self):
        rng = np.random.default_rng(0)
        masks = sample_masks(20, 16, rng)
        assert set(np.unique(masks)).issubset({0.0, 1.0})

    def test_odd_n_samples_rejected_when_paired(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            sample_masks(20, 15, rng, paired=True)

    def test_paired_complements(self):
        rng = np.random.default_rng(0)
        masks = sample_masks(10, 8, rng, paired=True)
        for i in range(0, 8, 2):
            np.testing.assert_array_equal(masks[i] + masks[i + 1], np.ones(10))

    def test_no_full_or_empty_subsets(self):
        rng = np.random.default_rng(1)
        masks = sample_masks(10, 200, rng, paired=True)
        sizes = masks.sum(axis=1)
        assert sizes.min() >= 1
        assert sizes.max() <= 9


class TestApplyMask:
    def test_keeps_masked_in(self):
        x = torch.ones(5, 4) * 3.0
        mask = torch.tensor([[1., 0., 1., 0.]] * 5)
        out = apply_mask(x, mask, baseline=0.0)
        assert torch.all(out[:, 0] == 3.0)
        assert torch.all(out[:, 1] == 0.0)

    def test_nonzero_baseline(self):
        x = torch.zeros(3, 2)
        mask = torch.tensor([[0., 0.]] * 3)
        out = apply_mask(x, mask, baseline=5.0)
        assert torch.all(out == 5.0)
