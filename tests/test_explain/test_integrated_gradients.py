"""Tests for Integrated Gradients attribution."""

import numpy as np
import torch
import pytest

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import per_row_mse
from autoencoder.explain.integrated_gradients import integrated_gradients, integrated_gradients_window


@pytest.fixture
def model():
    m = Autoencoder(n_sensors=8, latent_dim=4)
    m.eval()
    return m


def _true_gap(x, model, baseline):
    baseline_x = torch.full_like(x, baseline)
    with torch.no_grad():
        v_full = per_row_mse(x, model(x)).numpy()
        v_baseline = per_row_mse(baseline_x, model(baseline_x)).numpy()
    return v_full - v_baseline


class TestIntegratedGradients:
    def test_shape(self, model):
        x = torch.randn(10, 8)
        attributions = integrated_gradients(x, model, n_steps=10)
        assert attributions.shape == (10, 8)

    def test_completeness_axiom(self, model):
        # atol reflects genuine Riemann-sum discretization error at n_steps=200,
        # not numerical noise -- ReLU introduces gradient discontinuities
        # (kinks) along the baseline->input path that a finite sum can't
        # capture exactly; error shrinks roughly as O(1/n_steps) (verified:
        # ~0.02 mean error at 50 steps, ~0.0002 at 5000), so this is a
        # deliberately loose bound for a fast test, not the method's ceiling.
        x = torch.randn(15, 8)
        attributions = integrated_gradients(x, model, baseline=0.0, n_steps=200)
        gap = _true_gap(x, model, baseline=0.0)
        np.testing.assert_allclose(attributions.sum(axis=1), gap, atol=0.02)

    def test_zero_attribution_at_baseline(self, model):
        x = torch.zeros(5, 8)
        attributions = integrated_gradients(x, model, baseline=0.0, n_steps=10)
        np.testing.assert_allclose(attributions, 0.0, atol=1e-6)

    def test_nonzero_baseline_completeness(self, model):
        x = torch.randn(5, 8)
        attributions = integrated_gradients(x, model, baseline=1.0, n_steps=200)
        gap = _true_gap(x, model, baseline=1.0)
        np.testing.assert_allclose(attributions.sum(axis=1), gap, atol=0.02)

    def test_more_steps_reduces_completeness_error(self, model):
        torch.manual_seed(0)
        x = torch.randn(10, 8)
        gap = _true_gap(x, model, baseline=0.0)

        coarse = integrated_gradients(x, model, n_steps=2)
        fine = integrated_gradients(x, model, n_steps=200)

        coarse_err = np.abs(coarse.sum(axis=1) - gap).mean()
        fine_err = np.abs(fine.sum(axis=1) - gap).mean()
        assert fine_err <= coarse_err


class TestIntegratedGradientsWindow:
    def test_shape(self, model):
        x = torch.randn(120, 8)
        phi = integrated_gradients_window(x, model, n_steps=10)
        assert phi.shape == (8,)

    def test_matches_row_mean(self, model):
        x = torch.randn(20, 8)
        window_phi = integrated_gradients_window(x, model, n_steps=10)
        row_phi = integrated_gradients(x, model, n_steps=10)
        np.testing.assert_allclose(window_phi, row_phi.mean(axis=0), rtol=1e-6)
