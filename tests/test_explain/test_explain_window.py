"""Tests for per-row/window Shapley attribution via a trained FastSHAP explainer."""

import numpy as np
import torch
import pytest

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import per_row_mse
from autoencoder.explain.fastshap import FastSHAPExplainer
from autoencoder.explain.value_function import empty_value
from autoencoder.explain.explain_window import explain_rows_fastshap, explain_window_fastshap


@pytest.fixture
def model():
    m = Autoencoder(n_sensors=6, latent_dim=4)
    m.eval()
    return m


@pytest.fixture
def explainer():
    e = FastSHAPExplainer(n_sensors=6, hidden_dim=8)
    e.eval()
    return e


class TestExplainRowsFastshap:
    def test_shape(self, model, explainer):
        x = torch.randn(10, 6)
        phi = explain_rows_fastshap(x, model, explainer)
        assert phi.shape == (10, 6)

    def test_efficiency_property_holds_per_row(self, model, explainer):
        x = torch.randn(7, 6)
        phi = explain_rows_fastshap(x, model, explainer)
        v_full = per_row_mse(x, model(x)).detach().numpy()
        v_empty = empty_value(model, 6)
        np.testing.assert_allclose(phi.sum(axis=1), v_full - v_empty, rtol=1e-4, atol=1e-5)


class TestExplainWindowFastshap:
    def test_shape(self, model, explainer):
        x = torch.randn(120, 6)
        phi = explain_window_fastshap(x, model, explainer)
        assert phi.shape == (6,)

    def test_matches_row_mean(self, model, explainer):
        x = torch.randn(15, 6)
        window_phi = explain_window_fastshap(x, model, explainer)
        row_phi = explain_rows_fastshap(x, model, explainer)
        np.testing.assert_allclose(window_phi, row_phi.mean(axis=0), rtol=1e-6)
