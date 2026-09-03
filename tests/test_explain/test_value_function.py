"""Tests for the FastSHAP value function (masked reconstruction error)."""

import torch
import pytest

from autoencoder.model.architecture import Autoencoder
from autoencoder.model.loss import per_row_mse
from autoencoder.explain.value_function import masked_row_error, empty_value


@pytest.fixture
def model():
    m = Autoencoder(n_sensors=8, latent_dim=4)
    m.eval()
    return m


class TestMaskedRowError:
    def test_full_mask_matches_direct_error(self, model):
        x = torch.randn(5, 8)
        mask = torch.ones(5, 8)
        v_full = masked_row_error(model, x, mask)
        direct = per_row_mse(x, model(x))
        torch.testing.assert_close(v_full, direct)

    def test_shape(self, model):
        x = torch.randn(6, 8)
        mask = torch.randint(0, 2, (6, 8)).float()
        v = masked_row_error(model, x, mask)
        assert v.shape == (6,)

    def test_no_grad_by_default(self, model):
        x = torch.randn(4, 8, requires_grad=True)
        mask = torch.ones(4, 8)
        v = masked_row_error(model, x, mask)
        assert not v.requires_grad


class TestEmptyValue:
    def test_is_scalar(self, model):
        v0 = empty_value(model, n_sensors=8)
        assert isinstance(v0, float)

    def test_matches_zero_masked_row(self, model):
        x = torch.randn(3, 8)
        mask = torch.zeros(3, 8)
        v_masked = masked_row_error(model, x, mask)
        v0 = empty_value(model, n_sensors=8)
        assert v_masked[0].item() == pytest.approx(v0, rel=1e-5)
