"""Tests for loss functions and error computation."""

import numpy as np
import torch
import pytest

from autoencoder.model.loss import (
    mse_loss,
    per_row_mse,
    per_sensor_mse,
    window_anomaly_score,
    sensor_contributions,
)


class TestMSELoss:
    def test_zero_error(self):
        x = torch.randn(120, 20)
        assert mse_loss(x, x).item() == pytest.approx(0.0)

    def test_positive_error(self):
        x = torch.randn(120, 20)
        x_hat = x + 1.0
        assert mse_loss(x, x_hat).item() > 0


class TestPerRowMSE:
    def test_shape(self):
        x = torch.randn(120, 20)
        x_hat = torch.randn(120, 20)
        errors = per_row_mse(x, x_hat)
        assert errors.shape == (120,)

    def test_zero_for_identical(self):
        x = torch.randn(120, 20)
        errors = per_row_mse(x, x)
        assert torch.all(errors == 0)


class TestPerSensorMSE:
    def test_shape(self):
        x = torch.randn(120, 20)
        x_hat = torch.randn(120, 20)
        errors = per_sensor_mse(x, x_hat)
        assert errors.shape == (20,)


class TestWindowAnomalyScore:
    def test_basic(self):
        errors = np.random.rand(120)
        score = window_anomaly_score(errors)
        assert 0 <= score <= 1

    def test_with_tensor(self):
        errors = torch.rand(120)
        score = window_anomaly_score(errors)
        assert isinstance(score, float)

    def test_p95_is_high(self):
        # 90 low values + 30 high values => p95 well within the high region
        errors = np.concatenate([np.zeros(90), np.ones(30) * 10.0])
        score = window_anomaly_score(errors)
        assert score >= 9.0


class TestSensorContributions:
    def test_basic(self):
        errors = np.array([10.0, 5.0, 3.0, 2.0, 1.0])
        result = sensor_contributions(errors, top_k=3)
        assert len(result) == 3
        assert result[0][0] == 0  # sensor 0 has highest error
        assert sum(pct for _, pct in result) < 100.1

    def test_all_zero(self):
        errors = np.zeros(10)
        result = sensor_contributions(errors)
        assert result == []

    def test_default_top_k(self):
        errors = np.random.rand(100)
        result = sensor_contributions(errors)
        assert len(result) == 10  # min(10, 100//5) = 10
