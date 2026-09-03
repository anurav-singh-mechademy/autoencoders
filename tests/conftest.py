"""Shared test fixtures: synthetic sensor data, dummy windows, etc."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def n_sensors():
    return 20


@pytest.fixture
def window_size():
    return 120


@pytest.fixture
def sensor_columns(n_sensors):
    return [f"sensor_{i:02d}" for i in range(n_sensors)]


@pytest.fixture
def regime_feature_columns():
    return ["sensor_00", "sensor_01", "sensor_02"]  # proxy for speed, power, pressure


@pytest.fixture
def synthetic_df(sensor_columns, window_size):
    """Create a synthetic DataFrame with 10 complete windows of healthy data."""
    n_windows = 10
    n_rows = n_windows * window_size
    rng = np.random.default_rng(42)

    data = {"timestamp": pd.date_range("2025-01-01", periods=n_rows, freq="15s")}
    for col in sensor_columns:
        data[col] = rng.normal(loc=50.0, scale=5.0, size=n_rows)

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_windows(synthetic_df, sensor_columns, window_size):
    """List of numpy arrays, one per 30-min window."""
    values = synthetic_df[sensor_columns].values
    n_windows = len(values) // window_size
    return [values[i * window_size:(i + 1) * window_size] for i in range(n_windows)]


@pytest.fixture
def windows_with_outliers(synthetic_windows, n_sensors):
    """Synthetic windows with 2 injected outlier windows."""
    rng = np.random.default_rng(99)
    windows = [w.copy() for w in synthetic_windows]

    # Inject outlier in window 2
    windows[2] = rng.normal(loc=200.0, scale=50.0, size=(120, n_sensors))
    # Inject outlier in window 7
    windows[7] = rng.normal(loc=-100.0, scale=50.0, size=(120, n_sensors))

    return windows
