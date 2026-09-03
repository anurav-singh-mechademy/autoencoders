#!/usr/bin/env python
"""Generate synthetic sensor data for testing the autoencoder pipeline.

Creates a CSV with datetime + 160 sensor tags, 8 months of data at 15-sec intervals.
Includes realistic patterns: operating regimes, transients, sensor noise, and anomalies.

Usage:
    python scripts/generate_synthetic_data.py --output data/synthetic_8months.csv
    python scripts/generate_synthetic_data.py --output data/test.csv --tags 40 --months 2
    python scripts/generate_synthetic_data.py --output data/test.parquet --format parquet
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_tags: int = 160,
    months: int = 8,
    freq_seconds: int = 15,
    start_date: str = "2024-01-01",
    anomaly_pct: float = 3.0,
    shutdown_periods: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict]]:
    """Generate realistic synthetic industrial sensor data.

    Patterns included:
        - 3 operating regimes (low/medium/high load) cycling over time
        - Sensor correlations within groups (temperature, pressure, vibration, etc.)
        - Gradual drift in some sensors over months
        - Transient startup/shutdown periods
        - Injected anomaly windows (~anomaly_pct% of data)
        - A few constant/near-constant columns (boolean flags, stuck sensors)

    Args:
        n_tags: Number of sensor tags (columns).
        months: Duration of data in months.
        freq_seconds: Sampling frequency in seconds.
        start_date: Start datetime string.
        anomaly_pct: Approximate % of rows with injected anomalies.
        shutdown_periods: Number of shutdown events to inject.
        seed: Random seed for reproducibility.

    Returns:
        (df, events) where df has 'datetime' + tag_001..tag_N columns, and
        events is a list of dicts describing injected ground truth --
        {"type": "shutdown"|"anomaly", "start_idx", "end_idx", "sensors": [...]}
        (row-index ranges, half-open [start_idx, end_idx)). This is metadata
        for evaluating detection accuracy only -- it is never merged into df
        and must never be fed into cleaning/training/inference.
    """
    rng = np.random.default_rng(seed)

    # Time index
    total_days = months * 30
    total_seconds = total_days * 24 * 3600
    n_rows = total_seconds // freq_seconds
    datetime_index = pd.date_range(start=start_date, periods=n_rows, freq=f"{freq_seconds}s")
    logger.info("Generating %d rows (%d months, %d tags)", n_rows, months, n_tags)

    # Allocate data
    data = np.zeros((n_rows, n_tags), dtype=np.float32)

    # ── Operating regime signal (cycles every ~2 weeks) ──
    regime_period = 14 * 24 * 3600 // freq_seconds  # rows per cycle
    t = np.arange(n_rows)
    regime_signal = np.sin(2 * np.pi * t / regime_period)  # -1 to 1
    regime_level = (regime_signal + 1) / 2  # 0 to 1 (low to high load)

    # ── Define sensor groups ──
    # Distribute tags across realistic sensor types
    n_temp = n_tags // 5          # ~32 temperature sensors
    n_pressure = n_tags // 5      # ~32 pressure sensors
    n_vibration = n_tags // 5     # ~32 vibration sensors
    n_flow = n_tags // 10         # ~16 flow sensors
    n_power = n_tags // 10        # ~16 power/current sensors
    n_other = n_tags - n_temp - n_pressure - n_vibration - n_flow - n_power

    groups = {
        "temp": (0, n_temp),
        "pressure": (n_temp, n_temp + n_pressure),
        "vibration": (n_temp + n_pressure, n_temp + n_pressure + n_vibration),
        "flow": (n_temp + n_pressure + n_vibration, n_temp + n_pressure + n_vibration + n_flow),
        "power": (n_temp + n_pressure + n_vibration + n_flow, n_temp + n_pressure + n_vibration + n_flow + n_power),
        "other": (n_tags - n_other, n_tags),
    }

    # ── Generate base signals per group ──

    # Temperature: 40-120°C, rises with load
    start, end = groups["temp"]
    for i in range(start, end):
        base = 40 + 80 * regime_level + rng.normal(0, 2, n_rows)
        offset = rng.uniform(-10, 10)
        data[:, i] = base + offset + rng.normal(0, 0.5, n_rows)

    # Pressure: 5-50 bar, correlated with load
    start, end = groups["pressure"]
    for i in range(start, end):
        base = 5 + 45 * regime_level + rng.normal(0, 1, n_rows)
        offset = rng.uniform(-5, 5)
        data[:, i] = base + offset + rng.normal(0, 0.3, n_rows)

    # Vibration: 0.1-5 mm/s, increases with load, more noise
    start, end = groups["vibration"]
    for i in range(start, end):
        base = 0.1 + 4.9 * regime_level + rng.normal(0, 0.5, n_rows)
        data[:, i] = np.abs(base + rng.normal(0, 0.2, n_rows))

    # Flow: 100-500 m3/h, follows load
    start, end = groups["flow"]
    for i in range(start, end):
        base = 100 + 400 * regime_level + rng.normal(0, 10, n_rows)
        data[:, i] = base + rng.normal(0, 3, n_rows)

    # Power/Current: 50-300 kW, proportional to load
    start, end = groups["power"]
    for i in range(start, end):
        base = 50 + 250 * regime_level + rng.normal(0, 5, n_rows)
        data[:, i] = base + rng.normal(0, 2, n_rows)

    # Other: misc signals, some constant
    start, end = groups["other"]
    for i in range(start, end):
        if i < start + 3:
            # Boolean-like flags (pump status, valve open/close)
            data[:, i] = rng.choice([0.0, 1.0], size=n_rows, p=[0.1, 0.9])
        elif i < start + 5:
            # Near-constant (stuck sensor or setpoint)
            data[:, i] = 25.0 + rng.normal(0, 1e-6, n_rows)
        else:
            data[:, i] = rng.normal(50, 10, n_rows)

    # ── Gradual drift on a few sensors (simulates degradation) ──
    drift_sensors = rng.choice(n_tags, size=min(10, n_tags), replace=False)
    for s in drift_sensors:
        drift = np.linspace(0, rng.uniform(2, 8), n_rows)
        data[:, s] += drift

    # ── Shutdown periods (motor current → 0, temps drop) ──
    window_size = 120
    n_windows = n_rows // window_size
    events: list[dict] = []
    for _ in range(shutdown_periods):
        shutdown_start = rng.integers(0, n_rows - window_size * 4)
        shutdown_len = rng.integers(window_size * 2, window_size * 4)
        shutdown_end = min(shutdown_start + shutdown_len, n_rows)

        # Power drops to near zero
        pwr_start, pwr_end = groups["power"]
        data[shutdown_start:shutdown_end, pwr_start:pwr_end] = rng.normal(0.5, 0.2, (shutdown_end - shutdown_start, pwr_end - pwr_start))

        # Temps drop gradually
        tmp_start, tmp_end = groups["temp"]
        decay = np.linspace(1.0, 0.3, shutdown_end - shutdown_start).reshape(-1, 1)
        data[shutdown_start:shutdown_end, tmp_start:tmp_end] *= decay

        # Ramp back up (transient)
        ramp_end = min(shutdown_end + window_size, n_rows)
        if ramp_end > shutdown_end:
            ramp = np.linspace(0.3, 1.0, ramp_end - shutdown_end).reshape(-1, 1)
            data[shutdown_end:ramp_end, tmp_start:tmp_end] *= ramp

        events.append({
            "type": "shutdown",
            "start_idx": int(shutdown_start),
            "end_idx": int(ramp_end),
            "sensors": list(range(pwr_start, pwr_end)) + list(range(tmp_start, tmp_end)),
        })

    # ── Inject anomalies ──
    n_anomaly_rows = int(n_rows * anomaly_pct / 100)
    anomaly_starts = rng.choice(n_rows - window_size, size=n_anomaly_rows // window_size, replace=False)
    for a_start in anomaly_starts:
        a_end = a_start + window_size
        # Pick 5-15 random sensors to spike
        n_spike = rng.integers(5, min(16, n_tags))
        spike_sensors = rng.choice(n_tags, size=n_spike, replace=False)
        for s in spike_sensors:
            spike = rng.uniform(3, 8) * np.std(data[:, s])
            data[a_start:a_end, s] += spike * rng.normal(1, 0.3, window_size)

        events.append({
            "type": "anomaly",
            "start_idx": int(a_start),
            "end_idx": int(a_end),
            "sensors": sorted(int(s) for s in spike_sensors),
        })

    # ── Build DataFrame ──
    tag_names = [f"tag_{i+1:03d}" for i in range(n_tags)]
    df = pd.DataFrame(data, columns=tag_names)
    df.insert(0, "datetime", datetime_index)

    logger.info(
        "Generated: %d rows, %d tags, %.1f MB",
        len(df), n_tags, df.memory_usage(deep=True).sum() / 1e6,
    )
    logger.info("Sensor groups: temp=%d, pressure=%d, vibration=%d, flow=%d, power=%d, other=%d",
                n_temp, n_pressure, n_vibration, n_flow, n_power, n_other)
    logger.info("Shutdown periods: %d, Anomaly windows: %d", shutdown_periods, len(anomaly_starts))

    return df, events


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sensor data for autoencoder testing.")
    parser.add_argument("--output", required=True, help="Output file path (.csv or .parquet)")
    parser.add_argument("--tags", type=int, default=160, help="Number of sensor tags (default: 160)")
    parser.add_argument("--months", type=int, default=8, help="Months of data (default: 8)")
    parser.add_argument("--freq", type=int, default=15, help="Sampling frequency in seconds (default: 15)")
    parser.add_argument("--anomaly-pct", type=float, default=3.0, help="Anomaly percentage (default: 3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--format", choices=["csv", "parquet"], default=None,
                        help="Output format (auto-detected from extension)")
    parser.add_argument("--labels-output", default=None,
                        help="Path to save injected-event ground truth JSON "
                             "(default: <output>.events.json). Kept separate "
                             "from the sensor data file -- never fed to cleaning/training.")
    args = parser.parse_args()

    df, events = generate_synthetic_data(
        n_tags=args.tags,
        months=args.months,
        freq_seconds=args.freq,
        anomaly_pct=args.anomaly_pct,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = args.format or ("parquet" if str(output_path).endswith(".parquet") else "csv")

    if fmt == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    logger.info("Saved to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)

    labels_path = Path(args.labels_output) if args.labels_output else output_path.with_suffix(".events.json")
    with open(labels_path, "w") as f:
        json.dump({"n_rows": len(df), "window_size": 120, "events": events}, f, indent=2)
    logger.info("Saved %d ground-truth events to %s", len(events), labels_path)


if __name__ == "__main__":
    main()
