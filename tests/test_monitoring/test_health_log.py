"""Tests for weekly health logging."""

import json
import numpy as np
import pytest
from pathlib import Path

from autoencoder.monitoring.health_log import compute_weekly_metrics, log_weekly_health, read_health_log


class TestComputeWeeklyMetrics:
    def test_basic(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        zones = ["green", "green", "green", "yellow", "yellow", "red", "red"]
        m = compute_weekly_metrics(scores, zones)
        assert m["mean_score"] == pytest.approx(4.0)
        assert m["max_score"] == 7.0
        assert m["min_score"] == 1.0
        assert m["zone_counts"]["green"] == 3
        assert m["zone_counts"]["yellow"] == 2
        assert m["zone_counts"]["red"] == 2
        assert m["n_days"] == 7

    def test_alert_rate(self):
        scores = [1.0, 2.0]
        zones = ["green", "yellow"]
        m = compute_weekly_metrics(scores, zones)
        assert m["alert_rate"] == pytest.approx(0.5)

    def test_all_green(self):
        scores = [1.0] * 7
        zones = ["green"] * 7
        m = compute_weekly_metrics(scores, zones)
        assert m["alert_rate"] == 0.0


class TestLogAndReadHealth:
    def test_log_and_read(self, tmp_path):
        metrics = compute_weekly_metrics([1.0, 2.0], ["green", "yellow"])
        log_path = log_weekly_health("pump_01", "2025-01-06", metrics, tmp_path)
        assert log_path.exists()

        entries = read_health_log(log_path)
        assert len(entries) == 1
        assert entries[0]["equipment_id"] == "pump_01"
        assert entries[0]["week_start"] == "2025-01-06"

    def test_appends(self, tmp_path):
        metrics = compute_weekly_metrics([1.0], ["green"])
        log_weekly_health("pump_01", "2025-01-06", metrics, tmp_path)
        log_weekly_health("pump_01", "2025-01-13", metrics, tmp_path)
        entries = read_health_log(tmp_path / "pump_01_health.jsonl")
        assert len(entries) == 2

    def test_read_nonexistent(self, tmp_path):
        entries = read_health_log(tmp_path / "nonexistent.jsonl")
        assert entries == []
