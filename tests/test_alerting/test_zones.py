"""Tests for zone classification."""

import pytest

from autoencoder.alerting.zones import classify_zone, classify_batch


@pytest.fixture
def thresholds():
    return {"green_yellow": 5.0, "yellow_red": 10.0}


class TestClassifyZone:
    def test_green(self, thresholds):
        assert classify_zone(2.0, thresholds) == "green"

    def test_yellow(self, thresholds):
        assert classify_zone(7.0, thresholds) == "yellow"

    def test_red(self, thresholds):
        assert classify_zone(15.0, thresholds) == "red"

    def test_boundary_green_yellow(self, thresholds):
        assert classify_zone(5.0, thresholds) == "yellow"

    def test_boundary_yellow_red(self, thresholds):
        assert classify_zone(10.0, thresholds) == "red"


class TestClassifyBatch:
    def test_batch(self, thresholds):
        scores = [1.0, 6.0, 12.0, 4.9, 10.0]
        zones = classify_batch(scores, thresholds)
        assert zones == ["green", "yellow", "red", "green", "red"]
