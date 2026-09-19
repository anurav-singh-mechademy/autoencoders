"""Tests for persistence rules."""

import pytest

from autoencoder.alerting.persistence import apply_persistence, compute_alert_level_timeline


class TestComputeAlertLevelTimeline:
    def test_matches_apply_persistence_at_every_step(self):
        # The streaming version must agree with re-running apply_persistence
        # on every growing prefix -- that's the exact behavior it's meant to
        # replicate at O(n) instead of O(n^2).
        zones = ["green", "red", "red", "red", "green", "yellow", "yellow", "yellow", "yellow"]
        streaming = compute_alert_level_timeline(zones)
        expected = [apply_persistence(zones[: i + 1])["alert_level"] for i in range(len(zones))]
        assert streaming == expected

    def test_empty_timeline(self):
        assert compute_alert_level_timeline([]) == []

    def test_mixed_window_rule_matches_apply_persistence(self):
        zones = ["red", "green", "yellow", "green", "red", "green"]
        streaming = compute_alert_level_timeline(zones, mixed_window_hours=2, window_minutes=30)
        expected = [
            apply_persistence(zones[: i + 1], mixed_window_hours=2, window_minutes=30)["alert_level"]
            for i in range(len(zones))
        ]
        assert streaming == expected


class TestApplyPersistence:
    def test_empty_history(self):
        result = apply_persistence([])
        assert result["alert_level"] == "green"

    def test_single_red_no_alert(self):
        result = apply_persistence(["green", "green", "red"])
        assert result["alert_level"] == "green"
        assert result["consecutive_red"] == 1

    def test_two_reds_yellow_alert(self):
        result = apply_persistence(["green", "red", "red"])
        assert result["alert_level"] == "yellow"
        assert result["consecutive_red"] == 2

    def test_three_reds_red_alert(self):
        result = apply_persistence(["green", "red", "red", "red"])
        assert result["alert_level"] == "red"
        assert result["consecutive_red"] == 3

    def test_four_yellows_yellow_alert(self):
        result = apply_persistence(["green", "yellow", "yellow", "yellow", "yellow"])
        assert result["alert_level"] == "yellow"
        assert result["consecutive_yellow"] == 4

    def test_three_yellows_no_alert(self):
        result = apply_persistence(["yellow", "yellow", "yellow"])
        assert result["alert_level"] == "green"
        assert result["consecutive_yellow"] == 3

    def test_red_then_green_resets(self):
        result = apply_persistence(["red", "red", "red", "green"])
        assert result["alert_level"] == "green"
        assert result["consecutive_red"] == 0

    def test_all_green(self):
        result = apply_persistence(["green"] * 10)
        assert result["alert_level"] == "green"

    def test_custom_thresholds(self):
        # With custom: 1 red = yellow alert, 2 red = red alert
        result = apply_persistence(
            ["red"],
            red_to_yellow_alert=1,
            red_to_red_alert=2,
        )
        assert result["alert_level"] == "yellow"


class TestMixedWindowPersistence:
    def test_disabled_by_default(self):
        # 1 red + 3 yellow interleaved would satisfy the mixed rule, but it's
        # off unless mixed_window_hours is passed.
        result = apply_persistence(["red", "green", "yellow", "yellow", "yellow"])
        assert result["alert_level"] == "green"

    def test_interleaved_red_and_yellow_triggers_yellow(self):
        # 1 red (weight 2) + 2 yellow (weight 2) = 4, within a 2h/4-window span.
        result = apply_persistence(
            ["green", "red", "green", "yellow", "yellow"],
            mixed_window_hours=2,
            window_minutes=30,
        )
        assert result["alert_level"] == "yellow"
        assert result["mixed_red_count"] == 1
        assert result["mixed_yellow_count"] == 2
        assert "mixed" in result["reason"]

    def test_pure_yellow_alone_not_enough_without_both_colors(self):
        # 3 yellows, no red -- weighted score is 3, and single-color doesn't count as mixed.
        result = apply_persistence(
            ["yellow", "yellow", "yellow"],
            mixed_window_hours=2,
            window_minutes=30,
        )
        assert result["alert_level"] == "green"

    def test_single_red_single_yellow_below_weighted_threshold(self):
        # 1 red (weight 2) + 1 yellow (weight 1) = 3, below the threshold of 4.
        result = apply_persistence(
            ["green", "green", "red", "yellow"],
            mixed_window_hours=2,
            window_minutes=30,
        )
        assert result["alert_level"] == "green"

    def test_never_escalates_past_yellow(self):
        # Lots of red+yellow mixing, but never 3 consecutive reds -- stays yellow, not red.
        result = apply_persistence(
            ["red", "yellow", "red", "yellow", "red", "yellow"],
            mixed_window_hours=3,
            window_minutes=30,
        )
        assert result["alert_level"] == "yellow"

    def test_outside_lookback_span_not_counted(self):
        # The red is far outside the trailing 2h/4-window span -- shouldn't count.
        result = apply_persistence(
            ["red", "green", "green", "green", "green", "yellow", "yellow"],
            mixed_window_hours=2,
            window_minutes=30,
        )
        assert result["alert_level"] == "green"
        assert result["mixed_red_count"] == 0

    def test_consecutive_red_rule_still_wins_over_mixed(self):
        # 3 consecutive reds already trigger Red; mixed check shouldn't downgrade it.
        result = apply_persistence(
            ["yellow", "red", "red", "red"],
            mixed_window_hours=2,
            window_minutes=30,
        )
        assert result["alert_level"] == "red"
