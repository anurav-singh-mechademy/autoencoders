"""Tests for node/event-based ground-truth loading/resampling to the model's
window grid (see src/autoencoder/evaluation/ground_truth.py's module
docstring for the *_combined_with_events.parquet + *_event_labels_long.parquet
schema)."""

import numpy as np
import pandas as pd
import pytest

from autoencoder.evaluation.ground_truth import (
    load_window_ground_truth, all_slugs, slug_presence_series,
    load_rule_to_sensors, expand_sensors, compute_split_node_coverage,
)


@pytest.fixture
def gt_files(tmp_path):
    """A tiny synthetic ground-truth pair: 2 windows of 4 rows each
    (window_size=4 for a fast test).

    Window 0 (rows 0-3): node 1 (rule_a, severity 3, sensors s1) active at
        rows 0-1.
    Window 1 (rows 4-7): node 1 continues (rows 4), node 2 (rule_b,
        severity 2, sensors s2) active at rows 4-5. Node 1 therefore spans
        both windows (rows 1 and 4) -- used to test partial split coverage.
    """
    n_rows = 8
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="15s")

    n_active_nodes = [1, 1, 0, 0, 2, 1, 0, 0]
    combined = pd.DataFrame({"timestamp": timestamps, "n_active_nodes": n_active_nodes})
    combined_path = tmp_path / "combined.parquet"
    combined.to_parquet(combined_path)

    event_rows = [
        {"timestamp": timestamps[0], "node_id": 1, "equipment_tag": "EQ", "start_time": timestamps[0], "end_time": timestamps[4], "rule_slug": "rule_a", "severity": 3, "sensors": ["s1"], "status": "ALERT", "n_sensors": 1, "duration_minutes": 1.25},
        {"timestamp": timestamps[1], "node_id": 1, "equipment_tag": "EQ", "start_time": timestamps[0], "end_time": timestamps[4], "rule_slug": "rule_a", "severity": 3, "sensors": ["s1"], "status": "ALERT", "n_sensors": 1, "duration_minutes": 1.25},
        {"timestamp": timestamps[4], "node_id": 1, "equipment_tag": "EQ", "start_time": timestamps[0], "end_time": timestamps[4], "rule_slug": "rule_a", "severity": 3, "sensors": ["s1"], "status": "ALERT", "n_sensors": 1, "duration_minutes": 1.25},
        {"timestamp": timestamps[4], "node_id": 2, "equipment_tag": "EQ", "start_time": timestamps[4], "end_time": timestamps[5], "rule_slug": "rule_b", "severity": 2, "sensors": ["s2"], "status": "ALERT", "n_sensors": 1, "duration_minutes": 0.5},
        {"timestamp": timestamps[5], "node_id": 2, "equipment_tag": "EQ", "start_time": timestamps[4], "end_time": timestamps[5], "rule_slug": "rule_b", "severity": 2, "sensors": ["s2"], "status": "ALERT", "n_sensors": 1, "duration_minutes": 0.5},
    ]
    events = pd.DataFrame(event_rows)
    events_path = tmp_path / "events.parquet"
    events.to_parquet(events_path)

    return str(combined_path), str(events_path)


class TestLoadWindowGroundTruth:
    def test_two_windows_from_eight_rows(self, gt_files):
        combined_path, events_path = gt_files
        gt = load_window_ground_truth(combined_path, events_path, window_size=4)
        assert len(gt.window_ids) == 2
        assert gt.is_anomaly.tolist() == [True, True]
        assert gt.n_active_nodes.tolist() == [1, 2]  # peak within each window
        assert gt.max_event_level.tolist() == [3.0, 3.0]

    def test_rule_slug_union_per_window(self, gt_files):
        combined_path, events_path = gt_files
        gt = load_window_ground_truth(combined_path, events_path, window_size=4)
        assert gt.all_rule_slugs[0] == frozenset({"rule_a"})
        assert gt.all_rule_slugs[1] == frozenset({"rule_a", "rule_b"})

    def test_triggering_sensors_union_per_window(self, gt_files):
        combined_path, events_path = gt_files
        gt = load_window_ground_truth(combined_path, events_path, window_size=4)
        assert gt.all_triggering_sensors[0] == frozenset({"s1"})
        assert gt.all_triggering_sensors[1] == frozenset({"s1", "s2"})

    def test_node_ids_union_per_window(self, gt_files):
        combined_path, events_path = gt_files
        gt = load_window_ground_truth(combined_path, events_path, window_size=4)
        assert gt.all_node_ids[0] == frozenset({1})
        assert gt.all_node_ids[1] == frozenset({1, 2})

    def test_drops_incomplete_trailing_window(self, gt_files):
        combined_path, events_path = gt_files
        # window_size=3 over 8 rows -> 2 full windows, 2 leftover rows dropped
        gt = load_window_ground_truth(combined_path, events_path, window_size=3)
        assert len(gt.window_ids) == 2


class TestFilterToWindowIds:
    def test_restricts_and_reorders(self, gt_files):
        combined_path, events_path = gt_files
        gt = load_window_ground_truth(combined_path, events_path, window_size=4)
        filtered = gt.filter_to_window_ids([1])
        assert filtered.window_ids.tolist() == [1]
        assert filtered.n_active_nodes.tolist() == [2]

    def test_raises_on_unknown_id(self, gt_files):
        combined_path, events_path = gt_files
        gt = load_window_ground_truth(combined_path, events_path, window_size=4)
        with pytest.raises(ValueError):
            gt.filter_to_window_ids([99])


class TestAllSlugs:
    def test_includes_any_slug_first(self, gt_files):
        combined_path, events_path = gt_files
        gt = load_window_ground_truth(combined_path, events_path, window_size=4)
        slugs = all_slugs(gt)
        assert slugs[0] == "any_slug"
        assert set(slugs) == {"any_slug", "rule_a", "rule_b"}


class TestSlugPresenceSeries:
    def test_any_slug_matches_is_anomaly(self, gt_files):
        combined_path, events_path = gt_files
        gt = load_window_ground_truth(combined_path, events_path, window_size=4)
        np.testing.assert_array_equal(slug_presence_series(gt, "any_slug"), gt.is_anomaly)

    def test_specific_slug_presence(self, gt_files):
        combined_path, events_path = gt_files
        gt = load_window_ground_truth(combined_path, events_path, window_size=4)
        assert slug_presence_series(gt, "rule_b").tolist() == [False, True]
        assert slug_presence_series(gt, "rule_a").tolist() == [True, True]


class TestExpandSensors:
    def test_unions_across_rules(self):
        catalog = {"rule_a": ["s1", "s2"], "rule_b": ["s2", "s3"]}
        assert expand_sensors(["rule_a", "rule_b"], catalog) == frozenset({"s1", "s2", "s3"})

    def test_unmapped_rule_contributes_nothing(self):
        catalog = {"rule_a": ["s1"]}
        assert expand_sensors(["rule_a", "unknown_rule"], catalog) == frozenset({"s1"})

    def test_valid_sensors_scopes_plant_wide_catalog(self):
        catalog = {"rule_a": ["this_eq_s1", "other_eq_s99"]}
        valid = frozenset({"this_eq_s1", "this_eq_s2"})
        assert expand_sensors(["rule_a"], catalog, valid_sensors=valid) == frozenset({"this_eq_s1"})


class TestLoadRuleToSensors:
    def test_json_dict(self, tmp_path):
        path = tmp_path / "catalog.json"
        path.write_text('{"rule_a": ["s1", "s2"], "rule_b": ["s3"]}')
        catalog = load_rule_to_sensors(str(path))
        assert catalog == {"rule_a": ["s1", "s2"], "rule_b": ["s3"]}


class TestLoadWindowGroundTruthWithCatalog:
    def test_catalog_takes_priority_over_event_sensors_column(self, gt_files):
        combined_path, events_path = gt_files
        catalog = {"rule_a": ["catalog_sensor_a"], "rule_b": ["catalog_sensor_b"]}
        gt = load_window_ground_truth(combined_path, events_path, window_size=4, rule_to_sensors=catalog)
        assert gt.all_triggering_sensors[0] == frozenset({"catalog_sensor_a"})
        assert gt.all_triggering_sensors[1] == frozenset({"catalog_sensor_a", "catalog_sensor_b"})


class TestComputeSplitNodeCoverage:
    def test_full_vs_partial(self, gt_files):
        combined_path, events_path = gt_files
        # Window 0 -> train, window 1 -> test. Node 1 touches both windows
        # (rows 0,1 in window 0; row 4 in window 1) -- partial in both
        # splits it touches. Node 2 only touches window 1 -- full in test.
        split_window_ids = {"train": [0], "val": [], "test": [1]}
        result = compute_split_node_coverage(events_path, combined_path, split_window_ids, window_size=4)
        assert result["train"]["n_active_nodes_full"] == 0
        assert result["train"]["n_active_nodes_partial"] == 1
        assert result["test"]["n_active_nodes_full"] == 1  # node 2
        assert result["test"]["n_active_nodes_partial"] == 1  # node 1
        assert result["val"]["n_active_nodes_full"] == 0
        assert result["val"]["n_active_nodes_partial"] == 0
