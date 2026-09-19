"""Load node/event-based ground truth and resample it to the model's window grid.

Ground truth now comes from a PAIR of files per equipment:

  *_combined_with_events.parquet -- the wide sensor table at 15-second
      granularity (a `timestamp` column, sensor columns, plus):
          active_node_ids (list<int>) -- node ids active at this row
          n_active_nodes (int)         -- len(active_node_ids)

  *_event_labels_long.parquet -- long/event format, one row per
      (active node, 15-second tick) it was active for:
          timestamp (datetime)          -- matches a row in the combined file
          node_id (int)
          equipment_tag (str)
          start_time, end_time (datetime) -- the node's full active span
          rule_slug (str)
          severity (int)
          sensors (list<str>)            -- sensors implicated by this event
          status (str)
          n_sensors (int)
          duration_minutes (float)

This module chunks both files into the same fixed-size (120-row) windows the
autoencoder pipeline scores. Per window:
    is_anomaly / n_active_nodes  -- max() over the window's rows, from the
        combined file (peak concurrent active-node count).
    all_rule_slugs / all_triggering_sensors / all_node_ids / max_event_level
        -- derived from whichever event_labels_long rows fall inside the
        window (by timestamp), taking the set union (rule_slugs, sensors,
        node ids) or max (severity) across those rows.

`all_triggering_sensors` is a CANDIDATE pool -- every sensor tied to any rule
active in that window, i.e. sensors that COULD be implicated, not a precise
label of which ones actually are (a rule's sensor list, or an event's own
`sensors` field, is fixed regardless of which subset of those sensors is
truly anomalous in a given occurrence). Evaluation code must not treat
non-membership in this set as a hard negative -- see
autoencoder.evaluation.slug_metrics.sensor_attribution_metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder.data.ingestion import read_parquet_robust

logger = logging.getLogger(__name__)

REQUIRED_COMBINED_COLUMNS = ["n_active_nodes"]
REQUIRED_EVENT_COLUMNS = ["timestamp", "node_id", "rule_slug", "severity", "sensors"]


def load_rule_to_sensors(path: str) -> dict[str, list[str]]:
    """Load a rule_slug -> sensor-list catalog (the static rule catalog Step 6
    prefers over the union-of-concurrent-events fallback -- see
    expand_sensors()).

    Accepts either:
      - a JSON file already shaped as {rule_slug: [sensor, ...]}, or
      - a CSV with one row per (rule_slug, sensor) pair -- any column named
        "sensor", "internal_tag", or "external_tag" is used as the sensor
        name (first match wins), grouped by "rule_slug".
    """
    path_obj = Path(path)
    if path_obj.suffix == ".json":
        with open(path_obj) as f:
            data = json.load(f)
        return {k: list(v) for k, v in data.items()}

    df = pd.read_csv(path)
    if "rule_slug" not in df.columns:
        raise ValueError(f"{path}: expected a 'rule_slug' column, found {list(df.columns)}")
    sensor_col = next((c for c in ("sensor", "internal_tag", "external_tag") if c in df.columns), None)
    if sensor_col is None:
        raise ValueError(f"{path}: expected a 'sensor'/'internal_tag'/'external_tag' column, found {list(df.columns)}")

    return {
        rule_slug: sorted(group[sensor_col].dropna().unique().tolist())
        for rule_slug, group in df.groupby("rule_slug")
    }


def expand_sensors(
    rule_slugs,
    rule_to_sensors: dict[str, list[str]],
    valid_sensors: frozenset[str] | None = None,
) -> frozenset[str]:
    """Union of every sensor tied to any rule in `rule_slugs` (an iterable
    of rule_slug strings, e.g. one window's all_rule_slugs frozenset),
    per the static catalog. Unmapped rule_slugs contribute nothing (not an
    error -- a rule genuinely not linked to any sensor in the catalog).

    Args:
        valid_sensors: If given, restrict the result to this set -- the
            catalog is plant-wide (a generic rule like a control-valve check
            lists sensors from every valve instance it's deployed on, not
            just one equipment's), so without this filter the ground truth
            for one equipment gets polluted with dozens of unrelated other
            -equipment sensor names. Pass the model's own trained
            sensor_columns here.
    """
    sensors: set[str] = set()
    for rule in rule_slugs:
        sensors.update(rule_to_sensors.get(rule, []))
    if valid_sensors is not None:
        sensors &= valid_sensors
    return frozenset(sensors)


@dataclass
class EventSensorFilterResult:
    """Result of restricting a sensor-column list to only sensors implicated
    by at least one labeled event node."""

    kept_columns: list[str]
    removed_columns: list[str]
    event_sensors: list[str]  # every sensor named by >=1 node, before intersecting with sensor_columns


def restrict_to_event_sensors(
    sensor_columns: list[str],
    event_labels_parquet_path: str,
    equipment_tag: str | None = None,
) -> EventSensorFilterResult:
    """Restrict sensor_columns to only those implicated by >=1 labeled event
    node anywhere in event_labels_parquet_path. Unions across every node in
    the file (not just one split) so the sensor set doesn't depend on the
    train/val/test assignment, which is computed later on windows.

    Args:
        sensor_columns: The already-detected sensor columns to filter.
        event_labels_parquet_path: Path to a *_event_labels_long.parquet
            file (see module docstring) whose `sensors` column names, per
            node, the sensors that event implicates.
        equipment_tag: If given and the file has an `equipment_tag` column,
            restrict to this equipment's own rows first -- the file may in
            principle carry other equipment's events.

    Returns:
        EventSensorFilterResult. Raises ValueError if equipment_tag is given
        but matches zero rows (silently filtering on a mismatched tag would
        otherwise empty out the whole sensor set), or if the result would
        remove every sensor column.
    """
    events = read_parquet_robust(event_labels_parquet_path)
    if "sensors" not in events.columns:
        raise ValueError(f"{event_labels_parquet_path} has no 'sensors' column")

    if equipment_tag is not None and "equipment_tag" in events.columns:
        events = events[events["equipment_tag"] == equipment_tag]
        if events.empty:
            raise ValueError(
                f"{event_labels_parquet_path}: equipment_tag={equipment_tag!r} matched 0 rows -- "
                "refusing to filter sensors to an empty event set (check equipment_tag spelling)."
            )

    event_sensors: set[str] = set()
    for s in events["sensors"].dropna():
        event_sensors.update(s)

    kept = [c for c in sensor_columns if c in event_sensors]
    removed = [c for c in sensor_columns if c not in event_sensors]

    if not kept:
        raise ValueError(
            f"restrict_to_event_sensors: 0 / {len(sensor_columns)} sensor columns matched any "
            f"event-node sensor in {event_labels_parquet_path} -- refusing to leave the model "
            "with no input sensors."
        )

    logger.info(
        "Event-sensor filter: kept %d / %d columns implicated by >=1 labeled event node in %s. "
        "Removed: %s",
        len(kept), len(sensor_columns), event_labels_parquet_path, removed,
    )

    return EventSensorFilterResult(
        kept_columns=kept,
        removed_columns=removed,
        event_sensors=sorted(event_sensors),
    )


@dataclass
class WindowGroundTruth:
    """Per-window ground truth, aligned 1:1 by window_id with a model's own
    window scores computed from the same combined file at the same
    window_size -- both are a deterministic function of (file, window_size),
    so joining on window_id (not a timestamp match) is safe."""

    window_ids: np.ndarray
    start_times: list
    end_times: list
    is_anomaly: np.ndarray
    n_active_nodes: np.ndarray     # peak (max) within the window, from the combined file
    max_event_level: np.ndarray    # peak severity within the window; NaN if never set
    all_rule_slugs: list           # list[frozenset[str]], one per window
    all_triggering_sensors: list   # list[frozenset[str]], one per window -- CANDIDATE pool, see module docstring
    all_node_ids: list             # list[frozenset[int]], one per window

    def filter_to_window_ids(self, window_ids) -> "WindowGroundTruth":
        """Restrict (and reorder) this ground truth to exactly the given
        window ids, in the given order -- e.g. to align with a window_scores
        table that only covers one split (see score_full_timeline.py's
        --split-ids). Raises if any requested id isn't present.
        """
        pos = {int(w): i for i, w in enumerate(self.window_ids)}
        missing = [int(w) for w in window_ids if int(w) not in pos]
        if missing:
            raise ValueError(
                f"{len(missing)} window id(s) requested are not present in this ground truth "
                f"(e.g. {missing[:5]}) -- window_scores and --ground-truth must come from the "
                "same file at the same window size."
            )
        idx = [pos[int(w)] for w in window_ids]
        return WindowGroundTruth(
            window_ids=np.asarray([self.window_ids[i] for i in idx]),
            start_times=[self.start_times[i] for i in idx],
            end_times=[self.end_times[i] for i in idx],
            is_anomaly=self.is_anomaly[idx],
            n_active_nodes=self.n_active_nodes[idx],
            max_event_level=self.max_event_level[idx],
            all_rule_slugs=[self.all_rule_slugs[i] for i in idx],
            all_triggering_sensors=[self.all_triggering_sensors[i] for i in idx],
            all_node_ids=[self.all_node_ids[i] for i in idx],
        )


def _to_frozenset(v) -> frozenset:
    """A cell that may be a native list/array (pyarrow list<...>) or None."""
    if v is None:
        return frozenset()
    return frozenset(v)


def load_window_ground_truth(
    combined_parquet_path: str,
    event_labels_parquet_path: str,
    window_size: int = 120,
    timestamp_column: str = "timestamp",
    rule_to_sensors: dict[str, list[str]] | None = None,
    valid_sensors: frozenset[str] | None = None,
) -> WindowGroundTruth:
    """Chunk a combined + event-labels parquet pair into fixed-size windows.

    The trailing incomplete window (if any) is dropped -- same convention
    used by autoencoder.data.preprocessing.construct_windows(_with_metadata).

    Args:
        combined_parquet_path: Path to a *_combined_with_events.parquet file
            (timestamp column + n_active_nodes, at minimum).
        event_labels_parquet_path: Path to the matching *_event_labels_long.parquet
            file (one row per active node per 15s tick -- see module docstring).
        window_size: Rows per window (120 = 30 min at 15s intervals).
        timestamp_column: Name of the timestamp column in both files.
        rule_to_sensors: Static rule_slug -> sensor-list catalog (see
            load_rule_to_sensors()). When given, all_triggering_sensors is
            derived from all_rule_slugs via this catalog instead of the
            event file's own per-event `sensors` column -- the catalog
            attributes each active rule to exactly its own sensors, whereas
            the event file's column (unioned across every event active in a
            window) is confounded the moment more than one rule fires at
            once.
        valid_sensors: Restrict catalog-derived all_triggering_sensors to
            this set (typically the model's own trained sensor_columns) --
            see expand_sensors()'s docstring. Ignored when rule_to_sensors
            is None (the event file's own `sensors` column already only
            ever contains this equipment's own sensor names).
    """
    combined = read_parquet_robust(combined_parquet_path)
    if timestamp_column in combined.columns:
        combined = combined.set_index(timestamp_column)
    missing = [c for c in REQUIRED_COMBINED_COLUMNS if c not in combined.columns]
    if missing:
        raise ValueError(f"{combined_parquet_path} is missing column(s): {missing}")

    n_rows_full = len(combined)
    n_windows = n_rows_full // window_size
    if n_windows == 0:
        raise ValueError(f"{combined_parquet_path} has {n_rows_full} rows, fewer than one window ({window_size})")
    dropped = n_rows_full - n_windows * window_size

    # Row-position index built off the FULL (untrimmed) timestamp grid --
    # needed below to map event_labels_long's own timestamps to a row
    # position, before we trim the trailing incomplete window away.
    row_pos = pd.Series(np.arange(n_rows_full), index=combined.index)

    combined = combined.iloc[: n_windows * window_size]
    n_rows = len(combined)

    window_id_per_row = np.repeat(np.arange(n_windows), window_size)
    timestamps = combined.index
    start_times = [timestamps[i * window_size] for i in range(n_windows)]
    end_times = [timestamps[(i + 1) * window_size - 1] for i in range(n_windows)]

    n_active_nodes_col = combined["n_active_nodes"].to_numpy()
    tmp = pd.DataFrame({"n_active_nodes": n_active_nodes_col, "window_id": window_id_per_row})
    grouped = tmp.groupby("window_id", sort=True)["n_active_nodes"].max()
    n_active_nodes = grouped.reindex(range(n_windows), fill_value=0).to_numpy()
    is_anomaly = n_active_nodes > 0

    # Map every event-labels row's timestamp to a raw row position in the
    # combined file's own timestamp grid, then to a window_id -- this is the
    # authoritative row order (event_labels_long doesn't carry its own
    # contiguous row index), and matches exactly how the model's own window
    # scores are chunked from the same combined file.
    events = read_parquet_robust(event_labels_parquet_path)
    event_missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in events.columns]
    if event_missing:
        raise ValueError(f"{event_labels_parquet_path} is missing column(s): {event_missing}")

    event_ts = events[timestamp_column]
    positions = row_pos.reindex(event_ts).to_numpy()
    valid_event_rows = ~np.isnan(positions)
    n_unmatched = int((~valid_event_rows).sum())
    if n_unmatched:
        logger.warning(
            "%d / %d event_labels rows have a timestamp not found in %s -- dropped.",
            n_unmatched, len(events), combined_parquet_path,
        )
    events = events.loc[valid_event_rows].copy()
    positions = positions[valid_event_rows].astype(np.int64)
    events["window_id"] = positions // window_size
    events = events[events["window_id"] < n_windows]  # drop the trailing incomplete window, same convention as combined

    has_sensors_column = "sensors" in events.columns

    def _agg_window(g: pd.DataFrame) -> pd.Series:
        rule_slugs = frozenset(g["rule_slug"].dropna().unique().tolist())
        node_ids = frozenset(int(n) for n in g["node_id"].dropna().unique().tolist())
        max_level = float(g["severity"].max()) if g["severity"].notna().any() else np.nan
        if has_sensors_column:
            sensors: set[str] = set()
            for s in g["sensors"]:
                sensors.update(_to_frozenset(s))
            triggering_sensors = frozenset(sensors)
        else:
            triggering_sensors = frozenset()
        return pd.Series({
            "all_rule_slugs": rule_slugs,
            "all_node_ids": node_ids,
            "max_event_level": max_level,
            "all_triggering_sensors": triggering_sensors,
        })

    if len(events):
        per_window = events.groupby("window_id", sort=True).apply(_agg_window)
    else:
        per_window = pd.DataFrame(columns=["all_rule_slugs", "all_node_ids", "max_event_level", "all_triggering_sensors"])

    all_rule_slugs = []
    all_node_ids = []
    max_event_level = np.full(n_windows, np.nan)
    all_triggering_sensors = []
    for i in range(n_windows):
        if i in per_window.index:
            row = per_window.loc[i]
            all_rule_slugs.append(row["all_rule_slugs"])
            all_node_ids.append(row["all_node_ids"])
            max_event_level[i] = row["max_event_level"]
            all_triggering_sensors.append(row["all_triggering_sensors"])
        else:
            all_rule_slugs.append(frozenset())
            all_node_ids.append(frozenset())
            all_triggering_sensors.append(frozenset())

    # The catalog wins whenever given: it attributes each active rule to
    # exactly its own sensors, whereas the event file's own `sensors` column
    # is a blob unioned across every event concurrently active in a window --
    # confounded the moment more than one rule fires at once.
    if rule_to_sensors is not None:
        all_triggering_sensors = [expand_sensors(rules, rule_to_sensors, valid_sensors) for rules in all_rule_slugs]
        sensor_source = "rule_to_sensors catalog"
    elif has_sensors_column:
        sensor_source = "event_labels_long 'sensors' column (no catalog given -- confounded by concurrency)"
    else:
        sensor_source = "none available"

    logger.info(
        "Loaded %d ground-truth windows from %s + %s (%d raw rows, %d dropped as incomplete "
        "trailing window, %d events matched, sensor attribution source: %s)",
        n_windows, combined_parquet_path, event_labels_parquet_path, n_rows, dropped, len(events), sensor_source,
    )

    return WindowGroundTruth(
        window_ids=np.arange(n_windows),
        start_times=start_times,
        end_times=end_times,
        is_anomaly=is_anomaly,
        n_active_nodes=n_active_nodes,
        max_event_level=max_event_level,
        all_rule_slugs=all_rule_slugs,
        all_triggering_sensors=all_triggering_sensors,
        all_node_ids=all_node_ids,
    )


def all_slugs(gt: WindowGroundTruth) -> list[str]:
    """Every distinct rule_slug that appears anywhere in the timeline, plus
    "any_slug" (the OR-of-everything series, evaluated with the identical
    code path as a real slug -- not a separate "primary" ground truth)."""
    slugs: set[str] = set()
    for s in gt.all_rule_slugs:
        slugs.update(s)
    return ["any_slug"] + sorted(slugs)


def slug_presence_series(gt: WindowGroundTruth, slug: str) -> np.ndarray:
    """Binary per-window presence series for one slug (or "any_slug" == is_anomaly)."""
    if slug == "any_slug":
        return np.asarray(gt.is_anomaly, dtype=bool)
    return np.array([slug in s for s in gt.all_rule_slugs], dtype=bool)


def compute_split_node_coverage(
    event_labels_parquet_path: str,
    combined_parquet_path: str,
    split_window_ids: dict[str, list[int]],
    window_size: int = 120,
    timestamp_column: str = "timestamp",
) -> dict:
    """For each split (train/val/test), count how many distinct active nodes
    (events) fall inside it -- an event "falls fully" in a split if every
    window it touches belongs to that split's window_id set, or "falls
    partially" if only some of its windows do (it straddles a split
    boundary). A node absent from a split's windows entirely isn't counted
    for that split at all.

    Args:
        split_window_ids: e.g. the "train"/"val"/"test" lists from a
            training run's artefacts/split_window_ids.json.
    """
    full_combined = read_parquet_robust(combined_parquet_path)
    if timestamp_column in full_combined.columns:
        full_timestamps = pd.Index(full_combined[timestamp_column])
    else:
        full_timestamps = full_combined.index
    row_pos = pd.Series(np.arange(len(full_timestamps)), index=full_timestamps)

    events = read_parquet_robust(event_labels_parquet_path)
    positions = row_pos.reindex(events[timestamp_column]).to_numpy()
    valid = ~np.isnan(positions)
    events = events.loc[valid].copy()
    events["window_id"] = (positions[valid].astype(np.int64)) // window_size

    node_windows: dict[int, set[int]] = {}
    for node_id, g in events.groupby("node_id"):
        node_windows[int(node_id)] = set(int(w) for w in g["window_id"].unique())

    result = {}
    for split_name, ids in split_window_ids.items():
        if split_name == "window_size":
            continue
        split_ids = set(int(i) for i in ids)
        n_full = 0
        n_partial = 0
        for node_id, windows in node_windows.items():
            overlap = windows & split_ids
            if not overlap:
                continue
            if windows <= split_ids:
                n_full += 1
            else:
                n_partial += 1
        result[split_name] = {
            "n_active_nodes_full": n_full,
            "n_active_nodes_partial": n_partial,
            "n_windows_in_split": len(split_ids),
        }
    return result
