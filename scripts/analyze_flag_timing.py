#!/usr/bin/env python
"""CLI: For each model-flagged window in a scored test split, find the
nearest labeled ground-truth event node in time and report a signed
lead/lag distance, plus a distribution/count summary.

Sign convention (matches "how early/late is this flag relative to the
nearest real event"):
    positive -- the nearest event node is AHEAD of (starts after) this
        window ends. The model flagged BEFORE the fault happened: an early
        /leading detection.
    negative -- the nearest event node is BEHIND (ended before) this
        window starts. The model flagged AFTER the fault already happened:
        a lagging detection.
    zero     -- this window overlaps the node's own active span: a
        concurrent flag.

Prerequisite: run scripts/score_full_timeline.py first (against the "test"
split) to produce --window-scores.

Usage:
    python scripts/analyze_flag_timing.py \\
        --window-scores output_5K501M/evaluation/window_scores.parquet \\
        --event-labels data/5K501M/5K501M_event_labels_long.parquet \\
        --output output_5K501M/evaluation/flag_timing.csv
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging
from autoencoder.data.ingestion import read_parquet_robust

setup_logging()
logger = logging.getLogger(__name__)


def build_node_spans(event_labels_parquet_path: str, equipment_tag: str | None = None) -> pd.DataFrame:
    """One row per distinct node_id: its own [start_time, end_time] active
    span and rule_slug. event_labels_long carries one row per (node, 15s
    tick) it was active for, with start_time/end_time constant across a
    node's own rows -- min/max collapses that back to one row per node
    (equivalent to "first", just robust to any accidental non-constant rows).
    """
    events = read_parquet_robust(event_labels_parquet_path)
    if equipment_tag is not None and "equipment_tag" in events.columns:
        events = events[events["equipment_tag"] == equipment_tag]
    if events.empty:
        return pd.DataFrame(columns=["node_id", "start_time", "end_time", "rule_slug"])

    nodes = (
        events.groupby("node_id")
        .agg(start_time=("start_time", "min"), end_time=("end_time", "max"), rule_slug=("rule_slug", "first"))
        .reset_index()
        .sort_values("start_time")
        .reset_index(drop=True)
    )
    return nodes


def nearest_event_distances(flagged: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Append signed_distance_hours / nearest_node_id / nearest_rule_slug /
    overlap columns to `flagged` (must have window_start, window_end).

    Signed distance to the SINGLE nearest node by absolute time gap -- see
    module docstring for the sign convention. NaN distance (nearest_node_id
    None) means there are no labeled event nodes at all to compare against.
    """
    node_starts = nodes["start_time"].to_numpy()
    node_ends = nodes["end_time"].to_numpy()
    node_ids = nodes["node_id"].to_numpy()
    node_slugs = nodes["rule_slug"].to_numpy()

    signed_distance_hours = np.full(len(flagged), np.nan)
    nearest_node_id = np.full(len(flagged), -1, dtype=int)
    nearest_rule_slug = np.array([None] * len(flagged), dtype=object)
    overlap = np.zeros(len(flagged), dtype=bool)

    window_starts = flagged["window_start"].to_numpy()
    window_ends = flagged["window_end"].to_numpy()

    for i, (w_start, w_end) in enumerate(zip(window_starts, window_ends)):
        if len(nodes) == 0:
            continue

        overlap_mask = (node_starts <= w_end) & (node_ends >= w_start)
        if overlap_mask.any():
            idx = np.flatnonzero(overlap_mask)[0]
            signed_distance_hours[i] = 0.0
            nearest_node_id[i] = node_ids[idx]
            nearest_rule_slug[i] = node_slugs[idx]
            overlap[i] = True
            continue

        # Not overlapping -> every node is strictly ahead (node_start > w_end)
        # or strictly behind (node_end < w_start), never both.
        gap_ahead = (node_starts - w_end) / np.timedelta64(1, "h")
        gap_behind = (w_start - node_ends) / np.timedelta64(1, "h")
        signed = np.where(gap_ahead > 0, gap_ahead, -gap_behind)
        idx = int(np.argmin(np.abs(signed)))
        signed_distance_hours[i] = signed[idx]
        nearest_node_id[i] = node_ids[idx]
        nearest_rule_slug[i] = node_slugs[idx]

    out = flagged.reset_index(drop=True).copy()
    out["signed_distance_hours"] = signed_distance_hours
    out["nearest_node_id"] = nearest_node_id
    out["nearest_rule_slug"] = nearest_rule_slug
    out["overlap"] = overlap
    return out


def print_distribution(result: pd.DataFrame, bin_hours: float, max_hours: float):
    n_total = len(result)
    no_events = result["nearest_node_id"] == -1
    n_no_events = int(no_events.sum())
    scored = result.loc[~no_events]
    n_overlap = int(scored["overlap"].sum())
    non_overlap = scored.loc[~scored["overlap"], "signed_distance_hours"]

    print()
    print("=" * 78)
    print("FLAG TIMING vs. NEAREST LABELED EVENT NODE")
    print("=" * 78)
    print(f"Flagged windows analyzed: {n_total}")
    if n_no_events:
        print(f"  (of which {n_no_events} have no labeled event nodes at all to compare against)")
    print()
    print("positive = event node is ahead of the window (early/leading detection)")
    print("negative = event node is behind the window (lagging detection)")
    print("zero     = window overlaps the node's active span (concurrent)")
    print()

    print(f"  overlap (concurrent):{'':6}{n_overlap:>6d}  {'#' * min(60, n_overlap)}")

    edges = np.arange(-max_hours, max_hours + bin_hours, bin_hours)
    counts, _ = np.histogram(non_overlap.to_numpy(), bins=edges)
    max_count = max(counts.max(), 1) if len(counts) else 1
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        if c == 0:
            continue
        label = "leading" if lo >= 0 else "lagging"
        bar = "#" * max(1, int(60 * c / max_count)) if c > 0 else ""
        print(f"  {lo:>6.1f}h to {hi:>6.1f}h  ({label:>7s}): {c:>6d}  {bar}")

    n_beyond_ahead = int((non_overlap > max_hours).sum())
    n_beyond_behind = int((non_overlap < -max_hours).sum())
    if n_beyond_ahead:
        print(f"  > {max_hours:.0f}h ahead (leading, beyond range): {n_beyond_ahead:>6d}")
    if n_beyond_behind:
        print(f"  < -{max_hours:.0f}h behind (lagging, beyond range): {n_beyond_behind:>6d}")
    if n_no_events:
        print(f"  no labeled events at all for this equipment: {n_no_events:>6d}")

    print()
    print("-- Summary --")
    leading = non_overlap[non_overlap > 0]
    lagging = non_overlap[non_overlap < 0]
    print(f"  n leading (event still ahead): {len(leading)} ({100 * len(leading) / n_total:.1f}%)")
    print(f"  n lagging (event already passed): {len(lagging)} ({100 * len(lagging) / n_total:.1f}%)")
    print(f"  n overlapping (concurrent):     {n_overlap} ({100 * n_overlap / n_total:.1f}%)")
    if n_no_events:
        print(f"  n with no labeled event at all:  {n_no_events} ({100 * n_no_events / n_total:.1f}%)")
    if len(leading):
        print(f"  median lead time: {leading.median():.2f}h  (mean {leading.mean():.2f}h)")
    if len(lagging):
        print(f"  median lag time:  {-lagging.median():.2f}h  (mean {-lagging.mean():.2f}h)")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Signed lead/lag distance of each flagged window to the nearest labeled event node.",
    )
    parser.add_argument("--window-scores", required=True, help="Output of scripts/score_full_timeline.py")
    parser.add_argument("--event-labels", required=True, help="Path to the matching *_event_labels_long.parquet file")
    parser.add_argument("--equipment-tag", default=None, help="Restrict event-labels rows to this equipment_tag, if present")
    parser.add_argument(
        "--zones", default="yellow,red",
        help="Comma-separated zones counted as 'flagged' (default: yellow,red -- i.e. non-green)",
    )
    parser.add_argument("--bin-hours", type=float, default=1.0, help="Histogram bin width, in hours")
    parser.add_argument("--max-hours", type=float, default=72.0, help="Histogram range +/- this many hours")
    parser.add_argument("--output", default=None, help="Optional path to save the per-window result table (.csv or .parquet)")
    args = parser.parse_args()

    window_scores = pd.read_parquet(args.window_scores) if args.window_scores.endswith(".parquet") else pd.read_csv(args.window_scores)
    window_scores["window_start"] = pd.to_datetime(window_scores["window_start"])
    window_scores["window_end"] = pd.to_datetime(window_scores["window_end"])
    logger.info("Loaded %d scored windows from %s", len(window_scores), args.window_scores)

    zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    flagged = window_scores[window_scores["usable"] & window_scores["zone"].isin(zones)]
    logger.info("Flagged windows (zone in %s, usable): %d / %d", zones, len(flagged), len(window_scores))
    if flagged.empty:
        logger.warning("No flagged windows -- nothing to analyze.")
        return

    nodes = build_node_spans(args.event_labels, equipment_tag=args.equipment_tag)
    logger.info("Loaded %d distinct event node(s) from %s", len(nodes), args.event_labels)

    result = nearest_event_distances(flagged, nodes)
    print_distribution(result, bin_hours=args.bin_hours, max_hours=args.max_hours)

    if args.output:
        cols = [
            "window_id", "window_start", "window_end", "zone", "anomaly_score",
            "signed_distance_hours", "nearest_node_id", "nearest_rule_slug", "overlap",
        ]
        cols = [c for c in cols if c in result.columns]
        if args.output.endswith(".parquet"):
            result[cols].to_parquet(args.output, index=False)
        else:
            result[cols].to_csv(args.output, index=False)
        logger.info("Saved per-window result table to %s", args.output)


if __name__ == "__main__":
    main()
