"""Spike-vs-dip recall test using real labeled event data. For each equipment,
classifies each event's direction (majority vote of its attributed sensors'
during-event mean vs. a 3h pre-event baseline) and computes model recall
(zone != green on >=1 touched test window) separately for spike vs. dip events.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOW_SIZE = 120
BASELINE_HOURS = 3

EQUIPMENT = ["5P921A", "5ST901A", "5ST901B", "5K512B"]


def classify_event_direction(combined: pd.DataFrame, sensors: list[str], start_time, end_time):
    """Return (direction, n_spike_votes, n_dip_votes, n_sensors_used, n_sensors_missing_or_nodata)."""
    baseline_start = start_time - pd.Timedelta(hours=BASELINE_HOURS)
    baseline_end = start_time - pd.Timedelta(seconds=15)  # one tick before start, avoid overlap

    during = combined.loc[start_time:end_time]
    baseline = combined.loc[baseline_start:baseline_end]

    n_spike, n_dip, n_used, n_bad = 0, 0, 0, 0
    for s in sensors:
        if s not in combined.columns:
            n_bad += 1
            continue
        d_mean = during[s].mean()
        b_mean = baseline[s].mean()
        if pd.isna(d_mean) or pd.isna(b_mean) or len(baseline) < 30:  # need >=30 rows (7.5min) of baseline
            n_bad += 1
            continue
        n_used += 1
        if d_mean > b_mean:
            n_spike += 1
        elif d_mean < b_mean:
            n_dip += 1
        # exact tie contributes to neither

    if n_used == 0:
        return "ambiguous_no_data", n_spike, n_dip, n_used, n_bad
    if n_spike > n_dip:
        return "spike", n_spike, n_dip, n_used, n_bad
    if n_dip > n_spike:
        return "dip", n_spike, n_dip, n_used, n_bad
    return "ambiguous_tie", n_spike, n_dip, n_used, n_bad


def event_touched_window_ids(row_pos: pd.Series, n_windows: int, start_time, end_time) -> set[int]:
    """Map an event's [start_time, end_time] span to window_ids on the same
    grid load_window_ground_truth uses (row // WINDOW_SIZE), via the combined
    file's own timestamp->row-position index (row_pos), built off the FULL
    untrimmed grid exactly like the ground-truth loader does."""
    sl = row_pos.loc[start_time:end_time]
    if len(sl) == 0:
        return set()
    positions = sl.to_numpy()
    wids = set((positions // WINDOW_SIZE).tolist())
    return {w for w in wids if w < n_windows}


def run_equipment(eq: str) -> dict:
    combined_path = ROOT / "data" / eq / f"{eq}_combined_with_events.parquet"
    events_path = ROOT / "data" / eq / f"{eq}_event_labels_long.parquet"
    scores_path = ROOT / f"output_{eq}_regime_fix" / "evaluation" / "window_scores.parquet"

    combined = pd.read_parquet(combined_path)
    combined = combined.set_index("timestamp").sort_index()
    n_rows_full = len(combined)
    n_windows = n_rows_full // WINDOW_SIZE
    row_pos = pd.Series(np.arange(n_rows_full), index=combined.index)

    events = pd.read_parquet(events_path)
    nodes = events.drop_duplicates("node_id").copy()

    scores = pd.read_parquet(scores_path)
    test_window_ids = set(scores["window_id"].unique().tolist())
    flagged_window_ids = set(scores.loc[scores["zone"] != "green", "window_id"].unique().tolist())

    records = []
    for _, row in nodes.iterrows():
        sensors = list(row["sensors"]) if row["sensors"] is not None else []
        direction, n_spike, n_dip, n_used, n_bad = classify_event_direction(
            combined, sensors, row["start_time"], row["end_time"]
        )
        touched = event_touched_window_ids(row_pos, n_windows, row["start_time"], row["end_time"])
        touched_test = touched & test_window_ids
        in_test = len(touched_test) > 0
        detected = bool(touched_test & flagged_window_ids) if in_test else None
        records.append({
            "equipment": eq,
            "node_id": row["node_id"],
            "rule_slug": row["rule_slug"],
            "status": row["status"],
            "n_sensors_attributed": len(sensors),
            "n_sensor_votes_used": n_used,
            "n_sensor_votes_missing": n_bad,
            "n_spike_votes": n_spike,
            "n_dip_votes": n_dip,
            "direction": direction,
            "n_windows_touched": len(touched),
            "n_windows_touched_test": len(touched_test),
            "in_test_split": in_test,
            "detected": detected,
        })

    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eq, g in df.groupby("equipment"):
        n_total = len(g)
        n_ambiguous = (g["direction"].str.startswith("ambiguous")).sum()
        g_dir = g[g["direction"].isin(["spike", "dip"])]
        g_test = g_dir[g_dir["in_test_split"]]
        n_not_in_test = (g_dir["in_test_split"] == False).sum()
        for direction in ["spike", "dip"]:
            gd = g_test[g_test["direction"] == direction]
            n = len(gd)
            n_det = int(gd["detected"].sum()) if n else 0
            recall = n_det / n if n else float("nan")
            rows.append({
                "equipment": eq,
                "direction": direction,
                "n_events_total_labeled": n_total,
                "n_ambiguous_excluded": n_ambiguous,
                "n_directional_not_in_test": n_not_in_test,
                "n_events_scored": n,
                "n_detected": n_det,
                "recall": recall,
            })
    return pd.DataFrame(rows)


def main():
    all_events = []
    for eq in EQUIPMENT:
        print(f"=== {eq} ===")
        df = run_equipment(eq)
        all_events.append(df)
        print(df["direction"].value_counts(dropna=False))
        print(f"in_test_split: {df['in_test_split'].sum()} / {len(df)}")
        print()

    events_df = pd.concat(all_events, ignore_index=True)
    out_csv = ROOT / "scripts" / "_event_direction_recall_events.csv"
    events_df.to_csv(out_csv, index=False)
    print(f"Per-event detail written to {out_csv}")

    summary = summarize(events_df)
    print("\n=== SUMMARY: recall by direction ===")
    print(summary.to_string(index=False))
    out_summary = ROOT / "scripts" / "_event_direction_recall_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"\nSummary written to {out_summary}")


if __name__ == "__main__":
    main()
