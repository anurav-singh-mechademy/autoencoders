#!/usr/bin/env python
"""CLI: Recalibrate Green/Yellow/Red thresholds against the equipment's own
observed event prevalence (instead of fixed P90/P99) -- yellow_red targets
the prevalence itself, green_yellow a wider multiple of it -- and compare
window- and episode-level precision/recall between the old (as-saved) and
new thresholds. Reuses an existing run's saved artefacts and
window_scores.parquet; doesn't retrain or rescore anything.

Usage:
    python scripts/recalibrate_prevalence_thresholds.py \\
        --model-dir output_5P921A_regime_fix/artefacts \\
        --window-scores output_5P921A_regime_fix/evaluation/window_scores.parquet \\
        --combined-data data/5P921A/5P921A_combined_with_events.parquet \\
        --event-labels data/5P921A/5P921A_event_labels_long.parquet \\
        --equipment-tag 5P921A \\
        --output output_5P921A_regime_fix/evaluation/prevalence_recalibration.json
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging
from autoencoder.data.ingestion import read_parquet_robust
from autoencoder.data.preprocessing import compute_event_touched_windows
from autoencoder.alerting.thresholds import compute_thresholds
from autoencoder.alerting.zones import classify_batch
from autoencoder.alerting.persistence import compute_alert_level_timeline
from autoencoder.alerting.episodes import extract_episodes
from autoencoder.evaluation.ground_truth import load_window_ground_truth
from autoencoder.evaluation.slug_metrics import pointwise_metrics

setup_logging()
logger = logging.getLogger(__name__)


def build_node_spans(event_labels_parquet_path: str, equipment_tag: str | None = None) -> pd.DataFrame:
    """One row per distinct node_id: its own [start_time, end_time] span."""
    events = read_parquet_robust(event_labels_parquet_path)
    if equipment_tag is not None and "equipment_tag" in events.columns:
        events = events[events["equipment_tag"] == equipment_tag]
    if events.empty:
        return pd.DataFrame(columns=["node_id", "start_time", "end_time"])
    return (
        events.groupby("node_id")
        .agg(start_time=("start_time", "min"), end_time=("end_time", "max"))
        .reset_index()
    )


def episode_precision_recall(episodes, node_spans: pd.DataFrame) -> dict:
    """Episode-level precision (episodes overlapping >=1 node) / recall (nodes caught by >=1 episode)."""
    n_episodes = len(episodes)
    n_nodes = len(node_spans)

    # Normalize to numpy datetime64 -- episode/node start/end can arrive as a mix of timestamp types.
    node_starts = pd.to_datetime(node_spans["start_time"]).to_numpy()
    node_ends = pd.to_datetime(node_spans["end_time"]).to_numpy()
    ep_starts = pd.to_datetime([ep.start for ep in episodes]).to_numpy() if n_episodes else np.array([], dtype="datetime64[ns]")
    ep_ends = pd.to_datetime([ep.end for ep in episodes]).to_numpy() if n_episodes else np.array([], dtype="datetime64[ns]")

    tp_episodes = 0
    for e_start, e_end in zip(ep_starts, ep_ends):
        overlap = ((node_starts <= e_end) & (node_ends >= e_start)).any()
        if overlap:
            tp_episodes += 1
    precision = tp_episodes / n_episodes if n_episodes else float("nan")

    caught_nodes = 0
    for n_start, n_end in zip(node_starts, node_ends):
        overlap = ((ep_starts <= n_end) & (ep_ends >= n_start)).any() if n_episodes else False
        if overlap:
            caught_nodes += 1
    recall = caught_nodes / n_nodes if n_nodes else float("nan")

    return {
        "n_model_episodes": n_episodes,
        "n_relevant_nodes": n_nodes,
        "episode_precision": precision,
        "episode_recall": recall,
        "n_episodes_matched": tp_episodes,
        "n_nodes_caught": caught_nodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recalibrate thresholds against observed event prevalence; compare window- and episode-level precision/recall.",
    )
    parser.add_argument("--model-dir", required=True, help="Trained model's artefacts dir (needs training_error_distribution.npy + thresholds.json)")
    parser.add_argument("--window-scores", required=True, help="Output of scripts/score_full_timeline.py (test split)")
    parser.add_argument("--combined-data", required=True, help="*_combined_with_events.parquet -- used to measure true event prevalence over the WHOLE file")
    parser.add_argument("--event-labels", required=True, help="*_event_labels_long.parquet")
    parser.add_argument("--equipment-tag", default=None)
    parser.add_argument("--window-rows", type=int, default=120)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--yellow-band-multiple", type=float, default=3.0,
                         help="green_yellow targets this multiple of prevalence (a wider pre/post-alert band)")
    parser.add_argument("--red-to-yellow-alert", type=int, default=2)
    parser.add_argument("--red-to-red-alert", type=int, default=3)
    parser.add_argument("--yellow-consecutive", type=int, default=4)
    parser.add_argument("--mixed-window-hours", type=float, default=2)
    parser.add_argument(
        "--red-only", action="store_true",
        help="Treat only 'red' as flagged (recode 'yellow' to 'green') instead of the default 'yellow or red'.",
    )
    parser.add_argument("--output", default=None, help="Optional path to save the full comparison as JSON")
    args = parser.parse_args()

    # True event prevalence, measured over the whole raw file (not just test).
    full_df = read_parquet_robust(args.combined_data)
    event_touched = compute_event_touched_windows(full_df, window_size=args.window_rows)
    prevalence = float(event_touched.mean())
    logger.info(
        "Observed event prevalence: %d / %d windows touched by >=1 event node (%.3f%%)",
        int(event_touched.sum()), len(event_touched), 100 * prevalence,
    )

    with open(f"{args.model_dir}/thresholds.json") as f:
        old_thresholds = json.load(f)
    training_errors = np.load(f"{args.model_dir}/training_error_distribution.npy")

    yellow_red_pct_new = float(np.clip(100 * (1 - prevalence), 50.0, 99.9))
    green_yellow_pct_new = float(np.clip(100 * (1 - args.yellow_band_multiple * prevalence), 50.0, yellow_red_pct_new - 0.1))
    new_thresholds = compute_thresholds(
        training_errors,
        green_yellow_percentile=green_yellow_pct_new,
        yellow_red_percentile=yellow_red_pct_new,
        method=old_thresholds.get("method", "gpd_tail"),
        spread=old_thresholds.get("spread") or "mad",
    )
    logger.info(
        "Old thresholds: green_yellow=%.6g (P%.1f), yellow_red=%.6g (P%.1f)",
        old_thresholds["green_yellow"], old_thresholds["green_yellow_percentile"],
        old_thresholds["yellow_red"], old_thresholds["yellow_red_percentile"],
    )
    logger.info(
        "New (prevalence-aware) thresholds: green_yellow=%.6g (P%.2f), yellow_red=%.6g (P%.2f)",
        new_thresholds["green_yellow"], green_yellow_pct_new,
        new_thresholds["yellow_red"], yellow_red_pct_new,
    )

    window_scores = pd.read_parquet(args.window_scores) if args.window_scores.endswith(".parquet") else pd.read_csv(args.window_scores)
    window_scores["window_start"] = pd.to_datetime(window_scores["window_start"])
    window_scores["window_end"] = pd.to_datetime(window_scores["window_end"])
    usable = window_scores["usable"].to_numpy(dtype=bool)
    scores = window_scores["anomaly_score"].to_numpy()
    old_zones = window_scores["zone"].fillna("green").to_numpy()

    new_zones = np.array(["green"] * len(window_scores), dtype=object)
    new_zones[usable] = classify_batch(scores[usable].tolist(), new_thresholds)

    if args.red_only:
        # Recode 'yellow' -> 'green' so every downstream consumer sees a red/green-only sequence.
        old_zones = np.where(old_zones == "red", "red", "green")
        new_zones = np.where(new_zones == "red", "red", "green")
        logger.info("--red-only: treating only 'red' as flagged (yellow recoded to green).")

    gt = load_window_ground_truth(args.combined_data, args.event_labels, window_size=args.window_rows, timestamp_column=args.timestamp_column)
    gt = gt.filter_to_window_ids(window_scores["window_id"].to_numpy())
    y_true = gt.is_anomaly

    # PR-AUC is threshold-independent (anomaly_score unchanged) so it's not repeated here.
    old_window = pointwise_metrics(y_true, old_zones, scores, usable=usable)
    new_window = pointwise_metrics(y_true, new_zones, scores, usable=usable)

    relevant_node_ids = set()
    for ids in gt.all_node_ids:
        relevant_node_ids.update(ids)
    all_nodes = build_node_spans(args.event_labels, equipment_tag=args.equipment_tag)
    relevant_nodes = all_nodes[all_nodes["node_id"].isin(relevant_node_ids)]

    persistence_kwargs = dict(
        red_to_yellow_alert=args.red_to_yellow_alert, red_to_red_alert=args.red_to_red_alert,
        yellow_consecutive=args.yellow_consecutive, mixed_window_hours=args.mixed_window_hours,
    )
    old_alert_levels = compute_alert_level_timeline(old_zones.tolist(), **persistence_kwargs)
    new_alert_levels = compute_alert_level_timeline(new_zones.tolist(), **persistence_kwargs)
    old_episodes = extract_episodes(old_alert_levels, window_scores["window_start"].tolist(), window_scores["window_end"].tolist())
    new_episodes = extract_episodes(new_alert_levels, window_scores["window_start"].tolist(), window_scores["window_end"].tolist())

    old_episode_metrics = episode_precision_recall(old_episodes, relevant_nodes)
    new_episode_metrics = episode_precision_recall(new_episodes, relevant_nodes)

    print()
    print("=" * 78)
    print("PREVALENCE-AWARE THRESHOLD RECALIBRATION" + (" (RED ONLY)" if args.red_only else ""))
    print("=" * 78)
    print(f"Observed event prevalence (whole file): {100 * prevalence:.3f}%")
    print(f"  Old:  green_yellow_pct={old_thresholds['green_yellow_percentile']:.1f}  yellow_red_pct={old_thresholds['yellow_red_percentile']:.1f}")
    print(f"  New:  green_yellow_pct={green_yellow_pct_new:.2f}  yellow_red_pct={yellow_red_pct_new:.2f}")
    print()
    print("-- Window-level (any_slug) --")
    print(f"  {'':10s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'n_flagged(Y+R)':>15s}")
    for label, m, zones in [("old", old_window, old_zones), ("new", new_window, new_zones)]:
        n_flagged = int(np.sum(np.asarray(zones) != "green"))
        print(f"  {label:10s} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {n_flagged:>15d}")
    print()
    print("-- Episode-level (vs. relevant labeled event nodes) --")
    print(f"  {'':10s} {'precision':>10s} {'recall':>8s} {'n_episodes':>11s} {'n_nodes':>8s}")
    for label, m in [("old", old_episode_metrics), ("new", new_episode_metrics)]:
        print(f"  {label:10s} {m['episode_precision']:>10.3f} {m['episode_recall']:>8.3f} {m['n_model_episodes']:>11d} {m['n_relevant_nodes']:>8d}")
    print("=" * 78)

    if args.output:
        result = {
            "prevalence": prevalence,
            "old_thresholds": old_thresholds,
            "new_thresholds": new_thresholds,
            "old_window_metrics": old_window,
            "new_window_metrics": new_window,
            "old_episode_metrics": old_episode_metrics,
            "new_episode_metrics": new_episode_metrics,
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Saved full comparison to %s", args.output)


if __name__ == "__main__":
    main()
