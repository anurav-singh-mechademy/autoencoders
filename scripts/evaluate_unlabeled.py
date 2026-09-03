#!/usr/bin/env python
"""Evaluate a trained model against real data that has NO ground-truth labels.

Unlike scripts/evaluate_detection.py (which scores precision/recall/F1/AUC
against injected synthetic events), a real historian export has no known
anomaly windows. Precision, recall, F1, and AUC are undefined without ground
truth -- this script deliberately does NOT compute or report them, or any
proxy standing in for them. What it *can* report, from the model's zone/score
output alone:

    1. Zone distribution (Green/Yellow/Red) and score percentiles across the
       whole dataset, not just a held-out split -- how often the model would
       flag this equipment in practice.
    2. Persistence-alert behaviour over time (how many sustained alert
       episodes, not just noisy single-window flags).
    3. Which sensors most often drive the flagged windows -- a domain expert
       can sanity-check this against known equipment history even without
       row-level labels.
    4. Data-quality/usability stats (windows rejected for missing data).

Usage:
    python scripts/evaluate_unlabeled.py \\
        --data data/dvn_data_1_clean.csv \\
        --model-dir output_dvn/artefacts \\
        --config configs/dvn.yaml \\
        --output output_dvn/evaluation
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging
from autoencoder.data.ingestion import load_data
from autoencoder.data.preprocessing import construct_windows_with_metadata
from autoencoder.artefacts.serialisation import load_artefacts
from autoencoder.inference.pipeline import infer_window
from autoencoder.alerting.zones import classify_zone
from autoencoder.alerting.persistence import apply_persistence

setup_logging()
logger = logging.getLogger(__name__)


def run_inference(windows, model, scaler, sensor_baselines, sensor_columns, thresholds, config):
    inference_cfg = config.get("inference", {})
    flag_threshold = inference_cfg.get("sensor_flag_threshold", 3.0)
    anomaly_sensor_pct = inference_cfg.get("anomaly_sensor_pct", 10.0)
    top_k = min(10, max(1, len(sensor_columns) // 5))

    records = []
    for w in windows:
        result = infer_window(
            window=w["data"],
            model=model,
            scaler=scaler,
            sensor_names=sensor_columns,
            top_k=top_k,
            sensor_baselines=sensor_baselines,
            flag_threshold=flag_threshold,
            anomaly_sensor_pct=anomaly_sensor_pct,
            already_scaled=False,
        )
        zone = classify_zone(result.window_score, thresholds) if result.usable else None
        top_names = [c["name"] for c in result.top_contributors] if result.usable else []

        records.append({
            "window_id": w["window_id"],
            "start_time": w["start_time"],
            "end_time": w["end_time"],
            "score": result.window_score,
            "zone": zone,
            "usable": result.usable,
            "quality_flags": ",".join(result.quality_flags),
            "top_sensors": top_names,
        })
    return records


def apply_persistence_timeline(records, persistence_cfg):
    """Incrementally compute the alert level after each window (streaming,
    same rules as alerting.persistence.apply_persistence but O(n) total
    instead of re-scanning full history per window)."""
    red_to_yellow = persistence_cfg.get("red_to_yellow_alert", 2)
    red_to_red = persistence_cfg.get("red_to_red_alert", 3)
    yellow_consec = persistence_cfg.get("yellow_consecutive", 4)
    mixed_window_hours = persistence_cfg.get("mixed_window_hours")
    window_minutes = persistence_cfg.get("window_minutes", 30)

    lookback_windows = None
    trailing_span: deque = deque(maxlen=0)
    if mixed_window_hours is not None and mixed_window_hours > 0:
        lookback_windows = max(1, round(mixed_window_hours * 60 / window_minutes))
        trailing_span = deque(maxlen=lookback_windows)

    consecutive_red = 0
    consecutive_yellow = 0
    alert_levels = []
    for r in records:
        zone = r["zone"] if r["usable"] else "green"
        if zone == "red":
            consecutive_red += 1
            consecutive_yellow = 0
        elif zone == "yellow":
            consecutive_yellow += 1
            consecutive_red = 0
        else:
            consecutive_red = 0
            consecutive_yellow = 0

        if lookback_windows is not None:
            trailing_span.append(zone)
            mixed_red_count = sum(1 for z in trailing_span if z == "red")
            mixed_yellow_count = sum(1 for z in trailing_span if z == "yellow")
        else:
            mixed_red_count = mixed_yellow_count = 0

        if consecutive_red >= red_to_red:
            level = "red"
        elif consecutive_red >= red_to_yellow:
            level = "yellow"
        elif consecutive_yellow >= yellow_consec:
            level = "yellow"
        elif mixed_red_count >= 1 and mixed_yellow_count >= 1 and (2 * mixed_red_count + mixed_yellow_count) >= 4:
            level = "yellow"
        else:
            level = "green"
        alert_levels.append(level)
    return alert_levels


def count_episodes(levels: list[str], level: str) -> int:
    """Count contiguous runs of `level` in the alert-level timeline."""
    episodes = 0
    in_run = False
    for l in levels:
        if l == level and not in_run:
            episodes += 1
            in_run = True
        elif l != level:
            in_run = False
    return episodes


def summarize(records: list[dict], alert_levels: list[str], top_k: int) -> str:
    df = pd.DataFrame(records)
    n_total = len(df)
    n_usable = int(df["usable"].sum())
    n_unusable = n_total - n_usable

    lines = []
    lines.append(
        "=" * 78 + "\n"
        "IMPORTANT: This is a LABEL-FREE evaluation.\n"
        "dvn_data_1.csv has no ground-truth anomaly annotations. Precision, "
        "recall,\nF1, and ROC/PR-AUC all require knowing which windows are "
        "actually anomalous --\nthey are mathematically undefined here and "
        "are intentionally NOT reported\nanywhere in this file. Treat "
        "everything below as a description of how the\nmodel *behaves* on "
        "this data (how often / where it flags things), not as a\nmeasure of "
        "how *accurate* those flags are. Accuracy validation requires either\n"
        "labelled fault events or field engineer sign-off on specific flagged "
        "periods.\n" + "=" * 78
    )
    lines.append("")

    lines.append(f"Windows constructed: {n_total}")
    lines.append(f"  Usable:   {n_usable} ({n_usable / n_total * 100:.1f}%)")
    lines.append(f"  Unusable: {n_unusable} ({n_unusable / n_total * 100:.1f}%) -- excluded from all stats below")
    if n_unusable:
        reasons = Counter()
        for flags in df.loc[~df["usable"], "quality_flags"]:
            for f in str(flags).split(","):
                if f:
                    reasons[f] += 1
        lines.append(f"    Reasons: {dict(reasons)}")
    lines.append("")

    usable = df[df["usable"]]
    zone_counts = usable["zone"].value_counts()
    lines.append("Zone distribution (usable windows):")
    for z in ["green", "yellow", "red"]:
        n = int(zone_counts.get(z, 0))
        pct = n / len(usable) * 100 if len(usable) else 0.0
        lines.append(f"  {z.upper():7s} {n:6d}  ({pct:5.1f}%)")
    lines.append("")

    scores = usable["score"].dropna().values
    if len(scores):
        pctiles = np.percentile(scores, [50, 90, 95, 99, 100])
        lines.append(
            f"Score distribution: min={scores.min():.6f}  "
            f"p50={pctiles[0]:.6f}  p90={pctiles[1]:.6f}  p95={pctiles[2]:.6f}  "
            f"p99={pctiles[3]:.6f}  max={pctiles[4]:.6f}"
        )
    lines.append("")

    lines.append("Persistence-alert timeline (consecutive-window rules from config):")
    n_red_alert = alert_levels.count("red")
    n_yellow_alert = alert_levels.count("yellow")
    lines.append(f"  Windows at RED alert:    {n_red_alert} ({n_red_alert / n_total * 100:.1f}%)")
    lines.append(f"  Windows at YELLOW alert: {n_yellow_alert} ({n_yellow_alert / n_total * 100:.1f}%)")
    lines.append(f"  Distinct RED alert episodes:    {count_episodes(alert_levels, 'red')}")
    lines.append(f"  Distinct YELLOW alert episodes: {count_episodes(alert_levels, 'yellow')}")
    lines.append("")

    flagged = usable[usable["zone"].isin(["yellow", "red"])]
    sensor_counter = Counter()
    for sensors in flagged["top_sensors"]:
        sensor_counter.update(sensors[:top_k])
    lines.append(f"Top sensors implicated across {len(flagged)} flagged (Yellow/Red) windows:")
    if sensor_counter:
        for name, count in sensor_counter.most_common(15):
            lines.append(f"  {name:35s} appeared in top-{top_k} for {count} windows "
                         f"({count / len(flagged) * 100:.1f}% of flagged)")
    else:
        lines.append("  (no flagged windows)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Label-free evaluation of a trained model on real data.")
    parser.add_argument("--data", required=True, help="Path to raw (or cleaned) sensor data (CSV/Parquet)")
    parser.add_argument("--model-dir", required=True, help="Path to trained model artefacts directory")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument("--output", required=True, help="Output directory for per-window results + report")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    ts_col = config.get("data", {}).get("timestamp_column", "datetime")
    window_rows = config.get("data", {}).get("window_rows", 120)
    persistence_cfg = config.get("alerting", {}).get("persistence", {})

    model, scaler, thresholds, metadata, _, sensor_baselines = load_artefacts(args.model_dir)
    sensor_columns = metadata["sensor_columns"]

    df = load_data(args.data, timestamp_column=ts_col)
    windows = construct_windows_with_metadata(
        df, window_size=window_rows, timestamp_column=ts_col, sensor_columns=sensor_columns,
    )
    logger.info("Constructed %d windows from %d raw rows", len(windows), len(df))

    records = run_inference(windows, model, scaler, sensor_baselines, sensor_columns, thresholds, config)
    alert_levels = apply_persistence_timeline(records, persistence_cfg)

    top_k = min(10, max(1, len(sensor_columns) // 5))
    report = summarize(records, alert_levels, top_k)
    print(report)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(records)
    out_df["alert_level"] = alert_levels
    out_df["top_sensors"] = out_df["top_sensors"].map(",".join)
    out_df.to_csv(output_dir / "window_results_unlabeled.csv", index=False)

    with open(output_dir / "report_unlabeled.txt", "w") as f:
        f.write(report)

    logger.info("Saved per-window results and report to %s", output_dir)


if __name__ == "__main__":
    main()
