#!/usr/bin/env python
"""Evaluate anomaly-detection accuracy against injected ground truth labels.

The synthetic data generator (generate_synthetic_data.py) injects known
shutdown/transient periods and anomaly spikes, and can save their row ranges
and affected sensors to a separate `*.events.json` file. This script:

    1. Loads that ground truth (never fed to cleaning/training/inference).
    2. Loads a trained model's artefacts (scaler, weights, thresholds).
    3. Re-windows the *raw* full dataset (same 120-row grid used in training)
       and runs each window through the real inference path (infer_window +
       classify_zone) -- the exact code path production inference uses.
    4. Labels each window "normal" / "shutdown" / "anomaly" by row-range
       overlap with the injected events, and compares that label against
       the model's zone classification and continuous anomaly score.
    5. Reports precision/recall/F1/AUC for anomaly detection, and separately
       checks whether the top-contributing sensors the model reports match
       the sensors that were actually spiked.

Usage:
    python scripts/evaluate_detection.py \\
        --data data/synthetic_full.parquet \\
        --events data/synthetic_full.events.json \\
        --model-dir output_full_shuffled/artefacts \\
        --output output_full_shuffled/evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from autoencoder.logging_config import setup_logging
from autoencoder.data.ingestion import load_data
from autoencoder.data.preprocessing import construct_windows_with_metadata
from autoencoder.artefacts.serialisation import load_artefacts
from autoencoder.inference.pipeline import infer_window
from autoencoder.alerting.zones import classify_zone

setup_logging()
logger = logging.getLogger(__name__)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def label_windows(windows: list[dict], events: list[dict], sensor_columns: list[str]) -> list[dict]:
    """Attach a ground-truth label + matched spiked sensor names to each window.

    Priority: a window overlapping any "anomaly" event is labelled "anomaly"
    (even if it also overlaps a shutdown); otherwise "shutdown" if it overlaps
    a shutdown event; otherwise "normal".
    """
    anomaly_events = [e for e in events if e["type"] == "anomaly"]
    shutdown_events = [e for e in events if e["type"] == "shutdown"]
    sensor_set = set(sensor_columns)

    labelled = []
    for w in windows:
        s, e = w["start_idx"], w["end_idx"]
        matched_sensors: set[str] = set()
        label = "normal"

        for ev in anomaly_events:
            if _overlaps(s, e, ev["start_idx"], ev["end_idx"]):
                label = "anomaly"
                matched_sensors.update(f"tag_{i + 1:03d}" for i in ev["sensors"])

        if label != "anomaly":
            for ev in shutdown_events:
                if _overlaps(s, e, ev["start_idx"], ev["end_idx"]):
                    label = "shutdown"
                    break

        labelled.append({
            **w,
            "label": label,
            "matched_sensors": sorted(matched_sensors & sensor_set),
        })
    return labelled


def run_inference(
    labelled_windows, model, scaler, sensor_baselines, sensor_columns, thresholds, config,
    tail_compression_scale=None,
):
    inference_cfg = config.get("inference", {})
    flag_threshold = inference_cfg.get("sensor_flag_threshold", 3.0)
    anomaly_sensor_pct = inference_cfg.get("anomaly_sensor_pct", 10.0)
    missing_data_cfg = inference_cfg.get("missing_data", {})
    top_k = min(10, max(1, len(sensor_columns) // 5))

    records = []
    for w in labelled_windows:
        result = infer_window(
            window=w["data"],
            model=model,
            scaler=scaler,
            sensor_names=sensor_columns,
            top_k=top_k,
            sensor_baselines=sensor_baselines,
            flag_threshold=flag_threshold,
            anomaly_sensor_pct=anomaly_sensor_pct,
            max_null_pct=missing_data_cfg.get("max_null_pct_per_sensor", 5.0),
            max_consecutive_nulls=missing_data_cfg.get("max_consecutive_nulls_ffill", 3),
            max_null_dominant_sensor_pct=missing_data_cfg.get("max_null_dominant_sensor_pct", 30.0),
            already_scaled=False,
            tail_compression_scale=tail_compression_scale,
        )
        zone = classify_zone(result.window_score, thresholds) if result.usable else "green"
        top_names = [c["name"] for c in result.top_contributors] if result.usable else []
        masked_names = [sensor_columns[i] for i in result.masked_sensors] if result.masked_sensors else []

        records.append({
            "window_id": w["window_id"],
            "start_idx": w["start_idx"],
            "end_idx": w["end_idx"],
            "label": w["label"],
            "matched_sensors": w["matched_sensors"],
            "score": result.window_score,
            "zone": zone,
            "usable": result.usable,
            "top_sensors": top_names,
            "masked_sensors": masked_names,
        })
    return records


def compute_metrics(records: list[dict], top_k: int) -> dict:
    """Compute all detection-quality numbers from labelled inference records.

    Pure numeric core shared by `summarize()` (human-readable report) and any
    programmatic caller (e.g. the ablation runner) that wants structured
    values instead of parsing formatted text.
    """
    df = pd.DataFrame(records)

    n_normal = int((df["label"] == "normal").sum())
    n_anomaly = int((df["label"] == "anomaly").sum())
    n_shutdown = int((df["label"] == "shutdown").sum())

    det = df[df["label"].isin(["normal", "anomaly"])].copy()
    y_true = (det["label"] == "anomaly").astype(int).values
    y_score = det["score"].values

    zone_metrics = {}
    for pred_name, positive_zones in [("any_flag", {"yellow", "red"}), ("red_only", {"red"})]:
        y_pred = det["zone"].isin(positive_zones).astype(int).values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        zone_metrics[pred_name] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "accuracy": float(accuracy),
        }

    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))

    shutdown_flagged_pct = None
    shutdown = df[df["label"] == "shutdown"]
    if len(shutdown):
        shutdown_flagged_pct = float(shutdown["zone"].isin({"yellow", "red"}).mean() * 100)

    tp_anomaly = det[(det["label"] == "anomaly") & (det["zone"].isin({"yellow", "red"})) & (det["matched_sensors"].map(len) > 0)]
    sensor_attribution = None
    if len(tp_anomaly):
        recalls_at_k = []
        n_masked_truth = 0
        for _, row in tp_anomaly.iterrows():
            truth = set(row["matched_sensors"])
            masked = set(row.get("masked_sensors") or [])
            if truth & masked:
                # A genuinely spiked sensor was itself null-dominant in this
                # window -- diagnose_window() forces masked sensors' error to
                # 0 before ranking, so it structurally cannot appear in
                # top_sensors regardless of model quality. Counted separately
                # so a low recall@k here isn't misread as a model failure.
                n_masked_truth += 1
            top = set(row["top_sensors"][:top_k])
            recalls_at_k.append(len(truth & top) / len(truth))
        sensor_attribution = {
            "top_k": top_k,
            "n_windows": int(len(tp_anomaly)),
            "mean": float(np.mean(recalls_at_k)),
            "median": float(np.median(recalls_at_k)),
            "n_windows_with_masked_truth_sensor": n_masked_truth,
        }

    usable = df[df["usable"]]
    masking = None
    if "masked_sensors" in usable.columns:
        n_masked_windows = int(usable["masked_sensors"].map(len).gt(0).sum())
        if n_masked_windows:
            sensor_counts = Counter()
            for sensors in usable["masked_sensors"]:
                sensor_counts.update(sensors)
            masking = {
                "n_windows": n_masked_windows,
                "pct_of_usable": float(n_masked_windows / len(usable) * 100) if len(usable) else 0.0,
                "top_masked_sensors": sensor_counts.most_common(15),
            }

    return {
        "n_windows": len(df),
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "n_shutdown": n_shutdown,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "zones": zone_metrics,
        "shutdown_flagged_pct": shutdown_flagged_pct,
        "sensor_attribution": sensor_attribution,
        "masking": masking,
    }


def summarize(records: list[dict], top_k: int) -> str:
    m = compute_metrics(records, top_k)
    lines = []

    lines.append(
        f"Windows evaluated: {m['n_windows']}  "
        f"(normal={m['n_normal']}, anomaly={m['n_anomaly']}, shutdown={m['n_shutdown']})"
    )
    lines.append(
        "Caveat: ~85-90% of 'normal' windows were part of the training set (only cleaning-flagged "
        "windows were excluded), so normal-class precision is somewhat optimistic. Nearly all "
        "'anomaly' windows were excluded from training by the cleaning pipeline, so anomaly recall "
        "is a fair measure of generalized detection."
    )
    lines.append("")

    for pred_name, label in [("any_flag", "any-flag (Yellow+Red)"), ("red_only", "red-only")]:
        zm = m["zones"][pred_name]
        lines.append(f"[{label}]")
        lines.append(f"  Confusion matrix: TP={zm['tp']} FP={zm['fp']} FN={zm['fn']} TN={zm['tn']}")
        lines.append(
            f"  Precision={zm['precision']:.3f}  Recall={zm['recall']:.3f}  "
            f"F1={zm['f1']:.3f}  Accuracy={zm['accuracy']:.3f}"
        )
        lines.append("")

    lines.append(f"ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  (continuous score vs. true label, threshold-independent)")
    lines.append("")

    if m["shutdown_flagged_pct"] is not None:
        lines.append(
            f"Shutdown/transient windows flagged Yellow+Red: {m['shutdown_flagged_pct']:.1f}% of {m['n_shutdown']} "
            "(informational only -- these are not faults; spec excludes them from training)"
        )
        lines.append("")

    sa = m["sensor_attribution"]
    if sa:
        lines.append(
            f"Sensor attribution recall@{sa['top_k']} on {sa['n_windows']} correctly-flagged anomaly windows: "
            f"mean={sa['mean']:.3f}  median={sa['median']:.3f}"
        )
        if sa["n_windows_with_masked_truth_sensor"]:
            lines.append(
                f"  Caveat: {sa['n_windows_with_masked_truth_sensor']} of those windows had a truly-spiked "
                "sensor that was ALSO null-dominant (masked out of scoring) in that window -- it structurally "
                "cannot appear in top_sensors regardless of model quality, which pulls recall@k down for "
                "reasons unrelated to detection accuracy."
            )
    else:
        lines.append("No correctly-flagged anomaly windows with known spiked sensors to score attribution on.")
    lines.append("")

    mk = m["masking"]
    if mk:
        lines.append(
            f"Null-dominant sensors masked out (window still scored on the rest): "
            f"{mk['n_windows']} / usable windows ({mk['pct_of_usable']:.1f}%)"
        )
        for name, count in mk["top_masked_sensors"]:
            lines.append(f"  {name:35s} masked in {count} windows")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection accuracy against injected ground truth.")
    parser.add_argument("--data", required=True, help="Path to raw sensor data (CSV/Parquet)")
    parser.add_argument("--events", required=True, help="Path to *.events.json ground truth")
    parser.add_argument("--model-dir", required=True, help="Path to trained model artefacts directory")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument("--output", required=True, help="Output directory for per-window results + report")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    ts_col = config.get("data", {}).get("timestamp_column", "datetime")

    model, scaler, thresholds, metadata, _, sensor_baselines = load_artefacts(args.model_dir)
    sensor_columns = metadata["sensor_columns"]

    with open(args.events) as f:
        events_payload = json.load(f)
    events = events_payload["events"]

    df = load_data(args.data, timestamp_column=ts_col)
    windows = construct_windows_with_metadata(df, window_size=120, timestamp_column=ts_col, sensor_columns=sensor_columns)
    logger.info("Constructed %d windows from %d raw rows", len(windows), len(df))

    labelled = label_windows(windows, events, sensor_columns)
    records = run_inference(
        labelled, model, scaler, sensor_baselines, sensor_columns, thresholds, config,
        tail_compression_scale=metadata.get("tail_compression_scale"),
    )

    top_k = min(10, max(1, len(sensor_columns) // 5))
    report = summarize(records, top_k)
    print(report)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(records)
    out_df["matched_sensors"] = out_df["matched_sensors"].map(",".join)
    out_df["top_sensors"] = out_df["top_sensors"].map(",".join)
    out_df["masked_sensors"] = out_df["masked_sensors"].map(",".join)
    out_df.to_csv(output_dir / "window_results.csv", index=False)

    with open(output_dir / "report.txt", "w") as f:
        f.write(report)

    logger.info("Saved per-window results and report to %s", output_dir)


if __name__ == "__main__":
    main()
