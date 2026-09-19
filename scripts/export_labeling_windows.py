#!/usr/bin/env python
"""CLI: Export raw sensor data + sensor-attribution scores for flagged
(Yellow/Red) windows, for a manual SME labeling interface to consume.

This does NOT re-run cleaning or use any cached per-sensor error array (none
is persisted by score_full_timeline.py, to avoid bloating window_scores.parquet
with a per-sensor float for every sensor in every scored window). Instead, for
just the flagged windows -- a small subset -- it re-runs the trained model's
own inference from scratch, using the exact same artefacts (model/scaler/
thresholds/tail_compression_scale) score_full_timeline.py used, to recover the
full per-sensor error ranking with real attribution values attached.

Correctness is the whole point of this script, so every window is verified
two independent ways before being written out:
  1. The raw row-slice's own first/last timestamps must exactly match that
     window_id's window_start/window_end already recorded in window_scores.parquet.
  2. The freshly recomputed anomaly_score must match window_scores.parquet's
     stored anomaly_score for that window_id, within a tight tolerance.
Either check failing raises immediately -- it means --data, --window-scores,
or --model-dir don't actually correspond to the same run, and silently
exporting mismatched windows would be worse than not exporting at all.

Usage:
    python scripts/export_labeling_windows.py \\
        --data data/5K512B/5K512B_combined_with_events_clean.parquet \\
        --window-scores output_5K512B/evaluation/window_scores.parquet \\
        --model-dir output_5K512B/artefacts \\
        --output-dir output_5K512B/labeling \\
        --equipment-id 5K512B
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging
from autoencoder.data.ingestion import read_parquet_robust
from autoencoder.artefacts.serialisation import load_artefacts
from autoencoder.inference.pipeline import infer_window

setup_logging()
logger = logging.getLogger(__name__)

# Relative tolerance for the recomputed-anomaly-score cross-check. Inference
# is deterministic (no dropout/randomness at eval time), so a real match
# should agree to float32 precision -- this only needs to be loose enough to
# absorb float32 round-trip noise through parquet, not model drift.
_SCORE_RTOL = 1e-4


def _load_timestamped_data(path: str, timestamp_column: str) -> pd.DataFrame:
    """Identical convention to score_full_timeline.py's own loader -- window_id
    is only meaningful relative to this exact row order, so any divergence
    here (different sort, different index reset) would silently desync every
    window_id from what was actually scored."""
    if path.endswith(".parquet"):
        df = read_parquet_robust(path)
    else:
        df = pd.read_csv(path)

    if timestamp_column not in df.columns:
        df = df.reset_index()
        if timestamp_column not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": timestamp_column})

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(timestamp_column).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Export raw window data + sensor attribution for flagged windows, for SME labeling."
    )
    parser.add_argument("--data", required=True, help="The SAME *_combined_with_events_clean.parquet passed to score_full_timeline.py")
    parser.add_argument("--window-scores", required=True, help="Output of scripts/score_full_timeline.py, from that same --data")
    parser.add_argument("--model-dir", required=True, help="Path to the trained model artefacts directory used to produce --window-scores")
    parser.add_argument("--output-dir", required=True, help="Directory to write the index + per-window files into")
    parser.add_argument("--equipment-id", required=True, help="Equipment identifier, used only for output file naming")
    parser.add_argument("--zones", default="yellow,red", help="Comma-separated zones to export (default: yellow,red)")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--window-rows", type=int, default=120, help="Must match score_full_timeline.py's --window-rows")
    parser.add_argument("--preview-top-k", type=int, default=5, help="How many top sensors to include in the index's quick-preview list")
    parser.add_argument("--sensor-flag-threshold", type=float, default=3.0)
    parser.add_argument("--anomaly-sensor-pct", type=float, default=10.0)
    parser.add_argument("--max-null-pct", type=float, default=5.0)
    parser.add_argument("--max-consecutive-nulls", type=int, default=3)
    parser.add_argument("--max-null-dominant-sensor-pct", type=float, default=30.0)
    args = parser.parse_args()

    zones_wanted = {z.strip() for z in args.zones.split(",") if z.strip()}

    model, scaler, thresholds, metadata, _, sensor_baselines = load_artefacts(args.model_dir)
    sensor_columns = metadata["sensor_columns"]
    n_sensors = len(sensor_columns)
    tail_compression_scale = metadata.get("tail_compression_scale")
    if tail_compression_scale is None:
        logger.warning(
            "No tail_compression_scale in %s/training_metadata.json -- recomputed scores will be "
            "wrong if this model was trained with preprocessing.tail_compression_scale set.",
            args.model_dir,
        )

    window_scores = pd.read_parquet(args.window_scores) if args.window_scores.endswith(".parquet") else pd.read_csv(args.window_scores)
    target = window_scores[window_scores["usable"] & window_scores["zone"].isin(zones_wanted)].copy()
    logger.info(
        "Found %d / %d windows in zones %s to export.",
        len(target), len(window_scores), sorted(zones_wanted),
    )
    if len(target) == 0:
        logger.warning("Nothing to export -- exiting.")
        return

    df = _load_timestamped_data(args.data, args.timestamp_column)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), args.data)

    output_dir = Path(args.output_dir)
    windows_dir = output_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    index_entries = []
    for _, row in target.iterrows():
        window_id = int(row["window_id"])
        start = window_id * args.window_rows
        end = start + args.window_rows
        if end > len(df):
            raise ValueError(
                f"window_id={window_id} needs rows [{start}:{end}) but --data only has {len(df)} rows -- "
                "--data does not match the file --window-scores was produced from."
            )
        chunk = df.iloc[start:end]

        # Check 1: raw row-slice timestamps must exactly match what was
        # already recorded for this window_id at scoring time.
        actual_start = chunk[args.timestamp_column].iloc[0]
        actual_end = chunk[args.timestamp_column].iloc[-1]
        expected_start = pd.Timestamp(row["window_start"])
        expected_end = pd.Timestamp(row["window_end"])
        if actual_start != expected_start or actual_end != expected_end:
            raise ValueError(
                f"window_id={window_id}: raw row-slice spans [{actual_start} .. {actual_end}], but "
                f"window_scores.parquet recorded [{expected_start} .. {expected_end}] for this window_id -- "
                "--data does not correspond to the same file --window-scores was produced from."
            )

        raw_values = chunk[sensor_columns].to_numpy(dtype=np.float64)  # raw physical units, unscaled

        result = infer_window(
            window=raw_values,
            model=model,
            scaler=scaler,
            sensor_names=sensor_columns,
            top_k=n_sensors,  # every sensor, not the default top-10 -- the UI needs the full ranking
            sensor_baselines=sensor_baselines,
            flag_threshold=args.sensor_flag_threshold,
            anomaly_sensor_pct=args.anomaly_sensor_pct,
            max_null_pct=args.max_null_pct,
            max_consecutive_nulls=args.max_consecutive_nulls,
            max_null_dominant_sensor_pct=args.max_null_dominant_sensor_pct,
            already_scaled=False,
            tail_compression_scale=tail_compression_scale,
        )

        # Check 2: recomputed score must agree with what was already scored.
        stored_score = float(row["anomaly_score"])
        if not result.usable:
            raise ValueError(
                f"window_id={window_id} was usable=True in window_scores.parquet but re-inference now "
                "finds it unusable -- --data or --model-dir has changed since scoring."
            )
        if not np.isclose(result.window_score, stored_score, rtol=_SCORE_RTOL):
            raise ValueError(
                f"window_id={window_id}: recomputed anomaly_score={result.window_score:.6g} does not match "
                f"window_scores.parquet's stored value={stored_score:.6g} (rtol={_SCORE_RTOL}) -- "
                "--data, --window-scores, and --model-dir must all correspond to the same run."
            )

        error_by_sensor = {c["name"]: c["error"] for c in result.top_contributors}
        contribution_pct_by_sensor = {c["name"]: c["contribution_pct"] for c in result.top_contributors}
        ranked_sensors = [c["name"] for c in result.top_contributors]  # already sorted descending
        # Continuous error/baseline ratio per sensor (e.g. 3.2 = 3.2x this
        # sensor's own normal reconstruction error) -- a much richer
        # attribution signal for an SME than the binary flag derived from it.
        error_ratio_by_sensor = (
            {sensor_columns[i]: float(result.error_ratios[i]) for i in range(n_sensors)}
            if result.error_ratios is not None else {}
        )
        flagged_by_sensor = (
            {sensor_columns[i]: bool(result.sensor_flags[i]) for i in range(n_sensors)}
            if result.sensor_flags is not None else {}
        )

        timestamps = [t.isoformat() for t in chunk[args.timestamp_column]]
        raw_by_sensor = {
            sensor_columns[i]: raw_values[:, i].tolist() for i in range(n_sensors)
        }

        window_record = {
            "equipment_id": args.equipment_id,
            "window_id": window_id,
            "window_start": actual_start.isoformat(),
            "window_end": actual_end.isoformat(),
            "zone": row["zone"],
            "anomaly_score": stored_score,
            "sensor_columns": sensor_columns,
            "masked_sensors": [sensor_columns[i] for i in result.masked_sensors] if result.masked_sensors else [],
            "quality_flags": result.quality_flags,
            "timestamps": timestamps,
            "raw_values": raw_by_sensor,
            "ranked_sensors": ranked_sensors,
            "sensor_error": error_by_sensor,
            "sensor_contribution_pct": contribution_pct_by_sensor,
            "sensor_error_ratio": error_ratio_by_sensor,
            "sensor_flagged": flagged_by_sensor,
        }

        window_filename = f"{args.equipment_id}_w{window_id}.json"
        with open(windows_dir / window_filename, "w") as f:
            json.dump(window_record, f)

        index_entries.append({
            "window_id": window_id,
            "window_start": actual_start.isoformat(),
            "window_end": actual_end.isoformat(),
            "zone": row["zone"],
            "anomaly_score": stored_score,
            "top_sensors": ranked_sensors[: args.preview_top_k],
            "file": f"windows/{window_filename}",
        })

    index_entries.sort(key=lambda e: e["anomaly_score"], reverse=True)
    index = {
        "equipment_id": args.equipment_id,
        "sensor_columns": sensor_columns,
        "window_rows": args.window_rows,
        "zones_exported": sorted(zones_wanted),
        "thresholds": thresholds,
        "n_windows": len(index_entries),
        "windows": index_entries,
    }
    index_path = output_dir / f"{args.equipment_id}_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info("Verified and wrote %d window files under %s", len(index_entries), windows_dir)
    logger.info("Wrote index: %s", index_path)


if __name__ == "__main__":
    main()
