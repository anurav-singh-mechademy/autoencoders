#!/usr/bin/env python
"""Ablation harness: sweep config variants through the real pipeline, score
each against injected ground-truth events, and emit a JSONL + markdown report.

Reuses the production code paths end-to-end -- main.step_clean / main.step_train
for cleaning+training, evaluate_detection's label_windows/run_inference/
compute_metrics for scoring -- so ablation results reflect exactly what the
real pipeline would produce for that config, not a simplified re-implementation.

Variants that only change threshold/inference config (not cleaning, model
architecture, or training) skip the clean+train step entirely and reuse the
most recently trained model -- see `UPSTREAM_KEYS` / `upstream_hash`.

Usage:
    python scripts/run_ablation.py \\
        --data data/synthetic_full.parquet \\
        --events data/synthetic_full.events.json \\
        --config configs/default.yaml \\
        --output-root output_ablation \\
        --tiers A,B

    # Fast correctness check before a full run (small epoch cap, one tier):
    python scripts/run_ablation.py --data ... --events ... --output-root /tmp/smoke \\
        --tiers A --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from main import step_clean, step_train, compute_training_errors  # noqa: E402
from evaluate_detection import label_windows, run_inference, compute_metrics  # noqa: E402
from ablation.variants import variants_for_tiers  # noqa: E402

from autoencoder.data.ingestion import load_data  # noqa: E402
from autoencoder.data.preprocessing import construct_windows_with_metadata  # noqa: E402
from autoencoder.training.dataset import split_windows, windows_to_numpy  # noqa: E402
from autoencoder.alerting.thresholds import compute_thresholds  # noqa: E402
from autoencoder.logging_config import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger("run_ablation")

# Config subtree that determines whether cleaning+training must rerun. Deliberately
# excludes "thresholds", "inference", "alerting" -- those are cheap post-hoc knobs
# applied on top of an already-trained model.
UPSTREAM_KEYS = ["data", "preprocessing", "regime", "transient", "cleaning", "model", "training"]


def deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)

    def merge(dst: dict, src: dict):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    merge(result, overrides)
    return result


def upstream_hash(config: dict) -> str:
    subset = {k: config.get(k) for k in UPSTREAM_KEYS}
    return hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:12]


class UpstreamCache:
    """Single-slot cache: only the most recently trained model is kept in memory.

    Variants sharing an upstream config must be adjacent in the variant list
    for this to pay off -- true for the registry in ablation/variants.py.
    """

    def __init__(self):
        self._hash = None
        self._entry = None

    def get_or_build(self, config: dict, data_path: str, output_root: Path, equipment_id: str) -> dict:
        h = upstream_hash(config)
        if h == self._hash:
            logger.info("Reusing cached upstream model (hash=%s)", h)
            return self._entry

        logger.info("Upstream config changed (hash=%s) -- running clean+train", h)
        variant_dir = output_root / f"_upstream_{h}"

        cleaned_arr, sensor_columns, scaler_result, regime_labels = step_clean(
            data_path, variant_dir, config, equipment_id,
        )
        model, _thresholds, sensor_baselines, test_arr, sensor_columns = step_train(
            cleaned_arr, sensor_columns, scaler_result.scaler, variant_dir, config,
            regime_labels=regime_labels,
        )
        training_errors = np.load(variant_dir / "artefacts" / "training_error_distribution.npy")

        entry = {
            "hash": h,
            "model": model,
            "scaler": scaler_result.scaler,
            "sensor_columns": sensor_columns,
            "sensor_baselines": sensor_baselines,
            "training_errors": training_errors,
            "cleaned_windows_list": [cleaned_arr[i] for i in range(len(cleaned_arr))],
            "regime_labels": regime_labels,
        }
        self._hash = h
        self._entry = entry
        return entry


def compute_variant_thresholds(config: dict, entry: dict) -> dict:
    """Compute Green/Yellow/Red thresholds for this variant's config.

    `calibration_source: val` recomputes the split (deterministic -- same
    train_pct/val_pct/regime_labels as step_train used internally) and scores
    the cached model on the val windows instead of reusing train errors.
    """
    thresh_cfg = config.get("thresholds", {})
    source = thresh_cfg.get("calibration_source", "train")

    if source == "val":
        train_cfg = config.get("training", {})
        _, val_w, _ = split_windows(
            entry["cleaned_windows_list"],
            train_pct=train_cfg.get("splits", {}).get("train", 0.85),
            val_pct=train_cfg.get("splits", {}).get("val", 0.10),
            regime_labels=entry["regime_labels"],
        )
        val_arr = windows_to_numpy(val_w)
        errors = compute_training_errors(entry["model"], val_arr, torch.device("cpu"))
    else:
        errors = entry["training_errors"]

    return compute_thresholds(
        errors,
        green_yellow_percentile=thresh_cfg.get("green_yellow", 90),
        yellow_red_percentile=thresh_cfg.get("yellow_red", 99),
        method=thresh_cfg.get("method", "robust"),
        spread=thresh_cfg.get("spread", "mad"),
    )


def evaluate_variant(entry: dict, thresholds: dict, config: dict, raw_windows: list[dict], events: list[dict]) -> dict:
    sensor_columns = entry["sensor_columns"]
    labelled = label_windows(raw_windows, events, sensor_columns)
    records = run_inference(
        labelled, entry["model"], entry["scaler"], entry["sensor_baselines"],
        sensor_columns, thresholds, config,
    )
    top_k = min(10, max(1, len(sensor_columns) // 5))
    return compute_metrics(records, top_k)


def flatten_metrics(
    run_id: str, base_run_id: str, seed: int | None, tier: str, group: str,
    config_diff: dict, upstream_hash_: str, m: dict,
) -> dict:
    return {
        "run_id": run_id,
        "base_run_id": base_run_id,
        "seed": seed,
        "tier": tier,
        "group": group,
        "config_diff": config_diff,
        "upstream_hash": upstream_hash_,
        "n_windows": m["n_windows"],
        "n_normal": m["n_normal"],
        "n_anomaly": m["n_anomaly"],
        "n_shutdown": m["n_shutdown"],
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "red_only_precision": m["zones"]["red_only"]["precision"],
        "red_only_recall": m["zones"]["red_only"]["recall"],
        "red_only_f1": m["zones"]["red_only"]["f1"],
        "any_flag_precision": m["zones"]["any_flag"]["precision"],
        "any_flag_recall": m["zones"]["any_flag"]["recall"],
        "any_flag_f1": m["zones"]["any_flag"]["f1"],
        "shutdown_flagged_pct": m["shutdown_flagged_pct"],
        "sensor_attribution_mean": m["sensor_attribution"]["mean"] if m["sensor_attribution"] else None,
        "sensor_attribution_median": m["sensor_attribution"]["median"] if m["sensor_attribution"] else None,
    }


def build_markdown_report(rows: list[dict], baseline_pr_auc_by_tier: dict) -> str:
    lines = ["# Ablation results\n"]
    tiers = sorted(set(r["tier"] for r in rows))
    for tier in tiers:
        tier_rows = [r for r in rows if r["tier"] == tier]
        lines.append(f"\n## Tier {tier} ({tier_rows[0]['group']})\n")
        lines.append("| run_id | PR-AUC | ROC-AUC | Red F1 | Any-flag Prec | Any-flag Recall | Attribution recall@k | ΔPR-AUC |")
        lines.append("|---|---|---|---|---|---|---|---|")
        baseline = baseline_pr_auc_by_tier.get(tier)
        for r in tier_rows:
            delta = f"{r['pr_auc'] - baseline:+.4f}" if baseline is not None else "—"
            attr = f"{r['sensor_attribution_mean']:.3f}" if r["sensor_attribution_mean"] is not None else "n/a"
            lines.append(
                f"| {r['run_id']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | "
                f"{r['red_only_f1']:.3f} | {r['any_flag_precision']:.3f} | {r['any_flag_recall']:.3f} | "
                f"{attr} | {delta} |"
            )

    lines.append("\n## Leaderboard (all variants, PR-AUC descending)\n")
    lines.append("| rank | run_id | tier | PR-AUC | Red F1 |")
    lines.append("|---|---|---|---|---|")
    for i, r in enumerate(sorted(rows, key=lambda r: -r["pr_auc"]), start=1):
        lines.append(f"| {i} | {r['run_id']} | {r['tier']} | {r['pr_auc']:.4f} | {r['red_only_f1']:.3f} |")

    return "\n".join(lines)


def build_multiseed_report(rows: list[dict]) -> str:
    """Aggregate (mean +/- std) across seeds for each base run_id, and flag
    deltas that don't clear 2x the observed noise (std) of their own tier."""
    lines = ["# Multi-seed ablation results\n"]

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["base_run_id"], []).append(r)

    tiers = sorted(set(r["tier"] for r in rows))
    for tier in tiers:
        base_ids = sorted(set(r["base_run_id"] for r in rows if r["tier"] == tier))
        tier_group = next(r["group"] for r in rows if r["tier"] == tier)
        lines.append(f"\n## Tier {tier} ({tier_group})\n")
        lines.append("| run_id | n_seeds | PR-AUC mean±std | Red F1 mean±std | Any-flag Prec mean±std |")
        lines.append("|---|---|---|---|---|")
        for base_id in base_ids:
            g = groups[base_id]
            pr = np.array([r["pr_auc"] for r in g])
            f1 = np.array([r["red_only_f1"] for r in g])
            prec = np.array([r["any_flag_precision"] for r in g])
            lines.append(
                f"| {base_id} | {len(g)} | {pr.mean():.4f}±{pr.std():.4f} | "
                f"{f1.mean():.3f}±{f1.std():.3f} | {prec.mean():.3f}±{prec.std():.3f} |"
            )

        # Noise floor for this tier: largest per-variant std observed.
        noise_floor = max(np.std([r["pr_auc"] for r in groups[bid]]) for bid in base_ids)
        default_id = next((bid for bid in base_ids if groups[bid][0]["config_diff"] == {}), base_ids[0])
        default_mean = np.mean([r["pr_auc"] for r in groups[default_id]])
        lines.append(f"\nNoise floor (max within-variant PR-AUC std) for tier {tier}: {noise_floor:.4f}\n")
        for base_id in base_ids:
            if base_id == default_id:
                continue
            variant_mean = np.mean([r["pr_auc"] for r in groups[base_id]])
            delta = variant_mean - default_mean
            verdict = "likely real" if abs(delta) > 2 * noise_floor else "NOT distinguishable from noise"
            lines.append(f"- {base_id} vs {default_id}: ΔPR-AUC={delta:+.4f} -- {verdict}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the ablation study over registered config variants.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tiers", default="A,B,C,F,H", help="Comma-separated tier letters to run.")
    parser.add_argument(
        "--seeds", default="42",
        help="Comma-separated training seeds. Iteration is seed-outer / variant-inner so "
             "cache reuse still applies within each seed (e.g. B0/B1/B2 share one retrain "
             "per seed, not one retrain per (variant, seed) pair). With >1 seed, an "
             "aggregated mean+-std report is written instead of the single-run report.",
    )
    parser.add_argument("--equipment-id", default="ablation")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Cap max_epochs and early-stopping patience low, for a fast end-to-end correctness check.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    if args.smoke_test:
        base_config = deep_merge(base_config, {
            "training": {"max_epochs": 2, "early_stopping_patience": 2},
        })
        logger.warning("Smoke-test mode: max_epochs=2. Results are NOT meaningful, only for verifying the harness runs.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ts_col = base_config.get("data", {}).get("timestamp_column", "datetime")
    df = load_data(args.data, timestamp_column=ts_col)

    with open(args.events) as f:
        events = json.load(f)["events"]

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    variants = variants_for_tiers(tiers)
    if not variants:
        logger.error("No variants registered for tiers %s", tiers)
        sys.exit(1)
    logger.info("Running %d variants across tiers %s", len(variants), tiers)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    logger.info("Seeds: %s", seeds)

    cache = UpstreamCache()
    results_path = output_root / "ablation_results.jsonl"
    rows = []

    with open(results_path, "w") as jsonl_f:
        for seed in seeds:
            # Group variants by upstream hash before running, so identical
            # (cleaning, model, training) configs that appear in different tiers
            # -- e.g. every tier's "_default" placeholder -- share one retrain
            # instead of each nominal tier triggering its own. Stable sort keeps
            # each hash-group in its original registry order.
            hashed = []
            for variant in variants:
                overrides_preview = deep_merge(variant["overrides"], {"training": {"seed": seed}})
                config_preview = deep_merge(base_config, overrides_preview)
                hashed.append((upstream_hash(config_preview), variant))
            hashed.sort(key=lambda t: t[0])

            for _, variant in hashed:
                base_run_id = variant["run_id"]
                run_id = base_run_id if len(seeds) == 1 else f"{base_run_id}__seed{seed}"
                logger.info("=== Variant %s (tier %s, seed=%s) ===", run_id, variant["tier"], seed)

                overrides = deep_merge(variant["overrides"], {"training": {"seed": seed}})
                config = deep_merge(base_config, overrides)
                entry = cache.get_or_build(config, args.data, output_root, args.equipment_id)

                thresholds = compute_variant_thresholds(config, entry)

                raw_windows = construct_windows_with_metadata(
                    df, window_size=120, timestamp_column=ts_col, sensor_columns=entry["sensor_columns"],
                )
                metrics = evaluate_variant(entry, thresholds, config, raw_windows, events)

                row = flatten_metrics(
                    run_id, base_run_id, seed, variant["tier"], variant["group"],
                    variant["overrides"], entry["hash"], metrics,
                )
                rows.append(row)
                jsonl_f.write(json.dumps(row) + "\n")
                jsonl_f.flush()

                logger.info(
                    "%s: PR-AUC=%.4f  red-F1=%.3f  any-flag-precision=%.3f",
                    run_id, row["pr_auc"], row["red_only_f1"], row["any_flag_precision"],
                )

    if len(seeds) == 1:
        baseline_pr_auc_by_tier = {}
        for tier in set(r["tier"] for r in rows):
            tier_rows = [r for r in rows if r["tier"] == tier]
            default_row = next((r for r in tier_rows if not r["config_diff"]), tier_rows[0])
            baseline_pr_auc_by_tier[tier] = default_row["pr_auc"]
        report = build_markdown_report(rows, baseline_pr_auc_by_tier)
        report_path = output_root / "ablation_report.md"
    else:
        report = build_multiseed_report(rows)
        report_path = output_root / "ablation_report_multiseed.md"

    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    logger.info("Wrote %d results to %s and %s", len(rows), results_path, report_path)


if __name__ == "__main__":
    main()
