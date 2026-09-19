#!/usr/bin/env python
"""CLI: Evaluate a trained autoencoder's window scores against SME rule-engine
ground truth (real labeled events, not synthetic injected ones).

Unlike evaluate_detection.py (synthetic ground truth) and
evaluate_unlabeled.py (no ground truth at all), this evaluates per-rule-slug
detection quality against real rule-firing state, with NO hardcoded notion
of which slugs are "episodic" vs "persistent" -- every slug (plus "any_slug",
the OR-of-everything series) is scored with the identical code path; where
slug behaviour differs, it shows up as different numbers in one uniform
table, never a different code branch.

Prerequisite: run scripts/score_full_timeline.py first (against the "test"
split -- see its own docstring) to produce the window-scores table this
script consumes. Evaluation always runs against the test split only; there
is no option here to evaluate train/val windows.

Usage:
    python scripts/evaluate_against_labels.py \\
        --window-scores output_5K501M/evaluation/window_scores.parquet \\
        --ground-truth data/5K501M/5K501M_combined_with_events.parquet \\
        --event-labels data/5K501M/5K501M_event_labels_long.parquet \\
        --split-ids output_5K501M/artefacts/split_window_ids.json \\
        --output output_5K501M/evaluation \\
        --exclusion-list missing_sensor_data
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder.logging_config import setup_logging
from autoencoder.alerting.episodes import extract_episodes
from autoencoder.alerting.persistence import compute_alert_level_timeline
from autoencoder.evaluation.ground_truth import (
    load_window_ground_truth, load_rule_to_sensors, expand_sensors, all_slugs, slug_presence_series,
    compute_split_node_coverage,
)
from autoencoder.evaluation import slug_metrics as sm

setup_logging()
logger = logging.getLogger(__name__)


# ---- JSON-safety helpers ---------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return obj.total_seconds()
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _duration_str(td) -> str | None:
    if td is None:
        return None
    return str(td)


# ---- Step 4: episode-level false-alarm rate --------------------------------

def episode_false_alarm_rate(
    model_episodes: list,
    any_slug_series: np.ndarray,
    top_sensors_per_window: list[list[str]],
    known_slug_signatures: dict[str, frozenset],
    resemblance_threshold: float = 0.3,
) -> dict:
    """For each model episode (Yellow/Red run), check overlap with any_slug
    activity anywhere in its window range. For unlabeled (no-overlap)
    episodes, compute a sensor-attribution profile and flag any that closely
    resemble a known slug's aggregate sensor signature -- reported
    separately, never folded silently into the false-alarm rate.
    """
    n_episodes = len(model_episodes)
    false_episodes = []
    for ep in model_episodes:
        window_range = range(ep.start_window_idx, ep.end_window_idx + 1)
        overlaps = any(any_slug_series[i] for i in window_range)
        if not overlaps:
            false_episodes.append(ep)

    resembling = []
    for ep in false_episodes:
        window_range = range(ep.start_window_idx, ep.end_window_idx + 1)
        episode_sensors: set[str] = set()
        for i in window_range:
            episode_sensors.update(top_sensors_per_window[i])
        best_slug, best_score = None, 0.0
        for slug, sig in known_slug_signatures.items():
            if not sig or not episode_sensors:
                continue
            jaccard = len(episode_sensors & sig) / len(episode_sensors | sig)
            if jaccard > best_score:
                best_slug, best_score = slug, jaccard
        if best_slug is not None and best_score >= resemblance_threshold:
            resembling.append({
                "start": ep.start, "end": ep.end, "n_windows": ep.n_windows,
                "resembles_slug": best_slug, "jaccard": round(best_score, 3),
            })

    return {
        "n_episodes": n_episodes,
        "n_false_alarm_episodes": len(false_episodes),
        "episode_false_alarm_rate": (len(false_episodes) / n_episodes) if n_episodes else None,
        "unlabeled_episodes_resembling_known_slug": resembling,
    }


def zone_breakdown(zones: np.ndarray) -> dict:
    """Green/Yellow/Red window counts, plus Yellow's and Red's share of all
    non-Green windows -- reported by default on every run so the balance
    between the two alert tiers is visible without re-deriving it from
    window_scores.parquet by hand. A tier's share swinging heavily one way
    (e.g. Red dominating non-Green, rather than being the rarer of the two
    as green_yellow/yellow_red's P90/P99 design intends) is a useful signal
    that the gap between the two thresholds may be too narrow for how this
    equipment's scores actually distribute.
    """
    zones = np.asarray(zones)
    green = int(np.sum(zones == "green"))
    yellow = int(np.sum(zones == "yellow"))
    red = int(np.sum(zones == "red"))
    non_green = yellow + red
    return {
        "green": green,
        "yellow": yellow,
        "red": red,
        "non_green": non_green,
        "yellow_share_of_non_green": (yellow / non_green) if non_green else None,
        "red_share_of_non_green": (red / non_green) if non_green else None,
    }


def window_false_alarm_rate(zones: np.ndarray, usable: np.ndarray, any_slug_series: np.ndarray) -> dict:
    flagged = (np.asarray(zones) != "green") & np.asarray(usable, dtype=bool)
    n_flagged = int(flagged.sum())
    n_false = int((flagged & ~np.asarray(any_slug_series, dtype=bool)).sum())
    return {
        "n_flagged_windows": n_flagged,
        "n_false_alarm_windows": n_false,
        "window_false_alarm_rate": (n_false / n_flagged) if n_flagged else None,
    }


# ---- Step 7: per-slug summary row ------------------------------------------

def build_slug_row(slug: str, gt, window_scores: pd.DataFrame, args, rule_to_sensors=None, valid_sensors=None) -> dict:
    binary_series = slug_presence_series(gt, slug)
    zones = window_scores["zone"].fillna("green").to_numpy()
    scores = window_scores["anomaly_score"].to_numpy(dtype=float)
    usable = window_scores["usable"].to_numpy(dtype=bool)

    ep_stats = sm.episode_stats(binary_series, gt.start_times, gt.end_times)
    pw = sm.pointwise_metrics(binary_series, zones, scores, usable=usable)
    sev = sm.severity_correlation(binary_series, gt.max_event_level, gt.n_active_nodes, scores)
    conc = sm.concurrency_correlation(binary_series, gt.n_active_nodes, scores)

    # Sensor attribution: only on windows where the slug is active AND detected (Step 2/3),
    # and usable -- never on missed windows, which would just double-count the same failure.
    detected_mask = binary_series & (zones != "green") & usable
    eligible_idx = np.where(detected_mask)[0]
    ranked_sensors_list = [window_scores["ranked_sensors"].iloc[i] for i in eligible_idx]
    if rule_to_sensors is not None and slug != "any_slug":
        # A single named slug's official sensors are fixed, independent of
        # whatever else happens to be concurrently active in a given window
        # -- unlike gt.all_triggering_sensors[i] (a per-WINDOW union across
        # every rule active at that instant), which would otherwise mix in
        # unrelated co-firing rules' sensors and reintroduce exactly the
        # concurrency confound the catalog was meant to escape.
        fixed_sensors = expand_sensors([slug], rule_to_sensors, valid_sensors)
        gt_sensor_sets = [fixed_sensors for _ in eligible_idx]
    else:
        # any_slug has no single rule to look up -- its ground truth is
        # genuinely "whichever specific rules were active in that window",
        # so the per-window union is the correct (not confounded) choice here.
        gt_sensor_sets = [gt.all_triggering_sensors[i] for i in eligible_idx]
    concurrency = [gt.n_active_nodes[i] for i in eligible_idx]
    attribution = sm.sensor_attribution_by_concurrency(
        ranked_sensors_list, gt_sensor_sets, concurrency,
        concurrency_threshold=args.concurrency_threshold,
    )

    return {
        "rule_slug": slug,
        "n_windows_active": ep_stats["n_windows_active"],
        "n_episodes": ep_stats["n_episodes"],
        "median_episode_duration": _duration_str(ep_stats["median_duration"]),
        "min_episode_duration": _duration_str(ep_stats["min_duration"]),
        "max_episode_duration": _duration_str(ep_stats["max_duration"]),
        "tp": pw["tp"],
        "fp": pw["fp"],
        "fn": pw["fn"],
        "tn": pw["tn"],
        "precision": pw["precision"],
        "recall": pw["recall"],
        "f1": pw["f1"],
        "pr_auc": pw["pr_auc"],
        "accuracy": (
            (pw["tp"] + pw["tn"]) / pw["n_windows_total"] if pw["n_windows_total"] else float("nan")
        ),
        "n_windows_total_usable": pw["n_windows_total"],
        "severity_correlation": sev,
        "concurrency_correlation": conc,
        "sensor_attribution": attribution,
    }


def summarize_table(rows: list[dict]) -> str:
    lines = []
    header = f"{'rule_slug':45s} {'n_active':>9s} {'n_ep':>5s} {'med_dur':>10s} {'precision':>9s} {'recall':>7s} {'pr_auc':>7s} {'sev_corr':>18s} {'mean_ap':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        sev = r["severity_correlation"]
        sev_str = f"rho={sev['spearman_rho']:.2f}" if sev.get("status") == "ok" else "n/a: " + sev["status"].split("--")[-1].strip()[:12]
        mean_ap = r["sensor_attribution"]["overall"]["mean_average_precision"]
        mean_ap_str = f"{mean_ap:.3f}" if mean_ap is not None else "n/a"
        lines.append(
            f"{r['rule_slug'][:45]:45s} {r['n_windows_active']:>9d} {r['n_episodes']:>5d} "
            f"{str(r['median_episode_duration'])[:10]:>10s} "
            f"{r['precision']:>9.3f} {r['recall']:>7.3f} {r['pr_auc']:>7.3f} "
            f"{sev_str:>18s} {mean_ap_str:>8s}"
        )
    return "\n".join(lines)


def summarize_sensor_attribution(rows: list[dict]) -> str:
    """Dedicated sensor-attribution report -- rank-aware (mean_average_precision,
    mean_gt_rank_percentile), split by concurrency, since ground-truth sensors
    are a candidate pool (sensors that COULD be implicated), not a precise
    label of which ones actually are -- see slug_metrics.sensor_attribution_metrics.
    """
    lines = [
        f"{'rule_slug':45s} {'concurrency':>11s} {'n_win':>6s} {'n_skip':>7s} {'mean_ap':>8s} {'mean_rank_pctile':>17s}",
    ]
    lines.append("-" * len(lines[0]))
    for r in rows:
        attribution = r["sensor_attribution"]
        for bucket in ("overall", "low_concurrency", "high_concurrency"):
            b = attribution[bucket]
            mean_ap = f"{b['mean_average_precision']:.3f}" if b["mean_average_precision"] is not None else "n/a"
            mean_rank = f"{b['mean_gt_rank_percentile']:.3f}" if b["mean_gt_rank_percentile"] is not None else "n/a"
            lines.append(
                f"{r['rule_slug'][:45]:45s} {bucket:>11s} {b['n_windows']:>6d} {b['n_skipped_empty_gt']:>7d} "
                f"{mean_ap:>8s} {mean_rank:>17s}"
            )
    return "\n".join(lines)


def summarize_severity_correlation(rows: list[dict]) -> str:
    """Dedicated report on correlation between reconstruction (anomaly)
    score and ground-truth event severity, on windows where each slug is
    active -- see slug_metrics.severity_correlation."""
    lines = [
        f"{'rule_slug':45s} {'status':>10s} {'field_used':>16s} {'spearman_rho':>13s} {'p_value':>9s} {'n_windows':>10s}",
    ]
    lines.append("-" * len(lines[0]))
    for r in rows:
        sev = r["severity_correlation"]
        if sev.get("status") == "ok":
            lines.append(
                f"{r['rule_slug'][:45]:45s} {'ok':>10s} {sev['field_used']:>16s} "
                f"{sev['spearman_rho']:>13.3f} {sev['p_value']:>9.3g} {sev['n_windows']:>10d}"
            )
        else:
            reason = sev["status"].split("--")[-1].strip()[:16]
            lines.append(f"{r['rule_slug'][:45]:45s} {'n/a':>10s} {reason:>16s} {'':>13s} {'':>9s} {sev['n_windows']:>10d}")
    return "\n".join(lines)


def summarize_concurrency_correlation(rows: list[dict]) -> str:
    """Dedicated report on correlation between reconstruction (anomaly)
    score and n_active_nodes -- how many distinct labeled event nodes
    ("episodes") are concurrently active -- on windows where each slug is
    active. Reported independently of severity_correlation (which only
    falls back to n_active_nodes when severity itself is degenerate) -- see
    slug_metrics.concurrency_correlation."""
    lines = [
        f"{'rule_slug':45s} {'status':>10s} {'spearman_rho':>13s} {'p_value':>9s} {'n_windows':>10s}",
    ]
    lines.append("-" * len(lines[0]))
    for r in rows:
        conc = r["concurrency_correlation"]
        if conc.get("status") == "ok":
            lines.append(
                f"{r['rule_slug'][:45]:45s} {'ok':>10s} "
                f"{conc['spearman_rho']:>13.3f} {conc['p_value']:>9.3g} {conc['n_windows']:>10d}"
            )
        else:
            reason = conc["status"].split("--")[-1].strip()[:16]
            lines.append(f"{r['rule_slug'][:45]:45s} {'n/a':>10s} {'':>13s} {'':>9s} {conc['n_windows']:>10d}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate window scores against SME rule-engine ground truth.")
    parser.add_argument("--window-scores", required=True, help="Output of scripts/score_full_timeline.py, run against the 'test' split")
    parser.add_argument("--ground-truth", required=True, help="Path to a *_combined_with_events.parquet file")
    parser.add_argument("--event-labels", required=True, help="Path to the matching *_event_labels_long.parquet file")
    parser.add_argument(
        "--split-ids", default=None,
        help="Path to artefacts/split_window_ids.json (see main.py's step_train). When given, ground truth is "
             "restricted to exactly window_scores' window ids (which score_full_timeline.py should already have "
             "limited to the test split) instead of requiring the two to cover the same full file, and a "
             "per-split active-node coverage report is added to the output.",
    )
    parser.add_argument("--window-rows", type=int, default=120, help="Must match score_full_timeline.py's --window-rows")
    parser.add_argument("--output", required=True, help="Output directory for eval_results.json + report.txt")
    parser.add_argument(
        "--exclusion-list", default="",
        help="Comma-separated rule_slugs to treat as data-quality/non-fault (Step 8) -- "
             "an explicit, externally-supplied list, not inferred from this run's data.",
    )
    parser.add_argument("--concurrency-threshold", type=float, default=1,
                         help="n_active_nodes at or below this counts as 'low concurrency' for sensor attribution (Step 6).")
    parser.add_argument("--red-to-yellow-alert", type=int, default=2)
    parser.add_argument("--red-to-red-alert", type=int, default=3)
    parser.add_argument("--yellow-consecutive", type=int, default=4)
    parser.add_argument("--mixed-window-hours", type=float, default=2)
    parser.add_argument("--window-minutes", type=float, default=30)
    parser.add_argument("--resemblance-threshold", type=float, default=0.3,
                         help="Min Jaccard overlap for an unlabeled episode to be flagged as resembling a known slug (Step 4).")
    parser.add_argument(
        "--rule-to-sensors", default=None,
        help="Path to a static rule_slug->sensor catalog (JSON dict or CSV with rule_slug+sensor columns; "
             "see autoencoder.evaluation.ground_truth.load_rule_to_sensors). Used to derive Step 6's "
             "sensor-attribution ground truth from all_rule_slugs when the ground-truth file has no "
             "all_triggering_sensors column of its own.",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Path to the trained model's artefacts directory (for training_metadata.json's sensor_columns). "
             "Only needed alongside --rule-to-sensors: the catalog is plant-wide (a generic rule lists sensors "
             "from every equipment instance it's deployed on), so without this the sensor-attribution ground "
             "truth gets polluted with irrelevant other-equipment sensor names, crushing Jaccard toward zero "
             "regardless of the model's actual attribution quality.",
    )
    args = parser.parse_args()

    exclusion_list = [s.strip() for s in args.exclusion_list.split(",") if s.strip()]
    if not exclusion_list:
        logger.warning("No --exclusion-list supplied -- Step 8's 'excluded' table will be identical to the full one.")

    window_scores = pd.read_parquet(args.window_scores) if args.window_scores.endswith(".parquet") else pd.read_csv(args.window_scores)
    logger.info("Loaded %d scored windows from %s", len(window_scores), args.window_scores)

    rule_to_sensors = load_rule_to_sensors(args.rule_to_sensors) if args.rule_to_sensors else None
    if rule_to_sensors is not None:
        logger.info("Loaded rule_to_sensors catalog from %s: %d rule_slugs mapped", args.rule_to_sensors, len(rule_to_sensors))

    valid_sensors = None
    if args.model_dir:
        with open(Path(args.model_dir) / "training_metadata.json") as f:
            model_metadata = json.load(f)
        valid_sensors = frozenset(model_metadata["sensor_columns"])
        logger.info("Loaded %d sensor_columns from %s to scope the catalog to this equipment", len(valid_sensors), args.model_dir)
    elif rule_to_sensors is not None:
        logger.warning(
            "--rule-to-sensors given without --model-dir -- the catalog is plant-wide, so sensor-attribution "
            "ground truth will include irrelevant other-equipment sensor names unless scoped. Pass --model-dir "
            "to fix this."
        )

    gt = load_window_ground_truth(
        args.ground_truth, args.event_labels, window_size=args.window_rows,
        rule_to_sensors=rule_to_sensors, valid_sensors=valid_sensors,
    )

    # Ground truth is loaded from the WHOLE file; window_scores should already
    # be restricted to one split (score_full_timeline.py's --split, default
    # "test") -- filter_to_window_ids restricts + reorders gt to match rather
    # than requiring the two to cover the same full file.
    gt = gt.filter_to_window_ids(window_scores["window_id"].to_numpy())

    zones = window_scores["zone"].fillna("green").to_numpy()
    usable = window_scores["usable"].to_numpy(dtype=bool)
    top_sensors_per_window = window_scores["top_sensors"].tolist()

    # Step 1 onward: every distinct slug + any_slug, identical code path.
    slugs = all_slugs(gt)
    logger.info("Found %d distinct rule_slugs (+ any_slug) in ground truth", len(slugs) - 1)

    rows = [build_slug_row(slug, gt, window_scores, args, rule_to_sensors=rule_to_sensors, valid_sensors=valid_sensors) for slug in slugs]
    rows_excluded = [r for r in rows if r["rule_slug"] not in exclusion_list]

    # Step 4: episode-level false-alarm rate, using the model's own persistence-suppressed alert timeline.
    alert_levels = compute_alert_level_timeline(
        zones.tolist(),
        red_to_yellow_alert=args.red_to_yellow_alert, red_to_red_alert=args.red_to_red_alert,
        yellow_consecutive=args.yellow_consecutive, mixed_window_hours=args.mixed_window_hours,
        window_minutes=args.window_minutes,
    )
    model_episodes = extract_episodes(alert_levels, gt.start_times, gt.end_times)

    any_slug_series = slug_presence_series(gt, "any_slug")
    # Per-slug sensor "signature" used only for Step 4's resemblance heuristic.
    # Prefer the real static catalog directly (authoritative, not confounded by
    # whatever else happened to be concurrently active) when one was given;
    # otherwise fall back to a best-effort proxy aggregated from this same
    # run's attributed sensors on that slug's own detected windows.
    known_slug_signatures = {}
    for slug in slugs:
        if slug == "any_slug":
            continue
        if rule_to_sensors is not None:
            sig = frozenset(rule_to_sensors.get(slug, []))
            known_slug_signatures[slug] = (sig & valid_sensors) if valid_sensors is not None else sig
            continue
        binary = slug_presence_series(gt, slug)
        detected = binary & (zones != "green") & usable
        sig: set[str] = set()
        for i in np.where(detected)[0]:
            sig.update(gt.all_triggering_sensors[i])
        known_slug_signatures[slug] = frozenset(sig)

    step4 = episode_false_alarm_rate(
        model_episodes, any_slug_series, top_sensors_per_window, known_slug_signatures,
        resemblance_threshold=args.resemblance_threshold,
    )
    step4["window_level"] = window_false_alarm_rate(zones, usable, any_slug_series)

    # Split node coverage: for each train/val/test split, how many distinct
    # active nodes (events) fall fully inside it vs. straddle a split
    # boundary (see compute_split_node_coverage's docstring).
    split_node_coverage = None
    if args.split_ids:
        with open(args.split_ids) as f:
            split_window_ids = json.load(f)
        split_node_coverage = compute_split_node_coverage(
            args.event_labels, args.ground_truth, split_window_ids, window_size=args.window_rows,
        )
        logger.info("Split node coverage: %s", split_node_coverage)

    # Step 9: headline number.
    any_slug_row = next(r for r in rows if r["rule_slug"] == "any_slug")
    headline_pr_auc = any_slug_row["pr_auc"]

    zones_breakdown = zone_breakdown(zones)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "headline_pr_auc_any_slug": headline_pr_auc,
        "exclusion_list": exclusion_list,
        "per_slug_all": rows,
        "per_slug_excluding_list": rows_excluded,
        "episode_false_alarm": step4,
        "split_node_coverage": split_node_coverage,
        "zone_breakdown": zones_breakdown,
        "n_windows_total": len(window_scores),
        "n_windows_usable": int(usable.sum()),
    }
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("Wrote %s", output_dir / "eval_results.json")

    report_lines = [
        "=" * 100,
        "SME LABEL EVALUATION",
        "=" * 100,
        "",
        f"HEADLINE: PR-AUC of anomaly_score vs. any_slug = {headline_pr_auc:.4f}"
        if headline_pr_auc is not None and not (isinstance(headline_pr_auc, float) and np.isnan(headline_pr_auc))
        else "HEADLINE: PR-AUC of anomaly_score vs. any_slug = n/a (degenerate -- check n_windows_active)",
        f"  ({any_slug_row['n_windows_active']} / {len(window_scores)} windows have any_slug active)",
        "",
        f"Windows: {len(window_scores)} total, {int(usable.sum())} usable ({int(usable.sum())/len(window_scores)*100:.1f}%)",
        "",
        "-- Zone breakdown --",
        f"  Green={zones_breakdown['green']}  Yellow={zones_breakdown['yellow']}  Red={zones_breakdown['red']}"
        + (
            f"  (of {zones_breakdown['non_green']} non-green: "
            f"{zones_breakdown['yellow_share_of_non_green']*100:.1f}% yellow, "
            f"{zones_breakdown['red_share_of_non_green']*100:.1f}% red)"
            if zones_breakdown["non_green"] else "  (no non-green windows)"
        ),
        "",
        "-- Per-slug summary (ALL slugs) --",
        summarize_table(rows),
        "",
        f"-- Per-slug summary (excluding {exclusion_list or '[]'}) --",
        summarize_table(rows_excluded),
        "",
        "-- Sensor attribution performance (Step 6) --",
        summarize_sensor_attribution(rows),
        "",
        "-- Reconstruction error vs. event severity correlation --",
        summarize_severity_correlation(rows),
        "",
        "-- Reconstruction error vs. concurrent active episodes (n_active_nodes) --",
        summarize_concurrency_correlation(rows),
        "",
        "-- Active-node coverage per split --",
    ]
    if split_node_coverage:
        for split_name, cov in split_node_coverage.items():
            report_lines.append(
                f"  {split_name:6s}: {cov['n_active_nodes_full']} full, {cov['n_active_nodes_partial']} partial "
                f"(straddles a split boundary), over {cov['n_windows_in_split']} windows"
            )
    else:
        report_lines.append("  --split-ids not given -- skipped")
    report_lines += [
        "",
        "-- Episode-level false-alarm rate (Step 4) --",
        f"  {step4['n_false_alarm_episodes']} / {step4['n_episodes']} model episodes have no overlapping active slug "
        f"(rate={step4['episode_false_alarm_rate']:.3f})" if step4["n_episodes"] else "  no model episodes at all",
        f"  Window-level: {step4['window_level']['n_false_alarm_windows']} / {step4['window_level']['n_flagged_windows']} "
        f"flagged windows have no active slug (rate="
        f"{step4['window_level']['window_false_alarm_rate']:.3f})"
        if step4["window_level"]["window_false_alarm_rate"] is not None else "  Window-level: no flagged windows at all",
        "",
        f"  Unlabeled episodes resembling a known slug (best-effort, no static catalog available -- "
        f"see per-slug sensor signatures built from this same run): {len(step4['unlabeled_episodes_resembling_known_slug'])}",
    ]
    for r in step4["unlabeled_episodes_resembling_known_slug"][:20]:
        report_lines.append(f"    {r['start']} -> {r['end']} ({r['n_windows']} windows): resembles '{r['resembles_slug']}' (jaccard={r['jaccard']})")

    report = "\n".join(str(l) for l in report_lines)
    print(report)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report)
    logger.info("Wrote %s", output_dir / "report.txt")


if __name__ == "__main__":
    main()
