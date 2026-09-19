"""Per-slug detection metrics: point-wise agreement, severity correlation,
sensor attribution. Every function here is generic over the slug's own
behaviour (how long/often it fires) -- none of them branch on slug identity
or assumed category; differences show up as different numbers, not
different code paths.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, confusion_matrix

from autoencoder.alerting.episodes import extract_episodes


def episode_stats(binary_series: np.ndarray, start_times: list, end_times: list) -> dict:
    """Episode count and duration distribution for one binary presence series.

    Purely descriptive -- not used to branch any downstream logic, just
    reported so a reader can tell "rare short fault" from "persistent
    regime shift" without the code itself encoding that judgement.
    """
    labels = ["red" if b else "green" for b in binary_series]
    episodes = extract_episodes(labels, start_times, end_times)
    n_windows_active = int(np.sum(binary_series))
    if not episodes:
        return {
            "n_episodes": 0,
            "n_windows_active": n_windows_active,
            "min_duration": None,
            "median_duration": None,
            "max_duration": None,
        }
    durations = sorted(e.end - e.start for e in episodes)
    return {
        "n_episodes": len(episodes),
        "n_windows_active": n_windows_active,
        "min_duration": durations[0],
        "median_duration": durations[len(durations) // 2],
        "max_duration": durations[-1],
    }


def pointwise_metrics(
    binary_series: np.ndarray,
    zones: np.ndarray,
    scores: np.ndarray,
    usable: np.ndarray | None = None,
) -> dict:
    """Precision/recall/F1 of (zone != green) and PR-AUC of anomaly_score,
    both against the slug's active/inactive series. Valid whether the slug
    covers 20 windows or 20,000 -- no minimum count required.
    """
    mask = np.asarray(usable, dtype=bool) if usable is not None else np.ones(len(binary_series), dtype=bool)

    y_true = np.asarray(binary_series, dtype=bool)[mask].astype(int)
    zone_arr = np.asarray(zones)[mask]
    y_pred = (zone_arr != "green").astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    y_score = np.asarray(scores, dtype=float)[mask]
    valid = ~np.isnan(y_score)
    pr_auc = float("nan")
    if valid.sum() > 0 and len(set(y_true[valid])) > 1:
        pr_auc = float(average_precision_score(y_true[valid], y_score[valid]))

    return {
        "n_windows_active": int(y_true.sum()),
        "n_windows_total": int(mask.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "pr_auc": pr_auc,
    }


def _spearman_against_field(mask: np.ndarray, field: np.ndarray, scores: np.ndarray) -> dict | None:
    """Core Spearman-rho computation shared by severity_correlation and
    concurrency_correlation. Returns None (not a dict) when the field is
    degenerate (zero variance) or there's too little data -- the caller
    decides how to report that (try a fallback field, or report degeneracy
    directly) rather than this baking in one specific "not computable"
    message for every use site.
    """
    vals = np.asarray(field, dtype=float)[mask]
    sc = np.asarray(scores, dtype=float)[mask]
    valid = ~np.isnan(vals) & ~np.isnan(sc)
    vals, sc = vals[valid], sc[valid]
    if len(vals) < 2 or np.var(vals) == 0:
        return None
    rho, p_value = spearmanr(vals, sc)
    return {"spearman_rho": float(rho), "p_value": float(p_value), "n_windows": int(len(vals))}


def severity_correlation(
    binary_series: np.ndarray,
    max_event_level: np.ndarray,
    n_active_nodes: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Spearman correlation of anomaly_score against ground-truth severity,
    on windows where the slug is active. Tries max_event_level first, falls
    back to n_active_nodes, and reports degeneracy explicitly rather than
    fabricating a number or silently omitting the row.
    """
    mask = np.asarray(binary_series, dtype=bool)
    for field_name, field in [("max_event_level", max_event_level), ("n_active_nodes", n_active_nodes)]:
        result = _spearman_against_field(mask, field, scores)
        if result is not None:
            return {"status": "ok", "field_used": field_name, **result}
    return {"status": "not computable -- zero variance (or too few active windows) in both max_event_level and n_active_nodes", "n_windows": int(mask.sum())}


def concurrency_correlation(
    binary_series: np.ndarray,
    n_active_nodes: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Spearman correlation of RAW anomaly_score against n_active_nodes --
    how many distinct labeled event nodes ("episodes") are concurrently
    active -- on windows where the slug is active.

    Reported separately from severity_correlation, which only falls back to
    n_active_nodes when max_event_level itself is degenerate: a slug whose
    severity already correlates fine would otherwise never get its OWN
    concurrency number surfaced at all, even though concurrency (how many
    overlapping episodes) and severity (how bad any one of them is) are
    related but distinct signals -- a window could have many mild
    concurrent episodes, or one severe one, and those needn't score the
    same way against reconstruction error.
    """
    mask = np.asarray(binary_series, dtype=bool)
    result = _spearman_against_field(mask, n_active_nodes, scores)
    if result is not None:
        return {"status": "ok", "field_used": "n_active_nodes", **result}
    return {"status": "not computable -- zero variance (or too few active windows) in n_active_nodes", "n_windows": int(mask.sum())}


def sensor_attribution_metrics(
    ranked_sensors_list: list[list[str]],
    gt_sensor_sets: list[frozenset],
) -> dict:
    """Rank-aware sensor attribution over the given (already-filtered-to
    -eligible) windows.

    `gt_sensor_sets` is a CANDIDATE pool -- every sensor tied to whichever
    rule(s)/event(s) are active in a window, i.e. sensors that COULD be
    implicated, not a precise label of which ones actually are (see
    autoencoder.evaluation.ground_truth's module docstring). A symmetric
    set-overlap metric like Jaccard wrongly penalizes the model twice for
    this: once for any candidate it doesn't rank near the top (treated as a
    "miss" even though most candidates in a real occurrence are never the
    true cause), and it also can't reward a model that ranks the true
    sensor(s) 1st/2nd but includes a couple of unrelated ones further down.

    Instead, treat `ranked_sensors_list[i]` (the model's FULL per-window
    sensor ranking, best/highest-error first -- not just a truncated top-k)
    as a ranked retrieval result and `gt_sensor_sets[i]` as the relevant set:
    mean_average_precision rewards concentrating candidate sensors near the
    top of the ranking without penalizing the model for how it ranks
    everything else, and mean_gt_rank_percentile (0=candidates ranked first,
    1=ranked last) gives a directly interpretable secondary number. Windows
    with an empty ground-truth sensor set (nothing to compare against), or
    whose candidates don't intersect this window's ranked sensor universe,
    are skipped and counted in `n_skipped_empty_gt`.
    """
    average_precisions = []
    mean_rank_percentiles = []
    n_skipped = 0
    for ranked, gt in zip(ranked_sensors_list, gt_sensor_sets):
        # len(), not a bare truthiness check -- `ranked` may be a numpy array
        # (parquet round-trips a list column back as one), whose truth value
        # for >1 element raises rather than short-circuiting like a list.
        gt_in_universe = (gt & set(ranked)) if len(ranked) else frozenset()
        if not gt_in_universe:
            n_skipped += 1
            continue

        n = len(ranked)
        n_hits = 0
        ap_sum = 0.0
        gt_positions = []
        for i, sensor in enumerate(ranked):
            is_hit = sensor in gt_in_universe
            if is_hit:
                n_hits += 1
                ap_sum += n_hits / (i + 1)
                gt_positions.append(i)
        average_precisions.append(ap_sum / len(gt_in_universe))
        mean_rank_percentiles.append(
            float(np.mean(gt_positions)) / (n - 1) if n > 1 else 0.0
        )

    return {
        "n_windows": len(average_precisions),
        "n_skipped_empty_gt": n_skipped,
        "mean_average_precision": float(np.mean(average_precisions)) if average_precisions else None,
        "mean_gt_rank_percentile": float(np.mean(mean_rank_percentiles)) if mean_rank_percentiles else None,
    }


def sensor_attribution_by_concurrency(
    ranked_sensors_list: list[list[str]],
    gt_sensor_sets: list[frozenset],
    concurrency_counts: list[int],
    concurrency_threshold: float,
) -> dict:
    """sensor_attribution_metrics split into low/high concurrency buckets, so
    the report makes clear whether attribution quality is being measured on
    clean or confounded instances -- a comparison at or below the threshold
    counts as low, above as high.
    """
    low_idx = [i for i, c in enumerate(concurrency_counts) if c <= concurrency_threshold]
    high_idx = [i for i, c in enumerate(concurrency_counts) if c > concurrency_threshold]

    def _subset(idx):
        return sensor_attribution_metrics(
            [ranked_sensors_list[i] for i in idx], [gt_sensor_sets[i] for i in idx],
        )

    return {
        "overall": sensor_attribution_metrics(ranked_sensors_list, gt_sensor_sets),
        "low_concurrency": _subset(low_idx),
        "high_concurrency": _subset(high_idx),
        "concurrency_threshold": concurrency_threshold,
    }
