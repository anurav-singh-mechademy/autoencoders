"""Classify rule_slugs by authoritative high/low alarm direction (from the
slug text itself), then compute model precision/recall by direction group.
"""
import json
import pandas as pd
from autoencoder.evaluation.ground_truth import load_window_ground_truth

EQUIPMENT = ["5P921A", "5ST901A", "5ST901B", "5K512B"]

print("=" * 80)
print("STEP 1-2: event_labels_long schema + rule_to_sensors.json")
print("=" * 80)
for eq in EQUIPMENT:
    df = pd.read_parquet(f"data/{eq}/{eq}_event_labels_long.parquet")
    print(f"\n--- {eq} ---")
    print("columns:", list(df.columns))
    print("distinct rule_slug:", sorted(df.rule_slug.unique()))

# Manual classification from unambiguous high/low substrings in the slug text; anything else is left unclassified.
HIGH_SLUGS = {
    "emai_high_journal_bearing_temperatures_1spe",
    "puce_high_axial_displacements",
    "puce_lube_oil_supply_temperature_high",
    "stno_de_shaft_high_radial_vibration",
    "stno_exhaust_pressure_high",
    "stno_generator_de_shaft_high_radial_vibration",
    "stno_generator_nde_shaft_high_radial_vibration",
    "steam_turbine_generator_high_stator_winding_temperature",
    "coce_high_axial_displacement",
}
LOW_SLUGS = {
    "coce_dgs_pri_dp_low",
}

print("\n" + "=" * 80)
print("STEP 3: rule count per equipment by direction classification")
print("=" * 80)
for eq in EQUIPMENT:
    df = pd.read_parquet(f"data/{eq}/{eq}_event_labels_long.parquet")
    slugs = sorted(df.rule_slug.unique())
    high = [s for s in slugs if s in HIGH_SLUGS]
    low = [s for s in slugs if s in LOW_SLUGS]
    unclassified = [s for s in slugs if s not in HIGH_SLUGS and s not in LOW_SLUGS]
    print(f"\n{eq}: {len(slugs)} distinct rule_slugs")
    print("  HIGH:", high)
    print("  LOW:", low)
    print("  UNCLASSIFIED:", unclassified)

# A window buckets into HIGH/LOW if any of its active rule_slugs is classified that way (can be both).
print("\n" + "=" * 80)
print("STEP 4: precision/recall by direction group")
print("=" * 80)

WINDOW_SIZE = 120

for eq in EQUIPMENT:
    df = pd.read_parquet(f"data/{eq}/{eq}_event_labels_long.parquet")
    slugs_present = set(df.rule_slug.unique())
    has_high = bool(slugs_present & HIGH_SLUGS)
    has_low = bool(slugs_present & LOW_SLUGS)
    print(f"\n--- {eq} --- has_high={has_high} has_low={has_low}")
    if not (has_high and has_low):
        print("  SKIP: this equipment does not have both a HIGH-classified and a "
              "LOW-classified rule_slug present in its labels -> no within-"
              "equipment high-vs-low comparison is possible from ground truth.")
        continue

    gt = load_window_ground_truth(
        combined_parquet_path=f"data/{eq}/{eq}_combined_with_events.parquet",
        event_labels_parquet_path=f"data/{eq}/{eq}_event_labels_long.parquet",
        window_size=WINDOW_SIZE,
    )
    gt_df = pd.DataFrame({
        "window_id": gt.window_ids,
        "is_anomaly": gt.is_anomaly,
        "rule_slugs": gt.all_rule_slugs,
    })

    def direction(slug_list):
        s = set(slug_list) if slug_list else set()
        is_high = bool(s & HIGH_SLUGS)
        is_low = bool(s & LOW_SLUGS)
        if is_high and is_low:
            return "both"
        if is_high:
            return "high"
        if is_low:
            return "low"
        if s:
            return "unclassified_only"
        return "none"

    gt_df["direction"] = gt_df["rule_slugs"].apply(direction)

    ws = pd.read_parquet(f"output_{eq}_regime_fix/evaluation/window_scores.parquet")
    ws["predicted_anomaly"] = ws["zone"] != "green"

    merged = gt_df.merge(ws[["window_id", "predicted_anomaly", "zone"]], on="window_id", how="inner")
    print("  window_id overlap:", len(merged), "of", len(gt_df), "gt windows /", len(ws), "score windows")
    print("  direction value counts (all windows):")
    print("   ", merged["direction"].value_counts().to_dict())

    rows = []
    for direction_label in ["high", "low"]:
        sub = merged[merged["direction"] == direction_label]
        pos = sub[sub["is_anomaly"]]
        n_windows = len(sub)
        n_pos = len(pos)
        if n_pos == 0:
            rows.append((direction_label, n_windows, n_pos, None, None))
            continue
        tp = pos["predicted_anomaly"].sum()
        recall = tp / n_pos
        # precision restricted to this direction's positive windows plus "none"-direction windows
        neg = merged[(merged["direction"].isin(["none"]))]
        pred_pos_in_scope = pd.concat([sub, neg])
        precision_tp = pred_pos_in_scope[pred_pos_in_scope["is_anomaly"] & pred_pos_in_scope["predicted_anomaly"]].shape[0]
        precision_fp = pred_pos_in_scope[(~pred_pos_in_scope["is_anomaly"]) & pred_pos_in_scope["predicted_anomaly"]].shape[0]
        precision = precision_tp / (precision_tp + precision_fp) if (precision_tp + precision_fp) > 0 else None
        rows.append((direction_label, n_windows, n_pos, recall, precision))

    res = pd.DataFrame(rows, columns=["direction", "n_windows_this_direction", "n_positive_windows", "recall", "precision_vs_none"])
    print(res.to_string(index=False))
    if merged["direction"].isin(["high", "low"]).sum() == 0:
        print("  NOTE: none of the high/low-classified ground-truth episodes for "
              "this equipment fall inside the window_scores.parquet evaluation "
              "range (train/eval split excludes them) -> comparison is not "
              "computable even though direction labels exist.")
