"""Registry of ablation variants: (run_id, tier, group, config overrides).

Each `overrides` dict is deep-merged onto the base YAML config (see
`run_ablation.deep_merge`). `run_ablation.py` decides whether a variant needs
a full clean+train rerun or can reuse a cached upstream model, based on
whether `overrides` touches anything outside the threshold/inference config
subtrees (see `run_ablation.UPSTREAM_KEYS`).

Tiers implemented here (see the ablation plan for full rationale):
    A -- cleaning pipeline (which stages run, in what order)
    B -- window-score threshold calibration (percentile source & width)
    C -- latent dimension
    F -- dropout rate
    H -- sensor-level flag threshold (attribution only, not detection)

Tiers NOT yet wired into the config/codebase (deliberately excluded rather
than faked -- adding them requires real code changes beyond a config diff):
    D -- normalization (LayerNorm vs BatchNorm vs none) needs architecture.py
         to build conditional norm layers instead of hardcoded LayerNorm.
    E -- batch composition (shuffled rows vs one-window-per-step) needs a
         second training loop mode in trainer.py.
    G -- regime segmentation on/off needs a bypass path in main.step_clean.
    I -- persistence rules need an event-level (not per-window) metric --
         evaluate_detection.compute_metrics only scores per-window zones.
"""

from __future__ import annotations

VARIANTS = [
    # ── Tier A: cleaning pipeline ────────────────────────────────────────
    dict(
        run_id="A0_no_cleaning", tier="A", group="cleaning_pipeline",
        overrides={"cleaning": {"order": []}},
    ),
    dict(
        run_id="A1_isolation_forest_only", tier="A", group="cleaning_pipeline",
        overrides={"cleaning": {"order": ["isolation_forest"]}},
    ),
    dict(
        run_id="A2_pca_only", tier="A", group="cleaning_pipeline",
        overrides={"cleaning": {"order": ["pca"]}},
    ),
    dict(
        run_id="A3_mahalanobis_only", tier="A", group="cleaning_pipeline",
        overrides={"cleaning": {"order": ["mahalanobis"]}},
    ),
    dict(
        run_id="A4_all_default", tier="A", group="cleaning_pipeline",
        overrides={},  # exactly the base config -- current production default
    ),
    dict(
        run_id="A5_reordered", tier="A", group="cleaning_pipeline",
        overrides={"cleaning": {"order": ["mahalanobis", "pca", "isolation_forest"]}},
    ),

    # ── Tier B: window-score threshold calibration ──────────────────────
    # B0 is identical to A4 upstream-wise (same cleaning/model/training config)
    # so the runner reuses A4's trained model instead of retraining.
    dict(
        run_id="B0_train_p90_p99", tier="B", group="threshold_calibration",
        overrides={},  # current default: percentiles computed on TRAIN errors
    ),
    dict(
        run_id="B1_val_p90_p99", tier="B", group="threshold_calibration",
        overrides={"thresholds": {"calibration_source": "val"}},
    ),
    dict(
        run_id="B2_train_p85_p975", tier="B", group="threshold_calibration",
        overrides={"thresholds": {"green_yellow": 85, "yellow_red": 97.5}},
    ),

    # ── Tier C: latent dimension ─────────────────────────────────────────
    dict(
        run_id="C0_latent8", tier="C", group="architecture",
        overrides={"model": {"latent_dim": 8}},
    ),
    dict(
        run_id="C1_latent19_default", tier="C", group="architecture",
        overrides={},  # max(8, 158//8) == 19 -- current default
    ),
    dict(
        run_id="C2_latent32", tier="C", group="architecture",
        overrides={"model": {"latent_dim": 32}},
    ),

    # ── Tier F: dropout ──────────────────────────────────────────────────
    dict(
        run_id="F0_dropout0.1", tier="F", group="regularization",
        overrides={"model": {"dropout": 0.1}},
    ),
    dict(
        run_id="F1_dropout0.2_default", tier="F", group="regularization",
        overrides={},  # current default
    ),
    dict(
        run_id="F2_dropout0.3", tier="F", group="regularization",
        overrides={"model": {"dropout": 0.3}},
    ),

    # ── Tier H: sensor-level flag threshold (attribution metric only) ───
    dict(
        run_id="H0_flag2.0", tier="H", group="sensor_flag_threshold",
        overrides={"inference": {"sensor_flag_threshold": 2.0}},
    ),
    dict(
        run_id="H1_flag3.0_default", tier="H", group="sensor_flag_threshold",
        overrides={},  # current default
    ),
    dict(
        run_id="H2_flag4.0", tier="H", group="sensor_flag_threshold",
        overrides={"inference": {"sensor_flag_threshold": 4.0}},
    ),
]


def variants_for_tiers(tiers: list[str]) -> list[dict]:
    return [v for v in VARIANTS if v["tier"] in tiers]
