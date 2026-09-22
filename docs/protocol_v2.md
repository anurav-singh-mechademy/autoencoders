# Protocol v2 — established-practice re-evaluation of the autoencoder pipeline

Date: 2026-09-21. Code: `scripts/protocol_v2.py`, `scripts/summarize_protocol_v2.py`, `scripts/alarm_anatomy_v2.py`.

## 1. Goal

- Find out whether an autoencoder anomaly score can reach **high precision and high recall against the platform's rule-event nodes**.
- Use only established practice (semi-supervised training on normal data, steady-state filtering, residual-based scoring, control limits from normal data, event-level evaluation, simple baselines).
- Fix the flaws in the existing experimental design first, then measure.

## 2. Flaws found in the existing design and what was changed

### 2.1 Data used for training and calibration
- **Flaw:** train/val windows contained labelled anomalies; only unsupervised IF→PCA→Mahalanobis trimming was applied.
  - Why it matters: a model that sees anomalies learns to reconstruct them; thresholds calibrated on them are inflated.
  - Change: train/val = windows with no active rule node ±2 h.
- **Flaw:** no operating-state handling (running flag never wired; regimes = PCA+KMeans clusters).
  - Change: running mask from the speed/current tag; fully-running windows only; first hour after a start excluded.
- **Flaw:** dead-band tags with IQR ≈ 0 dominated the scaled variance (two 5ST901A vibration tags carried 96 %), forcing the PCA latent to the floor of 4.
  - Change: scale = max(IQR, winsorised std P1–P99); centre = median.
- **Flaw:** historian sentinel 1,709,896.75 and ±inf treated as values; filters computed on the whole file.
  - Change: sentinel/inf → NaN; null/variance filters on training rows only.

### 2.2 Inputs
- **Flaw:** `restrict_to_event_sensors` picked inputs from the labelled events (including test-period events) — label leakage, not deployable.
  - Change: all sensors except platform ML-output tags (`-MLO-`, `-MLD-`), hand switches (`5HS*`) and status bits (`5XI*`) — a tag-type rule, no labels.
  - Evidence: with ML tags included, one ML-output tag produced 963 of 1,789 false windows on 5P921A.

### 2.3 Model
- **Flaw:** the AE recipe under-fits — validation MSE 4–6× that of PCA with the same latent size (halving funnel to the bottleneck, dropout 0.2 everywhere, patience 15).
  - Change: `Autoencoder(hidden_widths=…)` and `TrainConfig.hidden_widths` added (backward compatible); v2 AE uses widths `[max(4·latent, n), max(2·latent, n/2)]`, dropout 0, patience 30, batch 1024 → 0.3–0.9× PCA error.
  - PCA reconstruction and a no-model per-sensor limit check (`level`) are always run alongside.

### 2.4 Score
- **Flaw:** mean per-row MSE over sensors, P95 over rows — single-sensor faults diluted 1/N, noisy sensors dominate, no smoothing.
  - Change: per-sensor standardised residuals (μ, σ from the clean validation month) → `maxz_med` = window median of max |z| over sensors; also mean z², Mahalanobis, EWMA (λ = 0.5), 2-window persistence.

### 2.5 Thresholds
- **Flaw:** Green/Yellow at P90 of contaminated validation errors → 10 % of normal windows flagged by construction (window false-alarm rate 0.77–0.995 in every old run).
  - Change: P99.5 (or GPD peaks-over-threshold) of the previous clean month's window scores; false episodes/day reported.

### 2.6 Evaluation
- **Flaw:** each config variant had a different regime-stratified test split (2–17 nodes), so PR-AUC columns were not comparable; positives = any Yellow/Red window.
  - Change: **rolling-origin monthly evaluation** — train on ≤ 8 previous months, calibrate on the previous month, test the next month; every window from month 7 on is scored exactly once by a model that never saw it.
  - Metrics: node-level recall (any alarm within the node ±30 min), window precision, alarm-episode precision, false episodes/day, composite F1 = harmonic mean of node recall and window precision, PR-AUC.
- Blocked 4-fold CV was tried first and rejected: 5ST901B relationships shift with season, so a model that never saw summer alarms on 100 % of summer windows.

## 3. Results

### 3.1 Old design vs new protocol (best new metric: PCA `maxz_med`, P99.5, process sensors)
| Equipment | Old precision / recall | Old F1 | New precision / recall | New F1 | Change | False episodes/day (new) | Nodes in new test set |
|---|---|---|---|---|---|---|---|
| 5P921A pump | 0.25 / 0.30 | 0.274 | 0.34 / 0.77 | 0.470 | +72 % | 0.20 | 56 of 71 |
| 5ST901A turbine A | 0.03 / 0.79 | 0.065 | 0.45 / 0.77 | 0.565 | +766 % | 0.32 | 87 of 103 |
| 5ST901B turbine B | 0.01 / 1.00 | 0.026 | 0.03 / 0.89 | 0.052 | +103 % | 1.03 | 18 of 41 |
| 5K512B compressor | 0.01 / 1.00 | 0.012 | 0.03 / 0.80 | 0.058 | +384 % | 0.25 | 30 of 36 |

- Old = base-model results on the 15 % stratified test split (positive-residual modality); precision is window-level, recall is rule nodes detected.
- Production-acceptable bar used: precision ≥ 0.7 and recall ≥ 0.8 → F1 ≥ 0.75. No unit reaches it.

### 3.2 Model comparison (new protocol, composite F1 / false episodes per day)
| Equipment | Level (no model) | PCA | AE (fixed recipe) | AE (repo recipe) |
|---|---|---|---|---|
| 5P921A | 0.39 / 0.28 | **0.47 / 0.20** | 0.42 / 0.33 | 0.46 / 0.56 |
| 5ST901A | 0.33 / 0.51 | **0.56 / 0.32** | 0.49 / 0.93 | 0.46 / 0.57 |
| 5ST901B | 0.05 / 1.12 | 0.05 / 1.03 | 0.06 / 1.21 | 0.05 / 1.05 |
| 5K512B | 0.02 / 0.45 | **0.06 / 0.25** | 0.03 / 0.66 | 0.03 / 0.35 |

- The autoencoder does not beat PCA on any unit; PCA is only modestly better than the no-model limit check.
- The old score (`mse_p95`) is the worst or near-worst aligned metric on every unit under the same protocol.

### 3.3 Where the false alarms come from (PCA `maxz_med`)
- 5P921A: 1,136 alarm windows → 385 on nodes, 122 within 1–6 h, 258 within 6–24 h, 371 farther; 29 of 43 detected nodes were alarmed in the 6 h **before** the rule fired.
- 5ST901A: 1,142 → 509 on nodes, 158 within 1–6 h, 422 farther; 45 of 67 detected nodes alarmed before the rule.
- 5ST901B: 66 % of June 2026 windows alarmed on a stator-winding temperature that rose seasonally below the rule limit; the next monthly retrain adapted.
- 5K512B: ~50 positive windows in 9,786; false alarms on gas analyzers and valve tags; removing them moved the false alarms elsewhere.
- Almost none of the false windows are attributed to a sensor of a nearby node → they are early warnings, sub-threshold conditions, or seasonal shifts, not label-adjacent noise.

## 4. Verdict

- **Recall: achievable** — 0.77–0.94 of rule nodes, typically hours ahead of the rule.
- **Precision: not achievable against these labels** — ≤ 0.45 window / ≤ 0.47 episode on event-rich units, ≤ 0.16 on sparse units.
- **Why:** rule nodes mark fixed-threshold crossings on specific sensors; a residual model marks deviations from expected behaviour. The two disagree by design, and only SME review of the ~50–90 unlabelled alarm episodes per unit can decide which alarms are real.
- **Autoencoder vs alternatives:** AE ≈ PCA ≈ per-sensor limit check on clean recent data; the deep model adds no measurable value here.
- **If deployed anyway:** PCA or AE residuals on running-state, event-free, process-sensor data; `maxz_med` with EWMA; monthly retrain; P99.5 of the previous clean month; 2-window persistence → ~0.75 recall, ~0.45 episode precision, ~0.15–0.2 false episodes/day on units with regular events.

## 5. How to run

```bash
OMP_NUM_THREADS=3 .venv/bin/python scripts/protocol_v2.py --equipment 5P921A --output output_v2_roll \
  --sensors process --scheme rolling --with-level --with-repo-ae
.venv/bin/python scripts/summarize_protocol_v2.py --root output_v2_roll --select [--tol 12]
.venv/bin/python scripts/alarm_anatomy_v2.py output_v2_roll/5P921A/process pca maxz_med
```

- Outputs per run: `fold_<month>/{window_scores.parquet, metrics.json, ae_history.json}`, `pooled_window_scores.parquet`, `summary.json`, `nodes.json`.
- Runtime: 20–45 min per pump/compressor, 2–3 h per turbine (CPU, 3 threads).

## 6. Files changed

- New: `scripts/protocol_v2.py`, `scripts/summarize_protocol_v2.py`, `scripts/alarm_anatomy_v2.py`, `docs/protocol_v2.md`.
- Edited (backward compatible, 280 tests pass): `src/autoencoder/model/architecture.py` (`hidden_widths` argument), `src/autoencoder/training/trainer.py` (`TrainConfig.hidden_widths`).
- Not versioned: `output_v2*/` run artefacts (git-ignored).
