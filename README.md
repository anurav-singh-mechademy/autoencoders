# Autoencoder — Industrial Equipment Health Monitoring

A feedforward-autoencoder pipeline for detecting anomalies in industrial equipment
sensor data (rotating machinery telemetry: speed, pressure, temperature, vibration,
etc.). The model learns to reconstruct normal operating behaviour from historian
data; reconstruction error is calibrated into Green/Yellow/Red health zones and
turned into alerts, with a per-sensor MSE-share heuristic to explain *why* a window
was flagged.

## Pipeline

```
clean → train → infer → alert
```

1. **Ingest** — load CSV/Parquet historian export, validate schema, detect sensor columns.
2. **Preprocess** — drop low-variance columns, fit a `RobustScaler` on the train split,
   construct 30-minute windows (120 rows at 15-second intervals).
3. **Segment regimes** — KMeans clustering on speed/power/pressure into Low/Medium/High
   operating regimes.
4. **Clean** — per-regime outlier removal: Isolation Forest → PCA → Mahalanobis distance.
5. **Train** — feedforward autoencoder (LayerNorm, dropout) with early stopping and
   a reduce-on-plateau LR schedule.
6. **Calibrate** — compute Green/Yellow/Red thresholds (P90/P99) from validation-set
   reconstruction error, plus per-sensor baselines.
7. **Infer & alert** — classify windows into zones, apply persistence rules (e.g.
   3+ consecutive Red windows → alert), and attribute the anomaly to the responsible
   sensors by their share of total reconstruction error.

Configuration for every stage lives in [configs/default.yaml](configs/default.yaml).

## Project layout

```
main.py                    End-to-end CLI: clean + train + infer + alert
src/autoencoder/
  data/                    Ingestion, preprocessing, windowing, regime/transient detection
  cleaning/                Isolation Forest, PCA, Mahalanobis outlier removal
  model/                   Autoencoder architecture and loss
  training/                Dataset splitting, training loop, hyperparameter search
  alerting/                Threshold calibration, zone classification, alert persistence
  inference/               Inference pipeline, missing-data handling, diagnosis
  monitoring/               Drift detection, retrain triggers, rolling health log
  artefacts/               Model (de)serialisation
  reporting/                Cleaning report generation
scripts/                    Standalone CLIs (see below)
configs/                    YAML configs (default.yaml, dvn.yaml)
tests/                      pytest suite, mirrors src/autoencoder/ structure
```

## Installation

Requires Python ≥3.10. Uses [uv](https://github.com/astral-sh/uv) (`uv.lock` is checked in),
but plain `pip` works too.

```bash
uv sync                 # or: pip install -e ".[dev]"
```

## Usage

### Full pipeline

```bash
python main.py --data sensor_data.csv --output output/ --config configs/default.yaml
```

Other `main.py` modes:

```bash
# Skip cleaning (data is already cleaned windows, .npy)
python main.py --data cleaned_windows.npy --output output/ --skip-cleaning

# Inference only, against an existing trained model
python main.py --data new_data.csv --output output/ --model-dir output/artefacts/ --infer-only
```

### Standalone scripts

| Script | Purpose |
|---|---|
| `scripts/generate_synthetic_data.py` | Generate synthetic sensor data (with injected anomaly ground truth) for testing |
| `scripts/prepare_real_data.py` | Clean up a raw real-world historian export (missing sensors, NaNs) before it hits the pipeline |
| `scripts/run_cleaning.py` | Run just the data-cleaning stage and emit a report |
| `scripts/train.py` | Train the autoencoder on pre-cleaned windows |
| `scripts/infer.py` | Run inference on a single window or batch |
| `scripts/evaluate_detection.py` | Score detection accuracy (precision/recall/F1/AUC) against synthetic ground-truth events |
| `scripts/evaluate_unlabeled.py` | Evaluate against real (unlabeled) data — no precision/recall, since there's no ground truth |
| `scripts/compare_thresholds.py` | Re-derive Green/Yellow/Red breakdowns under different threshold calibration methods without retraining |
| `scripts/run_ablation.py` | Sweep config variants through the real pipeline and score each against ground truth |
| `scripts/score_full_timeline.py` | Score one train/val/test split's windows across a dataset's full timeline |
| `scripts/evaluate_against_labels.py` | Precision/recall/PR-AUC of a scored test split against real labeled event data |
| `scripts/recalibrate_prevalence_thresholds.py` | Recalibrate Green/Yellow/Red thresholds against observed event prevalence instead of fixed P90/P99 |
| `scripts/build_sensor_event_timeline.py` | Build a self-contained HTML chart of raw sensor traces, model zone verdicts, and labeled events for one run's test split |
| `scripts/analyze_flag_timing.py` | Signed lead/lag distance of each flagged window to the nearest labeled event |
| `scripts/analyze_signal_alignment.py` | Correlate anomaly_score / ground truth against volatility and magnitude reference metrics |
| `scripts/analyze_directional_alignment.py` | Correlate anomaly_score / ground truth against signed drift and level (vs. training baseline) |
| `scripts/analyze_flagged_direction.py` | Split flagged windows by within-window sensor trend and compare precision |
| `scripts/analyze_flagged_direction_lookback.py` | Split flagged windows by trend over a lookback period and compare precision |
| `scripts/analyze_reconstruction_error_sign.py` | Retain the sign of the reconstruction residual per window and report precision/PR-AUC/recall split by it |
| `scripts/rescore_directional.py` | Sweep an asymmetric directional weight on the reconstruction residual without retraining |
| `scripts/simulate_modality_suppression.py` | Simulate suppressing one reconstruction-error-sign modality's flags and compare metrics |

Run any script with `--help` for its full argument list. The `scripts/investigate_*.py`
scripts are one-off, ad hoc analyses from a specific investigation (kept for reference,
not general-purpose tools).

### Ground-truth evaluation workflow

For equipment with real labeled fault events (`data/<eq>/<eq>_combined_with_events.parquet`
+ `<eq>_event_labels_long.parquet`), score the trained model's test split and evaluate it
against those labels:

```bash
python scripts/score_full_timeline.py \
    --data data/<eq>/<eq>_combined_with_events.parquet \
    --model-dir output_<eq>/artefacts \
    --split-ids output_<eq>/artefacts/split_window_ids.json \
    --output output_<eq>/evaluation/window_scores.parquet

python scripts/evaluate_against_labels.py \
    --window-scores output_<eq>/evaluation/window_scores.parquet \
    --ground-truth data/<eq>/<eq>_combined_with_events.parquet \
    --event-labels data/<eq>/<eq>_event_labels_long.parquet \
    --output output_<eq>/evaluation
```

`preprocessing.restrict_to_event_sensors` (see `configs/default.yaml`) restricts the
model's sensor columns to only those implicated by labeled events, computed once over
the whole file so it can't leak split information.

## Testing

```bash
pytest
```

## Notes

- Experiment outputs (`output*/`) and log files are git-ignored — they're large,
  regenerable, and specific to individual runs.
- Raw/large data files (`*.csv`, `*.parquet`) are also git-ignored; see
  [.gitignore](.gitignore) for the full list of exclusions.
