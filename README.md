# Autoencoder — Industrial Equipment Health Monitoring

A feedforward-autoencoder pipeline for detecting anomalies in industrial equipment
sensor data (rotating machinery telemetry: speed, pressure, temperature, vibration,
etc.). The model learns to reconstruct normal operating behaviour from historian
data; reconstruction error is calibrated into Green/Yellow/Red health zones and
turned into alerts, with optional per-sensor attribution to explain *why* a window
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
   3+ consecutive Red windows → alert), and optionally attribute the anomaly to the
   responsible sensors (heuristic, Integrated Gradients, or a trained FastSHAP explainer).

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
  explain/                 FastSHAP, Integrated Gradients, masking-based attribution
  monitoring/               Drift detection, retrain triggers, rolling health log
  artefacts/               Model/explainer (de)serialisation
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

# Also train a FastSHAP explainer for per-sensor attribution
python main.py --data sensor_data.csv --output output/ --explain
```

### Standalone scripts

| Script | Purpose |
|---|---|
| `scripts/generate_synthetic_data.py` | Generate synthetic sensor data (with injected anomaly ground truth) for testing |
| `scripts/prepare_real_data.py` | Clean up a raw real-world historian export (missing sensors, NaNs) before it hits the pipeline |
| `scripts/run_cleaning.py` | Run just the data-cleaning stage and emit a report |
| `scripts/train.py` | Train the autoencoder on pre-cleaned windows |
| `scripts/infer.py` | Run inference on a single window or batch |
| `scripts/explain.py` | Generate and sanity-check FastSHAP attributions |
| `scripts/evaluate_detection.py` | Score detection accuracy (precision/recall/F1/AUC) against synthetic ground-truth events |
| `scripts/evaluate_unlabeled.py` | Evaluate against real (unlabeled) data — no precision/recall, since there's no ground truth |
| `scripts/compare_thresholds.py` | Re-derive Green/Yellow/Red breakdowns under different threshold calibration methods without retraining |
| `scripts/run_ablation.py` | Sweep config variants through the real pipeline and score each against ground truth |

Run any script with `--help` for its full argument list.

## Testing

```bash
pytest
```

## Notes

- Experiment outputs (`output*/`) and log files are git-ignored — they're large,
  regenerable, and specific to individual runs.
- Raw/large data files (`*.csv`, `*.parquet`) are also git-ignored; see
  [.gitignore](.gitignore) for the full list of exclusions.
