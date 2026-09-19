# pipeline-overview

# Autoencoder Pipeline — Read-Through

A feedforward autoencoder learns what “normal” looks like for a piece of rotating
equipment from historian sensor data, then flags 30-minute windows whose
reconstruction error is unusually high. This is the pipeline end to end, in the
order it actually runs: `ingest → preprocess → segment regimes → clean → train → calibrate → infer → alert → (monitor, evaluate)`.

## 1. Ingest & preprocess

Raw sensor data (CSV or Parquet) is loaded, timestamp-sorted, and validated — a
bad or non-numeric column is rejected here rather than crashing deep inside the
scaler later. Sensor columns that are constant or near-constant (stuck sensors,
boolean run-status tags) are dropped; they’d add a model input dimension with no
real signal.

A `RobustScaler` (median/IQR, not mean/std — robust to outliers and skew) is fit
on only the earliest slice of the data sized to the configured train split, so no
information from validation or test data leaks into it. Every row, train and
otherwise, is then scaled with that same frozen scaler for the rest of the
pipeline’s life. Optionally, scaled values pass through a tail-compression
transform (`c·asinh(z/c)`) — real historian tags that sit in a narrow deadband
almost all the time can produce z-scores in the thousands on a single glitch,
which would otherwise swamp the training loss; the compression keeps relative
severity and gradient intact instead of clipping it flat.

The scaled series is then cut into fixed, non-overlapping 120-row windows (30
minutes at a 15-second sample rate) — one input shape the rest of the pipeline
assumes everywhere.

## 2. Regime segmentation & window filtering

Windows are clustered by KMeans into Low/Medium/High operating regimes, using
load-indicator sensors (speed, power, discharge pressure) if configured, or a
PCA-based fallback on the whole sensor set if those columns aren’t present for
a given equipment. This matters because a normal High-load reading looks like an
outlier next to a Low-load majority, and vice versa — everything downstream
(cleaning, later evaluation) is regime-aware because of this step.

Windows where more than 5% of sensors are null or “stuck” (frozen for the whole
window) are dropped, with the regime labels filtered in lockstep so the two
arrays never drift out of index alignment.

## 3. Outlier cleaning

Within each regime separately (so a regime’s own legitimate spread isn’t
compared against another regime’s), windows pass through three outlier
detectors in a fixed order: **Isolation Forest** (tree-based, catches gross
non-linear outliers first, contamination ~5%), **PCA-space reconstruction
error** (catches windows that break the normal cross-sensor correlation
structure), then **Mahalanobis distance** (covariance-normalized distance from
the regime centroid, run last because its own covariance estimate is more
reliable once the obvious outliers are already gone). Each detector only ever
sees what the previous one didn’t remove — this ordering exists specifically so
Mahalanobis’s covariance estimate isn’t itself distorted by the outliers it’s
trying to catch.

If more than 40% of a regime’s windows would be removed, a warning is logged
(advisory only — nothing blocks the pipeline). An HTML report and diagnostic
scatter plots are generated documenting exactly what each detector removed.
Rejected windows are kept as an audit trail, separate from the windows used to
train the model.

## 4. Train

Windows are split train/val/test (85/10/5% by default), sliced within each
regime before concatenating, so validation and test aren’t accidentally
sampled from only one regime.

The bottleneck (latent) dimension is picked automatically via PCA on the
training data — the smallest number of components that explains 95% of
variance, floored at 4 so degenerate sensor sets don’t collapse the bottleneck
to nothing. Hidden-layer widths are then derived by repeatedly halving the
sensor count down toward that latent size, capped at 3 hidden layers — an
uncapped, much deeper network (verified on a 476-sensor case) got permanently
stuck in a bad plateau after epoch 1 and never recovered. Every hidden layer
uses LayerNorm rather than BatchNorm, because each training batch is rows from
a single, highly autocorrelated 30-minute window — batch statistics would be a
poor, noisy stand-in for a sensor’s real global spread.

Training uses Adam with a `ReduceLROnPlateau` scheduler and early stopping,
where patience scales with network depth but always stays above the LR
scheduler’s own patience (otherwise early stopping could fire before the
scheduler ever gets a chance to reduce the learning rate — this was verified as
a real, silent failure mode at one point). Weight decay is set higher than the
dataclass default specifically because the lower value let validation loss
diverge badly from training loss (2–12x ratios were observed) — an
“overfit ratio” warning now surfaces this automatically every epoch. Rows are
shuffled across all training windows before batching (not one window per
gradient step) — the literal per-window version was verified not to converge
on this data, since a whole window’s rows are one long autocorrelated run.
After training, the model’s weights are rolled back to the best validation
epoch, not whatever epoch training happened to stop on.

## 5. Calibrate thresholds & save

Green/Yellow/Red cutoffs (P90/P99 by default) are calibrated from the model’s
reconstruction error on the **validation** split, not the training split — an
ablation study confirmed this meaningfully improves detection quality (Red F1
0.88→0.92) because training errors are optimistic by construction. Thresholds
are computed “robustly” by default: log-transforming scores, then reading the
target percentile off the median and a robust spread estimate (MAD or IQR)
rather than the raw empirical tail, which is a much smaller and noisier sample
than the bulk of the distribution.

A window’s many per-row errors are collapsed to one score via the 95th
percentile — resistant to a single noisy row (unlike mean) without letting one
spike dominate (unlike max). Per-sensor baselines are computed from the held-out
test split, giving each sensor its own “what’s normal for you” reference used
at inference time.

The trained model, scaler, thresholds, metadata, and baselines are all saved
together as one bundle, so `--infer-only` runs can load a previously trained
model and score new data without retraining.

## 6. Infer & diagnose

At inference, a window’s sensors are first quality-checked: short null runs (up
to 3 consecutive readings) are forward-filled; a sensor whose raw null
percentage is too high, or that still has unfillable gaps, is masked out of
scoring rather than trusted. If too large a fraction of all sensors are
masked (mirroring the same tolerance used when filtering training windows),
the whole window is rejected rather than scored — an anomaly score built from
mostly-fabricated data isn’t trustworthy. Masked sensors are neutralized to
their scaled median before the forward pass (never left as NaN, which would
poison every other reconstructed sensor through the model’s shared layers) and
excluded from both the window score and the per-sensor ranking.

An unusable window (rejected on quality grounds, or equipment not running) is
always treated as Green — never Red — because an untrustworthy score should
fail safe to “no alert,” not manufacture a false one.

Sensor-level diagnosis — *which* sensors are responsible for a flagged window —
is a single heuristic: each sensor’s share of total reconstruction error, plus
a ratio against that sensor’s own historical baseline to flag it as
individually anomalous. 

## 7. Alert

A window’s score is classified into Green/Yellow/Red against the calibrated
thresholds, then persistence rules turn that per-window zone history into an
actual alert level: 2 consecutive Red windows raises Yellow, 3+ consecutive Red
raises Red, 4+ consecutive Yellow (with no Red) raises Yellow on its own, and an
interleaved mix of Red and Yellow within a trailing ~2-hour span can also raise
Yellow even when no single streak rule fires. Single isolated Red windows never
alert alone — the point of all these rules is to require sustained evidence
before paging anyone. Separately, runs of consecutive non-Green windows are
grouped into “episodes” — the actual client-facing alert unit used for
reporting and evaluation, so a single six-hour event is counted once, not as a
dozen independent per-window alerts.

## 8. Monitoring (implemented, not wired in)

A separate module can compute weekly health summaries, detect gradual drift
(rolling-mean trend regression) and sudden distribution shifts (KS-test), and
recommend a retrain when either fires or the false-alarm rate climbs. None of
this is currently called from `main.py` or any script — it exists and is
tested, but nothing in the pipeline invokes it today, so no retrain decision is
ever made automatically. A related config key for flagging a sensor that’s been
dead for 48+ hours has no implementing code at all.

## 9. Offline evaluation

Separate from the live pipeline, evaluation scripts compare model output
against either injected synthetic ground truth (known event windows and which
sensors were spiked) or real SME-labeled rule firings. Metrics are computed
per rule “slug” (not just an overall pass/fail) using the identical code path
for every slug, so no slug’s behavior is hand-coded specially. Both pointwise
metrics (precision/recall/F1) and episode-aware ones (duration, episode-level false-alarm rate) are reported, because per-window metrics alone can’t distinguish “missed one long fault” from
“missed two hundred short ones.” When no ground truth exists at all, a
separate unlabeled-evaluation mode deliberately reports only what the model’s
own output implies and withholds any precision/recall number, rather than
fabricate one against pseudo-labels.

---

For the full per-decision breakdown — exact numbers, function names, and the
“what if this weren’t there” reasoning behind each step — see
[pipeline-design.md](pipeline-design.md).