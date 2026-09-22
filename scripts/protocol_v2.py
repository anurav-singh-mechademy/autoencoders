#!/usr/bin/env python
"""Protocol v2 -- established-practice evaluation of autoencoder anomaly scores against rule-event labels.

Design (each point maps to a documented practice; see kb/research/best_practices.md):
  * Semi-supervised (one-class) training: train/val windows are RUNNING, NOT event-touched (+/- margin), and
    NOT start-up transients. Test windows keep every running, non-transient, usable window (events included).
  * Strict blocked time cross-validation (4 contiguous blocks, 1-day gap): every window is scored exactly once
    by a model that never saw its block -> one fixed test set shared by every model/score variant.
  * Sensor selection is label-free (all sensors passing dead/null filters computed on the training rows only).
  * Preprocessing fit on training rows only: RobustScaler + c*asinh(z/c) tail compression, sentinel/inf -> NaN,
    ffill<=3 rows, window usable if <=5% cells missing.
  * Models: PCA reconstruction baseline (must be beaten) and the repo's feedforward Autoencoder (unchanged
    architecture/optimiser/early stopping; latent from PCA 95% variance).
  * Scores from the residual matrix R = x - x_hat with per-sensor residual standardisation on the normal
    validation split (z_j = (r_j - mu_j) / sigma_j): mean r^2 (current), mean z^2 (normalised SPE),
    max |z| (single-sensor sensitivity), Mahalanobis distance (Ledoit-Wolf) -- each aggregated per 30-min window
    by P95 and by median, each optionally EWMA-smoothed across consecutive windows.
  * Thresholds from the normal validation score distribution only (P99 / P99.5 / P99.9 / GPD-POT 99.5).
  * Metrics: window PR-AUC / ROC-AUC (threshold-free), event(node)-level recall, alarm-episode precision,
    false alarms per day, composite F1 (event recall x window precision), with optional 2-window persistence.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from autoencoder.alerting.thresholds import _gpd_tail_threshold  # noqa: E402
from autoencoder.model.architecture import Autoencoder, pick_latent_dim  # noqa: E402
from autoencoder.training.trainer import TrainConfig, train_model  # noqa: E402

log = logging.getLogger("protocol_v2")
W = 120
SENTINELS = (1709896.75,)
RUN_TAG = {"5P921A": "M921A-EMAI-IO", "5K512B": "5SI-6000ABC", "5ST901A": "STGA-SPEED", "5ST901B": "STGB-SPEED"}
LABEL_COLS = ("timestamp", "active_node_ids", "n_active_nodes")
SCORE_BASES = ("mse", "spez", "maxz", "md")
AGGS = ("p95", "med")


# ----------------------------------------------------------------------------- data
def is_derived_or_discrete(tag: str) -> bool:
    return ("-MLO-" in tag) or ("-MLD-" in tag) or tag.startswith("5HS") or tag.startswith("5XI")


def is_actuator_or_analyzer(tag: str) -> bool:
    """Composition analyzers (AI) and control-valve commands/positions (FV, ZI, *-OUT, *-PCT) are not equipment-health
    sensors: analyzers track feed composition, actuators move by design. Label-free, tag-type policy (ISA-5.1 letters)."""
    t = tag.upper()
    return t.startswith("5AI-") or t.startswith("5FV-") or t.startswith("5ZI-") or t.endswith("-OUT") or t.endswith("-PCT") or t.endswith("_OP")


def load(eq: str, data_dir: str):
    p = Path(data_dir) / eq / f"{eq}_combined_with_events.parquet"
    df = pd.read_parquet(p).sort_values("timestamp").reset_index(drop=True)
    sensors = [c for c in df.columns if c not in LABEL_COLS and pd.api.types.is_numeric_dtype(df[c])]
    X = np.array(df[sensors].to_numpy(dtype=np.float32), copy=True)
    X[~np.isfinite(X)] = np.nan
    for s in SENTINELS:
        X[X == np.float32(s)] = np.nan
    ts = df["timestamp"].to_numpy().astype("datetime64[ns]")
    n_active = df["n_active_nodes"].to_numpy()
    lab = pd.read_parquet(Path(data_dir) / eq / f"{eq}_event_labels_long.parquet")
    return ts, sensors, X, n_active, lab


def node_table(lab: pd.DataFrame, ts: np.ndarray, n_w: int) -> pd.DataFrame:
    """One row per rule node: slug, severity, sensors, sorted window ids where it is active."""
    pos = np.searchsorted(ts, lab["timestamp"].to_numpy().astype("datetime64[ns]"))
    pos = np.clip(pos, 0, len(ts) - 1)
    lab = lab.assign(window=pos // W)
    lab = lab[lab.window < n_w]
    g = lab.groupby("node_id")
    out = pd.DataFrame({
        "slug": g["rule_slug"].first(),
        "severity": g["severity"].max(),
        "sensors": g["sensors"].first().apply(lambda v: sorted(set(v))),
        "start": g["start_time"].min(),
        "end": g["end_time"].max(),
        "windows": g["window"].apply(lambda s: sorted(set(s))),
    }).reset_index()
    out["duration_h"] = (out["end"] - out["start"]).dt.total_seconds() / 3600
    return out


# ----------------------------------------------------------------------------- windows / masks
def window_masks(eq, ts, sensors, X, n_active, post_start_margin, event_margin):
    n_w = len(ts) // W
    tag = pd.Series(X[:, sensors.index(RUN_TAG[eq])]).ffill().bfill().to_numpy()
    running = tag > 0.5 * np.nanmedian(tag)
    run_frac = running[: n_w * W].reshape(n_w, W).mean(1)
    fully_running = run_frac >= 1.0
    # start-up transient: first `post_start_margin` fully-running windows after any non-fully-running window
    transient = np.zeros(n_w, bool)
    for k in range(1, post_start_margin + 1):
        prev = np.concatenate([np.ones(k, bool), ~fully_running[:-k]])
        transient |= fully_running & prev
    event = n_active[: n_w * W].reshape(n_w, W).max(1) > 0
    near_event = event.copy()
    for k in range(1, event_margin + 1):
        near_event[k:] |= event[:-k]
        near_event[:-k] |= event[k:]
    start = ts[: n_w * W : W]
    return n_w, fully_running, transient, event, near_event, start


def blocked_folds(n_w: int, n_folds: int, gap: int):
    """Blocked CV: (fold_id, test_mask, train_pool_mask, val_mask=None -> every-5th-day rule inside the pool)."""
    block = np.zeros(n_w, int)
    for b, idx in enumerate(np.array_split(np.arange(n_w), n_folds)):
        block[idx] = b
    folds = []
    for b in range(n_folds):
        test = block == b
        t_idx = np.flatnonzero(test)
        lo, hi = t_idx[0] - gap, t_idx[-1] + gap
        train_pool = ~test & ((np.arange(n_w) < lo) | (np.arange(n_w) > hi))
        folds.append((f"b{b}", test, train_pool, None))
    return folds


def rolling_folds(start: np.ndarray, min_hist_months: int, max_train_months: int, gap: int):
    """Rolling-origin (production-faithful) evaluation: for each calendar month m with >= min_hist_months of history,
    train on months [m-1-max_train_months, m-2], calibrate thresholds on month m-1 (most recent normal data, minus a
    `gap`-window buffer before the test month), test on month m. Every window from month `min_hist_months` on is
    scored exactly once by a model that has seen only earlier data."""
    months = pd.to_datetime(start).to_period("M")
    uniq = sorted(months.unique())
    mi = np.searchsorted(np.array(uniq), months)
    folds = []
    for i in range(min_hist_months, len(uniq)):
        test = mi == i
        val = mi == (i - 1)
        vidx = np.flatnonzero(val)
        if gap > 0 and len(vidx) > gap:
            val[vidx[-gap:]] = False
        train = (mi >= max(0, i - 1 - max_train_months)) & (mi <= i - 2)
        folds.append((f"m{uniq[i]}", test, train, val))
    return folds


def prep_windows(X, sensors_keep_idx, rows_train, scaler_center, scaler_scale, c, n_w):
    """Scale + compress + ffill + window; returns (windows[n_w,W,S] float32, missing_frac[n_w])."""
    Xs = X[: n_w * W][:, sensors_keep_idx]
    Xs = pd.DataFrame(Xs).ffill(limit=3).to_numpy(dtype=np.float32)
    Z = (Xs - scaler_center) / scaler_scale
    if c:
        Z = c * np.arcsinh(Z / c)
    miss = np.isnan(Z)
    Z[miss] = 0.0
    Z = Z.reshape(n_w, W, -1).astype(np.float32)
    missing_frac = miss.reshape(n_w, -1).mean(1)
    return Z, missing_frac


def fit_robust_scale(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sensor centre = median; scale = max(IQR, winsorised std [P1,P99]).

    Plain IQR scaling (RobustScaler) breaks on deadband-logged tags whose IQR is ~0 while they still move
    occasionally: 5ST901A vibration tags 5VT-67500A2/67506A1 had IQR 1e-5 and after IQR scaling carried 96 %
    of the total scaled variance, forcing the PCA latent to the floor and letting two tags dominate the MSE.
    The winsorised std is the classical unit-variance scaling of multivariate SPC made glitch-robust; for a
    Gaussian-like tag IQR (1.35 sigma) still wins, so well-behaved sensors are scaled exactly as before."""
    med = np.nanmedian(Xtr, axis=0)
    q25, q75 = np.nanpercentile(Xtr, [25, 75], axis=0)
    lo, hi = np.nanpercentile(Xtr, [1, 99], axis=0)
    wstd = np.clip(np.nan_to_num(Xtr, nan=med), lo, hi).std(axis=0)
    scale = np.maximum(q75 - q25, wstd)
    scale = np.where(scale > 0, scale, 1.0)
    return med.astype(np.float32), scale.astype(np.float32)


# ----------------------------------------------------------------------------- models
def fit_pca(train_rows: np.ndarray, latent: int) -> PCA:
    return PCA(n_components=latent, random_state=42).fit(train_rows)


def reconstruct_pca(pca: PCA, rows: np.ndarray) -> np.ndarray:
    return pca.inverse_transform(pca.transform(rows)).astype(np.float32)


def reconstruct_ae(model: Autoencoder, rows: np.ndarray, batch: int = 65536) -> np.ndarray:
    model.eval()
    out = np.empty_like(rows)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            out[i : i + batch] = model(torch.from_numpy(rows[i : i + batch])).numpy()
    return out


# ----------------------------------------------------------------------------- scores
class ResidualStats:
    def __init__(self, R_val: np.ndarray):
        self.mu = R_val.mean(0)
        self.sigma = np.maximum(R_val.std(0), 1e-6)
        sub = R_val[np.random.default_rng(0).choice(len(R_val), min(len(R_val), 100_000), replace=False)]
        self.prec = LedoitWolf().fit(sub - self.mu).precision_.astype(np.float32)

    def row_scores(self, R: np.ndarray) -> dict[str, np.ndarray]:
        Rc = R - self.mu
        z = Rc / self.sigma
        md = np.einsum("ij,jk,ik->i", Rc, self.prec, Rc, optimize=True)
        return {
            "mse": (R ** 2).mean(1),
            "spez": (z ** 2).mean(1),
            "maxz": np.abs(z).max(1),
            "md": np.sqrt(np.maximum(md, 0)),
            "argmaxz": np.abs(z).argmax(1),
        }


def window_scores(row: dict[str, np.ndarray], n_w: int) -> dict[str, np.ndarray]:
    out = {}
    for b in SCORE_BASES:
        m = row[b].reshape(n_w, W)
        out[f"{b}_p95"] = np.percentile(m, 95, axis=1)
        out[f"{b}_med"] = np.median(m, axis=1)
    am = row["argmaxz"].reshape(n_w, W)
    out["top_sensor_idx"] = np.array([np.bincount(r).argmax() for r in am])
    return out


def ewma(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = lam * v + (1 - lam) * acc
        out[i] = acc
    return out


# ----------------------------------------------------------------------------- metrics
def episodes(alarm: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True in an ordered boolean sequence -> (start_pos, end_pos) inclusive."""
    eps, start = [], None
    for i, a in enumerate(alarm):
        if a and start is None:
            start = i
        if not a and start is not None:
            eps.append((start, i - 1)); start = None
    if start is not None:
        eps.append((start, len(alarm) - 1))
    return eps


def event_metrics(wid: np.ndarray, score: np.ndarray, thr: float, event_pos: np.ndarray, nodes: pd.DataFrame,
                  tol: int, persistence: int) -> dict:
    """wid: evaluated window ids (ordered); score aligned; event_pos: bool per evaluated window (event +/- tol)."""
    alarm = score > thr
    if persistence > 1:
        a = alarm.copy()
        for k in range(1, persistence):
            a[k:] &= alarm[:-k]
            a[:k] = False
        alarm = a
    wid_set = {int(w): i for i, w in enumerate(wid)}
    det, det_by_slug, n_by_slug = 0, {}, {}
    for _, nd in nodes.iterrows():
        ws = set()
        for w in nd["windows"]:
            for d in range(-tol, tol + 1):
                ws.add(w + d)
        idx = [wid_set[w] for w in ws if w in wid_set]
        if not idx:
            continue  # node has no evaluated window (not running / other fold) -> not counted
        n_by_slug[nd["slug"]] = n_by_slug.get(nd["slug"], 0) + 1
        hit = bool(alarm[idx].any())
        det += hit
        det_by_slug[nd["slug"]] = det_by_slug.get(nd["slug"], 0) + hit
    n_nodes = sum(n_by_slug.values())
    eps = episodes(alarm)
    tp_ep = sum(1 for s, e in eps if event_pos[s : e + 1].any())
    n_alarm = int(alarm.sum())
    win_prec = float((alarm & event_pos).sum() / n_alarm) if n_alarm else float("nan")
    ev_rec = det / n_nodes if n_nodes else float("nan")
    days = len(wid) * (W * 15) / 86400
    comp_f1 = (2 * win_prec * ev_rec / (win_prec + ev_rec)) if n_alarm and n_nodes and (win_prec + ev_rec) > 0 else 0.0
    return {
        "threshold": float(thr), "n_eval_windows": int(len(wid)), "n_alarm_windows": n_alarm,
        "alarm_rate": n_alarm / len(wid), "n_nodes_evaluable": int(n_nodes), "nodes_detected": int(det),
        "event_recall": ev_rec, "window_precision": win_prec, "composite_f1": comp_f1,
        "n_alarm_episodes": len(eps), "tp_episodes": tp_ep,
        "episode_precision": tp_ep / len(eps) if eps else float("nan"),
        "false_alarms_per_day": (len(eps) - tp_ep) / days,
        "recall_by_slug": {s: f"{det_by_slug.get(s, 0)}/{n}" for s, n in sorted(n_by_slug.items())},
    }


# ----------------------------------------------------------------------------- main per equipment
def run_equipment(eq: str, a: argparse.Namespace) -> None:
    out_root = Path(a.output) / eq / a.sensors
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ts, sensors, X, n_active, lab = load(eq, a.data_dir)
    n_w, fully_running, transient, event, near_event, start = window_masks(eq, ts, sensors, X, n_active, a.post_start_margin, a.event_margin)
    nodes = node_table(lab, ts, n_w)
    folds = blocked_folds(n_w, a.n_folds, a.gap) if a.scheme == "blocked" else rolling_folds(start, a.min_hist_months, a.max_train_months, a.gap)
    if a.sensors == "event":
        keep_names = sorted(set(s for v in lab["sensors"] for s in v) & set(sensors))
    elif a.sensors == "process":
        # label-free, tag-type based: drop platform ML-model outputs (MLO/MLD), hand switches (HS) and status bits (XI);
        # keep instruments and physics component-model tags (PUCE/EMAI/COCE/STNO/GTSS)
        keep_names = [s for s in sensors if not is_derived_or_discrete(s)]
    elif a.sensors == "process2":
        keep_names = [s for s in sensors if not is_derived_or_discrete(s) and not is_actuator_or_analyzer(s)]
    else:
        keep_names = list(sensors)
    log.info("%s: %d rows, %d windows, %d nodes, %d candidate sensors; running %.3f transient %.3f event %.3f",
             eq, len(ts), n_w, len(nodes), len(keep_names), fully_running.mean(), transient.mean(), event.mean())
    nodes.to_json(out_root / "nodes.json", orient="records", date_format="iso")

    all_rows = []
    fold_summaries = []
    for fi, (b, test_mask, train_pool, val_mask) in enumerate(folds):
        fdir = out_root / f"fold_{b}"; fdir.mkdir(exist_ok=True)
        clean = fully_running & ~transient & ~near_event
        if val_mask is None:
            trainable = train_pool & clean
            # val = every 5th calendar day of the trainable pool (interleaved so val spans the same conditions as train)
            days = pd.to_datetime(start).normalize()
            day_rank = pd.factorize(days[trainable], sort=True)[0]
            is_val = np.zeros(n_w, bool); is_val[np.flatnonzero(trainable)[day_rank % 5 == 4]] = True
            is_train = trainable & ~is_val
        else:
            is_train = train_pool & clean; is_val = val_mask & clean
        if is_train.sum() < a.min_train_windows or is_val.sum() < 200:
            log.warning("%s fold %s skipped: train %d val %d windows", eq, b, is_train.sum(), is_val.sum()); continue
        # sensor filters on TRAIN rows only
        tr_rows_idx = (np.flatnonzero(is_train)[:, None] * W + np.arange(W)).ravel()
        Xtr = X[tr_rows_idx][:, [sensors.index(s) for s in keep_names]]
        null_frac = np.isnan(Xtr).mean(0); var = np.nanvar(Xtr, axis=0)
        ok = (null_frac <= a.max_null_frac) & (var > 1e-5)
        keep_idx = [sensors.index(s) for s, k in zip(keep_names, ok) if k]
        kept = [s for s, k in zip(keep_names, ok) if k]
        center, scale = fit_robust_scale(Xtr[:, ok])
        Z, missing_frac = prep_windows(X, keep_idx, None, center, scale, a.tail_c, n_w)
        usable = missing_frac <= a.max_missing_frac
        is_train &= usable; is_val &= usable
        evaluated = test_mask & fully_running & ~transient & usable
        tr, va = Z[is_train].reshape(-1, len(kept)), Z[is_val].reshape(-1, len(kept))
        te_idx = np.flatnonzero(evaluated); te = Z[te_idx].reshape(-1, len(kept))
        rng = np.random.default_rng(42)
        latent = pick_latent_dim(tr[rng.choice(len(tr), min(len(tr), 200_000), replace=False)], 0.95, 4)
        log.info("%s fold %s: sensors %d, train %d val %d test %d windows (test events %d), latent %d",
                 eq, b, len(kept), is_train.sum(), is_val.sum(), len(te_idx), int(event[te_idx].sum()), latent)
        if len(te_idx) < 50:
            log.warning("%s fold %s skipped: only %d evaluable test windows", eq, b, len(te_idx)); continue
        if a.trim_pct > 0:  # robust refinement: drop the worst trim_pct% of train windows by PCA-SPE
            p0 = fit_pca(tr[rng.choice(len(tr), min(len(tr), 200_000), replace=False)], latent)
            spe = ((Z[is_train].reshape(-1, len(kept)) - reconstruct_pca(p0, Z[is_train].reshape(-1, len(kept)))) ** 2).mean(1).reshape(-1, W).mean(1)
            keep_w = spe <= np.percentile(spe, 100 - a.trim_pct)
            tidx = np.flatnonzero(is_train); is_train[tidx[~keep_w]] = False
            tr = Z[is_train].reshape(-1, len(kept))
        models = {}
        if a.with_level:  # no-model baseline: residual = scaled value itself (per-sensor level monitoring / limit check)
            models["level"] = lambda rows: np.zeros_like(rows)
        pca = fit_pca(tr[rng.choice(len(tr), min(len(tr), 300_000), replace=False)], latent)
        models["pca"] = lambda rows, p=pca: reconstruct_pca(p, rows)
        ae_variants = {}
        if not a.skip_ae:
            # "ae": wide hidden layers, low dropout, patient early stopping -- must reconstruct at least as well as PCA
            n = len(kept)
            ae_variants["ae"] = dict(dropout=a.ae_dropout, patience=a.ae_patience, widths=[max(4 * latent, n), max(2 * latent, n // 2)], batch=a.ae_batch)
            if a.with_repo_ae:  # the repo's recipe: halving funnel to the bottleneck, dropout 0.2, patience 15, batch 256
                ae_variants["ae_repo"] = dict(dropout=0.2, patience=15, widths=None, batch=256)
        pca_val_mse = float(((va - reconstruct_pca(pca, va)) ** 2).mean())
        ae_info = {"pca_val_mse": pca_val_mse}
        for vname, v in ae_variants.items():
            cfg = TrainConfig(n_sensors=len(kept), latent_dim=latent, dropout=v["dropout"], lr=1e-3, weight_decay=3e-4,
                              max_epochs=a.max_epochs, patience=v["patience"], lr_patience=10, batch_size=v["batch"], seed=42,
                              hidden_widths=v["widths"])
            model, hist = train_model(Z[is_train], Z[is_val], cfg)
            ae_info[vname] = {k: hist[k] for k in hist if k.startswith("best")} | {"epochs": len(hist["train_loss"]), "widths": model.hidden_widths,
                                                                                  "dropout": v["dropout"], "val_mse_over_pca": hist["best_val_loss"] / pca_val_mse}
            log.info("%s fold %s %s: epochs %d best %d val %.4f (PCA val %.4f, ratio %.2f) widths %s", eq, b, vname, len(hist["train_loss"]),
                     hist["best_epoch"], hist["best_val_loss"], pca_val_mse, hist["best_val_loss"] / pca_val_mse, model.hidden_widths)
            models[vname] = lambda rows, m=model: reconstruct_ae(m, rows)
        json.dump(ae_info | {"latent": latent, "n_sensors": len(kept)}, open(fdir / "ae_history.json", "w"), indent=1, default=float)
        rec = {"window_id": te_idx, "block": fi, "fold": b, "start": start[te_idx], "event": event[te_idx], "event_tol": None,
               "missing_frac": missing_frac[te_idx]}
        ev_tol = near_event.copy() if a.event_margin >= a.tol else event.copy()
        # tolerance mask for alarm TP: event +/- tol windows
        et = event.copy()
        for k in range(1, a.tol + 1):
            et[k:] |= event[:-k]; et[:-k] |= event[k:]
        rec["event_tol"] = et[te_idx]
        fsum = {"fold": b, "fold_index": fi, "n_sensors": len(kept), "latent": latent, "n_train": int(is_train.sum()), "n_val": int(is_val.sum()),
                "n_test": int(len(te_idx)), "n_test_events": int(event[te_idx].sum()), "ae_info": ae_info, "models": {}}
        for mname, recon in models.items():
            Rv = va - recon(va); Rt = te - recon(te)
            stats = ResidualStats(Rv)
            wv = window_scores(stats.row_scores(Rv), int(is_val.sum()))
            wt = window_scores(stats.row_scores(Rt), len(te_idx))
            rec[f"{mname}_top_sensor"] = np.array(kept)[wt["top_sensor_idx"]]
            msum = {}
            for b_ in SCORE_BASES:
                for agg in AGGS:
                    for sm in ("", "_ewma"):
                        key = f"{b_}_{agg}{sm}"
                        v = wv[f"{b_}_{agg}"]; t = wt[f"{b_}_{agg}"]
                        if sm:
                            v = ewma(v, a.ewma_lambda); t = ewma(t, a.ewma_lambda)
                        rec[f"{mname}_{key}"] = t
                        thr = {"q99": np.percentile(v, 99), "q995": np.percentile(v, 99.5), "q999": np.percentile(v, 99.9)}
                        try:
                            thr["gpd995"] = _gpd_tail_threshold(v, u=float(np.percentile(v, 95)), percentile=99.5)
                        except Exception:  # noqa: BLE001
                            thr["gpd995"] = float("nan")
                        rec[f"{mname}_{key}_thr_q995"] = np.full(len(te_idx), thr["q995"])
                        y = event[te_idx]
                        m = {"val_thresholds": {k: float(x) for k, x in thr.items()},
                             "pr_auc": float(average_precision_score(y, t)) if y.any() else float("nan"),
                             "roc_auc": float(roc_auc_score(y, t)) if 0 < y.sum() < len(y) else float("nan"),
                             "prevalence": float(y.mean())}
                        for tk, tv in thr.items():
                            if np.isfinite(tv):
                                m[tk] = event_metrics(te_idx, t, tv, et[te_idx], nodes, a.tol, 1)
                                m[tk + "_persist2"] = event_metrics(te_idx, t, tv, et[te_idx], nodes, a.tol, 2)
                        msum[key] = m
            fsum["models"][mname] = msum
        pd.DataFrame(rec).to_parquet(fdir / "window_scores.parquet", index=False)
        json.dump(fsum, open(fdir / "metrics.json", "w"), indent=1, default=float)
        fold_summaries.append(fsum)
        all_rows.append(pd.DataFrame(rec))
        log.info("%s fold %s done (%.0f s elapsed)", eq, b, time.time() - t0)

    pooled = pd.concat(all_rows).sort_values("window_id").reset_index(drop=True)
    pooled.to_parquet(out_root / "pooled_window_scores.parquet", index=False)
    # pooled metrics: alarms decided per fold (own val threshold), AUC on score / fold q995 threshold
    summary = {"equipment": eq, "sensors": a.sensors, "scheme": a.scheme, "n_folds": len(fold_summaries), "folds": fold_summaries, "pooled": {}}
    y = pooled["event"].to_numpy(); et = pooled["event_tol"].to_numpy(); wid = pooled["window_id"].to_numpy()
    for mname in [c[: -len("_mse_p95")] for c in pooled.columns if c.endswith("_mse_p95")]:
        pm = {}
        for b_ in SCORE_BASES:
            for agg in AGGS:
                for sm in ("", "_ewma"):
                    key = f"{b_}_{agg}{sm}"
                    t = pooled[f"{mname}_{key}"].to_numpy()
                    ratio = t / pooled[f"{mname}_{key}_thr_q995"].to_numpy()
                    m = {"pr_auc_ratio": float(average_precision_score(y, ratio)), "roc_auc_ratio": float(roc_auc_score(y, ratio)),
                         "prevalence": float(y.mean()), "n_windows": int(len(y))}
                    for tk in ("q99", "q995", "q999", "gpd995"):
                        # per-fold thresholds -> pooled alarm decisions: express as ratio to each fold's threshold
                        thr_map = {fs["fold_index"]: fs["models"][mname][key]["val_thresholds"][tk] for fs in fold_summaries}
                        fold_thr = np.array([thr_map[bi] for bi in pooled["block"].to_numpy()])
                        if not np.all(np.isfinite(fold_thr)):
                            continue
                        m[tk] = event_metrics(wid, t / fold_thr, 1.0, et, nodes, a.tol, 1)
                        m[tk + "_persist2"] = event_metrics(wid, t / fold_thr, 1.0, et, nodes, a.tol, 2)
                    pm[key] = m
        summary["pooled"][mname] = pm
    json.dump(summary, open(out_root / "summary.json", "w"), indent=1, default=float)
    log.info("%s done in %.0f s -> %s", eq, time.time() - t0, out_root)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equipment", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data/data_for_testing")
    ap.add_argument("--output", default="output_v2")
    ap.add_argument("--sensors", choices=["all", "event", "process", "process2"], default="all")
    ap.add_argument("--scheme", choices=["blocked", "rolling"], default="blocked")
    ap.add_argument("--n-folds", type=int, default=4, help="blocked scheme: number of contiguous blocks")
    ap.add_argument("--min-hist-months", type=int, default=6, help="rolling scheme: first test month index")
    ap.add_argument("--max-train-months", type=int, default=8, help="rolling scheme: cap on training months before the val month")
    ap.add_argument("--min-train-windows", type=int, default=2000)
    ap.add_argument("--gap", type=int, default=48, help="windows excluded from training either side of the test block (48 = 1 day)")
    ap.add_argument("--event-margin", type=int, default=4, help="windows around any event excluded from train/val (4 = 2 h)")
    ap.add_argument("--post-start-margin", type=int, default=2, help="fully-running windows after a stop treated as transient (2 = 1 h)")
    ap.add_argument("--tol", type=int, default=1, help="tolerance (windows) around an event for alarm TP")
    ap.add_argument("--max-null-frac", type=float, default=0.2)
    ap.add_argument("--max-missing-frac", type=float, default=0.05)
    ap.add_argument("--tail-c", type=float, default=20.0)
    ap.add_argument("--ewma-lambda", type=float, default=0.5)
    ap.add_argument("--trim-pct", type=float, default=0.0)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--ae-dropout", type=float, default=0.0)
    ap.add_argument("--ae-patience", type=int, default=30)
    ap.add_argument("--ae-batch", type=int, default=1024)
    ap.add_argument("--with-repo-ae", action="store_true", help="also train the repo's recipe (halving funnel, dropout 0.2, patience 15) as model 'ae_repo'")
    ap.add_argument("--skip-ae", action="store_true")
    ap.add_argument("--with-level", action="store_true", help="also score a no-model level baseline (x_hat = scaled median)")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for eq in a.equipment:
        run_equipment(eq, a)


if __name__ == "__main__":
    main()
