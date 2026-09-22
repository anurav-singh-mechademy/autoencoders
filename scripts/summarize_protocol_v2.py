#!/usr/bin/env python
"""Collect output_v2/<eq>/<sensors>/summary.json into markdown tables (pooled blocked-CV metrics)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protocol_v2 import event_metrics  # noqa: E402


def event_tol_mask(wid: np.ndarray, event: np.ndarray, tol: int) -> np.ndarray:
    """event +/- tol windows, computed on window ids (handles gaps in the evaluated set)."""
    ev_ids = set(int(w) for w in wid[event])
    return np.array([any((int(w) + d) in ev_ids for d in range(-tol, tol + 1)) for w in wid])


def recompute_rows(sdir: Path, s: dict, scores: list[str], thrs: list[str], tol: int, choice: dict | None = None, model: str | None = None) -> dict:
    """Recompute pooled metrics from pooled_window_scores.parquet for one model (or per-fold `choice` of model) at
    tolerance `tol` windows around events (alarm TP / node hit)."""
    pooled = pd.read_parquet(sdir / "pooled_window_scores.parquet")
    nodes = pd.read_json(sdir / "nodes.json")
    folds = {f["fold_index"]: f for f in s["folds"]}
    y = pooled["event"].to_numpy(); wid = pooled["window_id"].to_numpy(); blk = pooled["block"].to_numpy()
    et = event_tol_mask(wid, y, tol)
    pick = (lambda b: choice[b]) if choice else (lambda b: model)
    out = {}
    for sc in scores:
        raw = np.array([pooled[f"{pick(b)}_{sc}"].iloc[i] for i, b in enumerate(blk)]) if choice else pooled[f"{model}_{sc}"].to_numpy()
        m = {"prevalence": float(y.mean()), "n_windows": int(len(y))}
        for tk in thrs:
            thr = np.array([folds[b]["models"][pick(b)][sc]["val_thresholds"][tk] for b in blk])
            if not np.all(np.isfinite(thr)): continue
            ratio = raw / thr
            if tk == thrs[0]:
                m["pr_auc_ratio"] = float(average_precision_score(y, ratio)); m["roc_auc_ratio"] = float(roc_auc_score(y, ratio))
            m[tk] = event_metrics(wid, ratio, 1.0, et, nodes, tol, 1); m[tk + "_persist2"] = event_metrics(wid, ratio, 1.0, et, nodes, tol, 2)
        out[sc] = m
    return out


def selected_model_rows(sdir: Path, s: dict, scores: list[str], thrs: list[str]) -> dict:
    """Pseudo-model 'sel': per fold use the AE if its validation MSE beat PCA's, else PCA (model selection on
    validation reconstruction error only -- no labels involved). Returns pooled metrics like summary['pooled'][m]."""
    pooled = pd.read_parquet(sdir / "pooled_window_scores.parquet")
    nodes = pd.read_json(sdir / "nodes.json")
    folds = {f["fold_index"]: f for f in s["folds"]}
    choice = {fi: ("ae" if f.get("ae_info", {}).get("ae", {}).get("val_mse_over_pca", 9) < 1.0 else "pca") for fi, f in folds.items()}
    y = pooled["event"].to_numpy(); et = pooled["event_tol"].to_numpy(); wid = pooled["window_id"].to_numpy(); blk = pooled["block"].to_numpy()
    out = {"_choice": {folds[fi]["fold"]: c for fi, c in choice.items()}}
    for sc in scores:
        raw = np.where([choice[b] == "ae" for b in blk], pooled.get(f"ae_{sc}", np.nan), pooled[f"pca_{sc}"])
        m = {"prevalence": float(y.mean()), "n_windows": int(len(y))}
        for tk in thrs:
            thr = np.array([folds[b]["models"][choice[b]][sc]["val_thresholds"][tk] for b in blk])
            if not np.all(np.isfinite(thr)): continue
            ratio = raw / thr
            if tk == thrs[0]:
                m["pr_auc_ratio"] = float(average_precision_score(y, ratio)); m["roc_auc_ratio"] = float(roc_auc_score(y, ratio))
            m[tk] = event_metrics(wid, ratio, 1.0, et, nodes, 1, 1); m[tk + "_persist2"] = event_metrics(wid, ratio, 1.0, et, nodes, 1, 2)
        out[sc] = m
    return out

def fmt(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="output_v2"); ap.add_argument("--out", default=None)
    ap.add_argument("--thr", default="q995"); ap.add_argument("--thrs", default="q995,gpd995", help="thresholds to tabulate"); ap.add_argument("--select", action="store_true", help="add pseudo-model sel (AE if val MSE < PCA else PCA, per fold)"); ap.add_argument("--tol", type=int, default=None, help="recompute event-level metrics with this tolerance (windows)"); ap.add_argument("--per-fold", default=None, help="model:score to print per-fold alarm rate / recall / FA/day")
    ap.add_argument("--scores", default="mse_p95,spez_med,maxz_med,maxz_med_ewma,md_med,md_med_ewma")
    a = ap.parse_args()
    lines = []
    for sj in sorted(Path(a.root).glob("*/*/summary.json")):
        s = json.load(open(sj)); eq, sens = s["equipment"], s["sensors"]
        if a.select and "ae" in s["pooled"]:
            sel = selected_model_rows(sj.parent, s, a.scores.split(","), a.thrs.split(","))
            s["pooled"]["sel"] = {k: v for k, v in sel.items() if not k.startswith("_")}
            lines.append(f"\nsel = per-fold choice by val MSE: {sel['_choice']}")
        if a.tol is not None and a.tol != 1:
            choice = {f["fold_index"]: ("ae" if f.get("ae_info", {}).get("ae", {}).get("val_mse_over_pca", 9) < 1.0 else "pca") for f in s["folds"]}
            for mname in list(s["pooled"]):
                s["pooled"][mname] = recompute_rows(sj.parent, s, a.scores.split(","), a.thrs.split(","), a.tol, choice=choice if mname == "sel" else None, model=None if mname == "sel" else mname)
            lines.append(f"(metrics recomputed with tolerance ±{a.tol} windows = ±{a.tol * 0.5:g} h around events)")
        f0 = s["folds"]
        lines.append(f"\n### {eq} — sensors={sens} (folds: " + "; ".join(f"f{f['fold']}: {f['n_sensors']}S/lat{f['latent']} tr{f['n_train']} va{f['n_val']} te{f['n_test']} ev{f['n_test_events']}" for f in f0) + ")")
        lines.append(f"| model | score | PR-AUC | ROC-AUC | thr | event recall | window prec | episode prec | FA/day | comp-F1 | alarm rate | +persist2: recall / FA/day |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in s["pooled"]:
            for sc in a.scores.split(","):
                r = s["pooled"][m].get(sc)
                if not r: continue
                for thr in a.thrs.split(","):
                    t = r.get(thr); p2 = r.get(thr + "_persist2")
                    if not t: continue
                    lines.append(f"| {m} | {sc} | {fmt(r['pr_auc_ratio'],3)} | {fmt(r['roc_auc_ratio'],3)} | {thr} | {fmt(t['event_recall'])} ({t['nodes_detected']}/{t['n_nodes_evaluable']}) | {fmt(t['window_precision'])} | {fmt(t['episode_precision'])} | {fmt(t['false_alarms_per_day'])} | {fmt(t['composite_f1'])} | {fmt(t['alarm_rate'],3)} | {fmt(p2['event_recall'])} / {fmt(p2['false_alarms_per_day'])} |")
        if a.per_fold:
            m, sc = a.per_fold.split(":")
            rows = []
            for f in f0:
                r = f["models"].get(m, {}).get(sc)
                if not r: continue
                t = r.get(a.thr, {})
                rows.append(f"{f['fold']}: ev{f['n_test_events']}/{f['n_test']} alarm {t.get('alarm_rate', float('nan')):.3f} rec {fmt(t.get('event_recall'))} ({t.get('nodes_detected','-')}/{t.get('n_nodes_evaluable','-')}) winP {fmt(t.get('window_precision'))} FA/d {fmt(t.get('false_alarms_per_day'))} PR {fmt(r.get('pr_auc'),3)}")
            lines.append(f"\nPer fold ({m} {sc} @{a.thr}): " + " | ".join(rows))
        # per-slug recall for the AE best score by composite F1 at chosen thr
        best = None
        for m in s["pooled"]:
            for sc, r in s["pooled"][m].items():
                t = r.get(a.thr)
                if t and (best is None or t["composite_f1"] > best[2]["composite_f1"]): best = (m, sc, t)
        if best:
            lines.append(f"\nBest composite-F1 @{a.thr}: **{best[0]} {best[1]}** → recall by slug: " + ", ".join(f"{k} {v}" for k, v in best[2]["recall_by_slug"].items()))
    txt = "\n".join(lines)
    print(txt)
    if a.out: Path(a.out).write_text(txt)

if __name__ == "__main__":
    main()
