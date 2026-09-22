#!/usr/bin/env python
"""Alarm anatomy for a protocol_v2 run: where do alarms fall relative to labelled nodes, which sensors carry them,
and how early/late are detections. Usage: alarm_anatomy_v2.py <run_dir e.g. output_v2_roll/5P921A/process> <model> <score> [thr=q995]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

run, model, score = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
thr_key = sys.argv[4] if len(sys.argv) > 4 else "q995"
p = pd.read_parquet(run / "pooled_window_scores.parquet"); s = json.load(open(run / "summary.json")); nodes = pd.read_json(run / "nodes.json")
folds = {f["fold_index"]: f for f in s["folds"]}
thr = p["block"].map({fi: f["models"][model][score]["val_thresholds"][thr_key] for fi, f in folds.items()})
p["alarm"] = (p[f"{model}_{score}"] / thr) > 1
p["top"] = p[f"{model}_top_sensor"]
wid = p["window_id"].to_numpy(); pos = {int(w): i for i, w in enumerate(wid)}
ev_ids = set(int(w) for w in wid[p["event"].to_numpy()])
# distance (windows) from each window to the nearest event window
ev_sorted = np.array(sorted(ev_ids))
def dist(w):
    if not len(ev_sorted): return 10**9
    i = np.searchsorted(ev_sorted, w); c = []
    if i < len(ev_sorted): c.append(abs(ev_sorted[i] - w))
    if i > 0: c.append(abs(ev_sorted[i - 1] - w))
    return min(c)
p["dist_ev"] = [dist(int(w)) for w in wid]
# sensors implicated by the nearest node(s) within 12 windows
node_sensors_by_window = {}
for _, nd in nodes.iterrows():
    for w in nd["windows"]:
        node_sensors_by_window.setdefault(int(w), set()).update(nd["sensors"])
def near_sensors(w, k=12):
    out = set()
    for d in range(-k, k + 1):
        out |= node_sensors_by_window.get(int(w) + d, set())
    return out
al = p[p.alarm]
fa1 = al[al.dist_ev > 1]
print(f"{run} {model} {score} @{thr_key}: alarm windows {len(al)} / {len(p)}; TP(±1) {int((al.dist_ev <= 1).sum())}; FA(±1) {len(fa1)}")
bins = [-1, 1, 6, 12, 48, 10**9]; labels = ["≤1 (TP)", "2–6 (1–3 h)", "7–12 (3–6 h)", "13–48 (6–24 h)", ">48 (>1 day)"]
d = pd.cut(al.dist_ev, bins=bins, labels=labels, right=True)
print("alarm windows by distance to nearest event window:", d.value_counts().reindex(labels).to_dict())
on_ev_sensor = [t in near_sensors(w) for t, w in zip(fa1.top, fa1.window_id)]
print(f"FA(±1) windows whose top sensor belongs to a node within ±6 h: {int(np.sum(on_ev_sensor))} / {len(fa1)}")
print("FA(±1) top sensors:", fa1.top.value_counts().head(10).to_dict())
far = fa1[fa1.dist_ev > 48]
print(f"FA(±1) farther than 1 day from any event: {len(far)}; top sensors: {far.top.value_counts().head(8).to_dict()}; by month: {far.groupby(pd.to_datetime(far.start).dt.to_period('M')).size().to_dict()}")
# detection latency per node: first alarm window within [start-1, end+1] relative to node start (windows)
rows = []
for _, nd in nodes.iterrows():
    ws = [w for w in nd["windows"] if w in pos]
    if not ws: continue
    lo, hi = min(nd["windows"]) - 1, max(nd["windows"]) + 1
    cand = p[(p.window_id >= lo) & (p.window_id <= hi) & p.alarm]
    first = int(cand.window_id.min()) if len(cand) else None
    # early warning: alarms in the 12 windows before the node start (not counted as TP at tol 1)
    pre = p[(p.window_id >= min(nd["windows"]) - 12) & (p.window_id < min(nd["windows"]) - 1) & p.alarm]
    rows.append(dict(slug=nd["slug"], node=nd["node_id"], n_win=len(nd["windows"]), detected=first is not None,
                     latency_h=(first - min(nd["windows"])) * 0.5 if first is not None else None, pre_alarm_h=len(pre) * 0.5,
                     top=cand.top.value_counts().index[0] if len(cand) else None, labelled="|".join(nd["sensors"])[:40]))
r = pd.DataFrame(rows)
print(f"\nnodes evaluable {len(r)}, detected {int(r.detected.sum())}; latency (h from node start, detected only): median {r.latency_h.median():.1f}, ≤0 h (at/before start) {int((r.latency_h <= 0).sum())}, >2 h {int((r.latency_h > 2).sum())}; nodes with alarms in the 6 h before start: {int((r.pre_alarm_h > 0).sum())}")
print(r.groupby("slug").agg(n=("node", "count"), det=("detected", "sum"), lat_med_h=("latency_h", "median"), pre_alarm=("pre_alarm_h", lambda x: int((x > 0).sum()))).round(1).to_string())
print("\ntop sensor of detections vs labelled sensors (detected nodes):")
print(r[r.detected][["slug", "top", "labelled"]].groupby(["slug", "top"]).size().sort_values(ascending=False).head(12).to_string())
