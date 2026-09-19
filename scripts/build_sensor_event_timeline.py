#!/usr/bin/env python
"""CLI: Build a single, self-contained HTML chart of raw sensor traces,
model zone verdicts, and labeled fault events for one run's test split.
All data is embedded inline (no fetch(), works from a plain file:// open).

The x-axis is an ordinal window index (not real time, since test windows
are rarely contiguous), each sensor gets its own independently-autoscaled
y-axis, and a small triangle above the zone strip marks each window's
reconstruction-error modality (up/blue = positive, down/amber = negative;
see analyze_reconstruction_error_sign.py).

Usage:
    python scripts/build_sensor_event_timeline.py \\
        --equipment-id 5P921A \\
        --data data/5P921A/5P921A_combined_with_events.parquet \\
        --event-labels data/5P921A/5P921A_event_labels_long.parquet \\
        --model-dir output_5P921A/artefacts \\
        --window-scores output_5P921A/evaluation/window_scores.parquet \\
        --output viz/5P921A_base_timeline.html
"""

from __future__ import annotations

import argparse
import json
import logging
import math

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from autoencoder.logging_config import setup_logging
from autoencoder.data.ingestion import read_parquet_robust
from autoencoder.evaluation.ground_truth import load_window_ground_truth
from autoencoder.artefacts.serialisation import load_artefacts

setup_logging()
logger = logging.getLogger(__name__)

# A categorical palette deliberately avoiding saturated green/yellow/red
# (reserved for zone/severity semantics) -- cycled if there are more
# sensors than colors.
SENSOR_PALETTE = [
    "#3B6FA0", "#8452A8", "#C2542E", "#5C6BC0", "#A34D74", "#3E7C87",
    "#8A6D3B", "#6B4FA0", "#4A8FB0", "#9C5C3E", "#5A6B7C", "#B0527A",
    "#7C8C3E", "#4F6B9C", "#8C4F6B",
]


def json_safe(v):
    """None for NaN/None (Plotly gap marker), else a plain rounded float --
    Python's json.dump would otherwise emit a bare `NaN` token for a real
    NaN, which is not valid JSON and fails in the browser's JSON.parse."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--equipment-id", required=True)
    parser.add_argument("--data", required=True, help="*_combined_with_events.parquet")
    parser.add_argument("--event-labels", required=True, help="*_event_labels_long.parquet")
    parser.add_argument("--model-dir", required=True, help="Trained model's artefacts dir (for sensor_columns)")
    parser.add_argument("--window-scores", required=True, help="Output of scripts/score_full_timeline.py (test split)")
    parser.add_argument("--output", required=True, help="Output .html path")
    parser.add_argument("--window-rows", type=int, default=120)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument(
        "--default-visible", default=None,
        help="Comma-separated sensor names visible by default (others start as legendonly). "
             "Default: first 3 sensors in the model's sensor_columns.",
    )
    parser.add_argument("--title", default=None, help="Page title (default: '<equipment-id> Anomaly Timeline')")
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="How many top-implicated sensors the modality (reconstruction-error sign) tick averages "
             "over -- matches the sensors shown in the zone-strip hover's 'top:' list.",
    )
    parser.add_argument(
        "--test-windows", default=None,
        help="Path to test_windows.npy (scaled, model's own input) for the modality tick. "
             "Default: <model-dir>/../test_windows.npy",
    )
    parser.add_argument(
        "--test-window-ids", default=None,
        help="Path to test_window_ids.npy. Default: <model-dir>/../test_window_ids.npy",
    )
    args = parser.parse_args()

    with open(f"{args.model_dir}/training_metadata.json") as f:
        metadata = json.load(f)
    sensor_columns = metadata["sensor_columns"]
    sensor_index = {name: i for i, name in enumerate(sensor_columns)}
    logger.info("Model sensors (%d): %s", len(sensor_columns), sensor_columns)

    # Forward pass for the row-tied modality tick (see analyze_reconstruction_error_sign.py's compute_row_tied_modality).
    model_dir_path = Path(args.model_dir)
    test_windows_path = args.test_windows or str(model_dir_path.parent / "test_windows.npy")
    test_window_ids_path = args.test_window_ids or str(model_dir_path.parent / "test_window_ids.npy")
    model, _scaler, _thresholds, _meta, _training_errors, _sensor_baselines = load_artefacts(args.model_dir)
    test_windows = np.load(test_windows_path)
    test_window_ids = np.load(test_window_ids_path)
    id_to_row = {int(wid): idx for idx, wid in enumerate(test_window_ids)}
    x = torch.tensor(test_windows, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        x_hat = model(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    residual = (x - x_hat).numpy()  # signed, (n_test_windows, window_rows, n_sensors)
    row_sq_error = (residual ** 2).mean(axis=2)  # (n_test_windows, window_rows) -- mirrors per_row_mse (all sensors)
    logger.info("Computed reconstruction-error modality for %d test windows", len(test_window_ids))

    default_visible = (
        set(s.strip() for s in args.default_visible.split(","))
        if args.default_visible else set(sensor_columns[:3])
    )

    ws = pd.read_parquet(args.window_scores) if args.window_scores.endswith(".parquet") else pd.read_csv(args.window_scores)
    ws["window_start"] = pd.to_datetime(ws["window_start"])
    ws = ws.sort_values("window_start").reset_index(drop=True)
    logger.info("Loaded %d test windows, span %s -> %s", len(ws), ws["window_start"].min(), ws["window_start"].max())

    gt = load_window_ground_truth(
        args.data, args.event_labels, window_size=args.window_rows, timestamp_column=args.timestamp_column,
    )
    gt = gt.filter_to_window_ids(ws["window_id"].to_numpy())

    window_meta = []
    n_modality_missing = 0
    for i, row in ws.iterrows():
        sev = gt.max_event_level[i]
        # Top-k sensors by reconstruction error, from the full ranked_sensors list (not the smaller top_sensors column).
        top_sensors = [str(s) for s in row["ranked_sensors"][:args.top_k]] if len(row["ranked_sensors"]) else []

        # Row-tied modality sign (see analyze_reconstruction_error_sign.py's signed_error_row_tied).
        modality = None
        wid = int(row["window_id"])
        if top_sensors and wid in id_to_row:
            idxs = [sensor_index[s] for s in top_sensors if s in sensor_index]
            if idxs:
                tw_idx = id_to_row[wid]
                target = np.percentile(row_sq_error[tw_idx], 95)
                row_idx = int(np.argmin(np.abs(row_sq_error[tw_idx] - target)))
                val = float(residual[tw_idx, row_idx, idxs].mean())
                modality = "positive" if val > 0 else ("negative" if val < 0 else None)
        if modality is None:
            n_modality_missing += 1

        window_meta.append({
            "window_id": wid,
            "start": row["window_start"].isoformat(),
            "zone": row["zone"] if pd.notna(row["zone"]) else "green",
            "usable": bool(row["usable"]),
            "touched": bool(gt.is_anomaly[i]),
            "node_ids": sorted(int(n) for n in gt.all_node_ids[i]),
            "rule_slugs": sorted(gt.all_rule_slugs[i]),
            "severity": (None if isinstance(sev, float) and math.isnan(sev) else int(sev)),
            "top_sensors": top_sensors,
            "modality": modality,
        })
    logger.info("Reconstruction-error modality unavailable for %d / %d windows (unusable or unscored)", n_modality_missing, len(window_meta))
    n_touched = sum(1 for w in window_meta if w["touched"])
    logger.info("Ground truth: %d / %d test windows touched by >=1 event node", n_touched, len(window_meta))

    logger.info("Loading raw sensor data from %s ...", args.data)
    df = read_parquet_robust(args.data)

    sensor_data = {}
    for s in sensor_columns:
        values = []
        for wid in ws["window_id"]:
            start = int(wid) * args.window_rows
            chunk = df[s].iloc[start:start + args.window_rows].to_numpy(dtype=np.float64)
            values.extend(json_safe(v) for v in chunk)
            values.append(None)  # inter-window gap marker
        sensor_data[s] = values
    logger.info("Extracted raw values for %d sensors x %d windows", len(sensor_columns), len(ws))

    sensor_colors = {s: SENSOR_PALETTE[i % len(SENSOR_PALETTE)] for i, s in enumerate(sensor_columns)}
    title = args.title or f"{args.equipment_id} Anomaly Timeline"

    html = _render_html(
        title=title,
        equipment_id=args.equipment_id,
        window_meta=window_meta,
        sensor_data=sensor_data,
        sensor_colors=sensor_colors,
        default_visible=default_visible,
        window_rows=args.window_rows,
    )

    with open(args.output, "w") as f:
        f.write(html)
    logger.info("Wrote %s (%.1f MB)", args.output, len(html) / 1e6)


def _render_html(title, equipment_id, window_meta, sensor_data, sensor_colors, default_visible, window_rows) -> str:
    # All data embedded inline as JSON literals -- no fetch(), works from a
    # plain file:// double-click with no server and no CORS restrictions.
    window_meta_json = json.dumps(window_meta, separators=(",", ":"))
    sensor_data_json = json.dumps(sensor_data, separators=(",", ":"))
    sensor_colors_json = json.dumps(sensor_colors)
    default_visible_json = json.dumps(sorted(default_visible))

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2/plotly.min.js"></script>
<style>
  :root{{
    color-scheme: light dark;
    --bg:#eef1f4; --surface:#ffffff; --surface-2:#f5f7f9; --border:#d6dce2;
    --text:#12181f; --text-muted:#5c6773;
    --zone-green:#2f8f52; --zone-yellow:#b8860b; --zone-red:#c23b32;
    --shadow: 0 1px 2px rgba(20,30,40,.06), 0 4px 14px rgba(20,30,40,.05);
  }}
  @media (prefers-color-scheme: dark){{
    :root{{
      --bg:#0a0e13; --surface:#11161d; --surface-2:#161d25; --border:#232b34;
      --text:#e7edf3; --text-muted:#8b98a5;
      --zone-green:#4ecb82; --zone-yellow:#e3b33e; --zone-red:#f2554a;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35);
    }}
  }}
  *{{box-sizing:border-box;}}
  body{{
    background:var(--bg); color:var(--text); margin:0;
    font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,sans-serif;
    padding:20px clamp(16px,3vw,40px) 40px;
    display:flex; flex-direction:column; gap:16px;
  }}
  h1{{font-size:1.05rem; font-weight:600; letter-spacing:.01em; margin:0; text-wrap:balance;}}
  .mono{{font-family:"IBM Plex Mono",ui-monospace,monospace;}}
  .muted{{color:var(--text-muted);}}
  header.panel{{
    background:var(--surface); border:1px solid var(--border); border-radius:6px;
    box-shadow:var(--shadow); padding:16px 20px;
    display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between; gap:12px 24px;
  }}
  header .id-line{{display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;}}
  .tag-chip{{
    font-family:"IBM Plex Mono",monospace; font-size:.72rem; font-weight:500;
    background:var(--surface-2); border:1px solid var(--border); border-radius:4px;
    padding:2px 7px; color:var(--text-muted); letter-spacing:.02em;
  }}
  .stat-row{{display:flex; gap:10px; flex-wrap:wrap; align-items:center;}}
  .stat{{
    display:flex; align-items:center; gap:6px;
    font-family:"IBM Plex Mono",monospace; font-size:.78rem; font-variant-numeric:tabular-nums;
    padding:4px 10px; border-radius:4px; border:1px solid var(--border); background:var(--surface-2);
  }}
  .dot{{width:8px; height:8px; border-radius:50%; flex:none;}}
  .dot.green{{background:var(--zone-green);}} .dot.yellow{{background:var(--zone-yellow);}} .dot.red{{background:var(--zone-red);}}
  section.panel{{
    background:var(--surface); border:1px solid var(--border); border-radius:6px;
    box-shadow:var(--shadow); padding:14px 16px 8px;
  }}
  .panel-label{{
    font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; color:var(--text-muted);
    font-weight:500; margin:0 0 10px;
  }}
  #chart{{width:100%; height:640px;}}
  .legend-note{{
    display:flex; flex-wrap:wrap; gap:16px; font-size:.74rem; color:var(--text-muted);
    padding:10px 4px 4px; border-top:1px solid var(--border); margin-top:6px;
  }}
  .legend-note b{{color:var(--text); font-weight:500;}}
  .callout{{
    font-size:.76rem; color:var(--text-muted); background:var(--surface-2);
    border:1px solid var(--border); border-radius:4px; padding:8px 12px;
  }}
  .callout b{{color:var(--text); font-weight:500;}}
  footer{{font-size:.72rem; color:var(--text-muted); padding:2px 4px; display:flex; flex-wrap:wrap; gap:6px 18px;}}
  #loading{{padding:60px 20px; text-align:center; color:var(--text-muted); font-size:.85rem;}}
</style>
</head>
<body>

<h1>{title}</h1>

<header class="panel">
  <div class="id-line">
    <span class="tag-chip">{equipment_id}</span>
    <span class="muted" style="font-size:.85rem;">test split</span>
    <span class="mono muted" style="font-size:.78rem;" id="span-label">loading&hellip;</span>
  </div>
  <div class="stat-row" id="zone-stats"></div>
</header>

<p class="callout">
  <b>Axis is compressed to available data only</b> &mdash; test windows are not contiguous in real
  time; each gets one equal-width slot here, back to back, in chronological order. Hover any point
  for its real timestamp; a thin break still marks every window boundary so two windows that
  aren't actually adjacent in time are never drawn as one continuous line.
</p>

<section class="panel">
  <p class="panel-label">Sensor traces (raw, unscaled) &middot; model zone &middot; labeled event nodes &mdash; click a legend entry to toggle a sensor</p>
  <div id="chart"><div id="loading">Rendering&hellip;</div></div>
  <div class="legend-note">
    <span><b>Sensor lines</b> &mdash; click a legend entry to show/hide; double-click to isolate one. Each sensor autoscales to its own raw range.</span>
    <span><b>Model zone</b> strip &mdash; green/yellow/red per window, from the trained model's own thresholds; hover for its top 3 implicated sensors.</span>
    <span><b>Modality tick</b> &mdash; small triangle above the zone strip: &#9650; blue = positive reconstruction error (actual &gt; reconstruction), &#9660; amber = negative (actual &lt; reconstruction), averaged over that window's top implicated sensors.</span>
    <span><b>Event</b> strip &mdash; marks windows touched by &ge;1 labeled fault node, shaded by severity; hover for rule slug(s).</span>
  </div>
</section>

<footer>
  <span class="mono" id="n-windows-label"></span>
  <span class="mono" id="n-touched-label"></span>
</footer>

<script>
const WINDOW_META = {window_meta_json};
const SENSOR_DATA = {sensor_data_json};
const SENSOR_COLORS = {sensor_colors_json};
const DEFAULT_VISIBLE = new Set({default_visible_json});
const WINDOW_ROWS = {window_rows};
const SENSOR_LIST = Object.keys(SENSOR_COLORS);

const ZONE_COLOR = {{green:"#2f8f52", yellow:"#b8860b", red:"#c23b32"}};
const SEVERITY_COLOR = {{1:"#e8c98f", 2:"#d99a4e", 3:"#b8562f"}};

function isDarkTheme(){{
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}}

// Force UTC parsing of naive timestamps (bare `new Date(iso)` parses as local time).
function parseUTC(iso){{ return new Date(iso + "Z"); }}

function fmtDateTime(ms){{
  const d = new Date(ms);
  const p = n => String(n).padStart(2, "0");
  return `${{d.getUTCFullYear()}}-${{p(d.getUTCMonth() + 1)}}-${{p(d.getUTCDate())}} ${{p(d.getUTCHours())}}:${{p(d.getUTCMinutes())}}`;
}}

function main(){{
  const n = WINDOW_META.length;
  const xShared = new Array(n * (WINDOW_ROWS + 1));
  // Pre-formatted date strings (Plotly's customdata|<d3-format> hovertemplate syntax only works on date-typed axes).
  const hoverText = new Array(n * (WINDOW_ROWS + 1));
  let cursor = 0;
  for (let k = 0; k < n; k++){{
    const t0 = parseUTC(WINDOW_META[k].start).getTime();
    for (let i = 0; i < WINDOW_ROWS; i++){{
      xShared[cursor] = k + i / WINDOW_ROWS;
      hoverText[cursor] = fmtDateTime(t0 + i * 15000);
      cursor++;
    }}
    xShared[cursor] = k + 1;
    hoverText[cursor] = fmtDateTime(t0 + WINDOW_ROWS * 15000);
    cursor++;
  }}

  const sensorTraces = SENSOR_LIST.map((s, i) => ({{
    x: xShared, y: SENSOR_DATA[s], customdata: hoverText,
    type: "scattergl", mode: "lines", name: s,
    line: {{color: SENSOR_COLORS[s], width: 1.3}},
    yaxis: i === 0 ? "y" : `y${{i + 3}}`, xaxis: "x",
    visible: DEFAULT_VISIBLE.has(s) ? true : "legendonly",
    hovertemplate: `${{s}}: %{{y:,.4g}} (raw)<br>%{{customdata}}<extra></extra>`,
  }}));

  const zoneTrace = {{
    x: WINDOW_META.map((w, k) => k + 0.5), y: WINDOW_META.map(() => 1), width: WINDOW_META.map(() => 1),
    type: "bar", xaxis: "x", yaxis: "y2", showlegend: false,
    marker: {{color: WINDOW_META.map(w => ZONE_COLOR[w.zone] || ZONE_COLOR.green), line: {{width: 0}}}},
    hovertemplate: WINDOW_META.map(w => {{
      const top = w.top_sensors.length ? `<br>top: ${{w.top_sensors.join(", ")}}` : "";
      return `zone: ${{w.zone}}${{top}}<br>${{w.start}}<extra></extra>`;
    }}),
  }};

  const modalityIdx = [];
  WINDOW_META.forEach((w, k) => {{ if (w.modality) modalityIdx.push(k); }});
  const MODALITY_COLOR = {{positive: "#2f6fed", negative: "#e0793c"}};
  const modalityTrace = {{
    x: modalityIdx.map(k => k + 0.5), y: modalityIdx.map(() => 1.72),
    type: "scattergl", mode: "markers", xaxis: "x", yaxis: "y2", showlegend: false,
    marker: {{
      symbol: modalityIdx.map(k => WINDOW_META[k].modality === "positive" ? "triangle-up" : "triangle-down"),
      size: 6, color: modalityIdx.map(k => MODALITY_COLOR[WINDOW_META[k].modality]),
      line: {{width: 0}},
    }},
    hovertemplate: modalityIdx.map(k => `modality: ${{WINDOW_META[k].modality}} (actual ${{WINDOW_META[k].modality === "positive" ? ">" : "<"}} reconstruction)<extra></extra>`),
  }};

  const touchedIdx = [];
  WINDOW_META.forEach((w, k) => {{ if (w.touched) touchedIdx.push(k); }});
  const nodeTrace = {{
    x: touchedIdx.map(k => k + 0.5), y: touchedIdx.map(() => 1), width: touchedIdx.map(() => 1),
    type: "bar", xaxis: "x", yaxis: "y3", showlegend: false,
    marker: {{
      color: touchedIdx.map(k => SEVERITY_COLOR[WINDOW_META[k].severity] || SEVERITY_COLOR[1]),
      line: {{width: 0}},
    }},
    hovertemplate: touchedIdx.map(k => {{
      const w = WINDOW_META[k];
      return `${{w.rule_slugs.join(", ")}}<br>sev ${{w.severity ?? "?"}} &middot; node ${{w.node_ids.join(", ")}}<br>${{w.start}}<extra></extra>`;
    }}),
  }};

  const dark = isDarkTheme();
  const gridColor = dark ? "#232b34" : "#e2e7eb";
  const textColor = dark ? "#8b98a5" : "#5c6773";
  const paperBg = dark ? "#11161d" : "#ffffff";

  const tickvals = [], ticktext = [];
  let lastMonth = null;
  WINDOW_META.forEach((w, k) => {{
    const d = parseUTC(w.start);
    const monthKey = `${{d.getUTCFullYear()}}-${{d.getUTCMonth()}}`;
    if (monthKey !== lastMonth){{ tickvals.push(k); ticktext.push(d.toISOString().slice(0, 7)); lastMonth = monthKey; }}
  }});

  const layout = {{
    paper_bgcolor: paperBg, plot_bgcolor: paperBg,
    font: {{family: "IBM Plex Mono, monospace", color: textColor, size: 11}},
    margin: {{l: 56, r: 20, t: 10, b: 40}},
    showlegend: true,
    legend: {{orientation: "h", y: 1.06, x: 0, font: {{size: 10}}, bgcolor: "rgba(0,0,0,0)"}},
    hovermode: "closest",
    xaxis: {{
      domain: [0, 1], gridcolor: gridColor, zeroline: false,
      tickmode: "array", tickvals, ticktext,
      rangeslider: {{thickness: 0.06, bgcolor: paperBg, bordercolor: gridColor, borderwidth: 1}},
    }},
    yaxis: {{domain: [0.34, 1], gridcolor: gridColor, zeroline: false, title: {{text: "raw value (each sensor autoscaled)", font: {{size: 10}}}}}},
    yaxis2: {{domain: [0.20, 0.30], anchor: "x", range: [0, 2], showticklabels: false, title: {{text: "zone", font: {{size: 10}}}}, fixedrange: true}},
    yaxis3: {{domain: [0.02, 0.16], anchor: "x", range: [0, 2], showticklabels: false, title: {{text: "event", font: {{size: 10}}}}, fixedrange: true}},
  }};
  for (let i = 1; i < SENSOR_LIST.length; i++){{
    layout[`yaxis${{i + 3}}`] = {{domain: [0.34, 1], overlaying: "y", anchor: "x", showticklabels: false, showgrid: false, zeroline: false}};
  }}

  Plotly.newPlot("chart", [...sensorTraces, zoneTrace, modalityTrace, nodeTrace], layout, {{responsive: true, displaylogo: false}});

  const zoneCounts = {{green: 0, yellow: 0, red: 0}};
  WINDOW_META.forEach(w => zoneCounts[w.zone] = (zoneCounts[w.zone] || 0) + 1);
  const modalityCounts = {{positive: 0, negative: 0}};
  WINDOW_META.forEach(w => {{ if (w.modality) modalityCounts[w.modality]++; }});
  document.getElementById("zone-stats").innerHTML = `
    <span class="stat"><span class="dot green"></span>${{zoneCounts.green}} green</span>
    <span class="stat"><span class="dot yellow"></span>${{zoneCounts.yellow}} yellow</span>
    <span class="stat"><span class="dot red"></span>${{zoneCounts.red}} red</span>
    <span class="stat mono" style="color:${{MODALITY_COLOR.positive}}">&#9650; ${{modalityCounts.positive}} positive</span>
    <span class="stat mono" style="color:${{MODALITY_COLOR.negative}}">&#9660; ${{modalityCounts.negative}} negative</span>`;
  const first = parseUTC(WINDOW_META[0].start), last = parseUTC(WINDOW_META[WINDOW_META.length - 1].start);
  document.getElementById("span-label").textContent = `${{first.toISOString().slice(0,10)}} → ${{last.toISOString().slice(0,10)}} (real span; axis is compressed)`;
  document.getElementById("n-windows-label").textContent = `${{WINDOW_META.length}} test windows plotted`;
  document.getElementById("n-touched-label").textContent = `${{touchedIdx.length}} touched by ≥1 event node`;
}}

main();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
