"""Generate HTML cleaning report with statistics, audit trail, and diagnostic plots."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_cleaning_report(
    pipeline_result,
    sensor_columns: list[str],
    equipment_id: str,
    output_path: str | Path,
) -> str:
    """Generate an HTML cleaning report from pipeline results.

    Args:
        pipeline_result: CleaningPipelineResult from the cleaning pipeline.
        sensor_columns: List of sensor column names.
        equipment_id: Equipment identifier.
        output_path: Path to write the HTML report.

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build step log table rows
    step_rows = ""
    for log in pipeline_result.step_logs:
        step_rows += (
            f"<tr><td>{log.step_name}</td><td>{log.regime}</td>"
            f"<td>{log.windows_before}</td><td>{log.windows_after}</td>"
            f"<td>{log.windows_removed}</td></tr>\n"
        )

    # Per-regime summary rows
    regime_rows = ""
    for regime, stats in pipeline_result.per_regime_results.items():
        orig = stats["original"]
        clean = stats["cleaned"]
        pct = (1 - clean / orig) * 100 if orig > 0 else 0
        regime_rows += (
            f"<tr><td>{regime}</td><td>{orig}</td><td>{clean}</td>"
            f"<td>{pct:.1f}%</td></tr>\n"
        )

    # Embed plot images
    plot_html = ""
    vr = pipeline_result.validation_report
    if vr and vr.plot_paths:
        for p in vr.plot_paths:
            rel = Path(p).name
            plot_html += f'<div class="plot"><img src="{rel}" alt="{rel}"></div>\n'

    removal_pct = (
        (1 - pipeline_result.cleaned_count / pipeline_result.original_count) * 100
        if pipeline_result.original_count > 0
        else 0
    )

    excessive_warning = ""
    if vr and vr.excessive_removal:
        excessive_warning = (
            '<div class="warning">WARNING: Excessive data removal detected '
            f'({removal_pct:.1f}% > {40.0}%). Investigate data quality before training.</div>'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cleaning Report - {equipment_id}</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #2980b9; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background-color: #3498db; color: white; }}
    tr:nth-child(even) {{ background-color: #f2f2f2; }}
    .summary {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
    .summary span {{ font-weight: bold; font-size: 1.2em; }}
    .warning {{ background: #e74c3c; color: white; padding: 15px; border-radius: 8px; margin: 15px 0; }}
    .plot {{ margin: 15px 0; }}
    .plot img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
    .meta {{ color: #7f8c8d; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Data Cleaning Report</h1>
<p class="meta">Equipment: <strong>{equipment_id}</strong> |
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
Sensors: {len(sensor_columns)}</p>

{excessive_warning}

<div class="summary">
    <p>Total windows before cleaning: <span>{pipeline_result.original_count}</span></p>
    <p>Total windows after cleaning: <span>{pipeline_result.cleaned_count}</span></p>
    <p>Overall removal: <span>{removal_pct:.1f}%</span></p>
</div>

<h2>Per-Step Removal Audit</h2>
<table>
<tr><th>Step</th><th>Regime</th><th>Before</th><th>After</th><th>Removed</th></tr>
{step_rows}
</table>

<h2>Aggregate Removals by Method</h2>
<table>
<tr><th>Method</th><th>Total Removed</th></tr>
{"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in pipeline_result.per_step_removals.items())}
</table>

<h2>Per-Regime Summary</h2>
<table>
<tr><th>Regime</th><th>Original</th><th>Cleaned</th><th>Removed %</th></tr>
{regime_rows}
</table>

<h2>Diagnostic Plots</h2>
{plot_html if plot_html else "<p>No plots generated (output_dir not specified during cleaning).</p>"}

<hr>
<p class="meta">Cleaning order: Isolation Forest &rarr; PCA &rarr; Mahalanobis Distance (per regime)</p>
</body>
</html>"""

    output_path.write_text(html)
    logger.info("Cleaning report written to %s", output_path)
    return str(output_path)
