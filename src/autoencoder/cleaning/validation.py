"""Cleaning validation -- scatter plots, histograms, and coverage checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Summary of cleaning validation checks."""

    total_original: int
    total_after_cleaning: int
    removal_pct: float
    per_step_removals: dict[str, int]
    excessive_removal: bool          # True if >max_removal_pct removed
    plot_paths: list[str]


def validate_cleaning(
    original_windows: list[np.ndarray],
    cleaned_windows: list[np.ndarray],
    per_step_removals: dict[str, int],
    sensor_columns: list[str],
    output_dir: str | Path,
    n_sensor_pairs: int = 10,
    max_removal_pct: float = 40.0,
) -> ValidationReport:
    """Run validation checks on cleaned data and generate diagnostic plots.

    Checks:
        1. Total removal percentage vs threshold.
        2. Scatter plots of random sensor pairs (before/after).
        3. Histogram of per-window mean sensor values (after cleaning).

    Args:
        original_windows: All windows before any cleaning.
        cleaned_windows: Windows remaining after full cleaning pipeline.
        per_step_removals: Dict mapping step name to count of windows removed.
        sensor_columns: Names of sensor columns.
        output_dir: Directory to save plots.
        n_sensor_pairs: Number of random sensor pairs to plot.
        max_removal_pct: Warn if more than this percentage of windows removed.

    Returns:
        ValidationReport with diagnostics and plot paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_original = len(original_windows)
    n_cleaned = len(cleaned_windows)
    removal_pct = (1 - n_cleaned / n_original) * 100 if n_original > 0 else 0.0
    excessive = removal_pct > max_removal_pct

    if excessive:
        logger.warning(
            "Excessive data removal: %.1f%% removed (threshold: %.1f%%). "
            "Investigate data quality before training.",
            removal_pct, max_removal_pct,
        )

    plot_paths: list[str] = []

    # Scatter plots: random sensor pairs before/after
    n_sensors = original_windows[0].shape[1] if original_windows else 0
    n_pairs = min(n_sensor_pairs, n_sensors * (n_sensors - 1) // 2)
    if n_pairs > 0 and n_sensors >= 2:
        rng = np.random.default_rng(42)
        pairs = set()
        while len(pairs) < n_pairs:
            i, j = sorted(rng.choice(n_sensors, 2, replace=False))
            pairs.add((i, j))

        orig_means = np.array([w.mean(axis=0) for w in original_windows])
        clean_means = np.array([w.mean(axis=0) for w in cleaned_windows])

        for idx, (si, sj) in enumerate(sorted(pairs)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            name_i = sensor_columns[si] if si < len(sensor_columns) else f"S{si}"
            name_j = sensor_columns[sj] if sj < len(sensor_columns) else f"S{sj}"

            axes[0].scatter(orig_means[:, si], orig_means[:, sj], alpha=0.3, s=10)
            axes[0].set_title("Before Cleaning")
            axes[0].set_xlabel(name_i)
            axes[0].set_ylabel(name_j)

            axes[1].scatter(clean_means[:, si], clean_means[:, sj], alpha=0.3, s=10, color="green")
            axes[1].set_title("After Cleaning")
            axes[1].set_xlabel(name_i)
            axes[1].set_ylabel(name_j)

            fig.suptitle(f"{name_i} vs {name_j}")
            fig.tight_layout()
            # Sanitize for the FILENAME only (title/labels above keep the real
            # tag) -- a tag containing "/" (a real historian naming
            # convention, e.g. "5LI-5071D/PV") would otherwise be read as a
            # path separator and fail with a missing-directory error.
            safe_i = name_i.replace("/", "-")
            safe_j = name_j.replace("/", "-")
            path = str(output_dir / f"scatter_{idx:02d}_{safe_i}_vs_{safe_j}.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            plot_paths.append(path)

    # Histogram of per-window mean reconstruction proxy
    if cleaned_windows:
        clean_means = np.array([w.mean(axis=0) for w in cleaned_windows])
        global_means = clean_means.mean(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(global_means, bins=50, edgecolor="black", alpha=0.7)
        ax.set_title("Distribution of Per-Window Mean (Post-Cleaning)")
        ax.set_xlabel("Window Mean")
        ax.set_ylabel("Count")
        fig.tight_layout()
        path = str(output_dir / "histogram_window_means.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        plot_paths.append(path)

    # Removal per step bar chart
    if per_step_removals:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = list(per_step_removals.keys())
        counts = list(per_step_removals.values())
        ax.barh(steps, counts, color="salmon", edgecolor="black")
        ax.set_xlabel("Windows Removed")
        ax.set_title("Windows Removed per Cleaning Step")
        fig.tight_layout()
        path = str(output_dir / "removal_per_step.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        plot_paths.append(path)

    logger.info(
        "Validation complete: %d -> %d windows (%.1f%% removed). %d plots saved.",
        n_original, n_cleaned, removal_pct, len(plot_paths),
    )

    return ValidationReport(
        total_original=n_original,
        total_after_cleaning=n_cleaned,
        removal_pct=removal_pct,
        per_step_removals=per_step_removals,
        excessive_removal=excessive,
        plot_paths=plot_paths,
    )
