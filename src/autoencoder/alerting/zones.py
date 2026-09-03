"""Classify a window anomaly score into Green / Yellow / Red zones."""

from __future__ import annotations


def classify_zone(score: float, thresholds: dict) -> str:
    """Classify a window score into a zone.

    Args:
        score: Window anomaly score (P95 of per-row MSE).
        thresholds: Dict with 'green_yellow' and 'yellow_red' keys.

    Returns:
        "green", "yellow", or "red".
    """
    if score >= thresholds["yellow_red"]:
        return "red"
    elif score >= thresholds["green_yellow"]:
        return "yellow"
    else:
        return "green"


def classify_batch(scores: list[float], thresholds: dict) -> list[str]:
    """Classify a batch of window scores into zones."""
    return [classify_zone(s, thresholds) for s in scores]
