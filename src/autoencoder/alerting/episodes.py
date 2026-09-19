"""Collapse a per-window alert-level timeline into episode records.

apply_persistence() (see persistence.py) gives a running alert LEVEL --
this module turns that per-window timeline into (start, end, peak_level,
n_windows) records, the actual client-facing alert unit referenced
throughout the evaluation plan.
"""

from __future__ import annotations

from dataclasses import dataclass

_SEVERITY = {"green": 0, "yellow": 1, "red": 2}


@dataclass
class Episode:
    """A contiguous run of non-"green" alert windows."""

    start_window_idx: int
    end_window_idx: int
    start: object   # window_start of the first window in the episode
    end: object     # window_end of the last window in the episode
    peak_level: str  # "yellow" or "red" -- the most severe level reached
    n_windows: int


def extract_episodes(
    alert_levels: list[str],
    window_starts: list,
    window_ends: list,
) -> list[Episode]:
    """Collapse a per-window alert-level timeline into episode records.

    An episode is a maximal contiguous run of non-"green" windows. Returns
    an empty list if there are no non-green windows at all (a slug/unit
    that never alerts is a valid, clean result, not an error).

    Args:
        alert_levels: Per-window alert level ("green"/"yellow"/"red"),
            already computed by apply_persistence (or equivalent).
        window_starts: Per-window start timestamp/value, same length/order.
        window_ends: Per-window end timestamp/value, same length/order.
    """
    if not (len(alert_levels) == len(window_starts) == len(window_ends)):
        raise ValueError("alert_levels, window_starts, window_ends must be the same length")

    episodes: list[Episode] = []
    n = len(alert_levels)
    i = 0
    while i < n:
        if alert_levels[i] == "green":
            i += 1
            continue
        start_i = i
        peak = alert_levels[i]
        while i < n and alert_levels[i] != "green":
            if _SEVERITY.get(alert_levels[i], 0) > _SEVERITY.get(peak, 0):
                peak = alert_levels[i]
            i += 1
        end_i = i - 1
        episodes.append(Episode(
            start_window_idx=start_i,
            end_window_idx=end_i,
            start=window_starts[start_i],
            end=window_ends[end_i],
            peak_level=peak,
            n_windows=end_i - start_i + 1,
        ))
    return episodes
