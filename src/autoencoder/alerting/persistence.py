"""Persistence rules: require consecutive anomalous windows before raising alerts."""

from __future__ import annotations

from collections import deque


def compute_alert_level_timeline(
    zones: list[str],
    red_to_yellow_alert: int = 2,
    red_to_red_alert: int = 3,
    yellow_consecutive: int = 4,
    mixed_window_hours: float | None = None,
    window_minutes: float = 30,
) -> list[str]:
    """Streaming per-window alert level over a full zone timeline -- the
    same rules as apply_persistence(), applied incrementally at each step
    instead of re-scanning full history per window (O(n) total for an
    O(lookback) per-step cost instead of O(n) per call).

    Returns one alert level ("green"/"yellow"/"red") per input zone.
    """
    lookback_windows = None
    trailing_span: deque = deque(maxlen=0)
    if mixed_window_hours is not None and mixed_window_hours > 0:
        lookback_windows = max(1, round(mixed_window_hours * 60 / window_minutes))
        trailing_span = deque(maxlen=lookback_windows)

    consecutive_red = 0
    consecutive_yellow = 0
    levels = []
    for zone in zones:
        if zone == "red":
            consecutive_red += 1
            consecutive_yellow = 0
        elif zone == "yellow":
            consecutive_yellow += 1
            consecutive_red = 0
        else:
            consecutive_red = 0
            consecutive_yellow = 0

        if lookback_windows is not None:
            trailing_span.append(zone)
            mixed_red_count = sum(1 for z in trailing_span if z == "red")
            mixed_yellow_count = sum(1 for z in trailing_span if z == "yellow")
        else:
            mixed_red_count = mixed_yellow_count = 0

        if consecutive_red >= red_to_red_alert:
            level = "red"
        elif consecutive_red >= red_to_yellow_alert:
            level = "yellow"
        elif consecutive_yellow >= yellow_consecutive:
            level = "yellow"
        elif mixed_red_count >= 1 and mixed_yellow_count >= 1 and (2 * mixed_red_count + mixed_yellow_count) >= 4:
            level = "yellow"
        else:
            level = "green"
        levels.append(level)

    return levels


def apply_persistence(
    zone_history: list[str],
    red_to_yellow_alert: int = 2,
    red_to_red_alert: int = 3,
    yellow_consecutive: int = 4,
    mixed_window_hours: float | None = None,
    window_minutes: float = 30,
) -> dict:
    """Apply persistence rules to a sequence of zone classifications.

    Rules (from config):
        - 2 consecutive Red → Yellow alert
        - 3+ consecutive Red → Red alert
        - 4+ consecutive Yellow → Yellow alert
        - Mixed Red+Yellow within `mixed_window_hours` → Yellow alert
          (every window in that trailing span counts, not just consecutive
          runs; never escalates to Red -- only the 3+ consecutive Red rule
          above does)
        - Otherwise → Green (no alert)

    Args:
        zone_history: List of zone strings, most recent last.
        red_to_yellow_alert: Consecutive reds needed for yellow alert.
        red_to_red_alert: Consecutive reds needed for red alert.
        yellow_consecutive: Consecutive yellows needed for yellow alert.
        mixed_window_hours: If set, also check the trailing span of this many
            hours (converted to a window count via `window_minutes`) for a
            mix of Red and Yellow windows. Weighted severity `2*red + yellow
            >= 4` triggers a Yellow alert -- the same threshold that a pure
            2-red or pure 4-yellow streak already hits on its own, so this
            just extends coverage to interleaved patterns those streak-only
            rules miss. None (default) disables the check.
        window_minutes: Assumed duration of one window, for converting
            `mixed_window_hours` into a window count.

    Returns:
        Dict with:
            alert_level: "green", "yellow", or "red"
            consecutive_red: count of trailing consecutive red windows
            consecutive_yellow: count of trailing consecutive yellow windows
            reason: explanation string
    """
    if not zone_history:
        return {
            "alert_level": "green",
            "consecutive_red": 0,
            "consecutive_yellow": 0,
            "mixed_red_count": 0,
            "mixed_yellow_count": 0,
            "reason": "no history",
        }

    # Count trailing consecutive reds
    consecutive_red = 0
    for zone in reversed(zone_history):
        if zone == "red":
            consecutive_red += 1
        else:
            break

    # Count trailing consecutive yellows (only if no trailing reds)
    consecutive_yellow = 0
    if consecutive_red == 0:
        for zone in reversed(zone_history):
            if zone == "yellow":
                consecutive_yellow += 1
            else:
                break

    # Count red/yellow anywhere in the trailing mixed-window span (not just
    # consecutive runs) for the mixed-alert check below.
    mixed_red_count = 0
    mixed_yellow_count = 0
    if mixed_window_hours is not None and mixed_window_hours > 0:
        lookback_windows = max(1, round(mixed_window_hours * 60 / window_minutes))
        trailing_span = zone_history[-lookback_windows:]
        mixed_red_count = trailing_span.count("red")
        mixed_yellow_count = trailing_span.count("yellow")

    # Apply rules (most severe first)
    if consecutive_red >= red_to_red_alert:
        return {
            "alert_level": "red",
            "consecutive_red": consecutive_red,
            "consecutive_yellow": consecutive_yellow,
            "mixed_red_count": mixed_red_count,
            "mixed_yellow_count": mixed_yellow_count,
            "reason": f"{consecutive_red} consecutive red windows",
        }

    if consecutive_red >= red_to_yellow_alert:
        return {
            "alert_level": "yellow",
            "consecutive_red": consecutive_red,
            "consecutive_yellow": consecutive_yellow,
            "mixed_red_count": mixed_red_count,
            "mixed_yellow_count": mixed_yellow_count,
            "reason": f"{consecutive_red} consecutive red windows (below red alert threshold)",
        }

    if consecutive_yellow >= yellow_consecutive:
        return {
            "alert_level": "yellow",
            "consecutive_red": consecutive_red,
            "consecutive_yellow": consecutive_yellow,
            "mixed_red_count": mixed_red_count,
            "mixed_yellow_count": mixed_yellow_count,
            "reason": f"{consecutive_yellow} consecutive yellow windows",
        }

    if mixed_red_count >= 1 and mixed_yellow_count >= 1 and (2 * mixed_red_count + mixed_yellow_count) >= 4:
        return {
            "alert_level": "yellow",
            "consecutive_red": consecutive_red,
            "consecutive_yellow": consecutive_yellow,
            "mixed_red_count": mixed_red_count,
            "mixed_yellow_count": mixed_yellow_count,
            "reason": (
                f"mixed {mixed_red_count} red + {mixed_yellow_count} yellow windows "
                f"within trailing {mixed_window_hours}h"
            ),
        }

    return {
        "alert_level": "green",
        "consecutive_red": consecutive_red,
        "consecutive_yellow": consecutive_yellow,
        "mixed_red_count": mixed_red_count,
        "mixed_yellow_count": mixed_yellow_count,
        "reason": "below persistence thresholds",
    }
