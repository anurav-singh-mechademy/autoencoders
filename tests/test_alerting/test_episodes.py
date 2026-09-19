"""Tests for episode extraction from an alert-level timeline."""

import pytest

from autoencoder.alerting.episodes import extract_episodes


class TestExtractEpisodes:
    def test_no_episodes_when_all_green(self):
        levels = ["green"] * 5
        starts = list(range(5))
        ends = list(range(1, 6))
        assert extract_episodes(levels, starts, ends) == []

    def test_single_episode(self):
        levels = ["green", "yellow", "yellow", "green"]
        starts = [0, 1, 2, 3]
        ends = [1, 2, 3, 4]
        episodes = extract_episodes(levels, starts, ends)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.start_window_idx == 1
        assert ep.end_window_idx == 2
        assert ep.start == 1
        assert ep.end == 3
        assert ep.peak_level == "yellow"
        assert ep.n_windows == 2

    def test_peak_level_picks_most_severe(self):
        levels = ["yellow", "red", "yellow"]
        starts = [0, 1, 2]
        ends = [1, 2, 3]
        episodes = extract_episodes(levels, starts, ends)
        assert len(episodes) == 1
        assert episodes[0].peak_level == "red"

    def test_multiple_separate_episodes(self):
        levels = ["red", "green", "yellow", "green", "red"]
        starts = list(range(5))
        ends = [s + 1 for s in starts]
        episodes = extract_episodes(levels, starts, ends)
        assert len(episodes) == 3
        assert [e.peak_level for e in episodes] == ["red", "yellow", "red"]
        assert [e.n_windows for e in episodes] == [1, 1, 1]

    def test_episode_covering_entire_timeline(self):
        # A slug/alert that never clears (n_edges=0 case upstream) still
        # yields exactly one episode spanning everything -- no special case.
        levels = ["red"] * 10
        starts = list(range(10))
        ends = [s + 1 for s in starts]
        episodes = extract_episodes(levels, starts, ends)
        assert len(episodes) == 1
        assert episodes[0].n_windows == 10
        assert episodes[0].start == 0
        assert episodes[0].end == 10

    def test_empty_input(self):
        assert extract_episodes([], [], []) == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            extract_episodes(["green", "red"], [0], [1])
