"""The replay viewer and the race/replay commands, headless."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from neurodrive.cli import main
from neurodrive.replay.format import ReplayError
from neurodrive.replay.lab import driver_entry
from neurodrive.replay.recorder import record_many
from neurodrive.ui.replay_viewer import ReplayViewer


@pytest.fixture(scope="module")
def replays():
    entries = [driver_entry(n) for n in ("heuristic", "autopilot", "smoothed_random")]
    return record_many(entries, "beginner_oval", 0, log=lambda _: None)


def test_playback_advances_and_stops_at_the_end(replays):
    viewer = ReplayViewer(replays, frame_dt=1 / 60)
    viewer.speed_index = len(viewer_speeds()) - 1  # 8x
    viewer.run(max_frames=60)
    assert viewer.t == pytest.approx(8.0, abs=0.05)
    viewer = ReplayViewer(replays, frame_dt=1.0)
    viewer.run(max_frames=200)
    assert viewer.t == viewer.duration and not viewer.playing


def viewer_speeds():
    from neurodrive.ui.clock import SPEEDS

    return SPEEDS


def test_controls_and_every_overlay_draw(replays):
    viewer = ReplayViewer(replays, frame_dt=1 / 60)
    viewer.key(pygame.K_RIGHT, shift=True)
    assert viewer.t == pytest.approx(5.0)
    viewer.key(pygame.K_LEFT)
    assert viewer.t == pytest.approx(4.0)
    for focus in range(len(replays)):  # rays, network panel, follow camera for each car
        viewer.focus = focus
        for flags in ((True, False), (False, True)):
            viewer.show_network, viewer.follow = flags
            viewer.draw()
    viewer.focus = 0
    viewer.key(pygame.K_TAB)
    assert viewer.focus == 1
    viewer.key(pygame.K_HOME)
    assert viewer.t == 0.0
    viewer.seek(viewer.duration)
    viewer.draw()
    statuses = {viewer.cars[k].status(viewer.cars[k].r.index_at(viewer.t)) for k in range(3)}
    assert any(s.startswith("FINISHED") for s in statuses)
    pygame.quit()


def test_ranking_puts_the_faster_lap_first(replays):
    viewer = ReplayViewer(replays, frame_dt=1 / 60)
    viewer.seek(viewer.duration)
    first = viewer.cars[viewer.ranking()[0]]
    assert first.r.label == "autopilot"  # 25.4 s vs the heuristic's 26.4 s on the oval
    pygame.quit()


def test_replays_on_different_tracks_are_refused(replays):
    other = record_many([driver_entry("autopilot")], "curved_circuit", 0, log=lambda _: None)
    with pytest.raises(ReplayError, match="different tracks"):
        ReplayViewer([replays[0], other[0]])


def test_race_and_replay_commands(tmp_path, capsys):
    out = tmp_path / "race"
    args = ["race", "--driver", "heuristic", "--driver", "fast=autopilot", "--track"]
    assert main([*args, "beginner_oval", "--save-dir", str(out), "--frames", "10"]) == 0
    files = sorted(str(p) for p in out.glob("*.npz"))
    assert len(files) == 2
    assert main(["replay", *files, "--verify", "--frames", "5"]) == 0
    text = capsys.readouterr().out
    assert "verified" in text and "FPS average" in text
    assert main(["record", "--driver", "heuristic", "--out", str(tmp_path / "one")]) == 0
    assert (tmp_path / "one.npz").is_file()
