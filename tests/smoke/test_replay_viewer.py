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


@pytest.fixture(scope="module")
def race_replays():
    from neurodrive.replay.race import record_race

    entries = [driver_entry(n) for n in ("autopilot", "heuristic", "smoothed_random")]
    return record_race(entries, "beginner_oval", laps=1, log=lambda _: None)


def test_race_opens_with_start_lights_and_cars_on_the_grid(race_replays):
    viewer = ReplayViewer(race_replays, frame_dt=1 / 60)
    assert viewer.is_race and viewer.t == viewer.t_start < 0
    viewer.draw()  # lights, grid
    assert all(car.r.index_at(viewer.t) == 0 for car in viewer.cars)
    viewer.seek(viewer.duration)
    viewer.key(pygame.K_SPACE)  # at the end, Space restarts from the lights
    assert viewer.t == viewer.t_start
    pygame.quit()


def test_camera_modes_minimap_and_gaps(race_replays):
    viewer = ReplayViewer(race_replays, frame_dt=1 / 60)
    for mode in ("track", "follow", "tv"):
        viewer.key(pygame.K_z) if mode != "track" else None
        assert viewer.camera_mode == mode
        viewer.seek(8.0)
        viewer.draw()
    viewer.key(pygame.K_z)
    assert viewer.camera_mode == "track"
    leader = viewer.ranking()[0]
    assert viewer.gap_to_leader(leader) is None
    others = [viewer.gap_to_leader(k) for k in viewer.ranking()[1:]]
    assert any(g is not None and g >= 0 for g in others)
    pygame.quit()


def test_skid_detection_from_a_path():
    import numpy as np

    from neurodrive.ui import art

    n = 30
    x = np.linspace(0, 29, n)
    heading = np.zeros(n)
    heading[15:] = 0.5  # car rotated but kept moving along x: a slide
    skid, wheels = art.skid_profile(x, np.zeros(n), heading, np.full(n, 30.0), 1 / 30)
    assert not skid[:15].any() and skid[15:].all()
    assert wheels.shape == (n, 2, 2)
    assert art.skid_runs(skid) == [(15, 30)]


def test_video_export(replays, tmp_path):
    import shutil

    from neurodrive.ui.video import export_video

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    viewer = ReplayViewer(replays[:1], frame_dt=1 / 60)
    out = export_video(viewer, tmp_path / "clip.mp4", fps=10, speed=8.0, hold_end_s=0.2,
                       log=lambda _: None)  # fmt: skip
    assert out.stat().st_size > 10_000
