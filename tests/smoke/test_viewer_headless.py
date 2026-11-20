"""Opens the real viewer with SDL's dummy video driver: no window, but all drawing runs."""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from neurodrive.agents.pure_pursuit import PurePursuitDriver
from neurodrive.sim.simulation import Simulation
from neurodrive.sim.track import load_track
from neurodrive.ui.app import PlaygroundApp


@pytest.mark.parametrize("speed", [0.25, 8.0])
def test_viewer_physics_match_headless_at_any_playback_speed(speed):
    track = load_track("beginner_oval")
    app = PlaygroundApp(track, controller="autopilot", frame_dt=1 / 60)
    app.clock.set_speed(speed)
    app.run(max_frames=240)
    n = app.sim.step_count
    assert n > 0

    headless = Simulation(track)
    driver = PurePursuitDriver()
    for _ in range(n):
        headless.step(driver.act(headless))
    assert headless.state == app.sim.state  # exact equality


def test_viewer_survives_crash_and_reset():
    app = PlaygroundApp(load_track("beginner_oval"), controller="keyboard", frame_dt=1 / 60)
    for _ in range(400):
        app.sim.step((1.0, 1.0)) if not app.sim.done else None
    assert app.sim.done
    app._draw()  # crash banner renders
    app.reset()
    assert not app.sim.done
    app.run(max_frames=5)
