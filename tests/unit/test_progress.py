import numpy as np
import pytest

from neurodrive.sim.progress import ProgressTracker
from neurodrive.sim.track import load_track


@pytest.fixture(scope="module")
def oval():
    return load_track("beginner_oval")


def drive(tracker, track, s_values, lateral=0.0):
    """Feed centerline positions at the given arc lengths; return all updates."""
    updates = []
    for s in s_values:
        p, t = track.point_at(s)
        n = np.array([-t[1], t[0]])
        updates.append(tracker.update(p + lateral * n))
    return updates


def test_one_full_forward_loop_is_exactly_one_lap(oval):
    tr = ProgressTracker(oval)
    tr.reset(oval.point_at(0.0)[0])
    ups = drive(tr, oval, np.arange(0.5, oval.length + 0.6, 0.5))
    assert tr.laps == 1
    assert sum(u.lap_completed for u in ups) == 1
    assert sum(u.checkpoints_passed for u in ups) == oval.spec.checkpoint_count
    assert tr.distance == pytest.approx(oval.length + 0.6, abs=0.6)


def test_lap_completes_exactly_on_the_finish_line(oval):
    tr = ProgressTracker(oval)
    tr.reset(oval.point_at(0.0)[0])
    drive(tr, oval, np.arange(0.5, oval.length - 0.4, 0.5))
    assert tr.laps == 0
    drive(tr, oval, [oval.length + 0.3])
    assert tr.laps == 1


def test_progress_is_continuous_across_the_line(oval):
    tr = ProgressTracker(oval)
    tr.reset(oval.point_at(0.0)[0])
    ups = drive(tr, oval, np.arange(0.5, oval.length + 5.0, 0.5))
    deltas = np.array([u.delta for u in ups])
    np.testing.assert_allclose(deltas, 0.5, atol=0.05)


def test_driving_backwards_across_the_line_never_counts_a_lap(oval):
    tr = ProgressTracker(oval)
    tr.reset(oval.point_at(0.0)[0])
    ups = drive(tr, oval, -np.arange(0.5, oval.length + 0.6, 0.5))
    assert tr.laps == 0
    assert sum(u.checkpoints_passed for u in ups) == 0
    assert tr.distance == pytest.approx(-(oval.length + 0.5), abs=0.6)


def test_rocking_over_a_checkpoint_counts_it_once(oval):
    tr = ProgressTracker(oval)
    tr.reset(oval.point_at(0.0)[0])
    gate = oval.checkpoint_s[0]
    forward = np.arange(0.5, gate + 1.0, 0.5)
    ups = drive(tr, oval, forward)
    for _ in range(5):  # back and forth over the gate
        ups += drive(tr, oval, [gate - 1.0, gate - 2.0, gate - 1.0, gate + 1.0])
    assert sum(u.checkpoints_passed for u in ups) == 1
    assert tr.next_checkpoint == 2
    # Net progress equals net displacement: five round trips added nothing.
    assert tr.distance == pytest.approx(gate + 1.0, abs=0.1)
    assert sum(u.delta for u in ups) == pytest.approx(tr.distance, abs=1e-9)


def test_teleporting_across_the_track_does_not_skip_progress(oval):
    tr = ProgressTracker(oval)
    tr.reset(oval.point_at(0.0)[0])
    drive(tr, oval, np.arange(0.5, 10.5, 0.5))
    # Jump to the far side of the oval (s ~ 250 m). The windowed search can't follow.
    far, _ = oval.point_at(250.0)
    u = tr.update(far)
    assert abs(u.delta) <= tr.window * oval.spec.sample_spacing * 1.5
    assert tr.laps == 0 and tr.next_checkpoint == 1


def test_lateral_offset_sign(oval):
    tr = ProgressTracker(oval)
    tr.reset(oval.point_at(0.0)[0])
    drive(tr, oval, [1.0], lateral=2.0)
    assert tr.lateral == pytest.approx(2.0, abs=1e-6)
    drive(tr, oval, [1.5], lateral=-3.0)
    assert tr.lateral == pytest.approx(-3.0, abs=1e-6)
