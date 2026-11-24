"""Phase 5: more tracks, procedural tracks, random starts, multi-track envs, split rules."""

import numpy as np
import pytest

from neurodrive.agents.baselines import HeuristicController, PrivilegedAutopilot
from neurodrive.cli import expand_tracks
from neurodrive.env.config import EnvConfig
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.evaluation.evaluator import evaluate
from neurodrive.experiments.splits import SplitError, check_track_splits
from neurodrive.sim.procedural import procedural_track
from neurodrive.sim.simulation import SimConfig, Simulation, Termination
from neurodrive.sim.track import builtin_track_ids, load_any_track, load_track


def drive_one_lap(env, controller, seed=0, **options):
    obs, _ = env.reset(seed=seed, options={"perturb": False} | options)
    while True:
        obs, _, term, trunc, info = env.step(controller.act(obs))
        if term or trunc:
            return info


@pytest.mark.parametrize("track_id", builtin_track_ids())
def test_every_builtin_track_is_drivable(track_id):
    """The scripted autopilot must lap every shipped track, or the track is broken."""
    env = DrivingEnv(EnvConfig(track=track_id, max_episode_seconds=120))
    info = drive_one_lap(env, PrivilegedAutopilot(env))
    assert info["termination_reason"] == "lap_complete", track_id


def test_track_splits_are_as_designed():
    splits = {t: load_track(t).spec.split for t in builtin_track_ids()}
    assert splits["beginner_oval"] == splits["curved_circuit"] == "train"
    assert splits["technical_circuit"] == "train"
    assert splits["validation_circuit"] == "validation"
    assert splits["unseen_a"] == splits["unseen_b"] == "test"


def test_both_driving_directions_are_represented():
    turning = {t: np.sign(load_track(t).curvature.mean()) for t in builtin_track_ids()}
    assert 1.0 in turning.values() and -1.0 in turning.values()  # CCW and CW


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_procedural_tracks_are_valid_drivable_and_test_only(seed):
    track = procedural_track(seed)
    assert track.id == f"proc_{seed}" and track.spec.split == "test"
    env = DrivingEnv(EnvConfig(track=track.id, max_episode_seconds=120))
    assert drive_one_lap(env, PrivilegedAutopilot(env))["termination_reason"] == "lap_complete"


def test_procedural_tracks_are_deterministic_and_different():
    a, b, c = procedural_track(7), procedural_track(7), procedural_track(8)
    np.testing.assert_array_equal(a.centerline, b.centerline)
    assert a.centerline.shape != c.centerline.shape or not np.allclose(a.centerline, c.centerline)
    assert load_any_track("proc_7").length == a.length


def test_random_start_lap_is_one_full_circuit_from_the_start_point():
    track = load_track("curved_circuit")
    sim = Simulation(track, config=SimConfig(laps_to_finish=1, max_time=120))
    sim.reset(seed=3, random_start=True)
    assert sim.start_s > 1.0  # not on the start line
    assert abs(sim.progress.lap_fraction) < 0.01  # a new lap starts here
    from neurodrive.agents.pure_pursuit import PurePursuitDriver

    driver = PurePursuitDriver()
    while not sim.done:
        sim.step(driver.act(sim))
    assert sim.termination is Termination.LAPS_COMPLETE
    assert sim.progress.distance == pytest.approx(track.length, abs=2.0)


def test_random_starts_are_seeded_and_varied():
    sim = Simulation(load_track("unseen_a"))
    starts = []
    for seed in range(5):
        sim.reset(seed=seed, random_start=True)
        starts.append(sim.start_s)
    sim.reset(seed=0, random_start=True)
    assert sim.start_s == starts[0]
    assert len({round(s, 3) for s in starts}) == 5


def test_multitrack_env_picks_tracks_by_seed():
    env = DrivingEnv(EnvConfig(tracks=("beginner_oval", "curved_circuit", "technical_circuit")))
    picks = [env.reset(seed=s)[1]["track"] for s in range(30)]
    assert set(picks) == {"beginner_oval", "curved_circuit", "technical_circuit"}
    assert picks == [env.reset(seed=s)[1]["track"] for s in range(30)]
    assert env.reset(seed=0, options={"track": "beginner_oval"})[1]["track"] == "beginner_oval"


def test_evaluate_over_tracks_runs_every_pair():
    env = DrivingEnv(EnvConfig(tracks=("unseen_a", "unseen_b")))
    records = evaluate(
        env, HeuristicController(), [1, 2], tracks=["unseen_a", "unseen_b"], random_start=True
    )
    assert [r.track for r in records] == ["unseen_a", "unseen_a", "unseen_b", "unseen_b"]


@pytest.mark.parametrize("bad", ["unseen_a", "validation_circuit", "proc_3"])
def test_training_on_non_train_tracks_is_refused(bad):
    with pytest.raises(SplitError, match="can't be used for training"):
        check_track_splits(EnvConfig(tracks=("beginner_oval", bad)), None)


def test_selecting_on_test_tracks_is_refused():
    with pytest.raises(SplitError, match="select checkpoints"):
        check_track_splits(EnvConfig(), ["unseen_b"])
    check_track_splits(EnvConfig(), ["validation_circuit"])  # allowed


def test_expand_tracks_shortcuts():
    assert expand_tracks("split:test") == ["unseen_a", "unseen_b"]
    assert expand_tracks("beginner_oval, proc:2") == ["beginner_oval", "proc_0", "proc_1"]
    with pytest.raises(ValueError):
        expand_tracks("split:nothing")
