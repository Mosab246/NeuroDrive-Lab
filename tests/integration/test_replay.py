"""Replays: recording, the file format, and exact re-simulation."""

import dataclasses
import json

import neat
import numpy as np
import pytest

from neurodrive.agents.baselines import HeuristicController, PrivilegedAutopilot
from neurodrive.agents.neat_agent import NEATController
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.replay.format import ReplayError, load_replay
from neurodrive.replay.lab import benchmark_entries, driver_entry, timeline_entries
from neurodrive.replay.recorder import (
    describe_kind,
    env_for,
    network_info,
    record_episode,
    record_many,
    verify_replay,
)


@pytest.fixture(scope="module")
def heuristic_replay():
    env = env_for(None, "beginner_oval")
    controller = HeuristicController(angles_deg=env.config.sensors.angles_deg)
    return record_episode(env, controller, seed=3, label="H")


def test_recording_matches_the_episode(heuristic_replay):
    r = heuristic_replay
    assert r.meta["termination_reason"] == "lap_complete"
    assert r.meta["controller_kind"] == "heuristic" and not r.meta["learned"]
    assert len(r.x) == r.steps + 1 and r.actions.shape == (r.steps, 2)
    assert r.meta["lap_times"][0] == pytest.approx(r.duration, abs=r.dt)
    assert r.laps[-1] == 1 and r.laps[0] == 0


def test_save_load_round_trip(heuristic_replay, tmp_path):
    path = heuristic_replay.save(tmp_path / "h")
    assert path.suffix == ".npz"
    back = load_replay(path)
    assert back.meta == heuristic_replay.meta
    for name in ("x", "y", "heading", "actions", "ray_dist", "ray_dirs", "laps"):
        np.testing.assert_array_equal(getattr(back, name), getattr(heuristic_replay, name))


def test_replay_re_simulates_exactly(heuristic_replay, tmp_path):
    verify_replay(load_replay(heuristic_replay.save(tmp_path / "h.npz")))


def test_tampered_replay_is_detected(heuristic_replay):
    actions = heuristic_replay.actions.copy()  # copy: keep the fixture intact
    actions[40] = [-1.0, -1.0]
    with pytest.raises(ReplayError, match="diverges"):
        verify_replay(dataclasses.replace(heuristic_replay, actions=actions))


def test_load_rejects_other_files(tmp_path):
    np.savez(tmp_path / "x.npz", meta=np.array(json.dumps({"format": "other"})))
    with pytest.raises(ReplayError, match="not a"):
        load_replay(tmp_path / "x.npz")
    with pytest.raises(ReplayError, match="not found"):
        load_replay(tmp_path / "missing.npz")


def test_same_seed_same_start_for_every_driver():
    reps = record_many(
        [driver_entry("heuristic"), driver_entry("autopilot")], "beginner_oval", 5, log=print
    )
    assert reps[0].x[0] == reps[1].x[0] and reps[0].heading[0] == reps[1].heading[0]
    assert {r.meta["kind_description"] for r in reps} == {
        "scripted (sensors only)",
        "scripted (privileged)",
    }


def test_neat_network_and_replay(tmp_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "configs/neat/default.ini",
    )
    genome = next(iter(neat.Population(config).population.values()))
    controller = NEATController(genome, config, name="tiny")
    info, png = network_info(controller)
    assert info is not None and info["type"] == "neat"
    assert png is not None and png[:4] == b"\x89PNG"
    env = env_for(None, "beginner_oval")
    rep = record_episode(env, controller, seed=1)
    assert rep.meta["learned"] and describe_kind("neat") == "learned (NEAT)"
    back = load_replay(rep.save(tmp_path / "n.npz"))
    assert back.network_png == png
    verify_replay(back)


def test_ppo_network_description():
    from stable_baselines3 import PPO

    from neurodrive.agents.ppo_agent import PPOController

    env = DrivingEnv()
    controller = PPOController(PPO("MlpPolicy", env, device="cpu", seed=0))
    info, png = network_info(controller)
    assert png is None
    assert info == {
        "type": "mlp",
        "inputs": 11,
        "hidden": [64, 64],
        "outputs": 2,
        "activation": "Tanh",
        "parameters": 11 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2,
    }


def test_autopilot_uses_the_recording_env():
    env = env_for(None, "curved_circuit", laps=2)
    rep = record_episode(env, PrivilegedAutopilot(env), seed=0)
    assert rep.track == "curved_circuit" and len(rep.meta["lap_times"]) == 2


def _fake_model(path, algorithm):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.with_suffix(".meta.json").write_text(json.dumps({"algorithm": algorithm}))


def test_timeline_picks_evenly_spaced_snapshots(tmp_path):
    for n in range(0, 1_100_000, 100_000):
        _fake_model(tmp_path / "checkpoints" / f"step_{n:08d}.zip", "ppo")
    _fake_model(tmp_path / "best_model.zip", "ppo")
    labels = [e[0] for e in timeline_entries(tmp_path, 6)]
    assert labels == ["step 0", "step 200k", "step 400k", "step 600k", "step 800k",
                      "step 1.00M", "best (selected)"]  # fmt: skip


def test_timeline_reads_neat_champions(tmp_path):
    for g in range(3):
        _fake_model(tmp_path / "champions" / f"gen_{g:04d}.pkl", "neat")
    assert [e[0] for e in timeline_entries(tmp_path, 6)] == ["gen 0", "gen 1", "gen 2"]


def test_benchmark_entries_from_manifest(tmp_path):
    runs = {}
    for algo, ext in (("neat", "pkl"), ("ppo", "zip")):
        for seed in (0, 1):
            d = tmp_path / f"{algo}{seed}"
            _fake_model(d / f"best_model.{ext}", algo)
            runs[f"{algo}/{seed}"] = {"algorithm": algo, "seed": seed, "run_dir": str(d)}
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": runs}))
    labels = [e[0] for e in benchmark_entries(tmp_path)]
    assert labels == ["PPO seed 0", "PPO seed 1", "NEAT seed 0", "NEAT seed 1"]
    assert [e[0] for e in benchmark_entries(tmp_path, [1])] == ["PPO seed 1", "NEAT seed 1"]
    with pytest.raises(FileNotFoundError):
        benchmark_entries(tmp_path / "nope")


def test_unknown_driver_is_refused():
    with pytest.raises(ValueError):
        driver_entry("missing_model.zip")
