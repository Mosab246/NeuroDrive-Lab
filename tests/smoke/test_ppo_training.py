"""A very short PPO run: checks the pipeline, not whether anything is learned."""

import json

import numpy as np
import pytest

from neurodrive.agents.ppo_agent import (
    ModelError,
    PPOController,
    check_compatible,
    read_model_meta,
)
from neurodrive.env.config import EnvConfig, ObservationConfig
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.training.config import PPOExperiment, PPOHyperparams, PPOTraining
from neurodrive.training.ppo import train_ppo

TINY = PPOExperiment(
    name="tiny",
    ppo=PPOHyperparams(n_steps=256, batch_size=64, n_epochs=2),
    training=PPOTraining(
        total_timesteps=1024,
        n_envs=2,
        vec_env="dummy",
        eval_every_steps=512,
        eval_episodes=1,
        checkpoint_every_steps=512,
    ),
)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return train_ppo(TINY, runs_root=tmp_path_factory.mktemp("runs"), log=lambda _: None)


def test_run_directory_contents(run_dir):
    for name in (
        "config.yaml",
        "metadata.json",
        "best_model.zip",
        "best_model.meta.json",
        "final_model.zip",
        "eval_history.csv",
        "train_summary.json",
    ):
        assert (run_dir / name).is_file(), name
    ckpts = sorted(p.name for p in (run_dir / "checkpoints").glob("*.zip"))
    assert ckpts[0] == "step_00000000.zip"  # the untrained policy is kept for comparison
    assert len(ckpts) >= 3
    assert any((run_dir / "tb").rglob("events.out.tfevents.*"))


def test_summary_reports_real_numbers(run_dir):
    s = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    assert s["total_timesteps"] >= 1024
    assert s["wall_seconds_this_session"] > 0
    assert s["best_checkpoint_step"] is not None


def test_model_metadata_describes_training(run_dir):
    meta = read_model_meta(run_dir / "final_model.zip")
    assert meta["algorithm"] == "ppo"
    assert meta["observation"]["names"][0] == "ray_-90"
    assert meta["env_config"]["track"] == "beginner_oval"


def test_saved_policy_loads_and_produces_valid_deterministic_actions(run_dir):
    controller = PPOController.load(run_dir / "final_model.zip")
    env = DrivingEnv()
    check_compatible(controller.meta, env)
    obs, _ = env.reset(seed=0)
    a1 = controller.act(obs)
    a2 = controller.act(obs)
    assert a1.shape == (2,) and np.all(np.abs(a1) <= 1.0)
    np.testing.assert_array_equal(a1, a2)  # deterministic policy at evaluation time


def test_incompatible_environment_is_refused(run_dir):
    controller = PPOController.load(run_dir / "final_model.zip")
    env = DrivingEnv(EnvConfig(observation=ObservationConfig(route_features=True)))
    with pytest.raises(ModelError, match="observation mismatch"):
        check_compatible(controller.meta, env)


def test_model_without_metadata_is_refused(run_dir, tmp_path):
    lone = tmp_path / "lone.zip"
    lone.write_bytes((run_dir / "final_model.zip").read_bytes())
    with pytest.raises(ModelError, match="no metadata"):
        PPOController.load(lone)


def test_resume_continues_the_same_run(run_dir):
    ckpt = sorted((run_dir / "checkpoints").glob("*.zip"))[-1]
    exp = TINY.model_copy(
        update={"training": TINY.training.model_copy(update={"total_timesteps": 2048})}
    )
    same = train_ppo(exp, resume_from=ckpt, log=lambda _: None)
    assert same == run_dir
    s = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    assert s["total_timesteps"] >= 2048


def test_batch_larger_than_rollout_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        PPOExperiment(
            ppo=PPOHyperparams(n_steps=16, batch_size=64),
            training=PPOTraining(n_envs=2),
        )


def test_trained_model_drives_in_the_playground(run_dir, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from neurodrive.sim.track import load_track
    from neurodrive.ui.app import PlaygroundApp

    app = PlaygroundApp(
        load_track("beginner_oval"),
        controller="model",
        frame_dt=1 / 60,
        model_controller=PPOController.load(run_dir / "final_model.zip"),
    )
    assert app.drivers[-1] == "model"
    app.run(max_frames=30)
    assert app.sim.step_count > 0


def test_checkpoint_selection_prefers_laps_then_progress_then_speed():
    from neurodrive.training.ppo import selection_key

    slow_lap = {"completion_rate": 1.0, "mean_lap_fraction": 1.0, "lap_time_mean": 25.0}
    fast_lap = {"completion_rate": 1.0, "mean_lap_fraction": 1.0, "lap_time_mean": 21.0}
    far = {"completion_rate": 0.0, "mean_lap_fraction": 0.9, "lap_time_mean": None}
    near = {"completion_rate": 0.0, "mean_lap_fraction": 0.2, "lap_time_mean": ""}
    ranked = sorted([near, slow_lap, far, fast_lap], key=selection_key, reverse=True)
    assert ranked == [fast_lap, slow_lap, far, near]
