"""Very short NEAT runs: checks the pipeline and reproducibility, not learning."""

import csv

import numpy as np
import pytest

from neurodrive.agents.model_io import ModelError, check_compatible, load_controller
from neurodrive.agents.neat_agent import NEATController
from neurodrive.env.config import EnvConfig, ObservationConfig
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.training.config import NEATExperiment, NEATSettings
from neurodrive.training.neat_trainer import episode_fitness, generation_seeds, train_neat

TINY = NEATExperiment(
    name="tiny",
    neat=NEATSettings(
        pop_size=8,
        generations=2,
        scenarios_per_genome=1,
        validate_every=1,
        validation_episodes=1,
        checkpoint_every=1,
    ),
)


def history(run_dir):
    with (run_dir / "generations.csv").open(encoding="utf-8") as fh:
        return [
            (r["generation"], r["env_steps"], r["best_fitness"], r["mean_fitness"])
            for r in csv.DictReader(fh)
        ]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return train_neat(TINY, runs_root=tmp_path_factory.mktemp("neat"), log=lambda _: None)


def test_fitness_rewards_faster_laps():
    slow = {"episode_return": 59.0, "termination_reason": "lap_complete", "lap_times": [30.0]}
    fast = {"episode_return": 59.0, "termination_reason": "lap_complete", "lap_times": [20.0]}
    crash = {"episode_return": 20.0, "termination_reason": "collision", "lap_times": []}
    f = [episode_fitness(i, 90.0, 1.0) for i in (slow, fast, crash)]
    assert f == [59.0 + 60.0, 59.0 + 70.0, 20.0]  # no speed bonus without a lap


def test_generation_seeds_are_shared_and_never_hit_evaluation_ranges():
    seeds = [s for g in range(500) for s in generation_seeds(7, g, 3)]
    assert max(seeds) < 1_000_000  # validation starts at 1,000,000
    assert generation_seeds(7, 4, 3) == generation_seeds(7, 4, 3)
    assert generation_seeds(7, 4, 3) != generation_seeds(7, 5, 3)


def test_run_directory_contents(run_dir):
    for name in (
        "config.yaml",
        "neat.ini",
        "metadata.json",
        "generations.csv",
        "eval_history.csv",
        "best_model.pkl",
        "best_model.meta.json",
        "fittest_genome.pkl",
        "fitness.png",
        "complexity.png",
        "best_model_network.png",
        "train_summary.json",
    ):
        assert (run_dir / name).is_file(), name
    assert sorted(p.name for p in (run_dir / "champions").glob("*.pkl")) == [
        "gen_0000.pkl",
        "gen_0001.pkl",
    ]
    assert any((run_dir / "checkpoints").iterdir())


def test_env_steps_are_counted(run_dir):
    rows = history(run_dir)
    assert len(rows) == 2
    assert all(int(r[1]) > 0 for r in rows)


def test_same_seed_gives_identical_evolution(run_dir, tmp_path):
    again = train_neat(TINY, runs_root=tmp_path, log=lambda _: None)
    assert history(again) == history(run_dir)


def test_champion_loads_through_the_shared_loader_and_acts(run_dir):
    controller = load_controller(run_dir / "best_model.pkl")
    assert isinstance(controller, NEATController) and controller.kind == "neat"
    env = DrivingEnv()
    check_compatible(controller.meta, env)
    obs, _ = env.reset(seed=0)
    a = controller.act(obs)
    assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)
    np.testing.assert_array_equal(a, controller.act(obs))


def test_neat_model_refused_in_mismatched_env(run_dir):
    controller = load_controller(run_dir / "best_model.pkl")
    env = DrivingEnv(EnvConfig(observation=ObservationConfig(route_features=True)))
    with pytest.raises(ModelError, match="observation mismatch"):
        check_compatible(controller.meta, env)


def test_ini_must_match_the_observation_size(tmp_path):
    exp = TINY.model_copy(
        update={"env": EnvConfig(observation=ObservationConfig(route_features=True))}
    )
    with pytest.raises(ValueError, match="num_inputs"):
        train_neat(exp, runs_root=tmp_path, log=lambda _: None)


def test_resume_continues_counting(run_dir):
    ckpt = sorted((run_dir / "checkpoints").iterdir())[-1]
    exp = TINY.model_copy(update={"neat": TINY.neat.model_copy(update={"generations": 3})})
    same = train_neat(exp, resume_from=ckpt, log=lambda _: None)
    assert same == run_dir
    rows = history(run_dir)
    assert [r[0] for r in rows][-1] == "2"


def test_show_network_cli(run_dir, tmp_path, capsys):
    from neurodrive.cli import main

    out = tmp_path / "net.png"
    assert main(["show-network", str(run_dir / "best_model.pkl"), "--out", str(out)]) == 0
    assert out.stat().st_size > 1000
