import json

import pytest

from neurodrive.agents.base import Controller
from neurodrive.agents.baselines import (
    HeuristicController,
    PrivilegedAutopilot,
    RandomController,
    SmoothedRandomController,
)
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.evaluation.evaluator import evaluate, write_results
from neurodrive.evaluation.metrics import EpisodeRecord, summarize, wilson_interval
from neurodrive.evaluation.scenarios import scenario_seeds


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.722, abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.278, abs=1e-3)
    lo, hi = wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.404, 0.596), abs=1e-3)


def test_scenario_splits_are_disjoint_and_fixed():
    v = scenario_seeds("validation", 100)
    t = scenario_seeds("test", 100)
    assert not set(v) & set(t)
    assert v == scenario_seeds("validation", 100)
    with pytest.raises(ValueError):
        scenario_seeds("train", 5)


def rec(completed, lap_time=None, reason="lap_complete", frac=1.0):
    return EpisodeRecord(
        "c", "heuristic", "t", 0, completed, reason, lap_time, frac, 10, 1.0, 1.0, 0.0
    )


def test_lap_time_only_over_completed_and_reported_with_count():
    s = summarize([rec(True, 20.0), rec(True, 22.0), rec(False, None, "collision", 0.4)])
    assert s["lap_time_mean"] == pytest.approx(21.0)
    assert s["lap_time_n"] == 2 and s["episodes"] == 3
    assert s["completion_rate"] == pytest.approx(2 / 3)
    assert s["collision_rate"] == pytest.approx(1 / 3)
    assert s["mean_lap_fraction"] == pytest.approx((1 + 1 + 0.4) / 3)


def test_no_completions_gives_no_lap_time():
    s = summarize([rec(False, None, "stalled", 0.1)])
    assert s["lap_time_mean"] is None and s["completed"] == 0


@pytest.mark.parametrize(
    "make",
    [
        lambda env: RandomController(0),
        lambda env: SmoothedRandomController(0),
        lambda env: HeuristicController(),
        lambda env: PrivilegedAutopilot(env),
    ],
)
def test_all_baselines_satisfy_the_controller_protocol(make):
    env = DrivingEnv()
    c = make(env)
    assert isinstance(c, Controller)
    obs, _ = env.reset(seed=0)
    a = c.act(obs)
    assert a.shape == (2,) and abs(a).max() <= 1.0


def test_heuristic_uses_only_the_observation():
    """Same observation in, same action out, with no access to the simulator."""
    import numpy as np

    h = HeuristicController()
    obs = np.array([0.1, 0.2, 0.3, 0.5, 1.0, 0.5, 0.3, 0.2, 0.1, 0.4, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(h.act(obs), h.act(obs.copy()))
    # Symmetric surroundings -> no steering.
    assert h.act(obs)[0] == pytest.approx(0.0, abs=1e-7)


def test_evaluate_writes_csv_and_summary(tmp_path):
    env = DrivingEnv()
    records = evaluate(env, HeuristicController(), scenario_seeds("validation", 3))
    summary = write_results(records, tmp_path, {"note": "test"})
    lines = (tmp_path / "episodes.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # header + 3 episodes
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["summary"]["episodes"] == 3 == summary["episodes"]
    assert payload["metadata"] == {"note": "test"}


def test_random_controller_is_seeded():
    import numpy as np

    a, b = RandomController(1), RandomController(1)
    obs = np.zeros(11, dtype=np.float32)
    assert np.array_equal(a.act(obs), b.act(obs))
