from neurodrive.cli import main


def test_list_and_validate(capsys):
    assert main(["list-tracks"]) == 0
    assert "beginner_oval" in capsys.readouterr().out
    assert main(["validate-track", "beginner_oval"]) == 0
    assert "OK" in capsys.readouterr().out


def test_bad_track_gives_one_line_error(capsys):
    assert main(["validate-track", "nope"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: unknown track id") and "Traceback" not in err


def test_bench_runs_small(capsys):
    assert main(["bench-sim", "--steps", "300"]) == 0
    assert "steps/s" in capsys.readouterr().out


def test_bench_rejects_zero_steps(capsys):
    assert main(["bench-sim", "--steps", "0"]) == 2


def test_evaluate_command(tmp_path, capsys):
    assert (
        main(["evaluate", "--controller", "heuristic", "--episodes", "2", "--out", str(tmp_path)])
        == 0
    )
    out = capsys.readouterr().out
    assert "completed" in out and (tmp_path / "summary.json").exists()


def test_evaluate_bad_config_is_one_line(tmp_path, capsys):
    bad = tmp_path / "env.yaml"
    bad.write_text("env:\n  max_episode_seconds: -5\n", encoding="utf-8")
    assert main(["evaluate", "--controller", "random", "--config", str(bad)]) == 2
    assert "max_episode_seconds" in capsys.readouterr().err


def test_train_config_errors_are_one_line(tmp_path, capsys):
    bad = tmp_path / "ppo.yaml"
    bad.write_text("training:\n  n_envs: 0\n", encoding="utf-8")
    assert main(["train", "ppo", "--config", str(bad)]) == 2
    assert "n_envs" in capsys.readouterr().err


def test_shipped_ppo_configs_are_valid():
    from neurodrive.training.config import load_neat_config, load_ppo_config

    for name in ("default", "smoke", "multitrack"):
        load_ppo_config(f"configs/ppo/{name}.yaml")
        load_neat_config(f"configs/neat/{name}.yaml")
