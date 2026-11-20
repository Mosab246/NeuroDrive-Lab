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
