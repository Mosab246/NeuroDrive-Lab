"""Procedural validation tracks and re-selecting the best checkpoint after training."""

import csv

import pytest

from neurodrive.agents.model_io import load_controller, read_model_meta
from neurodrive.benchmark.reselect import list_checkpoints, reselect_runs
from neurodrive.cli import expand_tracks
from neurodrive.experiments.splits import SplitError
from neurodrive.sim.track import load_any_track
from neurodrive.training.config import NEATExperiment, NEATSettings
from neurodrive.training.neat_trainer import train_neat


def test_validation_tracks_are_their_own_split():
    assert expand_tracks("pval:2") == ["pval_0", "pval_1"]
    val, test = load_any_track("pval_0"), load_any_track("proc_0")
    assert val.spec.split == "validation" and test.spec.split == "test"
    assert val.id == "pval_0" and abs(val.length - test.length) > 1.0


def test_test_tracks_did_not_change():
    # proc_3 was 551 m long before validation tracks existed; test sets must stay fixed.
    assert round(load_any_track("proc_3").length) == 551


@pytest.fixture(scope="module")
def neat_run(tmp_path_factory):
    exp = NEATExperiment(
        name="tiny",
        neat=NEATSettings(pop_size=8, generations=3, scenarios_per_genome=1, validate_every=1,
                          validation_episodes=1, checkpoint_every=1, workers=0),
    )  # fmt: skip
    return train_neat(exp, runs_root=tmp_path_factory.mktemp("neat"), log=lambda _: None)


def test_reselect_scores_every_checkpoint_and_saves_a_new_best(neat_run):
    checkpoints = list_checkpoints(neat_run)
    assert len(checkpoints) == 3
    reselect_runs(
        [neat_run], ["beginner_oval", "pval_0"], episodes=1, workers=1, log=lambda _: None
    )
    with (neat_run / "reselection.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    model = neat_run / "best_model_v2.pkl"
    meta = read_model_meta(model)
    assert meta["label"].startswith("reselected@")
    assert meta["reselection"]["tracks"] == ["beginner_oval", "pval_0"]
    assert load_controller(model).kind == "neat"


def test_reselect_refuses_test_tracks(neat_run):
    with pytest.raises(SplitError):
        reselect_runs([neat_run], ["unseen_a"], episodes=1, workers=1, log=lambda _: None)
