"""Runs any controller on a fixed list of scenarios and records every episode."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from neurodrive.agents.base import Controller
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.evaluation.metrics import EpisodeRecord, summarize


def run_episode(env: DrivingEnv, controller: Controller, seed: int) -> EpisodeRecord:
    obs, info = env.reset(seed=seed, options={"perturb": True})
    controller.reset(seed)
    speed_sum = 0.0
    while True:
        obs, _, terminated, truncated, info = env.step(controller.act(obs))
        speed_sum += info["speed"]
        if terminated or truncated:
            break
    completed = info["termination_reason"] == "lap_complete"
    return EpisodeRecord(
        controller=controller.name,
        controller_kind=controller.kind,
        track=env.track.id,
        scenario_seed=seed,
        completed=completed,
        termination_reason=info["termination_reason"],
        lap_time=float(info["lap_times"][0]) if completed else None,
        lap_fraction=float(info["lap_fraction"]) if not completed else 1.0,
        steps=int(info["steps"]),
        sim_time=float(info["time"]),
        mean_speed=speed_sum / max(1, int(info["steps"])),
        episode_return=float(info["episode_return"]),
    )


def evaluate(
    env: DrivingEnv,
    controller: Controller,
    seeds: Sequence[int],
    progress: Callable[[int, int], None] | None = None,
) -> list[EpisodeRecord]:
    records = []
    for i, seed in enumerate(seeds):
        records.append(run_episode(env, controller, seed))
        if progress is not None:
            progress(i + 1, len(seeds))
    return records


def write_results(records: list[EpisodeRecord], out_dir: Path, metadata: dict[str, object]) -> dict:
    """Write ``episodes.csv`` (one row per episode) and ``summary.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in records]
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(records)
    payload = {"summary": summary, "metadata": metadata}
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    return summary


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
