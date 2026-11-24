"""Per-episode records and honest summary statistics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeRecord:
    controller: str
    controller_kind: str
    track: str
    scenario_seed: int
    completed: bool
    termination_reason: str
    lap_time: float | None  # seconds, only when completed
    lap_fraction: float  # progress through the lap when the episode ended (1.0 = complete)
    steps: int
    sim_time: float
    mean_speed: float  # m/s over the episode
    episode_return: float

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a success rate. Better than +-1.96*sqrt(p(1-p)/n) when
    the rate is near 0% or 100% or n is small (it never goes below 0 or above 1)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(records: list[EpisodeRecord]) -> dict[str, object]:
    """Aggregate records. Lap-time statistics are only over completed laps, and always
    reported next to how many laps that is: '22.9 s over 48 of 50' is honest,
    '22.9 s' alone is not."""
    n = len(records)
    if n == 0:
        raise ValueError("no episodes to summarize")
    done = [r for r in records if r.completed]
    lo, hi = wilson_interval(len(done), n)
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r.termination_reason] = reasons.get(r.termination_reason, 0) + 1
    lap_times = np.array([r.lap_time for r in done if r.lap_time is not None])
    return {
        "controller": records[0].controller,
        "controller_kind": records[0].controller_kind,
        "track": records[0].track,
        "episodes": n,
        "completed": len(done),
        "completion_rate": len(done) / n,
        "completion_rate_ci95": [lo, hi],
        "collision_rate": reasons.get("collision", 0) / n,
        "termination_reasons": reasons,
        "lap_time_mean": float(lap_times.mean()) if lap_times.size else None,
        "lap_time_std": float(lap_times.std(ddof=1)) if lap_times.size > 1 else None,
        "lap_time_best": float(lap_times.min()) if lap_times.size else None,
        "lap_time_n": int(lap_times.size),
        "mean_lap_fraction": float(np.mean([min(r.lap_fraction, 1.0) for r in records])),
        "mean_episode_steps": float(np.mean([r.steps for r in records])),
        "mean_speed": float(np.mean([r.mean_speed for r in records])),
        "mean_return": float(np.mean([r.episode_return for r in records])),
    }


def selection_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """How checkpoints are ranked on validation: most laps completed, then most progress,
    then the fastest lap.

    Not episode return: a completed lap is always worth 0.1 * length + bonus however long
    it took, so return can't tell a 30 s lap from a 20 s one. (PPO still learns to go
    faster because it discounts future reward with gamma; the undiscounted return doesn't.)
    Works on summary dicts and on CSV rows (strings; empty lap time = no laps).
    """
    lap = row.get("lap_time_mean")
    lap_time = float(lap) if lap not in (None, "") else float("inf")
    return (float(row["completion_rate"]), float(row["mean_lap_fraction"]), -lap_time)
