"""The reward function, kept separate from the environment so it can be read and tested alone.

Per step::

    reward = progress_per_meter * (signed meters of progress this step)
           + lap_bonus       if the lap was completed this step
           - crash_penalty   if the car hit the wall this step
           - stall_penalty   if the car made too little progress recently

Design notes (see docs/ml-guide.md for the long version):

* Progress is **signed**: driving backwards pays back exactly what driving forwards
  earned, so oscillating gains nothing.
* There is **no per-step time penalty**. Faster driving already earns progress sooner.
  A time penalty would make "crash early to stop the bleeding" attractive whenever the
  crash penalty is too small, which is a classic way to teach an agent to suicide.
* There are **no checkpoint bonuses**: checkpoints only validate laps (in the simulator).
"""

from __future__ import annotations

from dataclasses import dataclass

from neurodrive.env.config import RewardConfig
from neurodrive.sim.simulation import StepResult


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    progress: float
    lap_bonus: float
    crash: float
    stall: float

    @property
    def total(self) -> float:
        return self.progress + self.lap_bonus + self.crash + self.stall


def compute_reward(result: StepResult, stalled: bool, cfg: RewardConfig) -> RewardBreakdown:
    return RewardBreakdown(
        progress=cfg.progress_per_meter * result.progress_delta,
        lap_bonus=cfg.lap_bonus if result.lap_completed else 0.0,
        crash=-cfg.crash_penalty if result.collided else 0.0,
        stall=-cfg.stall_penalty if stalled else 0.0,
    )
