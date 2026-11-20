"""Headless simulation throughput benchmark.

Measures how many decision steps per second ``Simulation.step`` sustains with no window.
Two modes are timed separately so the numbers mean something:

* ``sim``: the simulation alone, fed a fixed action sequence (what training will pay for
  per step, before observations, rewards and the policy network are added).
* ``sim+autopilot``: the simulation driven by the scripted pure-pursuit driver, which
  keeps the car on track for whole laps (closer to a realistic episode).
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass

import numpy as np

from neurodrive.agents.pure_pursuit import PurePursuitDriver
from neurodrive.sim.simulation import SimConfig, Simulation
from neurodrive.sim.track import Track


@dataclass(frozen=True)
class BenchResult:
    mode: str
    steps: int
    seconds: float
    resets: int

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds


def bench(track: Track, steps: int, seed: int = 0) -> list[BenchResult]:
    results = []

    sim = Simulation(track, config=SimConfig(), seed=seed)
    actions = np.random.default_rng(seed).uniform(-0.3, 1.0, size=(steps, 2))
    resets = 0
    t0 = time.perf_counter()
    for a in actions:
        if sim.done:
            sim.reset()
            resets += 1
        sim.step(a)
    results.append(BenchResult("sim (random actions)", steps, time.perf_counter() - t0, resets))

    sim = Simulation(track, config=SimConfig(laps_to_finish=3), seed=seed)
    driver = PurePursuitDriver()
    resets = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        if sim.done:
            sim.reset()
            resets += 1
        sim.step(driver.act(sim))
    results.append(BenchResult("sim+autopilot", steps, time.perf_counter() - t0, resets))
    return results


def environment_summary() -> str:
    return (
        f"Python {platform.python_version()} | NumPy {np.__version__} | "
        f"{platform.system()} {platform.release()} | {platform.processor() or platform.machine()}"
    )
