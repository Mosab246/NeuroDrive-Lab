"""The simulation loop: one car on one track, advanced in fixed time steps.

``Simulation`` knows nothing about rendering or learning. It takes an action
``[steering, throttle/brake]`` each step and reports what happened. It never reads the
wall clock, so the same seed and the same actions always produce exactly the same
trajectory, whether it runs headless or behind a window at any playback speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from neurodrive.sim.progress import ProgressTracker
from neurodrive.sim.track import Track
from neurodrive.sim.vehicle import (
    VehicleParams,
    VehicleState,
    body_center,
    body_corners,
    clamp_action,
    step_vehicle,
)


class Termination(StrEnum):
    COLLISION = "collision"
    LAPS_COMPLETE = "laps_complete"
    TIME_LIMIT = "time_limit"


@dataclass(frozen=True, slots=True)
class SimConfig:
    """How the simulation is stepped and when a run ends."""

    dt: float = 1.0 / 30.0  # seconds per decision step
    substeps: int = 2  # physics sub-steps per decision step (prevents wall tunnelling)
    max_time: float | None = None  # seconds; None = no limit
    laps_to_finish: int | None = None  # None = drive laps forever
    start_lateral_jitter: float = 1.5  # meters, used only when reset(perturb=True)
    start_heading_jitter: float = math.radians(5.0)  # radians, likewise

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.substeps < 1:
            raise ValueError("dt must be positive and substeps >= 1")
        if self.max_time is not None and self.max_time <= 0:
            raise ValueError("max_time must be positive or None")
        if self.laps_to_finish is not None and self.laps_to_finish < 1:
            raise ValueError("laps_to_finish must be >= 1 or None")


@dataclass(frozen=True, slots=True)
class StepResult:
    """What happened during one decision step."""

    progress_delta: float  # signed meters along the track
    checkpoints_passed: int
    lap_completed: bool
    collided: bool
    termination: Termination | None
    time: float  # simulated seconds since reset, at the end of this step


@dataclass
class _Run:
    step_count: int = 0
    substep_count: int = 0
    lap_start_substep: int = 0
    lap_times: list[float] = field(default_factory=list)
    termination: Termination | None = None


class Simulation:
    """A single-car simulation with deterministic, fixed-step physics."""

    def __init__(
        self,
        track: Track,
        vehicle: VehicleParams | None = None,
        config: SimConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.track = track
        self.params = vehicle or VehicleParams()
        self.config = config or SimConfig()
        self._rng = np.random.default_rng(seed)
        self.progress = ProgressTracker(track)
        self.state = VehicleState(0.0, 0.0, 0.0)
        self._run = _Run()
        self.reset()

    # ------------------------------------------------------------------ properties
    @property
    def sub_dt(self) -> float:
        return self.config.dt / self.config.substeps

    @property
    def time(self) -> float:
        """Simulated seconds since reset (computed from the step count, never summed)."""
        return self._run.substep_count * self.sub_dt

    @property
    def step_count(self) -> int:
        return self._run.step_count

    @property
    def done(self) -> bool:
        return self._run.termination is not None

    @property
    def termination(self) -> Termination | None:
        return self._run.termination

    @property
    def lap_times(self) -> list[float]:
        return list(self._run.lap_times)

    @property
    def current_lap_time(self) -> float:
        return (self._run.substep_count - self._run.lap_start_substep) * self.sub_dt

    # ------------------------------------------------------------------ control
    def reset(self, seed: int | None = None, perturb: bool = False) -> None:
        """Put the car on the start line, stationary.

        Args:
            seed: reseeds this simulation's random generator (only used for perturbation).
            perturb: shift the start sideways and rotate it slightly, using the generator.
                This is how evaluation gets genuinely different scenarios from a
                deterministic simulator.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        x, y, heading = self.track.start_pose()
        if perturb:
            lat = self._rng.uniform(-1.0, 1.0) * self.config.start_lateral_jitter
            dh = self._rng.uniform(-1.0, 1.0) * self.config.start_heading_jitter
            nx, ny = self.track.normals[0]
            x, y, heading = x + lat * nx, y + lat * ny, heading + dh
        # Place the *body center* on the start point; the state stores the rear axle.
        half = 0.5 * self.params.wheelbase
        self.state = VehicleState(
            x=x - half * math.cos(heading), y=y - half * math.sin(heading), heading=heading
        )
        self.progress.reset(np.array(body_center(self.state, self.params)))
        self._run = _Run()

    def step(self, action: np.ndarray | tuple[float, float]) -> StepResult:
        """Advance one decision step (``config.dt`` seconds) with the given action."""
        if self.done:
            raise RuntimeError(
                f"simulation already ended ({self._run.termination}); call reset() first"
            )
        steering, longitudinal = clamp_action(action)
        run = self._run
        delta = 0.0
        passed = 0
        lap_completed = False
        collided = False
        for _ in range(self.config.substeps):
            step_vehicle(self.state, steering, longitudinal, self.params, self.sub_dt)
            run.substep_count += 1
            # One projection for the body center (progress) and the 4 corners (collision).
            corners = body_corners(self.state, self.params)
            points = np.vstack([corners.mean(axis=0), corners])
            proj = self.track.project(points, hint=self.progress.segment)
            update = self.progress.apply(
                float(proj.s[0]), int(proj.segment[0]), float(proj.lateral[0])
            )
            delta += update.delta
            passed += update.checkpoints_passed
            if update.lap_completed:
                lap_completed = True
                run.lap_times.append((run.substep_count - run.lap_start_substep) * self.sub_dt)
                run.lap_start_substep = run.substep_count
                limit = self.config.laps_to_finish
                if limit is not None and self.progress.laps >= limit:
                    run.termination = Termination.LAPS_COMPLETE
                    break
            if np.any(np.abs(proj.lateral[1:]) > self.track.half_width[proj.segment[1:]]):
                collided = True
                run.termination = Termination.COLLISION
                break
        run.step_count += 1
        if (
            run.termination is None
            and self.config.max_time is not None
            and self.time >= self.config.max_time - 1e-9
        ):
            run.termination = Termination.TIME_LIMIT
        return StepResult(
            progress_delta=delta,
            checkpoints_passed=passed,
            lap_completed=lap_completed,
            collided=collided,
            termination=run.termination,
            time=self.time,
        )

    # ------------------------------------------------------------------ queries
    def corners(self) -> np.ndarray:
        return body_corners(self.state, self.params)

    def is_colliding(self) -> bool:
        """True if any corner of the car is outside the drivable corridor."""
        proj = self.track.project(self.corners(), hint=self.progress.segment)
        return bool(np.any(np.abs(proj.lateral) > self.track.half_width[proj.segment]))
