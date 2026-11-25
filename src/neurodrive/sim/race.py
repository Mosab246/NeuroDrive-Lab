"""Several cars on one track at the same time: a real race, not ghosts.

Each car is an ordinary :class:`Simulation` (its own physics, progress and laps). The race
adds what happens *between* cars:

* **Starting grid.** Cars line up behind the start/finish line in pairs, two abreast,
  ``grid_spacing`` meters apart. Everyone's laps count from the same line, so the cars at
  the back simply have a few more meters to drive, as in real racing.
* **Contact.** After every step, overlapping cars (checked exactly, rectangle against
  rectangle) are pushed apart and both lose speed. Contact does not end a car's race by
  itself, but a push can shove a car into the wall, which does.
* **Stalling.** A car that gains less than ``stall_min_progress`` meters in
  ``stall_seconds`` is out (same rule as the training environment).

Cars that finish, crash or stall leave the track and stop being obstacles. What each car
*sees* of the others is handled by the sensors (``RaySensor.cast(obstacles=...)``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from neurodrive.sim.simulation import Simulation, StepResult

CONTACT_SPEED_FACTOR = 0.6  # both cars keep this share of their speed after contact


@dataclass
class CarStatus:
    index: int
    outcome: str | None = None  # "finished", "collision", "stalled", "time_limit"
    finish_time: float | None = None
    contacts: int = 0  # separate hits (a touch that lasts several steps counts once)
    lap_times: list[float] = field(default_factory=list)

    @property
    def racing(self) -> bool:
        return self.outcome is None


def rectangle_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """If convex quadrilaterals ``a`` and ``b`` (corners, shape (4, 2)) overlap, return the
    shortest vector that moves ``a`` out of ``b`` (separating-axis test); else None."""
    best_depth, best_axis = np.inf, None
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for ex, ey in edges[:2]:  # a rectangle has two distinct edge directions
            axis = np.array([-ey, ex]) / np.hypot(ex, ey)
            pa, pb = a @ axis, b @ axis
            depth = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if depth <= 0:
                return None  # a separating axis: no overlap
            if depth < best_depth:
                best_depth, best_axis = depth, axis
    assert best_axis is not None
    direction = 1.0 if (a.mean(axis=0) - b.mean(axis=0)) @ best_axis >= 0 else -1.0
    return best_axis * best_depth * direction


class RaceWorld:
    def __init__(
        self,
        cars: list[Simulation],
        grid_spacing: float = 9.0,
        grid_lateral: float = 2.6,
        stall_seconds: float = 5.0,
        stall_min_progress: float = 2.0,
        stall_exempt: tuple[int, ...] = (),
    ) -> None:
        """``cars``: one simulation per car, all on the same track (each car keeps its own
        vehicle parameters and episode rules, e.g. from the environment it was trained in)."""
        if not cars:
            raise ValueError("a race needs at least one car")
        if len({sim.track.id for sim in cars}) > 1:
            raise ValueError("all cars in a race must be on the same track")
        if len({sim.params.model for sim in cars}) > 1:
            raise ValueError("all cars in a race must use the same physics model")
        if len({sim.config.dt for sim in cars}) > 1:
            raise ValueError("all cars must use the same time step")
        track = cars[0].track
        self.track = track
        self.cars = cars
        self.dt = cars[0].config.dt
        # Two cars can only touch if their centres are closer than one car diagonal.
        self._reach2 = max(s.params.length**2 + s.params.width**2 for s in cars) * 1.01
        self.grid_spacing = grid_spacing
        self.grid_lateral = min(grid_lateral, 0.5 * float(track.half_width.min()))
        self._stall_steps = max(1, round(stall_seconds / self.dt))
        self.stall_min_progress = stall_min_progress
        self.stall_exempt = set(stall_exempt)
        self.last_results: list[StepResult | None] = [None] * len(cars)
        self.new_hits: list[int] = [0] * len(cars)
        self.status: list[CarStatus] = []
        self._touching: set[tuple[int, int]] = set()
        self._history: list[deque[float]] = []
        self.time = 0.0

    def reset(self) -> None:
        """Line the cars up on the starting grid and start the race."""
        for i, sim in enumerate(self.cars):
            row, side = divmod(i, 2)
            s = -(row + 0.5) * self.grid_spacing - 0.5 * side * self.grid_spacing
            lateral = self.grid_lateral if side == 0 else -self.grid_lateral
            sim.place(s, lateral)
        self.start()

    def start(self) -> None:
        """Start racing from wherever the cars have been placed."""
        self.status = [CarStatus(i) for i in range(len(self.cars))]
        self._touching = set()
        self.time = 0.0
        self.last_results = [None] * len(self.cars)
        self.new_hits = [0] * len(self.cars)
        self._history = [
            deque([sim.progress.distance], maxlen=self._stall_steps + 1) for sim in self.cars
        ]

    @property
    def done(self) -> bool:
        return not any(st.racing for st in self.status)

    def obstacles_for(self, i: int) -> list[np.ndarray]:
        """Corners of every other car still on track."""
        return [
            sim.corners() for j, sim in enumerate(self.cars) if j != i and self.status[j].racing
        ]

    def step(self, actions: list[np.ndarray | None]) -> list[int]:
        """Advance every racing car one step (``None`` for cars no longer racing), then
        resolve contact. Returns the indices of cars that touched another car this step."""
        for i, (sim, action) in enumerate(zip(self.cars, actions, strict=True)):
            st = self.status[i]
            if not st.racing:
                continue
            if action is None:
                raise ValueError(f"car {i} is still racing and needs an action")
            self.last_results[i] = sim.step(action)
        self.time += self.dt
        touched = self._resolve_contacts()
        for i, sim in enumerate(self.cars):
            st = self.status[i]
            if not st.racing:
                continue
            hist = self._history[i]
            hist.append(sim.progress.distance)
            if sim.termination is not None:
                term = str(sim.termination)
                st.outcome = "finished" if term == "laps_complete" else term
                st.lap_times = list(sim.lap_times)
                if st.outcome == "finished":
                    st.finish_time = sum(sim.lap_times)
            elif (
                i not in self.stall_exempt
                and len(hist) > self._stall_steps
                and hist[-1] - hist[0] < self.stall_min_progress
            ):
                st.outcome = "stalled"
        return touched

    def _resolve_contacts(self) -> list[int]:
        racing = [i for i, st in enumerate(self.status) if st.racing]
        corners = {i: self.cars[i].corners() for i in racing}
        centers = {i: c.mean(axis=0) for i, c in corners.items()}
        touched: set[int] = set()
        pairs: set[tuple[int, int]] = set()
        for n, i in enumerate(racing):
            for j in racing[n + 1 :]:
                gap = centers[i] - centers[j]
                if gap @ gap > self._reach2:  # too far apart to touch: skip the exact test
                    continue
                push = rectangle_overlap(corners[i], corners[j])
                if push is None:
                    continue
                for k, sign in ((i, 0.5), (j, -0.5)):
                    state = self.cars[k].state
                    state.x += sign * push[0]
                    state.y += sign * push[1]
                    state.speed *= CONTACT_SPEED_FACTOR
                    state.lateral_speed *= CONTACT_SPEED_FACTOR
                    state.yaw_rate *= CONTACT_SPEED_FACTOR
                    corners[k] = self.cars[k].corners()
                touched.update((i, j))
                pairs.add((i, j))
        self.new_hits = [0] * len(self.cars)
        for i, j in pairs - self._touching:  # count each hit once, when it starts
            for k in (i, j):
                self.status[k].contacts += 1
                self.new_hits[k] += 1
        self._touching = pairs
        return sorted(touched)

    def standings(self) -> list[int]:
        """Car indices in finishing order: finishers by time, then everyone else by how
        far they got."""

        def key(i: int) -> tuple[int, float]:
            st = self.status[i]
            if st.outcome == "finished" and st.finish_time is not None:
                return (0, st.finish_time)
            return (1, -self.cars[i].progress.distance)

        return sorted(range(len(self.cars)), key=key)
