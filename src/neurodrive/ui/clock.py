"""Decouples simulation time from frame rate.

Each frame, the real time that passed (times the playback speed) is added to an
accumulator, and the simulation advances by as many *whole* fixed steps as fit. The
physics therefore always sees exactly ``dt`` per step: a slow frame rate or a 8x
playback speed changes how many steps run per frame, never what a step does.
"""

from __future__ import annotations

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


class FixedStepClock:
    def __init__(self, dt: float, max_steps_per_frame: int = 64) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self.max_steps_per_frame = max_steps_per_frame
        self.accumulator = 0.0
        self._speed_index = SPEEDS.index(1.0)

    @property
    def speed(self) -> float:
        return SPEEDS[self._speed_index]

    def faster(self) -> None:
        self._speed_index = min(self._speed_index + 1, len(SPEEDS) - 1)

    def slower(self) -> None:
        self._speed_index = max(self._speed_index - 1, 0)

    def set_speed(self, speed: float) -> None:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        self._speed_index = SPEEDS.index(speed)

    def advance(self, real_seconds: float) -> int:
        """Return how many fixed steps to run for a frame that took ``real_seconds``."""
        self.accumulator += max(0.0, real_seconds) * self.speed
        n = int((self.accumulator + 1e-12) // self.dt)
        if n > self.max_steps_per_frame:
            # Can't keep up: drop the backlog instead of spiralling into ever-longer frames.
            n = self.max_steps_per_frame
            self.accumulator = 0.0
        else:
            self.accumulator -= n * self.dt
        return n

    def reset(self) -> None:
        self.accumulator = 0.0
