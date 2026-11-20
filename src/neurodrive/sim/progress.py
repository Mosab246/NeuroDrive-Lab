"""Track progress, ordered checkpoints and lap counting.

Progress is the arc length of the car's projection onto the centerline. Internally we
keep an *unwrapped* distance ``distance`` that starts at 0 on the start line and
changes only by the small, signed movement along the track each update:

* forward movement increases it, backward movement decreases it by the same amount,
* crossing the start/finish line does not cause a jump (the delta is wrapped),
* the windowed projection means it can only move continuously.

Checkpoint ``k`` (1-based) of the current lap is passed the first time
``distance - lap_base`` reaches ``k / K`` of the lap length. The last checkpoint is the
finish line. Because ``distance`` is continuous and each checkpoint must be reached in
order, a lap cannot be completed by skipping part of the circuit, by reversing over the
line, or by rocking back and forth over a gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurodrive.sim.geometry import wrap_delta
from neurodrive.sim.track import Track


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    delta: float  # signed meters along the track since the previous update
    checkpoints_passed: int  # new checkpoints passed during this update
    lap_completed: bool


class ProgressTracker:
    """Follows one car around one track."""

    def __init__(self, track: Track) -> None:
        self.track = track
        self.segment = 0
        self.s = 0.0  # wrapped arc length in [0, length)
        self.lateral = 0.0
        self.distance = 0.0  # unwrapped distance from the start line
        self.lap_base = 0.0  # distance at which the current lap started
        self.laps = 0
        self.next_checkpoint = 1  # 1..K

    def reset(self, position: np.ndarray) -> None:
        """Start tracking from ``position`` (searched globally, not windowed)."""
        proj = self.track.project(np.asarray(position, dtype=np.float64)[None, :])
        self.segment = int(proj.segment[0])
        self.s = float(proj.s[0])
        self.lateral = float(proj.lateral[0])
        # A start slightly behind the line counts as a small negative distance.
        self.distance = wrap_delta(self.s, self.track.length)
        self.lap_base = 0.0
        self.laps = 0
        self.next_checkpoint = 1

    @property
    def lap_fraction(self) -> float:
        """Progress through the current lap, 0..1 (negative if behind the line)."""
        return (self.distance - self.lap_base) / self.track.length

    @property
    def window(self) -> int:
        return self.track.index.window

    def update(self, position: np.ndarray) -> ProgressUpdate:
        """Project ``position`` (windowed around the last segment) and update progress."""
        proj = self.track.project(np.asarray(position, dtype=np.float64), hint=self.segment)
        return self.apply(float(proj.s[0]), int(proj.segment[0]), float(proj.lateral[0]))

    def apply(self, s: float, segment: int, lateral: float) -> ProgressUpdate:
        """Update from an already computed projection (lets the simulation project the
        car's center and corners in a single call)."""
        delta = wrap_delta(s - self.s, self.track.length)
        self.segment = segment
        self.s = s
        self.lateral = lateral
        self.distance += delta

        passed = 0
        lap_completed = False
        k_total = len(self.track.checkpoint_s)
        while self.distance - self.lap_base >= self.track.checkpoint_s[self.next_checkpoint - 1]:
            passed += 1
            if self.next_checkpoint == k_total:
                self.laps += 1
                self.lap_base += self.track.length
                self.next_checkpoint = 1
                lap_completed = True
                break  # at most one lap per update
            self.next_checkpoint += 1
        return ProgressUpdate(delta=delta, checkpoints_passed=passed, lap_completed=lap_completed)
