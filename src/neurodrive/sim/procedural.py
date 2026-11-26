"""Track shapes from a formula: hand-designed circuits and random test circuits.

Every shape here is a *radial* curve: for each angle theta around a center, the
centerline is at distance

    r(theta) = R * (1 + sum_k a_k * sin(k * theta + phase_k))

from the center. Because r is positive at every angle, the curve can never cross itself:
it's "star-shaped". The bumps in r make bends; larger amplitudes and higher k make tighter,
more technical corners. Every generated track still goes through the normal validation
(turn radius vs width, edges not crossing), so an undrivable shape is rejected, never
silently used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neurodrive.sim.track import Track, TrackError, TrackSpec, build_track

PROCEDURAL_PREFIX = "proc_"
#: Procedural *validation* tracks ``pval_<n>``: same generator, a disjoint seed range, so they
#: never coincide with a test track ``proc_<seed>``.
VALIDATION_PREFIX = "pval_"
VALIDATION_SEED_OFFSET = 1_000_000


@dataclass(frozen=True)
class RadialShape:
    radius: float  # meters
    terms: tuple[tuple[int, float, float], ...]  # (k, amplitude, phase)
    clockwise: bool = False

    def points(self, n: int = 48) -> list[tuple[float, float]]:
        theta = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        r = np.ones_like(theta)
        for k, amp, phase in self.terms:
            r += amp * np.sin(k * theta + phase)
        r *= self.radius
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        if self.clockwise:
            pts = pts[::-1]
            pts = np.roll(pts, 1, axis=0)  # keep theta = 0 as the start line
        return [(round(float(x), 3), round(float(y), 3)) for x, y in pts]


def shape_spec(
    track_id: str,
    name: str,
    shape: RadialShape,
    width: float,
    split: str,
    difficulty: str,
    n_points: int = 48,
) -> TrackSpec:
    return TrackSpec(
        id=track_id,
        name=name,
        difficulty=difficulty,
        split=split,
        width=width,
        control_points=shape.points(n_points),
    )


def random_shape(rng: np.random.Generator, difficulty: float = 1.0) -> RadialShape:
    """A random radial shape. ``difficulty`` scales how wiggly it is (0.5 easy .. 1.5 hard)."""
    radius = float(rng.uniform(70.0, 100.0))
    terms = []
    for k in (2, 3, 4, 5):
        amp = float(rng.uniform(0.0, 0.16)) * difficulty / math.sqrt(k - 1)
        terms.append((k, amp, float(rng.uniform(0, 2 * math.pi))))
    return RadialShape(radius, tuple(terms), clockwise=bool(rng.integers(2)))


def procedural_track(
    seed: int, difficulty: float = 1.0, width: float = 11.0, validation: bool = False
) -> Track:
    """Deterministic random test track ``proc_<seed>``.

    If a draw is undrivable it is redrawn with gentler bends (up to 20 tries), so a given
    seed always gives the same valid track.
    """
    if seed < 0:
        raise ValueError("procedural track seed must be >= 0")
    prefix, split = (VALIDATION_PREFIX, "validation") if validation else (PROCEDURAL_PREFIX, "test")
    draw_seed = seed + VALIDATION_SEED_OFFSET if validation else seed
    rng = np.random.default_rng([draw_seed, 7919])
    scale = difficulty
    for _ in range(20):
        shape = random_shape(rng, scale)
        spec = shape_spec(
            f"{prefix}{seed}",
            f"Procedural {'validation ' if validation else ''}#{seed}",
            shape,
            width=width,
            split=split,
            difficulty=f"procedural x{difficulty:g}",
        )
        try:
            return build_track(spec)
        except TrackError:
            scale *= 0.85
    raise TrackError(f"could not generate a valid procedural track for seed {seed}")
