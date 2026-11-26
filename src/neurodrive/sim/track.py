"""Track representation: a closed centerline plus a width.

A track file lists control points in driving order. Loading a track:

1. fits a closed Catmull-Rom spline through the control points,
2. resamples it at a uniform arc-length spacing (the "centerline"),
3. offsets the centerline left/right by the half-width to get the two edges.

Everything else derives from that single centerline:

* **progress** = arc length ``s`` of the car's projection onto the centerline,
* **collision** = a car corner is further than ``half_width`` from the centerline,
* **checkpoints** = gates at evenly spaced arc lengths,
* the **start/finish line** is at ``s = 0`` (the first control point).

See ``docs/physics.md`` for the reasoning behind this design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from neurodrive.sim import geometry as geo
from neurodrive.sim.geometry import ClosedPolylineIndex, FloatArray, Projection

PROJECTION_WINDOW = 8  # segments searched either side of the previous position


class TrackError(ValueError):
    """A track file is missing, malformed, or describes an undrivable track."""


class TrackSpec(BaseModel):
    """The on-disk YAML schema of a track."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    difficulty: str = "unrated"
    split: str = Field(default="train", pattern=r"^(train|validation|test)$")
    width: float = Field(gt=0, description="Drivable width in meters.")
    sample_spacing: float = Field(default=1.0, gt=0, le=5.0)
    checkpoint_count: int = Field(default=10, ge=2, le=200)
    control_points: list[tuple[float, float]] = Field(min_length=3)

    @field_validator("control_points")
    @classmethod
    def _distinct(cls, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        arr = np.asarray(pts, dtype=float)
        if np.any(geo.closed_segment_lengths(arr) < 1e-6):
            raise ValueError("consecutive control points must be distinct")
        return pts


@dataclass(frozen=True, eq=False)
class Track:
    """An immutable, validated track. Build one with :func:`load_track`."""

    spec: TrackSpec
    centerline: FloatArray  # (N, 2)
    cum_s: FloatArray  # (N + 1,), cum_s[0] = 0, cum_s[N] = length
    tangents: FloatArray  # (N, 2) unit
    normals: FloatArray  # (N, 2) unit, pointing left
    curvature: FloatArray  # (N,) signed, 1/m
    half_width: FloatArray  # (N,)
    left_edge: FloatArray  # (N, 2)
    right_edge: FloatArray  # (N, 2)
    checkpoint_s: FloatArray  # (K,) arc lengths of gates; last one == length (finish line)
    index: ClosedPolylineIndex = field(repr=False)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def length(self) -> float:
        return float(self.cum_s[-1])

    @property
    def num_points(self) -> int:
        return len(self.centerline)

    def start_pose(self) -> tuple[float, float, float]:
        """Position of the start/finish line center and the travel heading there."""
        x, y = self.centerline[0]
        tx, ty = self.tangents[0]
        return float(x), float(y), float(np.arctan2(ty, tx))

    def point_at(self, s: float) -> tuple[FloatArray, FloatArray]:
        """Centerline point and unit tangent at arc length ``s`` (wraps around)."""
        s = float(s) % self.length
        i = int(np.searchsorted(self.cum_s, s, side="right") - 1)
        i = min(i, self.num_points - 1)
        j = (i + 1) % self.num_points
        seg = self.cum_s[i + 1] - self.cum_s[i]
        t = (s - self.cum_s[i]) / seg
        p = self.centerline[i] + t * (self.centerline[j] - self.centerline[i])
        d = self.centerline[j] - self.centerline[i]
        return p, d / np.linalg.norm(d)

    def curvature_at(self, s: float) -> float:
        i = int(np.searchsorted(self.cum_s, float(s) % self.length, side="right") - 1)
        return float(self.curvature[min(i, self.num_points - 1)])

    def max_abs_curvature(self, s_values: FloatArray) -> float:
        """``max(abs(curvature_at(s)) for s in s_values)``, in one vectorized lookup."""
        idx = np.searchsorted(self.cum_s, np.asarray(s_values) % self.length, side="right") - 1
        return float(np.abs(self.curvature[np.minimum(idx, self.num_points - 1)]).max())

    def project(self, points: FloatArray, hint: int | None = None) -> Projection:
        """Project points (``(K, 2)`` or ``(2,)``) onto the centerline.

        With ``hint`` (the previous segment index) only nearby segments are searched;
        see :class:`~neurodrive.sim.geometry.ClosedPolylineIndex`.
        """
        return self.index.project(np.atleast_2d(points), hint)


def build_track(spec: TrackSpec) -> Track:
    """Build geometry from a spec and run validation. Raises :class:`TrackError` if invalid."""
    dense = geo.catmull_rom_closed(np.asarray(spec.control_points, dtype=np.float64))
    # Vertex 0 is the first control point, which defines the start/finish line.
    centerline = geo.resample_closed(dense, spec.sample_spacing)
    seg = geo.closed_segment_lengths(centerline)
    cum_s = np.concatenate([[0.0], np.cumsum(seg)])
    tangents = geo.closed_tangents(centerline)
    normals = geo.left_normals(tangents)
    # Curvature over a ~3 m baseline (see closed_curvature for why).
    curvature = geo.closed_curvature(centerline, stride=max(1, round(3.0 / spec.sample_spacing)))
    half_width = np.full(len(centerline), spec.width / 2.0)
    left = centerline + normals * half_width[:, None]
    right = centerline - normals * half_width[:, None]
    k = spec.checkpoint_count
    checkpoint_s = cum_s[-1] * np.arange(1, k + 1) / k

    problems: list[str] = []
    max_k = float(np.max(np.abs(curvature)))
    if max_k * spec.width / 2.0 >= 0.95:
        problems.append(
            f"tightest turn radius {1.0 / max_k:.2f} m is not larger than the half-width "
            f"{spec.width / 2.0:.2f} m; the inner edge would fold over itself"
        )
    if geo.closed_polyline_self_intersects(centerline):
        problems.append("centerline crosses itself")
    if geo.closed_polyline_self_intersects(left):
        problems.append("left edge crosses itself")
    if geo.closed_polyline_self_intersects(right):
        problems.append("right edge crosses itself")
    if geo.closed_polylines_intersect(left, right):
        problems.append("left and right edges cross each other")
    if np.max(seg) > 2.0 * spec.sample_spacing:
        problems.append("centerline resampling is uneven; add more control points")
    if problems:
        raise TrackError(f"track '{spec.id}' is invalid: " + "; ".join(problems))

    track = Track(
        spec=spec,
        centerline=centerline,
        cum_s=cum_s,
        tangents=tangents,
        normals=normals,
        curvature=curvature,
        half_width=half_width,
        left_edge=left,
        right_edge=right,
        checkpoint_s=checkpoint_s,
        index=ClosedPolylineIndex(centerline, cum_s, PROJECTION_WINDOW),
    )
    return track


def builtin_track_ids() -> list[str]:
    folder = resources.files("neurodrive.tracks")
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".yaml"))


def load_any_track(id_or_path: str | Path) -> Track:
    """Like :func:`load_track`, plus procedural tracks: ``proc_<seed>`` (test split) and
    ``pval_<n>`` (validation split)."""
    name = str(id_or_path)
    if name.startswith("proc_") and name[5:].isdigit():
        from neurodrive.sim.procedural import procedural_track

        return procedural_track(int(name[5:]))
    if name.startswith("pval_") and name[5:].isdigit():
        from neurodrive.sim.procedural import procedural_track

        return procedural_track(int(name[5:]), validation=True)
    return load_track(id_or_path)


def load_track(id_or_path: str | Path) -> Track:
    """Load a built-in track by id (e.g. ``"beginner_oval"``) or a YAML file by path."""
    text: str
    candidate = Path(id_or_path)
    if candidate.suffix in {".yaml", ".yml"} or candidate.exists():
        if not candidate.is_file():
            raise TrackError(f"track file not found: {candidate}")
        text = candidate.read_text(encoding="utf-8")
        source = str(candidate)
    else:
        ref = resources.files("neurodrive.tracks").joinpath(f"{id_or_path}.yaml")
        if not ref.is_file():
            known = ", ".join(builtin_track_ids()) or "none"
            raise TrackError(f"unknown track id '{id_or_path}' (built-in tracks: {known})")
        text = ref.read_text(encoding="utf-8")
        source = f"built-in track '{id_or_path}'"
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TrackError(f"{source}: not valid YAML ({exc})") from exc
    if not isinstance(raw, dict):
        raise TrackError(f"{source}: expected a YAML mapping at the top level")
    try:
        spec = TrackSpec.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise TrackError(f"{source}: {details}") from exc
    return build_track(spec)
