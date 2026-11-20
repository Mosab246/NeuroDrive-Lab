"""Small, tested 2D geometry helpers used by tracks, progress tracking and collisions.

Conventions used throughout the simulator:

* World units are meters. The world is y-up (standard math axes).
* Angles are radians, measured counter-clockwise from the +x axis.
* Closed polylines are stored as ``(N, 2)`` arrays *without* repeating the first point;
  segment ``i`` runs from ``points[i]`` to ``points[(i + 1) % N]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def closed_segment_lengths(points: FloatArray) -> FloatArray:
    """Length of each segment of a closed polyline, shape ``(N,)``."""
    return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)


def catmull_rom_closed(control_points: FloatArray, samples_per_segment: int = 64) -> FloatArray:
    """Densely sample a closed, centripetal Catmull-Rom spline through ``control_points``.

    Centripetal parameterisation (alpha = 0.5) avoids the loops and cusps that the
    uniform variant produces when control points are unevenly spaced.
    The curve passes through every control point.
    """
    p = np.asarray(control_points, dtype=np.float64)
    n = len(p)
    if n < 3:
        raise ValueError("a closed spline needs at least 3 control points")
    out = []
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    for i in range(n):
        p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        # Knot spacing: sqrt of chord length (centripetal).
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0) ** 0.5, 1e-9)
        t2 = t1 + max(np.linalg.norm(p2 - p1) ** 0.5, 1e-9)
        t3 = t2 + max(np.linalg.norm(p3 - p2) ** 0.5, 1e-9)
        tt = (t1 + (t2 - t1) * t)[:, None]
        a1 = (t1 - tt) / (t1 - t0) * p0 + (tt - t0) / (t1 - t0) * p1
        a2 = (t2 - tt) / (t2 - t1) * p1 + (tt - t1) / (t2 - t1) * p2
        a3 = (t3 - tt) / (t3 - t2) * p2 + (tt - t2) / (t3 - t2) * p3
        b1 = (t2 - tt) / (t2 - t0) * a1 + (tt - t0) / (t2 - t0) * a2
        b2 = (t3 - tt) / (t3 - t1) * a2 + (tt - t1) / (t3 - t1) * a3
        out.append((t2 - tt) / (t2 - t1) * b1 + (tt - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def resample_closed(points: FloatArray, spacing: float) -> FloatArray:
    """Resample a closed polyline to (almost) uniform arc-length spacing.

    The number of output points is ``round(perimeter / spacing)``, so the actual spacing is
    within a fraction of a percent of the requested one. The first output point equals the
    first input point.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    seg = closed_segment_lengths(points)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    perimeter = cum[-1]
    n_out = max(round(float(perimeter) / spacing), 3)
    targets = np.linspace(0.0, perimeter, n_out, endpoint=False)
    closed = np.vstack([points, points[:1]])
    x = np.interp(targets, cum, closed[:, 0])
    y = np.interp(targets, cum, closed[:, 1])
    return np.column_stack([x, y])


def closed_tangents(points: FloatArray) -> FloatArray:
    """Unit tangent at each vertex of a closed polyline (central difference)."""
    d = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def left_normals(tangents: FloatArray) -> FloatArray:
    """Rotate unit tangents +90 degrees: the normal pointing to the driver's left."""
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def closed_curvature(points: FloatArray, stride: int = 1) -> FloatArray:
    """Signed curvature (1 / turn radius) at each vertex; positive = turning left.

    Uses the circle through each vertex and the vertices ``stride`` steps either side.
    A wider stride averages out tiny position errors, which matter because curvature is
    a second derivative: with 1 m spacing, a 1 mm wobble changes it by ~0.008 1/m.
    """
    a = np.roll(points, stride, axis=0)
    b = points
    c = np.roll(points, -stride, axis=0)
    ab = b - a
    bc = c - b
    ca = a - c
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ca, axis=1)
    return np.divide(2.0 * cross, denom, out=np.zeros_like(cross), where=denom > 0)


def signed_polygon_area(points: FloatArray) -> float:
    """Shoelace area; positive when the polygon winds counter-clockwise."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def segments_intersect(
    p1: FloatArray, p2: FloatArray, q1: FloatArray, q2: FloatArray, eps: float = 1e-12
) -> NDArray[np.bool_]:
    """Vectorised proper-intersection test between segments ``p1->p2`` and ``q1->q2``.

    Inputs broadcast against each other (shape ``(..., 2)``). Touching at an endpoint or
    collinear overlap counts as *not* intersecting, which is what we want when testing
    neighbouring segments of the same polyline.
    """
    r = p2 - p1
    s = q2 - q1
    qp = q1 - p1
    denom = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    t_num = qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]
    u_num = qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]
    parallel = np.abs(denom) < eps
    safe = np.where(parallel, 1.0, denom)
    t = t_num / safe
    u = u_num / safe
    return (~parallel) & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)


def closed_polyline_self_intersects(points: FloatArray) -> bool:
    """True if any two non-adjacent segments of a closed polyline cross."""
    a = points
    b = np.roll(points, -1, axis=0)
    n = len(points)
    for i in range(n - 2):
        js = np.arange(i + 2, n)
        if i == 0:
            js = js[js != n - 1]  # segment n-1 is adjacent to segment 0
        if js.size and np.any(segments_intersect(a[i], b[i], a[js], b[js])):
            return True
    return False


def closed_polylines_intersect(p: FloatArray, q: FloatArray) -> bool:
    """True if any segment of closed polyline ``p`` crosses any segment of ``q``."""
    pa, pb = p, np.roll(p, -1, axis=0)
    qa, qb = q, np.roll(q, -1, axis=0)
    return any(np.any(segments_intersect(pa[i], pb[i], qa, qb)) for i in range(len(p)))


@dataclass(frozen=True, slots=True)
class Projection:
    """Where points land on a closed polyline.

    Attributes:
        segment: index of the segment each point projects onto, shape ``(K,)``.
        s: arc length of the projected point from vertex 0, in ``[0, perimeter)``.
        lateral: signed perpendicular distance; positive = left of the travel direction.
    """

    segment: NDArray[np.int64]
    s: FloatArray
    lateral: FloatArray


class ClosedPolylineIndex:
    """Precomputed segment geometry of a closed polyline, for fast repeated projection.

    The simulator projects a handful of points onto the centerline several times per step,
    so per-call NumPy overhead dominates. This class computes every segment's start,
    direction and length once, stores them as flat component arrays, and pads them by
    ``window`` segments on each side so a search window is a plain slice (a view, no
    copying and no modulo arithmetic).
    """

    def __init__(self, vertices: FloatArray, cum_s: FloatArray, window: int = 8) -> None:
        n = len(vertices)
        if window < 1:
            raise ValueError("window must be at least 1")
        self.n = n
        self.window = window
        self.perimeter = float(cum_s[-1])
        a = vertices
        d = np.roll(vertices, -1, axis=0) - vertices
        pad = np.r_[np.arange(n - window, n), np.arange(n), np.arange(window)] % n
        self._idx = pad.astype(np.int64)
        self._ax = a[pad, 0].copy()
        self._ay = a[pad, 1].copy()
        self._dx = d[pad, 0].copy()
        self._dy = d[pad, 1].copy()
        len2 = d[:, 0] ** 2 + d[:, 1] ** 2
        self._len2 = len2[pad].copy()
        self._len = np.sqrt(self._len2)
        self._s0 = cum_s[:-1][pad].copy()

    def project(self, points: FloatArray, hint: int | None = None) -> Projection:
        """Project points (shape ``(K, 2)``) onto the nearest segment.

        With ``hint``, only segments ``hint - window .. hint + window`` are searched. This
        is what keeps progress continuous: on a hairpin the globally nearest segment can be
        on the other side of the infield, but a windowed search cannot jump there.
        """
        w = self.window
        if hint is None or 2 * w + 1 >= self.n:
            sl = slice(w, w + self.n)
        else:
            start = int(hint) % self.n
            sl = slice(start, start + 2 * w + 1)
        ax, ay, dx, dy = self._ax[sl], self._ay[sl], self._dx[sl], self._dy[sl]
        px = points[:, 0:1]  # (K, 1)
        py = points[:, 1:2]
        apx = px - ax  # (K, W)
        apy = py - ay
        t = (apx * dx + apy * dy) / self._len2[sl]
        np.clip(t, 0.0, 1.0, out=t)
        ex = apx - t * dx
        ey = apy - t * dy
        d2 = ex * ex + ey * ey
        best = np.argmin(d2, axis=1)
        k = np.arange(len(points))
        tb = t[k, best]
        s = self._s0[sl][best] + tb * self._len[sl][best]
        s = np.where(s >= self.perimeter, s - self.perimeter, s)
        cross = dx[best] * apy[k, best] - dy[best] * apx[k, best]
        lateral = np.sign(cross) * np.sqrt(d2[k, best])
        seg = self._idx[sl][best]
        return Projection(segment=seg, s=s, lateral=lateral)


def project_onto_closed_polyline(
    points: FloatArray,
    vertices: FloatArray,
    cum_s: FloatArray,
    hint: int | None = None,
    window: int = 8,
) -> Projection:
    """One-off projection; see :class:`ClosedPolylineIndex` for the details and fast path."""
    return ClosedPolylineIndex(vertices, cum_s, window).project(np.atleast_2d(points), hint)


def wrap_delta(delta: float, period: float) -> float:
    """Map a difference of positions on a loop into ``(-period/2, period/2]``."""
    return float(-((-delta + period / 2.0) % period) + period / 2.0)
