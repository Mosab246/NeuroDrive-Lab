import math

import numpy as np
import pytest

from neurodrive.sim import geometry as geo


def circle(radius: float, n: int) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def test_catmull_rom_passes_through_control_points():
    ctrl = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    dense = geo.catmull_rom_closed(ctrl, samples_per_segment=16)
    for i, p in enumerate(ctrl):
        np.testing.assert_allclose(dense[i * 16], p, atol=1e-12)


def test_resample_gives_uniform_spacing_and_keeps_first_point():
    pts = geo.resample_closed(circle(50.0, 400), spacing=1.0)
    seg = geo.closed_segment_lengths(pts)
    assert seg.max() / seg.min() < 1.01
    np.testing.assert_allclose(pts[0], [50.0, 0.0])
    assert abs(seg.sum() - 2 * math.pi * 50) < 0.1


def test_curvature_of_ccw_circle_is_positive_one_over_radius():
    k = geo.closed_curvature(circle(20.0, 360))
    np.testing.assert_allclose(k, 1 / 20.0, rtol=1e-3)
    k_cw = geo.closed_curvature(circle(20.0, 360)[::-1])
    np.testing.assert_allclose(k_cw, -1 / 20.0, rtol=1e-3)


def test_left_normal_of_east_is_north():
    np.testing.assert_allclose(geo.left_normals(np.array([[1.0, 0.0]])), [[0.0, 1.0]])


def test_signed_area_sign_follows_winding():
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert geo.signed_polygon_area(sq) == pytest.approx(1.0)
    assert geo.signed_polygon_area(sq[::-1]) == pytest.approx(-1.0)


def test_segment_intersection_cases():
    p = lambda *a: np.array(a, dtype=float)  # noqa: E731
    assert geo.segments_intersect(p(0, 0), p(2, 2), p(0, 2), p(2, 0))
    assert not geo.segments_intersect(p(0, 0), p(1, 1), p(2, 0), p(3, 1))  # apart
    assert not geo.segments_intersect(p(0, 0), p(1, 0), p(1, 0), p(2, 1))  # share endpoint
    assert not geo.segments_intersect(p(0, 0), p(2, 0), p(1, 0), p(3, 0))  # collinear


def test_self_intersection_detection():
    assert not geo.closed_polyline_self_intersects(circle(10.0, 64))
    bowtie = np.array([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    assert geo.closed_polyline_self_intersects(bowtie)


def test_projection_reports_arc_length_and_signed_lateral():
    sq = geo.resample_closed(
        np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]), spacing=1.0
    )
    cum = np.concatenate([[0.0], np.cumsum(geo.closed_segment_lengths(sq))])
    proj = geo.project_onto_closed_polyline(np.array([[3.0, 1.0], [3.0, -2.0]]), sq, cum)
    np.testing.assert_allclose(proj.s, [3.0, 3.0])
    np.testing.assert_allclose(proj.lateral, [1.0, -2.0])  # left of travel is positive


def test_windowed_projection_cannot_jump_across_the_track():
    # A thin hairpin: the point is closest to the far side, but the window is near s=0.
    pts = geo.resample_closed(
        np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 4.0], [0.0, 4.0]]), spacing=1.0
    )
    cum = np.concatenate([[0.0], np.cumsum(geo.closed_segment_lengths(pts))])
    point = np.array([[20.0, 3.5]])
    global_proj = geo.project_onto_closed_polyline(point, pts, cum)
    windowed = geo.project_onto_closed_polyline(point, pts, cum, hint=20, window=5)
    assert global_proj.s[0] > 60  # nearest point is on the return leg
    assert windowed.s[0] == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("delta", "expected"), [(1.0, 1.0), (-1.0, -1.0), (99.0, -1.0), (-99.0, 1.0), (50.0, 50.0)]
)
def test_wrap_delta(delta, expected):
    assert geo.wrap_delta(delta, 100.0) == pytest.approx(expected)
