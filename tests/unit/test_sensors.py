"""Raycasts checked against hand-computed distances on the Beginner Oval.

The car sits at (0, -40), the middle of the bottom straight, heading east (+x).
The inner edge of that straight is the line y = -34, the outer edge y = -46.
"""

import math

import numpy as np
import pytest

from neurodrive.sim.sensors import DEFAULT_RAY_ANGLES_DEG, RaySensor
from neurodrive.sim.track import load_track

ORIGIN = np.array([0.0, -40.0])


@pytest.fixture(scope="module")
def oval():
    return load_track("beginner_oval")


def cast(oval, angles, max_range=100.0, origin=ORIGIN, heading=0.0):
    sensor = RaySensor(oval, angles_deg=angles, max_range=max_range)
    seg = int(oval.project(origin).segment[0])
    return sensor.cast(origin, heading, seg)


def test_side_rays_measure_distance_to_each_edge(oval):
    hits = cast(oval, (-90, 90))
    # -90 = left = north (toward the infield), +90 = right = south.
    np.testing.assert_allclose(hits.distances, [6.0, 6.0], atol=1e-3)
    np.testing.assert_allclose(hits.directions, [[0, 1], [0, -1]], atol=1e-12)


def test_angled_ray_hits_inner_edge_where_trigonometry_says(oval):
    hits = cast(oval, (-20, 20))
    expected = 6.0 / math.sin(math.radians(20))  # 17.54 m to either straight edge
    np.testing.assert_allclose(hits.distances, [expected, expected], atol=1e-3)


def test_forward_ray_reaches_the_outer_wall_of_the_curve(oval):
    hits = cast(oval, (0,))
    # Outer edge of the right-hand curve: circle radius 46 around (60, 0).
    expected = 60 + math.sqrt(46**2 - 40**2)
    assert hits.distances[0] == pytest.approx(expected, abs=0.25)


def test_distances_are_capped_at_max_range(oval):
    hits = cast(oval, (0,), max_range=50.0)
    assert hits.distances[0] == 50.0


def test_off_center_car(oval):
    hits = cast(oval, (-90, 90), origin=np.array([0.0, -38.0]))
    np.testing.assert_allclose(hits.distances, [4.0, 8.0], atol=1e-3)


def test_heading_rotates_the_rays(oval):
    # Facing west, "left" is south (the outer edge) and "right" is north.
    hits = cast(oval, (-90, 90), heading=math.pi)
    np.testing.assert_allclose(hits.directions, [[0, -1], [0, 1]], atol=1e-12)
    np.testing.assert_allclose(hits.distances, [6.0, 6.0], atol=1e-3)


def test_endpoints_lie_on_the_edges(oval):
    hits = cast(oval, DEFAULT_RAY_ANGLES_DEG, max_range=200.0)
    ends = hits.endpoints()
    lateral = np.abs(oval.project(ends).lateral)
    np.testing.assert_allclose(lateral, oval.spec.width / 2, atol=0.05)


def test_all_positions_give_finite_capped_distances(oval):
    sensor = RaySensor(oval)
    rng = np.random.default_rng(0)
    for s in rng.uniform(0, oval.length, 200):
        p, t = oval.point_at(s)
        lat = rng.uniform(-5.5, 5.5)
        origin = p + lat * np.array([-t[1], t[0]])
        heading = math.atan2(t[1], t[0]) + rng.uniform(-1, 1)
        seg = int(oval.project(origin).segment[0])
        d = sensor.cast(origin, heading, seg).distances
        assert np.all(np.isfinite(d)) and np.all((d >= 0) & (d <= sensor.max_range))


def test_invalid_config(oval):
    with pytest.raises(ValueError):
        RaySensor(oval, max_range=0)
    with pytest.raises(ValueError):
        RaySensor(oval, angles_deg=())
