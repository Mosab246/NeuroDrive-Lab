import math
from pathlib import Path

import numpy as np
import pytest

from neurodrive.sim.track import TrackError, builtin_track_ids, load_track


@pytest.fixture(scope="module")
def oval():
    return load_track("beginner_oval")


def test_builtin_tracks_listed():
    assert "beginner_oval" in builtin_track_ids()


def test_oval_length_matches_analytic_value(oval):
    analytic = 2 * 120 + 2 * math.pi * 40
    assert oval.length == pytest.approx(analytic, rel=0.005)


def test_oval_start_is_first_control_point_heading_east(oval):
    x, y, heading = oval.start_pose()
    assert (x, y) == pytest.approx((0.0, -40.0))
    assert heading == pytest.approx(0.0, abs=1e-6)


def test_oval_is_counter_clockwise_so_left_edge_is_inside(oval):
    # Inside of a CCW oval is toward the origin.
    assert np.linalg.norm(oval.left_edge[0]) < np.linalg.norm(oval.right_edge[0])
    assert oval.curvature.mean() > 0


def test_edges_are_half_width_from_centerline(oval):
    d = np.linalg.norm(oval.left_edge - oval.centerline, axis=1)
    np.testing.assert_allclose(d, oval.spec.width / 2)


def test_checkpoints_evenly_spaced_and_last_is_finish(oval):
    cps = oval.checkpoint_s
    assert len(cps) == oval.spec.checkpoint_count
    assert cps[-1] == pytest.approx(oval.length)
    np.testing.assert_allclose(np.diff(cps), oval.length / len(cps))


def test_point_at_wraps(oval):
    p0, _ = oval.point_at(0.0)
    p1, _ = oval.point_at(oval.length)
    np.testing.assert_allclose(p0, p1, atol=1e-9)


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "t.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_rejects_turn_tighter_than_half_width(tmp_path):
    pts = "\n".join(
        f"  - [{5 * math.cos(a):.3f}, {5 * math.sin(a):.3f}]"
        for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)
    )
    path = write(tmp_path, f"id: tiny\nname: Tiny\nwidth: 20\ncontrol_points:\n{pts}\n")
    with pytest.raises(TrackError, match="turn radius"):
        load_track(path)


def test_rejects_figure_eight(tmp_path):
    path = write(
        tmp_path,
        "id: eight\nname: Eight\nwidth: 4\ncontrol_points:\n"
        "  - [0, 0]\n  - [40, 40]\n  - [80, 0]\n  - [40, -40]\n  - [0, 0.001]\n"
        "  - [-40, 40]\n  - [-80, 0]\n  - [-40, -40]\n",
    )
    with pytest.raises(TrackError, match="crosses"):
        load_track(path)


def test_schema_errors_name_the_field(tmp_path):
    path = write(tmp_path, "id: bad\nname: Bad\nwidth: -3\ncontrol_points: [[0,0],[1,0],[0,1]]\n")
    with pytest.raises(TrackError, match="width"):
        load_track(path)


def test_unknown_id_lists_known_tracks():
    with pytest.raises(TrackError, match="beginner_oval"):
        load_track("no_such_track")


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(TrackError, match="not found"):
        load_track(tmp_path / "missing.yaml")


def test_oval_tightest_turn_is_close_to_design_radius(oval):
    # Designed as 40 m arcs; the spline tightens slightly where straights meet curves.
    tightest = 1.0 / np.abs(oval.curvature).max()
    assert 30.0 < tightest <= 40.5
