"""Draws a track and a car with pygame-ce. Reads simulation state; never changes it."""

from __future__ import annotations

from collections import deque

import numpy as np
import pygame

from neurodrive.sim import geometry as geo
from neurodrive.sim.sensors import RayHits
from neurodrive.sim.simulation import Simulation, Termination
from neurodrive.sim.track import Track
from neurodrive.ui import art

# Palette: dark "lab" background, muted asphalt, high-contrast car.
GRASS = art.GRASS
ASPHALT = art.ASPHALT
EDGE = (210, 214, 222)
CHECKPOINT = (95, 130, 160)
FINISH = (245, 245, 245)
CAR = (255, 170, 40)
CAR_CRASHED = (230, 70, 70)
TRAIL = (255, 170, 40)
TEXT = (230, 233, 240)
TEXT_DIM = (140, 148, 160)
PANEL = (16, 19, 24)

FOLLOW_SCALE = 8.0  # pixels per meter in follow-camera mode
MAX_LAYER_PIXELS = 8000  # cap on a pre-rendered track surface's width/height


class Camera:
    """Maps world meters (y-up) to screen pixels (y-down)."""

    def __init__(self, scale: float, world_center: np.ndarray, screen_center: tuple[int, int]):
        self.scale = scale
        self.world_center = np.asarray(world_center, dtype=float)
        self.screen_center = np.asarray(screen_center, dtype=float)

    def to_screen(self, pts: np.ndarray) -> np.ndarray:
        out = (np.atleast_2d(pts) - self.world_center) * self.scale
        out[:, 1] *= -1
        return out + self.screen_center


def track_bounds(track: Track) -> tuple[np.ndarray, np.ndarray]:
    pts = np.vstack([track.left_edge, track.right_edge])
    return pts.min(axis=0), pts.max(axis=0)


def fit_camera(track: Track, viewport: pygame.Rect, margin: float = 24.0) -> Camera:
    """A camera showing the whole track inside ``viewport``."""
    lo, hi = track_bounds(track)
    span = np.maximum(hi - lo, 1e-6)
    scale = min((viewport.width - 2 * margin) / span[0], (viewport.height - 2 * margin) / span[1])
    return Camera(float(scale), (lo + hi) / 2, viewport.center)


class _TrackLayer:
    """The static track drawn once at a given scale, to be blitted each frame."""

    def __init__(self, track: Track, scale: float, pad: int = 40) -> None:
        lo, hi = track_bounds(track)
        self.scale = scale
        self.pad = pad
        self.top_left_world = np.array([lo[0], hi[1]])
        size = ((hi - lo) * scale + 2 * pad).astype(int)
        self.surface = pygame.Surface((int(size[0]), int(size[1])))
        self.surface.fill(art.GRASS)

        def px(p: np.ndarray) -> np.ndarray:
            p = np.atleast_2d(p)
            return np.column_stack(
                [(p[:, 0] - lo[0]) * scale + pad, (hi[1] - p[:, 1]) * scale + pad]
            )

        left, right = px(track.left_edge), px(track.right_edge)
        # The edge enclosing the larger area is the outside of the loop.
        left_area = abs(geo.signed_polygon_area(track.left_edge))
        right_area = abs(geo.signed_polygon_area(track.right_edge))
        outer, inner = (left, right) if left_area > right_area else (right, left)
        pygame.draw.polygon(self.surface, art.ASPHALT, outer.tolist())
        pygame.draw.polygon(self.surface, art.GRASS, inner.tolist())
        art.texture_layer(self.surface, scale)
        width = max(1, round(scale * 0.15))
        for s in track.checkpoint_s[:-1]:  # faint marks; the finish line is drawn below
            i = int(np.searchsorted(track.cum_s, s % track.length, side="right") - 1)
            i %= track.num_points
            a, b = px(track.left_edge[i])[0], px(track.right_edge[i])[0]
            pygame.draw.line(self.surface, art.CHECKPOINT_MARK, a, b, width)
        art.draw_kerbs(self.surface, track, px)
        art.draw_finish_line(self.surface, track, px)
        line_w = max(1, round(scale * 0.25))
        pygame.draw.lines(self.surface, art.EDGE_LINE, True, left.tolist(), line_w)
        pygame.draw.lines(self.surface, art.EDGE_LINE, True, right.tolist(), line_w)

    def blit(self, screen: pygame.Surface, camera: Camera) -> None:
        origin = camera.to_screen(self.top_left_world)[0] - self.pad
        screen.blit(self.surface, (round(origin[0]), round(origin[1])))


class TrackRenderer:
    """Whole-track view plus an optional zoomed view that follows the car."""

    def __init__(self, track: Track, viewport: pygame.Rect) -> None:
        self.track = track
        self.viewport = viewport
        self.fit = fit_camera(track, viewport)
        lo, hi = track_bounds(track)
        follow_scale = min(FOLLOW_SCALE, MAX_LAYER_PIXELS / float(max(hi - lo)))
        # The big follow-camera layer is built on first use (it takes a moment).
        # Pad the whole-track layer so its grass covers the entire viewport.
        fit_pad = max(viewport.width, viewport.height)
        self._layers: dict[str, _TrackLayer] = {
            "fit": _TrackLayer(track, self.fit.scale, pad=fit_pad)
        }
        self.follow_scale = follow_scale

    def camera(self, follow: bool, sim: Simulation) -> Camera:
        if not follow:
            return self.fit
        center = sim.corners().mean(axis=0)
        return Camera(self.follow_scale, center, self.viewport.center)

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        key = "fit" if camera is self.fit else "follow"
        if key not in self._layers:
            pad = max(self.viewport.width, self.viewport.height) // 2 + 40
            self._layers[key] = _TrackLayer(self.track, self.follow_scale, pad=pad)
        layer = self._layers[key]
        previous_clip = screen.get_clip()
        screen.set_clip(self.viewport)
        screen.fill(GRASS, self.viewport)
        layer.blit(screen, camera)
        screen.set_clip(previous_clip)


class CarRenderer:
    def __init__(self, trail_length: int = 400) -> None:
        self.trail: deque[tuple[float, float]] = deque(maxlen=trail_length)
        self.show_trail = True

    def clear_trail(self) -> None:
        self.trail.clear()

    def record(self, sim: Simulation) -> None:
        c = sim.corners().mean(axis=0)
        self.trail.append((float(c[0]), float(c[1])))

    def draw(self, screen: pygame.Surface, sim: Simulation, camera: Camera) -> None:
        if self.show_trail and len(self.trail) > 1:
            pts = camera.to_screen(np.array(self.trail))
            pygame.draw.aalines(screen, TRAIL, False, pts.tolist())
        corners = camera.to_screen(sim.corners())
        crashed = sim.termination is Termination.COLLISION
        p = sim.params
        color = CAR_CRASHED if crashed else CAR
        center = corners.mean(axis=0)
        art.blit_car(
            screen, center, sim.state.heading, color, None, camera.scale, p.length, p.width
        )
        if camera.scale < 6.0:
            # At whole-track zoom the car is only a few pixels long: ring it.
            center = corners.mean(axis=0)
            pygame.draw.circle(screen, CAR_CRASHED if crashed else CAR, center, 11, 1)


RAY = (90, 200, 255)
RAY_HIT = (255, 90, 120)


def draw_rays(screen: pygame.Surface, camera: Camera, hits: RayHits, max_range: float) -> None:
    """Draw what the car's distance sensors see: a line per ray, with a dot where it hits
    a wall (no dot when the ray reaches its maximum range without hitting anything)."""
    origin = camera.to_screen(hits.origin)[0]
    ends = camera.to_screen(hits.endpoints())
    for end, dist in zip(ends, hits.distances, strict=True):
        pygame.draw.aaline(screen, RAY, origin, end)
        if dist < max_range:
            pygame.draw.circle(screen, RAY_HIT, end, 3)
