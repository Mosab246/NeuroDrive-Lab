"""Draws a track and a car with pygame-ce. Reads simulation state; never changes it."""

from __future__ import annotations

from collections import deque

import numpy as np
import pygame

from neurodrive.sim import geometry as geo
from neurodrive.sim.simulation import Simulation, Termination
from neurodrive.sim.track import Track

# Palette: dark "lab" background, muted asphalt, high-contrast car.
GRASS = (22, 30, 28)
ASPHALT = (58, 62, 70)
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
        self.surface.fill(GRASS)

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
        pygame.draw.polygon(self.surface, ASPHALT, outer.tolist())
        pygame.draw.polygon(self.surface, GRASS, inner.tolist())
        width = max(1, round(scale * 0.25))
        for k, s in enumerate(track.checkpoint_s):
            i = int(np.searchsorted(track.cum_s, s % track.length, side="right") - 1)
            i %= track.num_points
            a, b = px(track.left_edge[i])[0], px(track.right_edge[i])[0]
            last = k == len(track.checkpoint_s) - 1
            color = FINISH if last else CHECKPOINT
            pygame.draw.line(self.surface, color, a, b, 4 * width if last else width)
        pygame.draw.aalines(self.surface, EDGE, True, left.tolist())
        pygame.draw.aalines(self.surface, EDGE, True, right.tolist())

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
        self._layers = {
            "fit": _TrackLayer(track, self.fit.scale),
            "follow": _TrackLayer(track, follow_scale),
        }
        self.follow_scale = follow_scale

    def camera(self, follow: bool, sim: Simulation) -> Camera:
        if not follow:
            return self.fit
        center = sim.corners().mean(axis=0)
        return Camera(self.follow_scale, center, self.viewport.center)

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        layer = self._layers["fit" if camera is self.fit else "follow"]
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
        pygame.draw.polygon(screen, CAR_CRASHED if crashed else CAR, corners.tolist())
        # Windscreen marker near the front edge so the heading is obvious at a glance.
        front = (corners[0] * 0.7 + corners[3] * 0.3, corners[1] * 0.7 + corners[2] * 0.3)
        pygame.draw.line(screen, PANEL, front[0], front[1], 2)
        if camera.scale < 6.0:
            # At whole-track zoom the car is only a few pixels long: ring it.
            center = corners.mean(axis=0)
            pygame.draw.circle(screen, CAR_CRASHED if crashed else CAR, center, 11, 1)
