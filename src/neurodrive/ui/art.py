"""Procedural art for the track and cars: no image files, everything is drawn in code.

* Grass with mowing stripes and grain; asphalt with grain (world-anchored, so the pattern
  stays put when the camera moves).
* Red-and-white kerbs on both edges wherever the track bends.
* A checkered start/finish line and white edge lines.
* Top-down car sprites: body, windows, wheels and a race number, cached per colour and
  zoom level and rotated each frame.

Purely visual: none of this changes the track geometry or the physics.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pygame

from neurodrive.sim.track import Track

GRASS = (38, 72, 46)  # must not occur anywhere else on a freshly drawn layer
GRASS_STRIPE = (8, 9, 7)  # added to every other mowing stripe
ASPHALT = (64, 66, 72)
EDGE_LINE = (226, 228, 232)
KERB_RED = (196, 40, 44)
KERB_WHITE = (236, 236, 236)
CHECK_DARK = (24, 24, 26)
CHECK_LIGHT = (240, 240, 240)
CHECKPOINT_MARK = (92, 96, 106)

KERB_CURVATURE = 1.0 / 70.0  # bends tighter than a 70 m radius get kerbs
KERB_WIDTH = 0.9  # meters, drawn inside the track edge
KERB_BLOCK = 2.0  # meters per red or white block
STRIPE_WIDTH = 7.0  # meters per mowing stripe
GRAIN = 5  # +- intensity of the asphalt/grass grain
CHUNK = 512  # columns processed at once when texturing (bounds memory use)


def _grain_tile(size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(12345)
    return rng.integers(-GRAIN, GRAIN + 1, size=(size, size), dtype=np.int16)


def texture_layer(surface: pygame.Surface, scale: float) -> None:
    """Add stripes to grass pixels and grain everywhere, in place. Call after the grass and
    asphalt areas are drawn in their flat colours, before lines and kerbs."""
    tile = _grain_tile()
    t = tile.shape[0]
    stripe_px = max(2.0, STRIPE_WIDTH * scale)
    pixels = pygame.surfarray.pixels3d(surface)  # (W, H, 3), a view into the surface
    width, height = pixels.shape[:2]
    ys = np.arange(height)[None, :]
    grain_rows = np.tile(tile, (1, height // t + 1))[:, :height]  # (t, H)
    for x0 in range(0, width, CHUNK):
        block = pixels[x0 : x0 + CHUNK].astype(np.int16)
        xs = np.arange(x0, x0 + block.shape[0])[:, None]
        grass = (
            (block[..., 0] == GRASS[0]) & (block[..., 1] == GRASS[1]) & (block[..., 2] == GRASS[2])
        )
        stripe = ((((xs + ys) / stripe_px).astype(np.int64)) % 2 == 0) & grass
        grain = grain_rows[np.arange(x0, x0 + block.shape[0]) % t]
        for c in range(3):
            block[..., c] += grain + stripe * GRASS_STRIPE[c]
        pixels[x0 : x0 + CHUNK] = np.clip(block, 0, 255).astype(np.uint8)
    del pixels  # unlock the surface


def draw_kerbs(
    surface: pygame.Surface, track: Track, px: Callable[[np.ndarray], np.ndarray]
) -> None:
    n = track.num_points
    bend = np.abs(track.curvature) > KERB_CURVATURE
    # Widen each bend a little so kerbs start before and end after the corner.
    grow = 4
    wide = bend.copy()
    for k in range(1, grow + 1):
        wide |= np.roll(bend, k) | np.roll(bend, -k)
    block = (track.cum_s[:n] // KERB_BLOCK).astype(int) % 2
    for edge, inward in ((track.left_edge, -track.normals), (track.right_edge, track.normals)):
        inner = edge + inward * KERB_WIDTH
        a, b = px(edge), px(inner)
        for i in np.flatnonzero(wide):
            j = (i + 1) % n
            color = KERB_RED if block[i] else KERB_WHITE
            pygame.draw.polygon(surface, color, [a[i], a[j], b[j], b[i]])


def draw_finish_line(
    surface: pygame.Surface, track: Track, px: Callable[[np.ndarray], np.ndarray]
) -> None:
    """A checkered band two squares deep across the start/finish line."""
    point, tangent, normal = track.centerline[0], track.tangents[0], track.normals[0]
    half = float(track.half_width[0])
    cells = max(4, round(2 * half / 1.0))
    size = 2 * half / cells
    for row in range(2):
        for col in range(cells):
            t0, t1 = (row - 1) * size, row * size
            l0, l1 = -half + col * size, -half + (col + 1) * size
            corners = np.array(
                [
                    point + tangent * t + normal * lat
                    for t, lat in ((t0, l0), (t1, l0), (t1, l1), (t0, l1))
                ]
            )
            color = CHECK_DARK if (row + col) % 2 else CHECK_LIGHT
            pygame.draw.polygon(surface, color, px(corners).tolist())


# ------------------------------------------------------------------ cars
_SPRITES: dict[tuple, pygame.Surface] = {}


def _shade(color: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, c * f))) for c in color)  # type: ignore[return-value]


def car_sprite(
    color: tuple[int, int, int], number: int | None, length_px: float, width_px: float
) -> pygame.Surface:
    """A top-down car facing +x (right), drawn at 4x and smoothly scaled down."""
    key = (color, number, round(length_px, 1), round(width_px, 1))
    if key in _SPRITES:
        return _SPRITES[key]
    k = 4
    w, h = max(8, int(length_px * k)), max(4, int(width_px * k))
    pad = int(0.12 * h)
    surf = pygame.Surface((w + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
    ox, oy = pad, pad
    # Wheels (slightly outside the body).
    wheel_l, wheel_w = int(0.18 * w), int(0.2 * h)
    for fx in (0.16, 0.7):
        for y in (oy - int(0.06 * h), oy + h - wheel_w + int(0.06 * h)):
            pygame.draw.rect(
                surf, (18, 18, 20), (ox + int(fx * w), y, wheel_l, wheel_w), border_radius=3
            )
    # Body with a darker outline.
    body = pygame.Rect(ox, oy + int(0.04 * h), w, h - int(0.08 * h))
    pygame.draw.rect(surf, _shade(color, 0.55), body, border_radius=int(0.35 * h))
    pygame.draw.rect(
        surf, color, body.inflate(-int(0.1 * h), -int(0.1 * h)), border_radius=int(0.3 * h)
    )
    # Windscreen (front) and rear window.
    glass = (30, 38, 52)
    front = [
        (ox + 0.56 * w, oy + 0.2 * h),
        (ox + 0.70 * w, oy + 0.27 * h),
        (ox + 0.70 * w, oy + 0.73 * h),
        (ox + 0.56 * w, oy + 0.8 * h),
    ]
    rear = [
        (ox + 0.16 * w, oy + 0.26 * h),
        (ox + 0.24 * w, oy + 0.22 * h),
        (ox + 0.24 * w, oy + 0.78 * h),
        (ox + 0.16 * w, oy + 0.74 * h),
    ]
    pygame.draw.polygon(surf, glass, front)
    pygame.draw.polygon(surf, glass, rear)
    # Race number on the roof.
    if number is not None and h >= 24:
        r = int(0.24 * h)
        center = (ox + int(0.4 * w), oy + h // 2)
        pygame.draw.circle(surf, (245, 245, 245), center, r)
        font = pygame.font.SysFont("arial", int(r * 1.5), bold=True)
        text = font.render(str(number), True, (15, 15, 15))
        text = pygame.transform.rotate(text, -90)  # upright when the car drives up the screen
        surf.blit(text, text.get_rect(center=center))
    sprite = pygame.transform.smoothscale(
        surf, (max(2, round(surf.get_width() / k)), max(2, round(surf.get_height() / k)))
    )
    _SPRITES[key] = sprite
    return sprite


def blit_car(
    screen: pygame.Surface,
    center: tuple[float, float] | np.ndarray,
    heading: float,
    color: tuple[int, int, int],
    number: int | None,
    scale: float,
    length_m: float,
    width_m: float,
) -> None:
    sprite = car_sprite(color, number, length_m * scale, width_m * scale)
    rotated = pygame.transform.rotate(sprite, math.degrees(heading))
    screen.blit(rotated, rotated.get_rect(center=(round(center[0]), round(center[1]))))


# ------------------------------------------------------------------ skid marks and smoke
SKID_SLIP_DEG = 5.0  # sliding: travel direction differs from heading by more than this
SKID_DECEL = 10.0  # m/s^2: braking hard enough to leave marks
SKID_MIN_SPEED = 3.0  # m/s
SKID_COLOR = (30, 30, 33)
REAR_TRACK = 0.8  # meters from the car's axis to each rear wheel


def skid_profile(
    x: np.ndarray, y: np.ndarray, heading: np.ndarray, speed: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """From a recorded path: which steps leave skid marks, and the two rear-wheel positions
    at every step. ``x, y`` is the rear axle; a sliding car's rear axle moves sideways."""
    move = np.diff(np.column_stack([x, y]), axis=0)
    travel = np.arctan2(move[:, 1], move[:, 0])
    slip = np.abs((travel - heading[1:] + np.pi) % (2 * np.pi) - np.pi)
    fast = speed[1:] > SKID_MIN_SPEED
    braking = -np.diff(speed) / dt > SKID_DECEL
    skid = np.r_[False, (fast & (slip > math.radians(SKID_SLIP_DEG))) | (braking & fast)]
    left = np.column_stack([-np.sin(heading), np.cos(heading)]) * REAR_TRACK
    axle = np.column_stack([x, y])
    wheels = np.stack([axle + left, axle - left], axis=1)  # (N, 2 wheels, 2)
    return skid, wheels


def skid_runs(skid: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` index ranges of consecutive skidding steps."""
    edges = np.flatnonzero(np.diff(np.r_[0, skid.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2], strict=True))


_PUFFS: dict[int, pygame.Surface] = {}


def smoke_puff(radius: int) -> pygame.Surface:
    radius = max(2, radius)
    if radius not in _PUFFS:
        surf = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
        for r in range(radius, 0, -1):  # soft edge: more opaque towards the middle
            alpha = int(150 * (1 - r / radius) ** 0.6)
            pygame.draw.circle(surf, (215, 215, 215, alpha), (radius, radius), r)
        _PUFFS[radius] = surf
    return _PUFFS[radius]
