"""Replay viewer: play back one or more recorded episodes on the same track.

With several replays it is a ghost race: every car drives its own recorded episode, cars
pass through each other (no car-to-car collisions), and the panel ranks them. Used by the
``replay``, ``race``, ``timeline`` and ``demo`` commands.

Controls:
    Space               play / pause
    Left / Right        seek 1 s back / forward (hold Shift for 5 s)
    + / -               playback speed 0.25x .. 8x
    Home or R           back to the start
    Tab                 focus the next car (its sensor rays and network are shown)
    Z                   camera: whole track -> follow the focused car -> TV (follows the
                        closest battle)
    V                   show/hide the focused car's sensor rays
    N                   show/hide the focused car's network
    T                   show/hide trails
    E                   show/hide skid marks and tyre smoke
    click the bar       jump to that moment
    Esc                 quit
"""

from __future__ import annotations

import io
import itertools
import math
import time

import numpy as np
import pygame

from neurodrive.replay.format import Replay, ReplayError
from neurodrive.sim.track import load_any_track
from neurodrive.sim.vehicle import VehicleParams, VehicleState, body_corners
from neurodrive.ui import art
from neurodrive.ui import renderer as R
from neurodrive.ui.app import _enable_dpi_awareness, _fmt_time
from neurodrive.ui.clock import SPEEDS

START_LIGHTS_S = 4.0  # races: seconds of start lights before the cars go
LIGHT_STEP_S = 0.75  # one more red light every this many seconds
GO_SHOWN_S = 1.0  # "GO" stays on screen this long after lights out
PANEL_WIDTH = 360
BAR_HEIGHT = 44
TRAIL_STEPS = 240

# Distinct car colours (colour-blind-friendly order first).
COLORS = [
    (255, 170, 40),
    (80, 170, 255),
    (120, 220, 120),
    (240, 100, 200),
    (250, 240, 110),
    (170, 130, 255),
    (90, 225, 215),
    (255, 120, 90),
]
LEARNED_TAG = (120, 220, 160)
SCRIPTED_TAG = (180, 186, 200)


class _Car:
    def __init__(self, replay: Replay, color: tuple[int, int, int], params: VehicleParams):
        self.r = replay
        self.color = color
        self.params = params
        self.learned = bool(replay.meta.get("learned"))
        self.end = replay.meta["termination_reason"]
        times = replay.meta["lap_times"]
        # A race's result is its total time (several laps, from the grid).
        self.lap_time = replay.meta.get("finish_time") or (times[0] if times else None)
        self.network_surface: pygame.Surface | None = None
        self.skid, self.wheels = art.skid_profile(
            replay.x, replay.y, replay.heading, replay.speed, replay.dt
        )
        self.skid_runs = art.skid_runs(self.skid)
        # Laps driven (fractional) at every step, for gaps between cars.
        self.dist = replay.laps.astype(float) + replay.lap_fraction

    def corners(self, i: int) -> np.ndarray:
        r = self.r
        return body_corners(VehicleState(r.x[i], r.y[i], r.heading[i]), self.params)

    def center(self, i: int) -> np.ndarray:
        return self.corners(i).mean(axis=0)

    def finished(self, i: int) -> bool:
        return i >= self.r.steps

    def distance(self, i: int) -> float:
        """Laps done + fraction of the current lap, for ranking."""
        return float(self.r.laps[i]) + max(0.0, float(self.r.lap_fraction[i]))

    def status(self, i: int) -> str:
        if not self.finished(i):
            return f"lap {100 * max(0.0, self.r.lap_fraction[i]):5.1f} %"
        if self.end == "lap_complete" and self.lap_time is not None:
            return f"FINISHED {self.lap_time:6.2f} s"
        return f"{self.end.upper().replace('_', ' ')} {self.r.duration:5.1f} s"


class ReplayViewer:
    def __init__(
        self,
        replays: list[Replay],
        title: str = "Replay",
        size: tuple[int, int] = (1400, 860),
        frame_dt: float | None = None,
        fps: int = 60,
    ) -> None:
        if not replays:
            raise ReplayError("no replays to show")
        tracks = {r.track for r in replays}
        if len(tracks) > 1:
            raise ReplayError(f"replays are on different tracks: {', '.join(sorted(tracks))}")
        if len(replays) > len(COLORS):
            raise ReplayError(f"at most {len(COLORS)} cars at once")
        _enable_dpi_awareness()
        pygame.init()
        self.title = title
        self.track = load_any_track(replays[0].track)
        pygame.display.set_caption(f"NeuroDrive Lab - {title} - {self.track.spec.name}")
        self.screen = pygame.display.set_mode(size)
        mono = "consolas,menlo,dejavusansmono,monospace"
        self.font = pygame.font.SysFont(mono, 16)
        self.font_small = pygame.font.SysFont(mono, 13)
        self.font_big = pygame.font.SysFont(mono, 26, True)
        viewport = pygame.Rect(0, 0, size[0] - PANEL_WIDTH, size[1] - BAR_HEIGHT)
        self.bar = pygame.Rect(0, size[1] - BAR_HEIGHT, size[0] - PANEL_WIDTH, BAR_HEIGHT)
        self.track_view = R.TrackRenderer(self.track, viewport)
        params = VehicleParams()
        self.cars = [_Car(r, COLORS[i], params) for i, r in enumerate(replays)]
        self.dt = replays[0].dt
        self.duration = max(r.duration for r in replays)
        self.is_race = any(r.meta.get("multi_car") for r in replays)
        self.race_laps = int(replays[0].meta.get("race", {}).get("laps", 1))
        # A race opens with the start lights: playback begins before time 0, cars on the grid.
        self.t_start = -START_LIGHTS_S if self.is_race else 0.0
        self.t = self.t_start
        self.camera_mode = "track"  # "track", "follow" or "tv"
        self._tv_pair: tuple[int, ...] = (0,)
        self._tv_since = -1e9
        self._cam_center: np.ndarray | None = None
        self._minimap: pygame.Surface | None = None
        self.speed_index = SPEEDS.index(1.0)
        self.playing = True
        self.running = True
        self.focus = 0
        self.show_rays = True
        self.show_network = False
        self.show_trails = True
        self.show_effects = True  # skid marks and tyre smoke (E)
        self.frame_dt = frame_dt
        self.fps = fps
        self.measured_fps = 0.0
        self.frame_times: list[float] = []
        self.wall_seconds = 0.0

    # ------------------------------------------------------------------ loop
    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_index]

    def run(self, max_frames: int | None = None) -> int:
        frames = 0
        tick = pygame.time.Clock()
        start = time.perf_counter()
        try:
            while self.running and (max_frames is None or frames < max_frames):
                real_dt = tick.tick(self.fps) / 1000.0
                self.measured_fps = tick.get_fps()
                self._handle_events()
                if self.playing:
                    step = self.frame_dt if self.frame_dt is not None else real_dt
                    self.seek(self.t + step * self.speed)
                    if self.t >= self.duration:
                        self.playing = False
                t0 = time.perf_counter()
                self.draw()
                self.frame_times.append(time.perf_counter() - t0)
                pygame.display.flip()
                frames += 1
        finally:
            self.wall_seconds = time.perf_counter() - start
            pygame.quit()
        return frames

    @property
    def follow(self) -> bool:
        return self.camera_mode == "follow"

    @follow.setter
    def follow(self, on: bool) -> None:
        self.camera_mode = "follow" if on else "track"

    def seek(self, t: float) -> None:
        self.t = min(max(self.t_start, t), self.duration)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.bar.collidepoint(event.pos):
                    x = (event.pos[0] - self.bar.x - 16) / max(1, self.bar.width - 32)
                    span = self.duration - self.t_start
                    self.seek(self.t_start + min(max(x, 0.0), 1.0) * span)
            elif event.type == pygame.KEYDOWN:
                self.key(event.key, bool(event.mod & pygame.KMOD_SHIFT))

    def key(self, key: int, shift: bool = False) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            if self.t >= self.duration:
                self.seek(self.t_start)
            self.playing = not self.playing
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            jump = 5.0 if shift else 1.0
            self.seek(self.t + (jump if key == pygame.K_RIGHT else -jump))
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.speed_index = min(self.speed_index + 1, len(SPEEDS) - 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_index = max(self.speed_index - 1, 0)
        elif key in (pygame.K_HOME, pygame.K_r):
            self.seek(self.t_start)
            self.playing = True
        elif key == pygame.K_TAB:
            self.focus = (self.focus + 1) % len(self.cars)
        elif key == pygame.K_z:
            modes = ["track", "follow"] + (["tv"] if len(self.cars) > 1 else [])
            self.camera_mode = modes[(modes.index(self.camera_mode) + 1) % len(modes)]
            self._cam_center = None
        elif key == pygame.K_v:
            self.show_rays = not self.show_rays
        elif key == pygame.K_n:
            self.show_network = not self.show_network
        elif key == pygame.K_t:
            self.show_trails = not self.show_trails
        elif key == pygame.K_e:
            self.show_effects = not self.show_effects

    # ------------------------------------------------------------------ drawing
    def _camera(self) -> R.Camera:
        if self.camera_mode == "track":
            return self.track_view.fit
        if self.camera_mode == "follow":
            car = self.cars[self.focus]
            center = car.center(car.r.index_at(self.t))
        else:
            target = self._tv_target()
            # Glide towards the action instead of cutting, so the picture stays watchable.
            if self._cam_center is None or np.hypot(*(target - self._cam_center)) > 120.0:
                self._cam_center = target
            else:
                self._cam_center = self._cam_center + 0.15 * (target - self._cam_center)
            center = self._cam_center
        return R.Camera(self.track_view.follow_scale, center, self.track_view.viewport.center)

    TV_HOLD_S = 3.0  # keep a shot at least this long before switching battles

    def _tv_target(self) -> np.ndarray:
        """Centre of the closest fight between two cars still racing (consecutive in the
        standings); the leader when nobody is close. Re-chosen at most every few seconds."""
        racing = [
            k for k in self.ranking() if not self.cars[k].finished(self.cars[k].r.index_at(self.t))
        ]
        if abs(self.t - self._tv_since) >= self.TV_HOLD_S or not all(
            k in racing for k in self._tv_pair
        ):
            pair: tuple[int, ...] = (racing[0],) if racing else (self.ranking()[0],)
            best = 0.25  # only pairs within a quarter lap count as a battle
            for a, b in itertools.pairwise(racing):
                gap = self._lap_distance(a) - self._lap_distance(b)
                if gap < best:
                    best, pair = gap, (a, b)
            self._tv_pair, self._tv_since = pair, self.t
        points = [self.cars[k].center(self.cars[k].r.index_at(self.t)) for k in self._tv_pair]
        return np.mean(points, axis=0)

    def _lap_distance(self, k: int) -> float:
        car = self.cars[k]
        return float(car.dist[car.r.index_at(self.t)])

    def gap_to_leader(self, k: int) -> float | None:
        """Seconds behind the leader: how long ago the leader was where car ``k`` is now."""
        lead = self.ranking()[0]
        if k == lead or self.t <= 0:
            return None
        here = self._lap_distance(k)
        leader = self.cars[lead]
        j = int(np.searchsorted(np.maximum.accumulate(leader.dist), here))
        if j >= len(leader.dist):
            return None
        return max(0.0, self.t - j * self.dt)

    def ranking(self, t: float | None = None) -> list[int]:
        """Car indices, leader first: finished laps by lap time, then by distance."""
        at = self.t if t is None else t

        def key(k: int) -> tuple[int, float]:
            car = self.cars[k]
            i = car.r.index_at(at)
            if car.finished(i) and car.lap_time is not None:
                return (0, car.lap_time)
            if car.finished(i):
                return (2, -car.distance(i))
            return (1, -car.distance(i))

        return sorted(range(len(self.cars)), key=key)

    def draw(self) -> None:
        self.screen.fill(R.GRASS)
        camera = self._camera()
        self.track_view.draw(self.screen, camera)
        clip = self.screen.get_clip()
        self.screen.set_clip(self.track_view.viewport)
        self._labels: list[pygame.Rect] = []
        order = [self.focus] + [k for k in range(len(self.cars)) if k != self.focus]
        for car in self.cars:  # marks on the asphalt first, under every car
            self._draw_skids(car, camera)
        for k in order:
            self._draw_car(k, camera)
        for car in self.cars:  # smoke above the cars
            self._draw_smoke(car, camera)
        self.screen.set_clip(clip)
        if camera is not self.track_view.fit:
            self._draw_minimap()
        if self.is_race:
            self._draw_start_lights()
        if self.show_network:
            self._draw_network()
        self._draw_panel()
        self._draw_bar()

    def _draw_start_lights(self) -> None:
        """Five red lights come on one by one; all go out at time 0 (the race starts)."""
        if self.t >= GO_SHOWN_S:
            return
        view = self.track_view.viewport
        r, gap = 18, 14
        width = 5 * 2 * r + 4 * gap + 40
        box = pygame.Rect(0, 0, width, 2 * r + 30)
        box.midtop = (view.centerx, view.top + 20)
        pygame.draw.rect(self.screen, (12, 12, 14), box, border_radius=10)
        lit = 0 if self.t >= 0 else min(5, int((self.t - self.t_start) / LIGHT_STEP_S) + 1)
        for n in range(5):
            center = (box.left + 20 + r + n * (2 * r + gap), box.centery)
            color = (230, 30, 30) if n < lit else (55, 20, 20)
            pygame.draw.circle(self.screen, color, center, r)
        if self.t >= 0:
            go = self.font_big.render("GO", True, (90, 230, 110))
            self.screen.blit(go, go.get_rect(midtop=(box.centerx, box.bottom + 6)))

    def _draw_minimap(self) -> None:
        """Whole-track overview with every car, shown while the camera is zoomed in."""
        size = 210
        view = self.track_view.viewport
        if self._minimap is None:
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            surf.fill((10, 12, 16, 238))
            self._mini_cam = R.fit_camera(self.track, surf.get_rect(), margin=12)
            for edge in (self.track.left_edge, self.track.right_edge):
                pts = self._mini_cam.to_screen(edge)
                pygame.draw.aalines(surf, (200, 204, 212), True, pts.tolist())
            self._minimap = surf
        origin = (view.right - size - 12, view.top + 12)
        self.screen.blit(self._minimap, origin)
        for k, car in enumerate(self.cars):
            i = car.r.index_at(self.t)
            p = self._mini_cam.to_screen(car.center(i))[0] + origin
            radius = 5 if k == self.focus else 4
            color = R.CAR_CRASHED if car.finished(i) and car.end == "collision" else car.color
            pygame.draw.circle(self.screen, color, (round(p[0]), round(p[1])), radius)

    def _draw_skids(self, car: _Car, camera: R.Camera) -> None:
        if not self.show_effects:
            return
        i = car.r.index_at(self.t)
        width = max(1, round(0.25 * camera.scale))
        for start, end in car.skid_runs:
            if start >= i:
                break
            stop = min(end, i + 1)
            if stop - start < 2:
                continue
            for w in range(2):
                pts = camera.to_screen(car.wheels[start:stop, w])
                pygame.draw.lines(self.screen, art.SKID_COLOR, False, pts.tolist(), width)

    SMOKE_STEPS = 20  # how long a puff lingers (steps)

    def _draw_smoke(self, car: _Car, camera: R.Camera) -> None:
        if not self.show_effects or camera.scale < 3.0:
            return
        i = car.r.index_at(self.t)
        for j in range(max(0, i - self.SMOKE_STEPS), i + 1, 2):
            if not car.skid[j]:
                continue
            age = (i - j) / self.SMOKE_STEPS
            puff = art.smoke_puff(round((0.8 + 2.2 * age) * camera.scale))
            puff.set_alpha(int(255 * (1 - age)))
            for w in range(2):
                p = camera.to_screen(car.wheels[j, w])[0]
                self.screen.blit(puff, puff.get_rect(center=(round(p[0]), round(p[1]))))

    def _draw_car(self, k: int, camera: R.Camera) -> None:
        car = self.cars[k]
        i = car.r.index_at(self.t)
        focused = k == self.focus
        if self.show_trails and i > 1:
            lo = max(0, i - TRAIL_STEPS)
            pts = camera.to_screen(np.column_stack([car.r.x[lo : i + 1], car.r.y[lo : i + 1]]))
            pygame.draw.aalines(self.screen, car.color, False, pts.tolist())
        if focused and self.show_rays:
            origin = camera.to_screen(car.r.ray_origin[i])[0]
            dists = car.r.ray_dist[i]
            ends = camera.to_screen(car.r.ray_origin[i] + car.r.ray_dirs[i] * dists[:, None])
            max_range = float(car.r.meta.get("ray_max_range", 50.0))
            for end, d in zip(ends, dists, strict=True):
                pygame.draw.aaline(self.screen, R.RAY, origin, end)
                if d < max_range:
                    pygame.draw.circle(self.screen, R.RAY_HIT, end, 3)
        corners = camera.to_screen(car.corners(i))
        if not self.track_view.viewport.inflate(80, 80).collidepoint(*corners.mean(axis=0)):
            return  # off screen (follow camera): draw nothing, not even the label
        crashed = car.finished(i) and car.end == "collision"
        color = R.CAR_CRASHED if crashed else car.color
        center = corners.mean(axis=0)
        p = car.params
        heading = float(car.r.heading[i])
        art.blit_car(self.screen, center, heading, color, k + 1, camera.scale, p.length, p.width)
        if camera.scale < 6.0:
            pygame.draw.circle(
                self.screen, color, center, 13 if focused else 10, 2 if focused else 1
            )
        tag = self.font_small.render(car.r.label, True, color)
        bg = pygame.Rect((center[0] + 14, center[1] - 22), tag.get_size()).inflate(6, 2)
        moved = False
        for _ in range(len(self.cars)):  # step down past labels already drawn this frame
            hit = bg.collidelist(self._labels)
            if hit < 0:
                break
            bg.top = self._labels[hit].bottom + 1
            moved = True
        self._labels.append(bg)
        if moved:
            pygame.draw.aaline(self.screen, color, center, (bg.left, bg.centery))
        pygame.draw.rect(self.screen, R.PANEL, bg, border_radius=3)
        self.screen.blit(tag, (bg.x + 3, bg.y + 1))

    def _draw_panel(self) -> None:
        w, h = self.screen.get_size()
        panel = pygame.Rect(w - PANEL_WIDTH, 0, PANEL_WIDTH, h)
        pygame.draw.rect(self.screen, R.PANEL, panel)
        x0, x1 = panel.x + 16, panel.right - 16
        y = 14
        self.screen.blit(self.font_big.render(self.title, True, R.TEXT), (x0, y))
        y += 38
        for label, value in (
            ("track", self.track.spec.name),
            (
                "time",
                ("start lights" if self.t < 0 else _fmt_time(self.t))
                + f" / {_fmt_time(self.duration)}",
            ),
            ("playback", f"{self.speed:g}x {'playing' if self.playing else 'paused'}"),
            (
                "camera",
                {"track": "whole track", "follow": "follow car", "tv": "TV"}[self.camera_mode],
            ),
            ("fps", f"{self.measured_fps:4.0f}" if self.measured_fps > 0 else "-"),
        ):
            self.screen.blit(self.font.render(label, True, R.TEXT_DIM), (x0, y))
            img = self.font.render(value, True, R.TEXT)
            self.screen.blit(img, (x1 - img.get_width(), y))
            y += 22
        y += 10
        self.screen.blit(self.font.render("standings", True, R.TEXT_DIM), (x0, y))
        y += 24
        before = {k: n for n, k in enumerate(self.ranking(max(self.t_start, self.t - 1.5)))}
        for place, k in enumerate(self.ranking(), start=1):
            car = self.cars[k]
            i = car.r.index_at(self.t)
            moved = before[k] - (place - 1)  # > 0: gained places in the last 1.5 s
            if k == self.focus:
                pygame.draw.rect(
                    self.screen, (34, 40, 50), (panel.x + 6, y - 3, PANEL_WIDTH - 12, 46)
                )
            pygame.draw.rect(self.screen, car.color, (x0, y + 2, 12, 12))
            name = self.font.render(f"{place}. {car.r.label}", True, R.TEXT)
            self.screen.blit(name, (x0 + 20, y))
            if moved and not car.finished(i):
                ax, ay = x0 + 26 + name.get_width(), y + 8
                if moved > 0:
                    tri = [(ax, ay + 4), (ax + 10, ay + 4), (ax + 5, ay - 5)]
                    pygame.draw.polygon(self.screen, (90, 220, 110), tri)
                else:
                    tri = [(ax, ay - 4), (ax + 10, ay - 4), (ax + 5, ay + 5)]
                    pygame.draw.polygon(self.screen, (235, 90, 80), tri)
            status = car.status(i)
            if self.is_race and not car.finished(i):
                lap_no = min(int(car.r.laps[i]) + 1, self.race_laps)
                status = f"L{lap_no}/{self.race_laps} {100 * max(0.0, car.r.lap_fraction[i]):3.0f}%"
            st = self.font.render(status, True, R.TEXT)
            self.screen.blit(st, (x1 - st.get_width(), y))
            kind = car.r.meta.get("kind_description", car.r.meta.get("controller_kind", "?"))
            tag_color = LEARNED_TAG if car.learned else SCRIPTED_TAG
            speed = f"{car.r.speed[i] * 3.6:5.0f} km/h"
            gap = self.gap_to_leader(k) if self.is_race and not car.finished(i) else None
            if gap is not None:
                speed = f"+{gap:.1f} s   {speed}"
            self.screen.blit(self.font_small.render(kind, True, tag_color), (x0 + 20, y + 21))
            sp = self.font_small.render(speed, True, R.TEXT_DIM)
            self.screen.blit(sp, (x1 - sp.get_width(), y + 21))
            y += 50
        help_lines = [
            "Space play/pause   Left/Right seek",
            "+/- speed   R restart   Tab focus",
            "Z camera   V rays   N network",
            "T trails   E skids   click bar   Esc",
        ]
        y = h - 12 - 20 * len(help_lines)
        for line in help_lines:
            self.screen.blit(self.font_small.render(line, True, R.TEXT_DIM), (x0, y))
            y += 20

    def _draw_bar(self) -> None:
        pygame.draw.rect(self.screen, R.PANEL, self.bar)
        track = self.bar.inflate(-32, -30)
        pygame.draw.rect(self.screen, (50, 56, 66), track, border_radius=4)
        span = max(self.duration, 1e-9)
        for car in self.cars:  # tick where each car's episode ended
            x = track.x + track.width * car.r.duration / span
            pygame.draw.line(self.screen, car.color, (x, track.y - 6), (x, track.bottom + 6), 2)
        x = track.x + track.width * self.t / span
        filled = pygame.Rect(track.x, track.y, max(0, x - track.x), track.height)
        pygame.draw.rect(self.screen, R.TEXT_DIM, filled, border_radius=4)
        pygame.draw.circle(self.screen, R.TEXT, (round(x), track.centery), 7)

    # ------------------------------------------------------------------ network panel
    def _draw_network(self) -> None:
        car = self.cars[self.focus]
        view = self.track_view.viewport.inflate(-60, -60)
        pygame.draw.rect(self.screen, (245, 246, 248), view, border_radius=8)
        title = f"{car.r.label}: {car.r.meta.get('kind_description', '')}"
        self.screen.blit(self.font.render(title, True, (20, 24, 30)), (view.x + 16, view.y + 10))
        body = view.inflate(-32, -60).move(0, 14)
        graph = (car.r.meta.get("network") or {}).get("graph")
        if graph is not None and car.r.node_values is not None and len(car.r.node_values):
            self._draw_live_network(car, body, graph)
            return
        if car.r.network_png is not None:
            if car.network_surface is None:
                car.network_surface = pygame.image.load(io.BytesIO(car.r.network_png), "n.png")
            img = car.network_surface
            scale = min(body.width / img.get_width(), body.height / img.get_height())
            img = pygame.transform.smoothscale(
                img, (int(img.get_width() * scale), int(img.get_height() * scale))
            )
            self.screen.blit(img, img.get_rect(center=body.center))
            return
        info = car.r.meta.get("network") or {}
        if info.get("type") == "mlp":
            self._draw_mlp(body, info)
            return
        msg = "hand-written rules: no neural network"
        if not car.learned:
            msg = f"{car.r.meta.get('kind_description', 'scripted')}: {msg}"
        img = self.font.render(msg, True, (20, 24, 30))
        self.screen.blit(img, img.get_rect(center=body.center))

    def _draw_live_network(self, car: _Car, body: pygame.Rect, graph: dict) -> None:
        """The evolved network at this moment: each node shows its current value, each
        connection how strongly it is pushing right now (weight x the value it carries).
        Blue = positive, red = negative; brighter / thicker = stronger."""
        nodes = car.r.node_values
        assert nodes is not None
        i = min(car.r.index_at(self.t), len(nodes) - 1)
        values = dict(zip(graph["order"], nodes[i].tolist(), strict=True))
        # Room on the left for input labels and on the right for output labels.
        area = pygame.Rect(body.left + 170, body.top + 20, body.width - 420, body.height - 60)

        def at(k: int) -> tuple[int, int]:
            x, y = graph["pos"][str(k)] if str(k) in graph["pos"] else graph["pos"][k]
            return (round(area.left + x * area.width), round(area.top + (1 - y) * area.height))

        pos = {k: at(k) for k in graph["order"]}
        flows = [w * values.get(a, 0.0) for a, _, w in graph["edges"]]
        top = max([abs(f) for f in flows] + [1e-6])
        dark = (20, 24, 30)
        for (a, b, _w), f in sorted(
            zip(graph["edges"], flows, strict=True), key=lambda e: abs(e[1])
        ):
            s = min(1.0, abs(f) / top)
            base = (43, 123, 186) if f >= 0 else (217, 83, 79)
            color = tuple(int(235 - (235 - c) * (0.15 + 0.85 * s)) for c in base)
            pygame.draw.line(self.screen, color, pos[a], pos[b], max(1, round(1 + 5 * s)))
        names = graph.get("input_names") or [str(k) for k in graph["inputs"]]
        out_names = ["steering", "throttle/brake"]
        for k in graph["order"]:
            v = values.get(k, 0.0)
            s = min(1.0, abs(v))
            base = (43, 123, 186) if v >= 0 else (217, 83, 79)
            fill = tuple(int(235 - (235 - c) * s) for c in base)
            radius = 9 if k in graph["outputs"] else 7
            pygame.draw.circle(self.screen, fill, pos[k], radius)
            pygame.draw.circle(self.screen, dark, pos[k], radius, 1)
        for k, name in zip(graph["inputs"], names, strict=False):
            img = self.font_small.render(f"{name} {values.get(k, 0.0):+.2f}", True, dark)
            self.screen.blit(img, img.get_rect(midright=(pos[k][0] - 14, pos[k][1])))
        for k, name in zip(graph["outputs"], out_names, strict=False):
            img = self.font.render(f"{name} {values.get(k, 0.0):+.2f}", True, dark)
            self.screen.blit(img, img.get_rect(midleft=(pos[k][0] + 16, pos[k][1])))
        note = (
            f"{len(graph['hidden'])} hidden neurons, {len(graph['edges'])} connections; "
            "line strength = weight x signal flowing through it right now"
        )
        img = self.font_small.render(note, True, dark)
        self.screen.blit(img, img.get_rect(midbottom=(body.centerx, body.bottom)))

    def _draw_mlp(self, body: pygame.Rect, info: dict) -> None:
        """PPO's policy network: input, hidden layers, output, drawn as columns of dots."""
        sizes = [info["inputs"], *info["hidden"], info["outputs"]]
        names = ["inputs (sensors)", *[f"hidden {n}" for n in info["hidden"]], "actions"]
        cols = len(sizes)
        dark = (20, 24, 30)
        xs = [body.x + 80 + (body.width - 160) * c / max(1, cols - 1) for c in range(cols)]
        points: list[list[tuple[float, float]]] = []
        for c, n in enumerate(sizes):
            top, bottom = body.y + 40, body.bottom - 60
            ys = [top + (bottom - top) * (j + 0.5) / n for j in range(n)]
            points.append([(xs[c], y) for y in ys])
        for c in range(cols - 1):  # sample of connections, faint
            for j, a in enumerate(points[c]):
                for k, b in enumerate(points[c + 1]):
                    if (j * 7 + k * 3) % max(1, (len(points[c]) * len(points[c + 1])) // 120) == 0:
                        pygame.draw.aaline(self.screen, (190, 200, 215), a, b)
        for c, pts in enumerate(points):
            r = 6 if len(pts) <= 16 else 2
            color = (
                (154, 165, 177) if c == 0 else (92, 184, 92) if c == cols - 1 else (240, 173, 78)
            )
            for p in pts:
                pygame.draw.circle(self.screen, color, (round(p[0]), round(p[1])), r)
            lab = self.font_small.render(names[c], True, dark)
            self.screen.blit(lab, lab.get_rect(midtop=(xs[c], body.bottom - 48)))
        text = (
            f"fixed architecture chosen before training: {' -> '.join(map(str, sizes))}, "
            f"{info['activation']} activations, {info['parameters']:,} policy weights"
        )
        img = self.font_small.render(text, True, dark)
        self.screen.blit(img, img.get_rect(midtop=(body.centerx, body.bottom - 22)))

    # ------------------------------------------------------------------ stats
    def frame_time_stats(self) -> dict[str, float]:
        ft = sorted(self.frame_times) or [0.0]
        p95 = ft[min(len(ft) - 1, math.ceil(0.95 * len(ft)) - 1)]
        fps = len(self.frame_times) / self.wall_seconds if self.wall_seconds > 0 else 0.0
        return {
            "frames": len(self.frame_times),
            "mean_ms": 1000 * float(np.mean(ft)),
            "p95_ms": 1000 * p95,
            "average_fps": fps,
        }
