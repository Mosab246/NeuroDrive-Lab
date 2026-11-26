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
    Z                   camera: whole track <-> follow the focused car
    V                   show/hide the focused car's sensor rays
    N                   show/hide the focused car's network
    T                   show/hide trails
    E                   show/hide skid marks and tyre smoke
    click the bar       jump to that moment
    Esc                 quit
"""

from __future__ import annotations

import io
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
        self.t = 0.0
        self.speed_index = SPEEDS.index(1.0)
        self.playing = True
        self.running = True
        self.focus = 0
        self.follow = False
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

    def seek(self, t: float) -> None:
        self.t = min(max(0.0, t), self.duration)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.bar.collidepoint(event.pos):
                    x = (event.pos[0] - self.bar.x - 16) / max(1, self.bar.width - 32)
                    self.seek(min(max(x, 0.0), 1.0) * self.duration)
            elif event.type == pygame.KEYDOWN:
                self.key(event.key, bool(event.mod & pygame.KMOD_SHIFT))

    def key(self, key: int, shift: bool = False) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            if self.t >= self.duration:
                self.seek(0.0)
            self.playing = not self.playing
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            jump = 5.0 if shift else 1.0
            self.seek(self.t + (jump if key == pygame.K_RIGHT else -jump))
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.speed_index = min(self.speed_index + 1, len(SPEEDS) - 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_index = max(self.speed_index - 1, 0)
        elif key in (pygame.K_HOME, pygame.K_r):
            self.seek(0.0)
            self.playing = True
        elif key == pygame.K_TAB:
            self.focus = (self.focus + 1) % len(self.cars)
        elif key == pygame.K_z:
            self.follow = not self.follow
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
        if not self.follow:
            return self.track_view.fit
        car = self.cars[self.focus]
        center = car.center(car.r.index_at(self.t))
        return R.Camera(self.track_view.follow_scale, center, self.track_view.viewport.center)

    def ranking(self) -> list[int]:
        """Car indices, leader first: finished laps by lap time, then by distance."""

        def key(k: int) -> tuple[int, float]:
            car = self.cars[k]
            i = car.r.index_at(self.t)
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
        if self.show_network:
            self._draw_network()
        self._draw_panel()
        self._draw_bar()

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
            ("time", f"{_fmt_time(self.t)} / {_fmt_time(self.duration)}"),
            ("playback", f"{self.speed:g}x {'playing' if self.playing else 'paused'}"),
            ("fps", f"{self.measured_fps:4.0f}" if self.measured_fps > 0 else "-"),
        ):
            self.screen.blit(self.font.render(label, True, R.TEXT_DIM), (x0, y))
            img = self.font.render(value, True, R.TEXT)
            self.screen.blit(img, (x1 - img.get_width(), y))
            y += 22
        y += 10
        self.screen.blit(self.font.render("standings", True, R.TEXT_DIM), (x0, y))
        y += 24
        for place, k in enumerate(self.ranking(), start=1):
            car = self.cars[k]
            i = car.r.index_at(self.t)
            if k == self.focus:
                pygame.draw.rect(
                    self.screen, (34, 40, 50), (panel.x + 6, y - 3, PANEL_WIDTH - 12, 46)
                )
            pygame.draw.rect(self.screen, car.color, (x0, y + 2, 12, 12))
            name = self.font.render(f"{place}. {car.r.label}", True, R.TEXT)
            self.screen.blit(name, (x0 + 20, y))
            st = self.font.render(car.status(i), True, R.TEXT)
            self.screen.blit(st, (x1 - st.get_width(), y))
            kind = car.r.meta.get("kind_description", car.r.meta.get("controller_kind", "?"))
            tag_color = LEARNED_TAG if car.learned else SCRIPTED_TAG
            speed = f"{car.r.speed[i] * 3.6:5.0f} km/h"
            self.screen.blit(self.font_small.render(kind, True, tag_color), (x0 + 20, y + 21))
            sp = self.font_small.render(speed, True, R.TEXT_DIM)
            self.screen.blit(sp, (x1 - sp.get_width(), y + 21))
            y += 50
        help_lines = [
            "Space play/pause   Left/Right seek",
            "+/- speed   R restart   Tab focus",
            "Z follow   V rays   N network",
            "T trails   click bar to jump   Esc",
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
