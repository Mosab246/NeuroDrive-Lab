"""The Phase 1 playground: drive the car yourself, or watch the scripted autopilot.

Controls:
    Arrow keys / WASD   steer, throttle, brake
    C                   switch driver: keyboard <-> autopilot (scripted, privileged)
    P                   pause / resume
    R                   reset to the start line
    + / -               simulation speed 0.25x .. 8x
    T                   toggle the driven-path trail
    Z                   toggle camera: whole track <-> follow the car
    Esc                 quit
"""

from __future__ import annotations

import sys

import numpy as np
import pygame

from neurodrive.agents.pure_pursuit import PurePursuitDriver
from neurodrive.sim.simulation import SimConfig, Simulation
from neurodrive.sim.track import Track
from neurodrive.ui import renderer as R
from neurodrive.ui.clock import FixedStepClock

PANEL_WIDTH = 300


class KeyboardDriver:
    name = "keyboard"
    kind = "human"

    def act(self, keys: pygame.key.ScancodeWrapper) -> np.ndarray:
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        return np.array([float(right) - float(left), float(up) - float(down)])


def _enable_dpi_awareness() -> None:
    """Stop Windows from bitmap-scaling the window on high-DPI laptop screens."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "--:--.--"
    m, s = divmod(seconds, 60.0)
    return f"{int(m):02d}:{s:05.2f}"


class PlaygroundApp:
    def __init__(
        self,
        track: Track,
        controller: str = "keyboard",
        size: tuple[int, int] = (1280, 800),
        frame_dt: float | None = None,
        fps: int = 60,
    ) -> None:
        """
        Args:
            frame_dt: if set, pretend every frame took this long instead of reading the
                real clock (used by tests to make the viewer deterministic).
        """
        _enable_dpi_awareness()
        pygame.init()
        pygame.display.set_caption(f"NeuroDrive Lab - {track.spec.name}")
        self.screen = pygame.display.set_mode(size)
        self.font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 17)
        self.font_big = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 30, True)
        self.sim = Simulation(track, config=SimConfig())
        self.clock = FixedStepClock(self.sim.config.dt)
        viewport = pygame.Rect(0, 0, size[0] - PANEL_WIDTH, size[1])
        self.track_view = R.TrackRenderer(track, viewport)
        self.car_view = R.CarRenderer()
        self.follow_camera = False
        self.keyboard = KeyboardDriver()
        self.autopilot = PurePursuitDriver()
        if controller not in ("keyboard", "autopilot"):
            raise ValueError("controller must be 'keyboard' or 'autopilot'")
        self.use_autopilot = controller == "autopilot"
        self.paused = False
        self.running = True
        self.frame_dt = frame_dt
        self.fps = fps
        self.last_action = np.zeros(2)
        self.measured_fps = 0.0

    # ------------------------------------------------------------------ loop
    def run(self, max_frames: int | None = None) -> int:
        frames = 0
        tick = pygame.time.Clock()
        try:
            while self.running and (max_frames is None or frames < max_frames):
                real_dt = tick.tick(self.fps) / 1000.0
                self.measured_fps = tick.get_fps()
                self._handle_events()
                n = self.clock.advance(self.frame_dt if self.frame_dt is not None else real_dt)
                if not self.paused:
                    for _ in range(n):
                        if self.sim.done:
                            break
                        self._step()
                self._draw()
                pygame.display.flip()
                frames += 1
        finally:
            pygame.quit()
        return frames

    def _step(self) -> None:
        if self.use_autopilot:
            action = self.autopilot.act(self.sim)
        else:
            action = self.keyboard.act(pygame.key.get_pressed())
        self.last_action = action
        self.sim.step(action)
        self.car_view.record(self.sim)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_ESCAPE:
                    self.running = False
                elif key == pygame.K_p:
                    self.paused = not self.paused
                elif key == pygame.K_r:
                    self.reset()
                elif key == pygame.K_c:
                    self.use_autopilot = not self.use_autopilot
                elif key == pygame.K_z:
                    self.follow_camera = not self.follow_camera
                elif key == pygame.K_t:
                    self.car_view.show_trail = not self.car_view.show_trail
                elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.clock.faster()
                elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.clock.slower()

    def reset(self) -> None:
        self.sim.reset()
        self.clock.reset()
        self.car_view.clear_trail()
        self.last_action = np.zeros(2)

    # ------------------------------------------------------------------ drawing
    def _draw(self) -> None:
        self.screen.fill(R.GRASS)
        camera = self.track_view.camera(self.follow_camera, self.sim)
        self.track_view.draw(self.screen, camera)
        self.car_view.draw(self.screen, self.sim, camera)
        self._draw_panel()
        if self.sim.termination is not None:
            self._banner(f"{self.sim.termination.upper()}  -  press R to reset")
        elif self.paused:
            self._banner("PAUSED  -  press P to resume")

    def _banner(self, text: str) -> None:
        img = self.font_big.render(text, True, R.TEXT)
        rect = img.get_rect(center=self.track_view.viewport.center)
        bg = rect.inflate(32, 18)
        pygame.draw.rect(self.screen, R.PANEL, bg, border_radius=8)
        self.screen.blit(img, rect)

    def _draw_panel(self) -> None:
        w, h = self.screen.get_size()
        panel = pygame.Rect(w - PANEL_WIDTH, 0, PANEL_WIDTH, h)
        pygame.draw.rect(self.screen, R.PANEL, panel)
        sim = self.sim
        best = min(sim.lap_times) if sim.lap_times else None
        last = sim.lap_times[-1] if sim.lap_times else None
        driver = self.autopilot if self.use_autopilot else self.keyboard
        steer, pedal = self.last_action
        rows: list[tuple[str, str] | None] = [
            ("track", sim.track.spec.name),
            ("driver", driver.name),
            ("type", driver.kind),
            None,
            ("speed", f"{sim.state.speed * 3.6:6.1f} km/h"),
            ("steering cmd", f"{steer:+.2f}"),
            ("throttle/brake", f"{pedal:+.2f}"),
            None,
            ("lap", f"{sim.progress.laps + 1}"),
            ("lap progress", f"{100 * max(0.0, sim.progress.lap_fraction):5.1f} %"),
            (
                "next checkpoint",
                f"{sim.progress.next_checkpoint}/{len(sim.track.checkpoint_s)}",
            ),
            ("lap time", _fmt_time(sim.current_lap_time)),
            ("last lap", _fmt_time(last)),
            ("best lap", _fmt_time(best)),
            None,
            ("sim time", _fmt_time(sim.time)),
            ("sim speed", f"{self.clock.speed:g}x"),
            ("fps", f"{self.measured_fps:4.0f}"),
            ("status", sim.termination or ("paused" if self.paused else "running")),
        ]
        y = 18
        title = self.font_big.render("NeuroDrive Lab", True, R.TEXT)
        self.screen.blit(title, (panel.x + 16, y))
        y += 48
        for row in rows:
            if row is None:
                y += 10
                continue
            label, value = row
            self.screen.blit(self.font.render(label, True, R.TEXT_DIM), (panel.x + 16, y))
            img = self.font.render(value, True, R.TEXT)
            self.screen.blit(img, (panel.right - 16 - img.get_width(), y))
            y += 24
        help_lines = [
            "arrows/WASD drive   C driver",
            "P pause   R reset   +/- speed",
            "T trail   Z camera   Esc quit",
        ]
        y = h - 16 - 22 * len(help_lines)
        for line in help_lines:
            self.screen.blit(self.font.render(line, True, R.TEXT_DIM), (panel.x + 16, y))
            y += 22
