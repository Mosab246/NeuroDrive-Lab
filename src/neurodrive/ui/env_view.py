"""Rendering for ``DrivingEnv.render()``: track, car and sensor rays, no HUD.

``rgb_array`` mode draws off-screen and returns an ``(H, W, 3)`` uint8 array (for videos
and replays); ``human`` mode shows it in a window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygame

from neurodrive.ui import renderer as R

if TYPE_CHECKING:
    from neurodrive.env.driving_env import DrivingEnv


class EnvView:
    def __init__(self, env: DrivingEnv, window: bool, size: tuple[int, int] = (960, 600)):
        self.env = env
        self.window = window
        pygame.init()
        if window:
            pygame.display.set_caption(f"NeuroDrive Lab - {env.track.spec.name}")
            self.surface = pygame.display.set_mode(size)
        else:
            self.surface = pygame.Surface(size)
        self.rect = pygame.Rect(0, 0, *size)
        self.track_view = R.TrackRenderer(env.track, self.rect)
        self.car_view = R.CarRenderer()
        self.car_view.show_trail = False

    def render(self) -> np.ndarray | None:
        env = self.env
        if self.track_view.track is not env.track:  # multi-track env switched tracks
            self.track_view = R.TrackRenderer(env.track, self.rect)
        camera = self.track_view.fit
        self.track_view.draw(self.surface, camera)
        hits = env.obs_builder.last_hits
        if hits is not None:
            R.draw_rays(self.surface, camera, hits, env.obs_builder.sensor.max_range)
        self.car_view.draw(self.surface, env.sim, camera)
        if self.window:
            pygame.event.pump()
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2)).copy()

    def close(self) -> None:
        if self.window:
            pygame.display.quit()
