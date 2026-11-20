"""Baseline drivers. None of these learn anything.

* :class:`RandomController`: a new uniform random action every step. Mostly vibrates.
* :class:`SmoothedRandomController`: random actions that drift smoothly, a fairer
  "no knowledge" baseline because it actually moves.
* :class:`HeuristicController`: hand-written rules that use **only the same observation
  vector the learning agents get**. It answers "how far does simple logic get with
  exactly this information?"
* :class:`PrivilegedAutopilot`: the pure-pursuit driver, which reads the track geometry
  directly. Not comparable with the others; it proves a track is drivable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neurodrive.agents.pure_pursuit import PurePursuitDriver

if TYPE_CHECKING:
    from neurodrive.env.driving_env import DrivingEnv


class RandomController:
    name = "random"
    kind = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=2)


class SmoothedRandomController:
    """Random walk in action space: each step moves a little toward a random target, and
    the target is re-drawn every ``hold_steps`` steps."""

    name = "smoothed_random"
    kind = "random"

    def __init__(self, seed: int | None = None, hold_steps: int = 15, smoothing: float = 0.2):
        self._rng = np.random.default_rng(seed)
        self.hold_steps = hold_steps
        self.smoothing = smoothing
        self._action = np.zeros(2)
        self._target = np.zeros(2)
        self._t = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._action = np.zeros(2)
        self._t = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._t % self.hold_steps == 0:
            self._target = self._rng.uniform(-1.0, 1.0, size=2)
        self._t += 1
        self._action += self.smoothing * (self._target - self._action)
        return self._action.copy()


class HeuristicController:
    """Steer toward open space, slow down when the road ahead is short.

    Uses only the default sensors-only observation: rays (ordered left to right, angles
    given in ``angles_deg``), normalized speed, and normalized wheel angle.

    * **Steering:** compare how much room there is on the right vs the left, weighting each
      ray by ``sin(angle)`` (side rays count most). More room on the right -> steer right.
    * **Speed:** the target speed grows with the free distance straight ahead.
    """

    name = "heuristic"
    kind = "heuristic"

    def __init__(
        self,
        angles_deg: tuple[float, ...] = (-90, -60, -40, -20, 0, 20, 40, 60, 90),
        steer_gain: float = 3.0,
        speed_gain: float = 4.0,
        min_target_speed: float = 0.3,
        max_target_speed: float = 0.8,
    ) -> None:
        self.angles = np.radians(np.asarray(angles_deg, dtype=float))
        self.weights = np.sin(self.angles)
        self.forward = int(np.argmin(np.abs(self.angles)))
        self.steer_gain = steer_gain
        self.speed_gain = speed_gain
        self.min_target_speed = min_target_speed
        self.max_target_speed = max_target_speed

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        n = len(self.angles)
        rays, speed = obs[:n], float(obs[n])
        balance = float(np.dot(self.weights, rays) / np.sum(np.abs(self.weights)))
        steering = self.steer_gain * balance
        ahead = float(rays[self.forward])
        lo, hi = self.min_target_speed, self.max_target_speed
        target = lo + (hi - lo) * ahead
        pedal = self.speed_gain * (target - speed)
        return np.clip(np.array([steering, pedal]), -1.0, 1.0)


class PrivilegedAutopilot:
    """Adapter: the pure-pursuit driver behind the Controller interface. It ignores the
    observation and reads the environment's simulator directly."""

    name = "pure_pursuit"
    kind = "scripted (privileged)"

    def __init__(self, env: DrivingEnv) -> None:
        self.env = env
        self.driver = PurePursuitDriver()

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.driver.act(self.env.sim)
