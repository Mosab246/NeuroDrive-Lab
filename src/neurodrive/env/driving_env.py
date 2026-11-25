"""The Gymnasium environment: the contract every learning algorithm trains against.

One episode = one attempt at ``laps_to_finish`` laps (default 1) from the start line.

Episode ends:

=================  ===========  ==========================================
reason             flag         meaning
=================  ===========  ==========================================
``lap_complete``   terminated   success
``collision``      terminated   a car corner left the track
``stalled``        terminated   < ``stall_min_progress`` m in ``stall_seconds``
``time_limit``     truncated    ran out of time; the car could have continued
=================  ===========  ==========================================

With ``traffic.cars > 0`` every episode also has opponent cars on track: scripted
drivers at random paces, starting ahead of the agent. The agent's distance sensors see
them, contact slows both cars (and costs ``reward.contact_penalty``), and a push into the
wall is a crash. This is how an agent learns to race rather than only to lap.

``terminated`` vs ``truncated`` matters for learning: after a real ending there is no
future reward, but after a time cut-off the agent should still value the state it was in.
Mixing them up teaches an agent that running out of time is as bad as crashing.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from neurodrive.agents.pure_pursuit import PurePursuitDriver
from neurodrive.env.config import EnvConfig
from neurodrive.env.observations import ObservationBuilder
from neurodrive.env.rewards import compute_reward
from neurodrive.sim.race import RaceWorld
from neurodrive.sim.simulation import SimConfig, Simulation, Termination
from neurodrive.sim.track import Track, load_any_track
from neurodrive.sim.vehicle import VehicleParams


class DrivingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}  # noqa: RUF012

    def __init__(
        self,
        config: EnvConfig | None = None,
        track: Track | None = None,
        vehicle: VehicleParams | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.render_mode = render_mode
        self._vehicle = vehicle or self.config.physics.vehicle_params()
        # One simulation per track, built on first use and kept: switching tracks between
        # episodes is then just a pointer change.
        self._worlds: dict[str, tuple[Track, Simulation, ObservationBuilder]] = {}
        if track is not None:
            self.track_ids: tuple[str, ...] = (track.id,)
            self._add_world(track)
        else:
            self.track_ids = tuple(self.config.tracks or (self.config.track,))
        self._use(self.track_ids[0])
        spec = self.obs_builder.spec
        self.observation_space = spaces.Box(spec.low, spec.high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._stall_steps = max(1, round(self.config.reward.stall_seconds / self.sim.config.dt))
        self._history: deque[float] = deque(maxlen=self._stall_steps + 1)
        self._episode_return = 0.0
        self._renderer: Any = None
        self._opponents: dict[str, list[Simulation]] = {}
        self._opponent_drivers: list[PurePursuitDriver] = []
        self.race: RaceWorld | None = None

    def _add_world(self, track: Track) -> None:
        sim = Simulation(
            track,
            vehicle=self._vehicle,
            config=SimConfig(
                max_time=self.config.max_episode_seconds,
                laps_to_finish=self.config.laps_to_finish,
            ),
        )
        builder = ObservationBuilder(sim, self.config.sensors, self.config.observation)
        self._worlds[track.id] = (track, sim, builder)

    def _use(self, track_id: str) -> None:
        if track_id not in self._worlds:
            self._add_world(load_any_track(track_id))
        self.track, self.sim, self.obs_builder = self._worlds[track_id]

    # ------------------------------------------------------------------ gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        perturb = bool(options.get("perturb", self.config.perturb_start))
        random_start = bool(options.get("random_start", self.config.random_start))
        if "track" in options:
            track_id = str(options["track"])
        elif len(self.track_ids) > 1:
            track_id = self.track_ids[int(self.np_random.integers(len(self.track_ids)))]
        else:
            track_id = self.track_ids[0]
        self._use(track_id)
        self.sim.reset(
            seed=int(self.np_random.integers(2**31 - 1)),
            perturb=perturb,
            random_start=random_start,
        )
        self.race = self._start_traffic() if self.config.traffic.cars > 0 else None
        self._history.clear()
        self._history.append(self.sim.progress.distance)
        self._episode_return = 0.0
        obs = self.obs_builder.build(self._obstacles())
        if self.render_mode == "human":
            self.render()
        return obs, self._info(reason=None, reward_parts=None)

    def _start_traffic(self) -> RaceWorld:
        cfg = self.config.traffic
        track = self.track
        if track.id not in self._opponents:
            sim_cfg = SimConfig(max_time=None, laps_to_finish=None)
            self._opponents[track.id] = [
                Simulation(track, vehicle=self._vehicle, config=sim_cfg) for _ in range(cfg.cars)
            ]
        opponents = self._opponents[track.id]
        rng = self.np_random
        gaps = rng.uniform(*cfg.gap_range, size=len(opponents))
        room = track.length - 2.0 * cfg.gap_range[0]  # never wrap round onto the agent
        if gaps.sum() > room:
            gaps *= room / gaps.sum()
        max_lateral = max(0.0, min(2.5, float(track.half_width.min()) - 1.5))
        s = self.sim.start_s
        self._opponent_drivers = []
        for opp, gap in zip(opponents, gaps, strict=True):
            s += float(gap)
            opp.place(s, float(rng.uniform(-1.0, 1.0)) * max_lateral)
            pace = float(rng.uniform(*cfg.speed_range))
            self._opponent_drivers.append(
                PurePursuitDriver(speed_margin=pace, dynamic_grip_share=0.625 * pace)
            )
        world = RaceWorld([self.sim, *opponents], stall_exempt=(0,))
        world.start()
        return world

    def _obstacles(self) -> list[np.ndarray] | None:
        return self.race.obstacles_for(0) if self.race is not None else None

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        new_hits = 0
        if self.race is None:
            result = self.sim.step(action)
        else:
            race = self.race
            actions: list[np.ndarray | None] = [np.asarray(action)]
            for k, driver in enumerate(self._opponent_drivers, start=1):
                actions.append(driver.act(race.cars[k]) if race.status[k].racing else None)
            race.step(actions)
            last = race.last_results[0]
            assert last is not None
            result = last
            new_hits = race.new_hits[0]
        self._history.append(self.sim.progress.distance)
        stalled = False
        if result.termination is None and len(self._history) > self._stall_steps:
            stalled = self._history[-1] - self._history[0] < self.config.reward.stall_min_progress
        parts = compute_reward(result, stalled, self.config.reward, new_hits)
        reward = parts.total
        self._episode_return += reward

        reason: str | None = None
        terminated = truncated = False
        if result.termination is Termination.COLLISION:
            reason, terminated = "collision", True
        elif result.termination is Termination.LAPS_COMPLETE:
            reason, terminated = "lap_complete", True
        elif stalled:
            reason, terminated = "stalled", True
        elif result.termination is Termination.TIME_LIMIT:
            reason, truncated = "time_limit", True

        obs = self.obs_builder.build(self._obstacles())
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, self._info(reason, parts)

    def _info(self, reason: str | None, reward_parts: Any) -> dict[str, Any]:
        sim = self.sim
        info: dict[str, Any] = {
            "track": self.track.id,
            "time": sim.time,
            "steps": sim.step_count,
            "distance": sim.progress.distance,
            "lap_fraction": sim.progress.lap_fraction,
            "laps": sim.progress.laps,
            "speed": sim.state.speed,
            "lap_times": sim.lap_times,
            "episode_return": self._episode_return,
        }
        if self.race is not None:
            info["contacts"] = self.race.status[0].contacts
        if reward_parts is not None:
            info["reward_parts"] = {
                "progress": reward_parts.progress,
                "lap_bonus": reward_parts.lap_bonus,
                "crash": reward_parts.crash,
                "stall": reward_parts.stall,
                "contact": reward_parts.contact,
            }
        if reason is not None:
            info["termination_reason"] = reason
        return info

    # ------------------------------------------------------------------ rendering
    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from neurodrive.ui.env_view import EnvView  # pygame only when rendering

            self._renderer = EnvView(self, window=self.render_mode == "human")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
