"""Record a real race (cars share the track, see and bump into each other) as replays.

Each car drives in its own ``DrivingEnv`` (the environment its model was trained in: same
physics, sensors and observation), and :class:`RaceWorld` puts all those simulations on
one track. The result is one replay per car, in the normal replay format, so the replay
viewer shows a race exactly like a ghost race; the metadata records positions, outcomes
and contacts.

Replays of a race cannot be re-simulated one car at a time (each car's path depends on the
others), so they are marked ``multi_car`` and ``verify_replay`` refuses them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from neurodrive.experiments.metadata import git_commit
from neurodrive.replay.format import FORMAT, Replay, ReplayError
from neurodrive.replay.lab import Entry
from neurodrive.replay.recorder import describe_kind, env_for, is_learned, network_info
from neurodrive.sim.race import RaceWorld

OUTCOME_TO_ENDING = {"finished": "lap_complete"}


def record_race(
    entries: list[Entry],
    track: str,
    laps: int = 2,
    max_seconds: float | None = None,
    log: Callable[[str], None] = print,
    with_network: bool = True,
) -> list[Replay]:
    if not entries:
        raise ReplayError("nothing to race")
    envs, controllers = [], []
    for _label, make, model_meta, _ in entries:
        env = env_for(model_meta, track, laps, max_seconds or 45.0 * laps + 30.0)
        env.reset(seed=0, options={"track": track, "perturb": False, "random_start": False})
        controllers.append(make(env))
        envs.append(env)
    world = RaceWorld([env.sim for env in envs])
    world.reset()
    for c in controllers:
        c.reset(0)
    n = len(envs)
    rows: list[list[list[float]]] = [[] for _ in range(n)]
    rays: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(n)]
    actions: list[list[np.ndarray]] = [[] for _ in range(n)]

    def observe(i: int) -> np.ndarray:
        envs[i].obstacles = world.obstacles_for(i)  # the autopilot reads these
        return envs[i].obs_builder.build(envs[i].obstacles)

    obs = [observe(i) for i in range(len(envs))]

    def keep(i: int) -> None:
        sim = envs[i].sim
        s = sim.state
        rows[i].append(
            [s.x, s.y, s.heading, s.speed, s.steer, sim.progress.lap_fraction, sim.progress.laps]
        )
        hits = envs[i].obs_builder.last_hits
        assert hits is not None
        rays[i].append((hits.origin, hits.distances, hits.directions))

    for i in range(n):
        keep(i)
    while not world.done:
        step_actions: list[np.ndarray | None] = []
        for i in range(n):
            if world.status[i].racing:
                a = np.asarray(controllers[i].act(obs[i]), dtype=np.float64)
                actions[i].append(a)
                step_actions.append(a)
            else:
                step_actions.append(None)
        was_racing = [st.racing for st in world.status]
        world.step(step_actions)
        for i in range(n):
            if was_racing[i]:
                obs[i] = observe(i)
                keep(i)
    standings = world.standings()
    position = {car: p for p, car in enumerate(standings, start=1)}
    replays = []
    for i, ((label, _, _, path), env, controller) in enumerate(
        zip(entries, envs, controllers, strict=True)
    ):
        st = world.status[i]
        table = np.asarray(rows[i], dtype=np.float64)
        network, png = network_info(controller) if with_network else (None, None)
        meta: dict[str, Any] = {
            "format": FORMAT,
            "label": label,
            "controller_name": controller.name,
            "controller_kind": controller.kind,
            "kind_description": describe_kind(controller.kind),
            "learned": is_learned(controller.kind),
            "model_path": path or None,
            "track": env.track.id,
            "track_name": env.track.spec.name,
            "multi_car": True,
            "env_config": env.config.model_dump(mode="json"),
            "dt": env.sim.config.dt,
            "ray_max_range": env.obs_builder.sensor.max_range,
            "termination_reason": OUTCOME_TO_ENDING.get(str(st.outcome), str(st.outcome)),
            "lap_times": st.lap_times,
            "finish_time": st.finish_time,
            "steps": len(actions[i]),
            "race": {
                "cars": n,
                "laps": laps,
                "position": position[i],
                "outcome": st.outcome,
                "contacts": st.contacts,
            },
            "network": network,
            "git_commit": git_commit(),
        }
        replays.append(
            Replay(
                meta=meta,
                x=table[:, 0],
                y=table[:, 1],
                heading=table[:, 2],
                speed=table[:, 3],
                steer=table[:, 4],
                lap_fraction=table[:, 5],
                laps=table[:, 6].astype(np.int32),
                actions=np.asarray(actions[i], dtype=np.float64).reshape(-1, 2),
                ray_origin=np.asarray([r[0] for r in rays[i]]),
                ray_dist=np.asarray([r[1] for r in rays[i]]),
                ray_dirs=np.asarray([r[2] for r in rays[i]]),
                network_png=png,
            )
        )
    for i in standings:
        st = world.status[i]
        result = f"{st.finish_time:6.2f} s" if st.finish_time is not None else st.outcome
        log(f"  P{position[i]}  {entries[i][0]:<20} {result:<12} contacts {st.contacts}")
    return replays
