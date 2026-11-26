"""Record episodes into replays, and check that a replay re-simulates exactly.

Recording runs headless through the same ``DrivingEnv`` the evaluator uses, so a recorded
lap time is the lap time the evaluator would report for that scenario.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from neurodrive.agents.base import Controller
from neurodrive.env.config import EnvConfig
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.experiments.metadata import git_commit
from neurodrive.replay.format import FORMAT, Replay, ReplayError

#: How each controller kind is described on screen: learned or hand-written.
KIND_DESCRIPTION = {
    "ppo": "learned (PPO)",
    "sac": "learned (SAC)",
    "neat": "learned (NEAT)",
    "heuristic": "scripted (sensors only)",
    "scripted (privileged)": "scripted (privileged)",
    "random": "random",
    "human": "human",
}


def describe_kind(kind: str) -> str:
    return KIND_DESCRIPTION.get(kind, kind)


def is_learned(kind: str) -> bool:
    return kind in ("ppo", "sac", "neat")


def network_info(controller: Controller) -> tuple[dict[str, Any] | None, bytes | None]:
    """A description of the controller's network for the viewer: a drawing of the evolved
    NEAT network, or the layer sizes of PPO's policy network."""
    if controller.kind == "neat":
        from neurodrive.viz.network import draw_network

        meta = getattr(controller, "meta", {}) or {}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "net.png"
            draw_network(
                controller.genome,  # type: ignore[attr-defined]
                controller.config,  # type: ignore[attr-defined]
                out,
                input_names=meta.get("observation", {}).get("names"),
                title=controller.name,
            )
            png = out.read_bytes()
        hidden, conns = controller.size()  # type: ignore[attr-defined]
        from neurodrive.viz.network import network_layout

        graph = network_layout(controller.genome, controller.config)  # type: ignore[attr-defined]
        graph["order"] = graph["inputs"] + graph["hidden"] + graph["outputs"]
        graph["input_names"] = meta.get("observation", {}).get("names")
        info = {"type": "neat", "hidden": hidden, "connections": conns, "graph": graph}
        return info, png
    if controller.kind == "ppo":
        policy = controller.model.policy  # type: ignore[attr-defined]
        layers = [m for m in policy.mlp_extractor.policy_net if hasattr(m, "out_features")]
        actor = sum(p.numel() for p in policy.mlp_extractor.policy_net.parameters()) + sum(
            p.numel() for p in policy.action_net.parameters()
        )
        info = {
            "type": "mlp",
            "inputs": int(policy.observation_space.shape[0]),
            "hidden": [int(m.out_features) for m in layers],
            "outputs": int(policy.action_net.out_features),
            "activation": policy.activation_fn.__name__,
            "parameters": int(actor),
        }
        return info, None
    if controller.kind == "sac":
        actor = controller.model.policy.actor  # type: ignore[attr-defined]
        hidden = [int(m.out_features) for m in actor.latent_pi if hasattr(m, "out_features")]
        info = {
            "type": "mlp",
            "inputs": int(actor.observation_space.shape[0]),
            "hidden": hidden,
            "outputs": int(actor.mu.out_features),
            "activation": "ReLU",
            "parameters": int(sum(p.numel() for p in actor.parameters())),
        }
        return info, None
    return None, None


def node_snapshot(controller: Controller, order: list[int] | None) -> list[float] | None:
    """The NEAT network's node values after its last ``act`` (None for other drivers)."""
    net = getattr(controller, "net", None)
    if order is None or net is None:
        return None
    return [float(net.values.get(k, 0.0)) for k in order]


def _frame(env: DrivingEnv) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    s = env.sim.state
    hits = env.obs_builder.last_hits
    assert hits is not None
    row = [s.x, s.y, s.heading, s.speed, s.steer, env.sim.progress.lap_fraction]
    return [*row, env.sim.progress.laps], hits.origin, hits.distances, hits.directions


def record_episode(
    env: DrivingEnv,
    controller: Controller,
    seed: int,
    *,
    track: str | None = None,
    random_start: bool = False,
    perturb: bool = True,
    label: str | None = None,
    model_path: str | None = None,
    with_network: bool = True,
) -> Replay:
    """Run one episode and return it as a replay."""
    options: dict[str, Any] = {"perturb": perturb, "random_start": random_start}
    if track is not None:
        options["track"] = track
    obs, info = env.reset(seed=seed, options=options)
    controller.reset(seed)
    rows, origins, dists, dirs = [], [], [], []
    actions: list[np.ndarray] = []

    def keep() -> None:
        row, o, d, u = _frame(env)
        rows.append(row)
        origins.append(o)
        dists.append(d)
        dirs.append(u)

    network, png = network_info(controller) if with_network else (None, None)
    order = (network or {}).get("graph", {}).get("order")
    nodes: list[list[float]] = []
    keep()
    while True:
        action = np.asarray(controller.act(obs), dtype=np.float64)
        actions.append(action)
        snap = node_snapshot(controller, order)
        if snap is not None:
            nodes.append(snap)
        obs, _, terminated, truncated, info = env.step(action)
        keep()
        if terminated or truncated:
            break
    table = np.asarray(rows, dtype=np.float64)
    model_meta = getattr(controller, "meta", None) or {}
    meta = {
        "format": FORMAT,
        "label": label or controller.name,
        "controller_name": controller.name,
        "controller_kind": controller.kind,
        "kind_description": describe_kind(controller.kind),
        "learned": is_learned(controller.kind),
        "model_path": model_path,
        "model_label": model_meta.get("label"),
        "track": env.track.id,
        "track_name": env.track.spec.name,
        "scenario_seed": seed,
        "random_start": random_start,
        "perturb": perturb,
        "env_config": env.config.model_dump(mode="json"),
        "dt": env.sim.config.dt,
        "ray_max_range": env.obs_builder.sensor.max_range,
        "termination_reason": info["termination_reason"],
        "lap_times": [float(t) for t in info["lap_times"]],
        "steps": int(info["steps"]),
        "network": network,
        "git_commit": git_commit(),
    }
    return Replay(
        meta=meta,
        x=table[:, 0],
        y=table[:, 1],
        heading=table[:, 2],
        speed=table[:, 3],
        steer=table[:, 4],
        lap_fraction=table[:, 5],
        laps=table[:, 6].astype(np.int32),
        actions=np.asarray(actions, dtype=np.float64).reshape(-1, 2),
        ray_origin=np.asarray(origins, dtype=np.float64),
        ray_dist=np.asarray(dists, dtype=np.float64),
        ray_dirs=np.asarray(dirs, dtype=np.float64),
        network_png=png,
        node_values=np.asarray(nodes, dtype=np.float32) if nodes else None,
    )


def env_for(
    meta: dict[str, Any] | None,
    track: str,
    laps: int | None = None,
    max_seconds: float | None = None,
) -> DrivingEnv:
    """The environment a controller should be recorded in: the model's own environment
    config (sensors, observation) if it has one, on the requested track."""
    base = EnvConfig.model_validate(meta["env_config"]) if meta else EnvConfig()
    update: dict[str, Any] = {"tracks": (track,), "track": track}
    if laps is not None:
        update["laps_to_finish"] = laps
    if max_seconds is not None:
        update["max_episode_seconds"] = max_seconds
    return DrivingEnv(base.model_copy(update=update))


def verify_replay(replay: Replay) -> None:
    """Re-simulate the stored actions from the stored scenario; raise ``ReplayError`` if any
    stored state differs from the re-simulated one."""
    m = replay.meta
    if m.get("multi_car"):
        raise ReplayError(
            "a race replay can't be re-simulated on its own: each car's path depends on "
            "the other cars"
        )
    env = DrivingEnv(EnvConfig.model_validate(m["env_config"]))
    options = {"perturb": m["perturb"], "random_start": m["random_start"], "track": m["track"]}
    env.reset(seed=m["scenario_seed"], options=options)
    info: dict[str, Any] = {}
    for i in range(replay.steps + 1):
        if i > 0:
            _, _, _, _, info = env.step(replay.actions[i - 1])
        s = env.sim.state
        got = (s.x, s.y, s.heading, s.speed, s.steer)
        want = (replay.x[i], replay.y[i], replay.heading[i], replay.speed[i], replay.steer[i])
        if got != want:
            raise ReplayError(f"replay diverges from the simulator at step {i}")
    if info.get("termination_reason") != m["termination_reason"]:
        raise ReplayError("replay ending does not match the re-simulated ending")


def record_many(
    entries: list[tuple[str, Callable[[DrivingEnv], Controller], dict[str, Any] | None, str]],
    track: str,
    seed: int,
    *,
    random_start: bool = False,
    laps: int | None = None,
    log: Callable[[str], None] = print,
) -> list[Replay]:
    """Record several drivers on the same scenario. Each entry is
    ``(label, make_controller(env), model_meta or None, model_path or "")``."""
    replays = []
    for label, make, meta, path in entries:
        env = env_for(meta, track, laps)
        controller = make(env)
        rep = record_episode(
            env,
            controller,
            seed,
            random_start=random_start,
            label=label,
            model_path=path or None,
        )
        end = rep.meta["termination_reason"]
        lap = f" {rep.meta['lap_times'][0]:.2f} s" if rep.meta["lap_times"] else ""
        log(f"  recorded {label:<22} {end}{lap}")
        replays.append(rep)
    if not replays:
        raise ReplayError("nothing to record")
    return replays
