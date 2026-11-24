"""Command-line entry point: ``python -m neurodrive <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from neurodrive import __version__
from neurodrive.env.config import ConfigError

if TYPE_CHECKING:
    from neurodrive.agents.base import Controller
    from neurodrive.env.driving_env import DrivingEnv
from neurodrive.sim.track import TrackError, builtin_track_ids, load_track


def _cmd_play(args: argparse.Namespace) -> int:
    track = load_track(args.track)
    from neurodrive.ui.app import PlaygroundApp  # imported lazily: pygame is UI-only

    model = None
    driver = args.driver
    if args.model:
        from neurodrive.agents.model_io import load_controller

        model = load_controller(args.model)
        driver = "model"
    PlaygroundApp(
        track, controller=driver, size=(args.width, args.height), model_controller=model
    ).run()
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir or f"runs/{args.algorithm}")
    if args.algorithm == "neat":
        from neurodrive.training.config import NEATExperiment, load_neat_config
        from neurodrive.training.neat_trainer import train_neat

        if args.timesteps is not None:
            raise ValueError("--timesteps is for PPO; NEAT uses --generations")
        nexp = load_neat_config(args.config) if args.config else NEATExperiment()
        ndata = nexp.model_dump()
        overrides = {"seed": args.seed, "generations": args.generations}
        ndata["neat"].update({k: v for k, v in overrides.items() if v is not None})
        if args.name:
            ndata["name"] = args.name
        train_neat(
            NEATExperiment.model_validate(ndata), runs_root=runs_dir, resume_from=args.resume
        )
        return 0

    from neurodrive.training.config import PPOExperiment, load_ppo_config
    from neurodrive.training.ppo import train_ppo

    if args.generations is not None:
        raise ValueError("--generations is for NEAT; PPO uses --timesteps")
    exp = load_ppo_config(args.config) if args.config else PPOExperiment()
    overrides = {"seed": args.seed, "total_timesteps": args.timesteps}
    data = exp.model_dump()
    data["training"].update({k: v for k, v in overrides.items() if v is not None})
    if args.name:
        data["name"] = args.name
    exp = PPOExperiment.model_validate(data)
    train_ppo(exp, runs_root=runs_dir, resume_from=args.resume)
    return 0


def _cmd_show_network(args: argparse.Namespace) -> int:
    from neurodrive.agents.model_io import ModelError, load_controller
    from neurodrive.agents.neat_agent import NEATController
    from neurodrive.viz.network import draw_network

    controller = load_controller(args.model)
    if not isinstance(controller, NEATController):
        raise ModelError("show-network draws NEAT networks; this model is not one")
    out = Path(args.out or Path(args.model).with_suffix(".png"))
    draw_network(
        controller.genome,
        controller.config,
        out,
        input_names=controller.meta.get("observation", {}).get("names"),
        title=controller.name,
    )
    print(f"wrote {out}")
    return 0


def _cmd_bench_sim(args: argparse.Namespace) -> int:
    from neurodrive.bench import bench, environment_summary

    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    track = load_track(args.track)
    print(f"track: {track.id} ({track.length:.1f} m)   steps per mode: {args.steps}")
    print(environment_summary())
    for r in bench(track, args.steps, seed=args.seed):
        print(
            f"  {r.mode:<22} {r.steps_per_second:>10,.0f} steps/s   "
            f"({r.seconds:.2f} s, {r.resets} resets)"
        )
    return 0


def _cmd_validate_track(args: argparse.Namespace) -> int:
    track = load_track(args.track)
    print(
        f"OK: '{track.id}' ({track.spec.name}) length {track.length:.1f} m, "
        f"width {track.spec.width:g} m, {track.num_points} centerline points, "
        f"{len(track.checkpoint_s)} checkpoints, tightest turn radius "
        f"{1.0 / max(abs(track.curvature).max(), 1e-9):.1f} m"
    )
    return 0


CONTROLLERS = ("random", "smoothed_random", "heuristic", "autopilot")


def make_controller(name: str, env: DrivingEnv, seed: int) -> Controller:
    from neurodrive.agents import baselines as b

    if name == "random":
        return b.RandomController(seed)
    if name == "smoothed_random":
        return b.SmoothedRandomController(seed)
    if name == "heuristic":
        return b.HeuristicController(angles_deg=env.config.sensors.angles_deg)
    if name == "autopilot":
        return b.PrivilegedAutopilot(env)
    raise ValueError(f"unknown controller '{name}' (choose from {', '.join(CONTROLLERS)})")


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from neurodrive.env.config import EnvConfig, load_env_config
    from neurodrive.env.driving_env import DrivingEnv
    from neurodrive.evaluation.evaluator import evaluate, write_results
    from neurodrive.evaluation.scenarios import scenario_seeds
    from neurodrive.experiments.metadata import run_metadata

    meta = None
    if args.model:
        from neurodrive.agents.model_io import read_model_meta

        meta = read_model_meta(Path(args.model))
        config = (
            load_env_config(args.config)
            if args.config
            else EnvConfig.model_validate(meta["env_config"])
        )
    else:
        config = load_env_config(args.config) if args.config else EnvConfig()
    if args.track:
        config = config.model_copy(update={"track": args.track})
    env = DrivingEnv(config)
    controller: Controller
    if meta is not None:
        from neurodrive.agents.model_io import check_compatible, load_controller

        check_compatible(meta, env)
        controller = load_controller(args.model)
        label = f"{meta.get('experiment', 'model')}_{meta.get('label', '').replace('@', '-')}"
    else:
        controller = make_controller(args.controller, env, args.seed)
        label = args.controller
    seeds = scenario_seeds(args.split, args.episodes)
    out = Path(args.out or f"runs/eval/{label}_{config.track}_{args.split}")

    def show(i: int, n: int) -> None:
        if i == n or i % max(1, n // 10) == 0:
            print(f"  {i}/{n} episodes", flush=True)

    print(f"evaluating {controller.name} [{controller.kind}] on {config.track} ({args.split})")
    records = evaluate(env, controller, seeds, progress=show)
    meta = run_metadata() | {
        "env_config": config.model_dump(mode="json"),
        "observation": env.obs_builder.spec.to_dict(),
        "split": args.split,
        "scenario_seeds": [seeds[0], seeds[-1]],
        "controller_seed": args.seed,
        "model": str(args.model) if args.model else None,
        "model_meta": meta,
    }
    s = write_results(records, out, meta)
    lo, hi = s["completion_rate_ci95"]
    lap = (
        f"{s['lap_time_mean']:.2f} s mean, {s['lap_time_best']:.2f} s best "
        f"(over {s['lap_time_n']} completed laps)"
        if s["lap_time_n"]
        else "no completed laps"
    )
    print(
        f"completed {s['completed']}/{s['episodes']} "
        f"({100 * s['completion_rate']:.1f}%, 95% CI {100 * lo:.1f}-{100 * hi:.1f}%)\n"
        f"lap time: {lap}\n"
        f"mean lap progress: {100 * s['mean_lap_fraction']:.1f}%   "
        f"collision rate: {100 * s['collision_rate']:.1f}%   "
        f"endings: {s['termination_reasons']}\n"
        f"wrote {out / 'episodes.csv'} and {out / 'summary.json'}"
    )
    return 0


def _cmd_list_tracks(_: argparse.Namespace) -> int:
    for tid in builtin_track_ids():
        print(tid)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurodrive", description="NeuroDrive Lab")
    p.add_argument("--version", action="version", version=f"neurodrive {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="open the interactive driving playground")
    play.add_argument("--track", default="beginner_oval", help="built-in track id or YAML path")
    play.add_argument(
        "--driver",
        choices=["keyboard", "heuristic", "smoothed_random", "autopilot"],
        default="keyboard",
    )
    play.add_argument("--width", type=int, default=1280)
    play.add_argument("--height", type=int, default=800)
    play.add_argument("--model", default=None, help="drive with a trained model .zip")
    play.set_defaults(func=_cmd_play)

    bench = sub.add_parser("bench-sim", help="measure headless simulation steps per second")
    bench.add_argument("--track", default="beginner_oval")
    bench.add_argument("--steps", type=int, default=100_000)
    bench.add_argument("--seed", type=int, default=0)
    bench.set_defaults(func=_cmd_bench_sim)

    val = sub.add_parser("validate-track", help="load a track and run all validity checks")
    val.add_argument("track", help="built-in track id or YAML path")
    val.set_defaults(func=_cmd_validate_track)

    lst = sub.add_parser("list-tracks", help="list built-in tracks")
    lst.set_defaults(func=_cmd_list_tracks)

    ev = sub.add_parser("evaluate", help="evaluate a controller on fixed scenarios")
    who = ev.add_mutually_exclusive_group(required=True)
    who.add_argument("--controller", choices=CONTROLLERS, help="a baseline driver")
    who.add_argument("--model", help="a trained model .zip (uses its training env config)")
    ev.add_argument("--track", default=None, help="override the config's track")
    ev.add_argument("--split", choices=["validation", "test"], default="validation")
    ev.add_argument("--episodes", type=int, default=50)
    ev.add_argument("--config", default=None, help="environment YAML (default: built-in)")
    ev.add_argument("--seed", type=int, default=0, help="seed for random controllers")
    ev.add_argument("--out", default=None, help="output folder (default under runs/eval/)")
    ev.set_defaults(func=_cmd_evaluate)

    tr = sub.add_parser("train", help="train a learning agent")
    tr.add_argument("algorithm", choices=["ppo", "neat"])
    tr.add_argument("--config", default=None, help="experiment YAML, e.g. configs/ppo/default.yaml")
    tr.add_argument("--generations", type=int, default=None, help="NEAT: override generations")
    tr.add_argument("--seed", type=int, default=None, help="override training.seed")
    tr.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    tr.add_argument("--name", default=None, help="override the experiment name")
    tr.add_argument("--runs-dir", default=None, help="parent folder (default runs/<algo>)")
    tr.add_argument("--resume", default=None, help="continue from a checkpoint file")
    tr.set_defaults(func=_cmd_train)

    sn = sub.add_parser("show-network", help="draw a NEAT model's network to a PNG")
    sn.add_argument("model", help="a NEAT model .pkl")
    sn.add_argument("--out", default=None, help="PNG path (default: next to the model)")
    sn.set_defaults(func=_cmd_show_network)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (TrackError, ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
