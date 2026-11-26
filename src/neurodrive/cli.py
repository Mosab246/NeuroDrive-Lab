"""Command-line entry point: ``python -m neurodrive <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        overrides = {
            "seed": args.seed,
            "generations": args.generations,
            "validation_tracks": _val_tracks(args),
            "validation_episodes": args.validation_episodes,
        }
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
    overrides = {
        "seed": args.seed,
        "total_timesteps": args.timesteps,
        "validation_tracks": _val_tracks(args),
        "eval_episodes": args.validation_episodes,
    }
    data = exp.model_dump()
    data["training"].update({k: v for k, v in overrides.items() if v is not None})
    if args.name:
        data["name"] = args.name
    exp = PPOExperiment.model_validate(data)
    train_ppo(exp, runs_root=runs_dir, resume_from=args.resume)
    return 0


def _val_tracks(args: argparse.Namespace) -> tuple[str, ...] | None:
    spec = getattr(args, "validation_tracks", None)
    return tuple(expand_tracks(spec)) if spec else None


def _cmd_reselect(args: argparse.Namespace) -> int:
    from neurodrive.benchmark.reselect import reselect_runs

    reselect_runs([Path(r) for r in args.runs], expand_tracks(args.validation_tracks),
                  args.episodes, args.workers)  # fmt: skip
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from neurodrive.benchmark.report import build_report

    if args.action == "reselect":
        from neurodrive.benchmark.reselect import reselect_benchmark

        if not args.dir:
            raise ValueError("benchmark reselect needs --dir runs/benchmark/<name>")
        bench_dir = Path(args.dir)
        out = Path(args.out or f"results/benchmark/{bench_dir.name}")
        reselect_benchmark(
            bench_dir, out, expand_tracks(args.validation_tracks), args.episodes, args.workers
        )
        return 0
    if args.action == "run":
        from neurodrive.benchmark.config import load_benchmark_config
        from neurodrive.benchmark.runner import Benchmark

        if not args.config:
            raise ValueError("benchmark run needs --config")
        cfg = load_benchmark_config(args.config)
        bench_dir = Benchmark(cfg, root=Path(args.root)).run()
    else:
        if not args.dir:
            raise ValueError("benchmark report needs --dir runs/benchmark/<name>")
        bench_dir = Path(args.dir)
        if not (bench_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"no manifest.json in {bench_dir}")
    out = Path(args.out or f"results/benchmark/{bench_dir.name}")
    build_report(bench_dir, out)
    print(f"report written to {out / 'report.md'}")
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
        config = config.model_copy(update={"track": args.track, "tracks": None})
    if args.traffic is not None:
        traffic = config.traffic.model_copy(update={"cars": args.traffic})
        config = config.model_copy(update={"traffic": traffic})
    track_list = expand_tracks(args.tracks) if args.tracks else None
    random_start = {"random": True, "line": False}.get(args.start or "")
    env = DrivingEnv(
        config.model_copy(update={"tracks": tuple(track_list)}) if track_list else config
    )
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
    where = "multi" if track_list else config.track
    out = Path(args.out or f"runs/eval/{label}_{where}_{args.split}")

    def show(i: int, n: int) -> None:
        if i == n or i % max(1, n // 10) == 0:
            print(f"  {i}/{n} episodes", flush=True)

    on = f"{len(track_list)} tracks" if track_list else config.track
    print(f"evaluating {controller.name} [{controller.kind}] on {on} ({args.split} scenarios)")
    records = evaluate(
        env, controller, seeds, progress=show, tracks=track_list, random_start=random_start
    )
    meta = run_metadata() | {
        "env_config": config.model_dump(mode="json"),
        "observation": env.obs_builder.spec.to_dict(),
        "split": args.split,
        "tracks": track_list or [config.track],
        "random_start": random_start,
        "scenario_seeds": [seeds[0], seeds[-1]],
        "controller_seed": args.seed,
        "model": str(args.model) if args.model else None,
        "model_meta": meta,
    }
    s = write_results(records, out, meta)
    if track_list:
        from neurodrive.evaluation.metrics import summarize

        print(f"{'track':<20} {'laps':>9}  {'lap time':>9}  {'progress':>8}  {'hits':>5}")
        for tid in track_list:
            t: dict[str, Any] = summarize([r for r in records if r.track == tid])
            lap_t = f"{t['lap_time_mean']:.2f} s" if t["lap_time_n"] else "-"
            print(
                f"{tid:<20} {t['completed']:>4}/{t['episodes']:<4} {lap_t:>9}  "
                f"{100 * t['mean_lap_fraction']:7.1f}%  {t['mean_contacts']:5.2f}"
            )
        print("overall:")
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
        f"endings: {s['termination_reasons']}"
    )
    if config.traffic.cars:
        print(f"hits on other cars: {s['mean_contacts']:.2f} per episode")
    print(f"wrote {out / 'episodes.csv'} and {out / 'summary.json'}")
    return 0


def expand_tracks(spec: str) -> list[str]:
    """Comma-separated track ids, plus shortcuts: 'split:test' = every built-in track in that
    split, 'proc:20' = procedural tracks proc_0 .. proc_19."""
    from neurodrive.sim.track import load_track

    out: list[str] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part.startswith("split:"):
            split = part[6:]
            ids = [t for t in builtin_track_ids() if load_track(t).spec.split == split]
            if not ids:
                raise ValueError(f"no built-in tracks with split '{split}'")
            out += ids
        elif part.startswith("pval:"):
            n = int(part[5:])
            if n < 1:
                raise ValueError("pval:N needs N >= 1")
            out += [f"pval_{i}" for i in range(n)]
        elif part.startswith("proc:"):
            n = int(part[5:])
            if n < 1:
                raise ValueError("proc:N needs N >= 1")
            out += [f"proc_{i}" for i in range(n)]
        else:
            out.append(part)
    return list(dict.fromkeys(out))


def _show_replays(replays: list, title: str, args: argparse.Namespace) -> int:
    if getattr(args, "save_dir", None):
        out = Path(args.save_dir)
        for i, rep in enumerate(replays):
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in rep.label)
            print(f"wrote {rep.save(out / f'{i:02d}_{safe}.npz')}")
    if getattr(args, "no_window", False):
        return 0
    if args.video:
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # render off-screen
        from neurodrive.ui.replay_viewer import ReplayViewer
        from neurodrive.ui.video import export_video

        viewer = ReplayViewer(replays, title=title, size=(args.width, args.height))
        viewer.camera_mode = args.camera
        export_video(viewer, args.video, fps=args.fps, speed=args.video_speed)
        return 0
    from neurodrive.ui.replay_viewer import ReplayViewer

    viewer = ReplayViewer(replays, title=title, size=(args.width, args.height))
    viewer.camera_mode = args.camera
    frames = viewer.run(max_frames=args.frames)
    s = viewer.frame_time_stats()
    print(
        f"viewer: {frames} frames at {s['average_fps']:.1f} FPS average; draw time mean "
        f"{s['mean_ms']:.1f} ms, 95th percentile {s['p95_ms']:.1f} ms"
    )
    return 0


def _record(entries: list, args: argparse.Namespace) -> list:
    from neurodrive.replay.recorder import record_many

    if len(entries) > 8:
        raise ValueError(f"{len(entries)} drivers chosen; the viewer shows at most 8")
    if getattr(args, "real", False):
        from neurodrive.replay.race import record_race

        laps = args.laps or 2
        print(f"racing {len(entries)} cars on {args.track}, {laps} laps (cars can touch)")
        return record_race(entries, args.track, laps=laps)
    print(f"recording {len(entries)} driver(s) on {args.track}, scenario seed {args.seed}")
    return record_many(
        entries, args.track, args.seed, random_start=args.start == "random", laps=args.laps
    )


def _cmd_replay(args: argparse.Namespace) -> int:
    from neurodrive.replay.format import load_replay
    from neurodrive.replay.recorder import verify_replay

    replays = [load_replay(f) for f in args.files]
    if args.verify:
        for f, rep in zip(args.files, replays, strict=True):
            verify_replay(rep)
            print(f"verified {f}: re-simulation matches all {rep.steps} steps")
    return _show_replays(replays, "Replay", args)


def _cmd_record(args: argparse.Namespace) -> int:
    from neurodrive.replay.lab import driver_entry

    rep = _record([driver_entry(args.driver)], args)[0]
    print(f"wrote {rep.save(args.out)}")
    return 0


def _cmd_race(args: argparse.Namespace) -> int:
    from neurodrive.replay.lab import benchmark_entries, driver_entry

    entries = []
    if args.benchmark:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        entries += benchmark_entries(args.benchmark, seeds)
    entries += [driver_entry(d) for d in args.driver or []]
    if not entries:
        raise ValueError("choose drivers with --driver and/or --benchmark")
    title = "Race" if args.real else "Ghost race"
    return _show_replays(_record(entries, args), title, args)


def _cmd_timeline(args: argparse.Namespace) -> int:
    from neurodrive.replay.lab import timeline_entries

    entries = timeline_entries(args.run, args.count)
    if args.timelapse:
        return _timelapse(_record(entries, args), args)
    return _show_replays(_record(entries, args), "Learning timeline", args)


def _timelapse(replays: list, args: argparse.Namespace) -> int:
    """Each snapshot on its own, one after another, captioned: how the driver improved."""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from neurodrive.ui.replay_viewer import ReplayViewer
    from neurodrive.ui.video import export_timelapse

    clips = []
    for rep in replays:
        end = rep.meta["termination_reason"]
        if end == "lap_complete":
            result = f"lap in {rep.meta['lap_times'][0]:.2f} s"
        else:
            result = f"{end.replace('_', ' ')} after {rep.duration:.1f} s"
        viewer = ReplayViewer([rep], title="Learning time-lapse", size=(args.width, args.height))
        viewer.camera_mode = args.camera
        clips.append((viewer, f"{rep.label}: {result}"))
    export_timelapse(clips, args.timelapse, fps=args.fps, speed=args.video_speed)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from neurodrive.replay.lab import DEFAULT_BENCHMARK, baseline_entry, benchmark_entries

    entries = []
    if (DEFAULT_BENCHMARK / "manifest.json").is_file():
        entries = benchmark_entries(DEFAULT_BENCHMARK, [0, 1, 2])
    else:
        print(
            "no trained benchmark models found; showing the scripted drivers only. "
            "Train some with `neurodrive benchmark run --config configs/benchmark/default.yaml`."
        )
    entries += [baseline_entry("heuristic"), baseline_entry("autopilot")]
    title = "Race" if args.real else "Ghost race"
    return _show_replays(_record(entries, args), title, args)


def _cmd_grand_prix(args: argparse.Namespace) -> int:
    from neurodrive.replay.grand_prix import run_grand_prix
    from neurodrive.replay.lab import benchmark_entries, driver_entry

    entries = []
    if args.benchmark:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        entries += benchmark_entries(args.benchmark, seeds)
    entries += [driver_entry(d) for d in args.driver or []]
    tracks = expand_tracks(args.tracks)
    run_grand_prix(entries, tracks, args.laps, Path(args.out))
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from neurodrive.benchmark.dashboard import build_dashboard

    out = build_dashboard(Path(args.results), Path(args.out))
    print(f"wrote {out}")
    return 0


def _cmd_export_web(args: argparse.Namespace) -> int:
    import json as _json

    from neurodrive.web_export import export_web

    models: list[tuple[str, Path]] = []
    for spec in args.model or []:
        label, _, path = spec.partition("=")
        models.append((label, Path(path)))
    if args.benchmark:
        manifest = _json.loads((Path(args.benchmark) / "manifest.json").read_text(encoding="utf-8"))
        for r in sorted(
            manifest["runs"].values(), key=lambda r: (r["algorithm"] != "ppo", r["seed"])
        ):
            run = Path(r["run_dir"])
            ext = ".zip" if r["algorithm"] == "ppo" else ".pkl"
            model = run / f"best_model_v2{ext}"
            model = model if model.is_file() else run / f"best_model{ext}"
            models.append((f"{r['algorithm'].upper()} #{r['seed'] + 1}", model))
    out = export_web(models, Path(args.out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(models)} drivers)")
    return 0


def _cmd_list_tracks(_: argparse.Namespace) -> int:
    for tid in builtin_track_ids():
        print(tid)
    return 0


def _reselect_args(a: argparse.ArgumentParser) -> None:
    from neurodrive.benchmark.reselect import DEFAULT_VALIDATION

    a.add_argument("--validation-tracks", default=DEFAULT_VALIDATION, help="re-selection tracks")
    a.add_argument("--episodes", type=int, default=5, help="validation scenarios per track")
    a.add_argument("--workers", type=int, default=8, help="parallel processes")


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
    ev.add_argument(
        "--tracks",
        default=None,
        help="evaluate on several tracks: ids, 'split:test', 'proc:20' (comma-separated)",
    )
    ev.add_argument(
        "--start", choices=["line", "random"], default=None, help="start position (default: config)"
    )
    ev.add_argument(
        "--traffic", type=int, default=None, help="opponent cars on track (default: config)"
    )
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
    tr.add_argument(
        "--validation-tracks",
        default=None,
        help="tracks for picking the best checkpoint, e.g. 'validation_circuit,pval:3,split:train'",
    )
    tr.add_argument(
        "--validation-episodes", type=int, default=None, help="validation scenarios per track"
    )
    tr.set_defaults(func=_cmd_train)

    bm = sub.add_parser("benchmark", help="run the PPO-vs-NEAT benchmark, or rebuild its report")
    bm.add_argument("action", choices=["run", "report", "reselect"])
    bm.add_argument("--config", default=None, help="benchmark YAML (for run)")
    bm.add_argument("--dir", default=None, help="benchmark folder (for report)")
    bm.add_argument("--root", default="runs/benchmark", help="parent folder for benchmarks")
    bm.add_argument("--out", default=None, help="report folder (default results/benchmark/<name>)")
    _reselect_args(bm)
    bm.set_defaults(func=_cmd_benchmark)

    rs = sub.add_parser("reselect", help="re-pick the best checkpoint of runs, wider validation")
    rs.add_argument("runs", nargs="+", help="PPO or NEAT run folders")
    _reselect_args(rs)
    rs.set_defaults(func=_cmd_reselect)

    sn = sub.add_parser("show-network", help="draw a NEAT model's network to a PNG")
    sn.add_argument("model", help="a NEAT model .pkl")
    sn.add_argument("--out", default=None, help="PNG path (default: next to the model)")
    sn.set_defaults(func=_cmd_show_network)

    def viewer_args(a: argparse.ArgumentParser, record: bool = True) -> None:
        if record:
            a.add_argument("--track", default="unseen_a", help="track id (default unseen_a)")
            a.add_argument("--seed", type=int, default=0, help="scenario seed (same for all)")
            a.add_argument("--start", choices=["line", "random"], default="line")
            a.add_argument("--laps", type=int, default=None, help="laps per car (default 1)")
            a.add_argument("--save-dir", default=None, help="also save the replays here")
            a.add_argument("--no-window", action="store_true", help="record only")
        a.add_argument("--width", type=int, default=1400)
        a.add_argument("--height", type=int, default=860)
        a.add_argument("--frames", type=int, default=None, help=argparse.SUPPRESS)
        a.add_argument(
            "--camera", choices=["track", "follow", "tv"], default="track", help="start camera"
        )
        a.add_argument("--video", default=None, help="write an MP4 instead of opening a window")
        a.add_argument("--fps", type=int, default=30, help="video frame rate")
        a.add_argument("--video-speed", type=float, default=1.0, help="video playback speed")

    rp = sub.add_parser("replay", help="play back saved replay files (several = a race)")
    rp.add_argument("files", nargs="+", help="replay .npz files on the same track")
    rp.add_argument("--verify", action="store_true", help="re-simulate and check each replay")
    viewer_args(rp, record=False)
    rp.set_defaults(func=_cmd_replay)

    rec = sub.add_parser("record", help="record one driver's episode to a replay file")
    rec.add_argument("--driver", required=True, help="model file, baseline name, or LABEL=either")
    rec.add_argument("--out", required=True, help="replay .npz path")
    viewer_args(rec)
    rec.set_defaults(func=_cmd_record)

    race = sub.add_parser("race", help="record several drivers on one scenario and watch them")
    race.add_argument(
        "--driver", action="append", help="model file, baseline name, or LABEL=either (repeat)"
    )
    race.add_argument("--benchmark", default=None, help="add every best model of a benchmark")
    race.add_argument("--seeds", default=None, help="with --benchmark: only these seeds, e.g. 0,1")
    race.add_argument(
        "--real", action="store_true", help="a real race: cars see and bump into each other"
    )
    viewer_args(race)
    race.set_defaults(func=_cmd_race)

    tl = sub.add_parser("timeline", help="race snapshots of one training run against each other")
    tl.add_argument("run", help="a PPO or NEAT run folder")
    tl.add_argument("--count", type=int, default=6, help="how many snapshots (default 6)")
    tl.add_argument(
        "--timelapse", default=None, help="write an MP4 playing the snapshots one after another"
    )
    viewer_args(tl)
    tl.set_defaults(func=_cmd_timeline)

    ex = sub.add_parser("export-web", help="export tracks and drivers for the browser game")
    ex.add_argument("--benchmark", default=None, help="add every best model of a benchmark")
    ex.add_argument("--model", action="append", help="LABEL=path to a model (repeat)")
    ex.add_argument("--out", default="web/data.js")
    ex.set_defaults(func=_cmd_export_web)

    db = sub.add_parser("dashboard", help="interactive HTML results dashboard")
    db.add_argument("--results", default="results", help="results folder")
    db.add_argument("--out", default="results/dashboard.html")
    db.set_defaults(func=_cmd_dashboard)

    gp = sub.add_parser("grand-prix", help="a championship of real races, grid rotated")
    gp.add_argument("--driver", action="append", help="model file, baseline, or LABEL=either")
    gp.add_argument("--benchmark", default=None, help="add every best model of a benchmark")
    gp.add_argument("--seeds", default=None, help="with --benchmark: only these seeds")
    gp.add_argument("--tracks", default="split:test,proc:10", help="one race per track")
    gp.add_argument("--laps", type=int, default=2)
    gp.add_argument("--out", default="results/grand_prix/latest", help="output folder")
    gp.set_defaults(func=_cmd_grand_prix)

    dm = sub.add_parser("demo", help="ghost race of the benchmark models and scripted drivers")
    dm.add_argument(
        "--real", action="store_true", help="a real race: cars see and bump into each other"
    )
    viewer_args(dm)
    dm.set_defaults(func=_cmd_demo)
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
