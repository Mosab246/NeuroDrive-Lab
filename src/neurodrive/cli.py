"""Command-line entry point: ``python -m neurodrive <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from neurodrive import __version__
from neurodrive.sim.track import TrackError, builtin_track_ids, load_track


def _cmd_play(args: argparse.Namespace) -> int:
    track = load_track(args.track)
    from neurodrive.ui.app import PlaygroundApp  # imported lazily: pygame is UI-only

    PlaygroundApp(track, controller=args.driver, size=(args.width, args.height)).run()
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
    play.add_argument("--driver", choices=["keyboard", "autopilot"], default="keyboard")
    play.add_argument("--width", type=int, default=1280)
    play.add_argument("--height", type=int, default=800)
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
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (TrackError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
