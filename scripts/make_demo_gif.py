"""Render the README demo GIF: a ghost race, drawn off-screen by the real replay viewer.

    uv run python scripts/make_demo_gif.py [--benchmark runs/benchmark/ppo_vs_neat_1m]
    uv run python scripts/make_demo_gif.py --real --driver "PPO=model.zip" --driver autopilot ...

Needs a finished benchmark (for the trained models); writes docs/images/demo-ghost-race.gif.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from PIL import Image

from neurodrive.replay.lab import baseline_entry, benchmark_entries, driver_entry
from neurodrive.replay.race import record_race
from neurodrive.replay.recorder import record_many
from neurodrive.ui import art
from neurodrive.ui import renderer as R
from neurodrive.ui.clock import SPEEDS
from neurodrive.ui.replay_viewer import (
    COLORS,
    LEARNED_TAG,
    SCRIPTED_TAG,
    ReplayViewer,
)


def _palette(sample: Image.Image) -> Image.Image:
    """A 256-colour palette that always contains the exact car and UI colours (a plain
    median cut drops them: cars are a tiny fraction of the pixels), filled up from a
    sample frame."""
    keys = [*COLORS, LEARNED_TAG, SCRIPTED_TAG, (34, 40, 50), (50, 56, 66)]
    keys += [getattr(R, n) for n in dir(R) if n.isupper() and isinstance(getattr(R, n), tuple)]
    keys = list(dict.fromkeys(k for k in keys if len(k) == 3))
    rest = sample.quantize(colors=256 - len(keys), method=Image.Quantize.MEDIANCUT)
    flat = [c for k in keys for c in k] + rest.getpalette()[: 3 * (256 - len(keys))]
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat)
    return pal


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="runs/benchmark/ppo_vs_neat_1m")
    p.add_argument(
        "--driver", action="append", help="instead of --benchmark: model, baseline or LABEL=either"
    )
    p.add_argument("--track", default="unseen_a")
    p.add_argument("--out", default="docs/images/demo-ghost-race.gif")
    p.add_argument("--speed", type=float, default=2.0, help="playback speed")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--real", action="store_true", help="a real race (cars collide)")
    p.add_argument("--laps", type=int, default=1)
    p.add_argument("--no-baselines", action="store_true", help="only the benchmark models")
    p.add_argument("--camera", choices=["track", "follow", "tv"], default="track")
    p.add_argument("--focus", type=int, default=0, help="car to follow (follow camera)")
    p.add_argument("--start", type=float, default=None, help="first second to include")
    p.add_argument("--end", type=float, default=None, help="last second to include")
    p.add_argument("--crop", action="store_true", help="track view only (no side panel)")
    args = p.parse_args()

    entries = (
        [driver_entry(d) for d in args.driver] if args.driver else benchmark_entries(args.benchmark)
    )
    if not args.no_baselines and not args.driver:
        entries += [baseline_entry("heuristic"), baseline_entry("autopilot")]
    if args.real:
        replays = record_race(entries, args.track, laps=args.laps, with_network=False)
        title = "Race"
    else:
        replays = record_many(entries, args.track, 0)
        title = "Ghost race"
    art.GRAIN = 0  # fine texture grain is invisible in a small GIF but ruins compression
    viewer = ReplayViewer(replays, title=title, size=(1400, 860))
    viewer.camera_mode = args.camera
    viewer.focus = args.focus
    if args.speed in SPEEDS:
        viewer.speed_index = SPEEDS.index(args.speed)
    frames = []
    step = args.speed / args.fps
    t = viewer.t_start
    first = viewer.t_start if args.start is None else args.start
    last = viewer.duration if args.end is None else min(args.end, viewer.duration)
    while True:
        viewer.seek(t)
        viewer.draw()
        if t >= first:
            raw = pygame.image.tobytes(viewer.screen, "RGB")
            img = Image.frombytes("RGB", viewer.screen.get_size(), raw)
            if args.crop:
                v = viewer.track_view.viewport
                img = img.crop((v.left, v.top, v.right, v.bottom))
            h = round(img.height * args.width / img.width)
            frames.append(img.resize((args.width, h), Image.Resampling.LANCZOS))
        if t >= last:
            break
        t = min(t + step, last)
    pygame.quit()
    palette = _palette(frames[len(frames) // 2])
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    durations = [round(1000 / args.fps)] * (len(quantized) - 1) + [2500]  # hold the result
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        out, save_all=True, append_images=quantized[1:], duration=durations, loop=0, optimize=True
    )
    print(f"wrote {out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
