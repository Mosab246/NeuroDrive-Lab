"""Render the README demo GIF: a ghost race, drawn off-screen by the real replay viewer.

    uv run python scripts/make_demo_gif.py [--benchmark runs/benchmark/ppo_vs_neat_1m]

Needs a finished benchmark (for the trained models); writes docs/images/demo-ghost-race.gif.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from PIL import Image

from neurodrive.replay.lab import baseline_entry, benchmark_entries
from neurodrive.replay.recorder import record_many
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
    p.add_argument("--track", default="unseen_a")
    p.add_argument("--out", default="docs/images/demo-ghost-race.gif")
    p.add_argument("--speed", type=float, default=2.0, help="playback speed")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=900)
    args = p.parse_args()

    entries = benchmark_entries(args.benchmark)
    entries += [baseline_entry("heuristic"), baseline_entry("autopilot")]
    replays = record_many(entries, args.track, 0)
    viewer = ReplayViewer(replays, title="Ghost race", size=(1400, 860))
    if args.speed in SPEEDS:
        viewer.speed_index = SPEEDS.index(args.speed)
    frames = []
    step = args.speed / args.fps
    t = 0.0
    while True:
        viewer.seek(t)
        viewer.draw()
        raw = pygame.image.tobytes(viewer.screen, "RGB")
        img = Image.frombytes("RGB", viewer.screen.get_size(), raw)
        h = round(img.height * args.width / img.width)
        frames.append(img.resize((args.width, h), Image.Resampling.LANCZOS))
        if t >= viewer.duration:
            break
        t = min(t + step, viewer.duration)
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
