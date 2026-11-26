"""Who goes into a race or a learning timeline: turns command-line choices into a list of
drivers to record.

Each entry is ``(label, make_controller(env), model_meta or None, model_path)``, the shape
:func:`neurodrive.replay.recorder.record_many` takes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neurodrive.agents.base import Controller
from neurodrive.env.driving_env import DrivingEnv

Entry = tuple[str, Callable[[DrivingEnv], Controller], dict[str, Any] | None, str]
BASELINES = ("heuristic", "autopilot", "smoothed_random", "random")
DEFAULT_BENCHMARK = Path("runs/benchmark/ppo_vs_neat_1m")


def model_entry(path: str | Path, label: str | None = None) -> Entry:
    from neurodrive.agents.model_io import load_controller, read_model_meta

    path = Path(path)
    meta = read_model_meta(path)
    algo = str(meta.get("algorithm", "?")).upper()
    return (label or f"{algo} {path.stem}", lambda env: load_controller(path), meta, str(path))


def baseline_entry(name: str, label: str | None = None) -> Entry:
    from neurodrive.cli import make_controller

    if name not in BASELINES:
        raise ValueError(f"unknown driver '{name}' (a model file or one of {', '.join(BASELINES)})")
    return (label or name, lambda env: make_controller(name, env, 0), None, "")


def driver_entry(spec: str) -> Entry:
    """``heuristic``, ``path/to/model.zip``, or ``LABEL=either``."""
    label = None
    if "=" in spec:
        label, spec = spec.split("=", 1)
    if spec in BASELINES:
        return baseline_entry(spec, label)
    return model_entry(spec, label)


def benchmark_entries(bench_dir: str | Path, seeds: list[int] | None = None) -> list[Entry]:
    """The best model of every (algorithm, seed) in a finished benchmark."""
    bench_dir = Path(bench_dir)
    manifest = bench_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{manifest} not found: run `neurodrive benchmark run` first, or pass --driver"
        )
    runs = json.loads(manifest.read_text(encoding="utf-8"))["runs"]
    out = []
    for key in sorted(runs, key=lambda k: (runs[k]["algorithm"] != "ppo", runs[k]["seed"])):
        r = runs[key]
        if seeds is not None and r["seed"] not in seeds:
            continue
        ext = ".zip" if r["algorithm"] == "ppo" else ".pkl"
        run = Path(r["run_dir"])
        # Prefer the re-selected checkpoint (`neurodrive benchmark reselect`) when there is one.
        model = run / f"best_model_v2{ext}"
        if not model.is_file():
            model = run / f"best_model{ext}"
        if model.is_file():
            out.append(model_entry(model, f"{r['algorithm'].upper()} seed {r['seed']}"))
    return out


def _checkpoint_label(path: Path) -> tuple[int, str]:
    m = re.fullmatch(r"step_(\d+)", path.stem)
    if m:
        n = int(m.group(1))
        if n < 1000:
            return n, f"step {n}"
        return n, f"step {n // 1000}k" if n < 1_000_000 else f"step {n / 1e6:.2f}M"
    m = re.fullmatch(r"gen_(\d+)", path.stem)
    if m:
        return int(m.group(1)), f"gen {int(m.group(1))}"
    raise ValueError(f"not a checkpoint name: {path.name}")


def timeline_entries(run_dir: str | Path, count: int = 6) -> list[Entry]:
    """Evenly spaced snapshots from one training run (first and last included), oldest
    first, plus the run's selected best model."""
    run_dir = Path(run_dir)
    files = sorted((run_dir / "checkpoints").glob("step_*.zip"))  # PPO
    if not files:
        files = sorted((run_dir / "champions").glob("gen_*.pkl"))  # NEAT
    if not files:
        raise FileNotFoundError(f"{run_dir}: no PPO checkpoints or NEAT champions found")
    files.sort(key=lambda p: _checkpoint_label(p)[0])
    count = max(2, min(count, len(files)))
    picks = sorted({round(i * (len(files) - 1) / (count - 1)) for i in range(count)})
    out = [model_entry(files[i], _checkpoint_label(files[i])[1]) for i in picks]
    for best in ("best_model.zip", "best_model.pkl"):
        if (run_dir / best).is_file():
            out.append(model_entry(run_dir / best, "best (selected)"))
    return out
