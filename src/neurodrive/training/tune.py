"""Hyperparameter search for PPO: random search, several seeds per setting, scored on
validation tracks only (never on test tracks).

Each trial is one set of PPO hyperparameters, trained once per seed with the normal
``neurodrive train ppo`` command (in a subprocess, so the sweep can run several trainings
at once and be stopped and resumed: a finished run is never trained again).

A trial's score rewards *reliable* learning, not one lucky checkpoint:

* ``stability`` = laps completed on validation, averaged over every evaluation in the
  second half of training, for the *worst* seed. A setting that learns to lap and then
  forgets, or that only works for some seeds, scores low.
* ``best_lap`` = the mean over seeds of the best checkpoint's validation lap time,
  used to break ties.

Trial 0 is always Stable-Baselines3's defaults, so the search is compared with the
baseline under exactly the same budget, seeds and validation.
"""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from neurodrive.evaluation.metrics import selection_key
from neurodrive.training.config import PPOHyperparams, load_ppo_config

#: Values tried for each hyperparameter. ``learning_rate`` is drawn log-uniformly.
SEARCH_SPACE: dict[str, list[Any]] = {
    "n_steps": [512, 1024, 2048],
    "batch_size": [64, 128, 256],
    "n_epochs": [5, 10, 20],
    "gamma": [0.98, 0.99, 0.995],
    "gae_lambda": [0.9, 0.95, 0.98],
    "clip_range": [0.1, 0.2, 0.3],
    "ent_coef": [0.0, 0.001, 0.01],
    "net_arch": [[64, 64], [128, 128]],
    "log_std_init": [0.0, -0.5, -1.0],
}
LEARNING_RATE = (1e-4, 1e-3)


def sample_trials(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Trial 0 = SB3 defaults; the rest are random draws from :data:`SEARCH_SPACE`."""
    defaults = PPOHyperparams().model_dump(mode="json")
    trials = [defaults]
    rng = random.Random(seed)
    while len(trials) < n:
        t = dict(defaults)
        lo, hi = LEARNING_RATE
        t["learning_rate"] = float(f"{math.exp(rng.uniform(math.log(lo), math.log(hi))):.2e}")
        for key, values in SEARCH_SPACE.items():
            t[key] = rng.choice(values)
        if t not in trials:
            trials.append(t)
    return trials


def _finished_run(runs: Path, name: str, seed: int) -> Path | None:
    for d in sorted(runs.glob(f"{name}_seed{seed}_*")):
        if (d / "train_summary.json").is_file():
            return d
    return None


def run_sweep(
    base_config: Path,
    out: Path,
    trials: int = 8,
    seeds: tuple[int, ...] = (0, 1),
    timesteps: int = 1_000_000,
    workers: int = 4,
    validation_tracks: str = "validation_circuit,pval:3,split:train",
    validation_episodes: int = 5,
    eval_every: int = 100_000,
    search_seed: int = 0,
    log: Any = print,
) -> Path:
    base = load_ppo_config(base_config)
    out.mkdir(parents=True, exist_ok=True)
    runs = out / "runs"
    params = sample_trials(trials, search_seed)
    jobs: list[tuple[str, int, Path]] = []
    for i, hp in enumerate(params):
        data = base.model_dump(mode="json")
        data["name"] = f"tune_t{i:02d}"
        data["ppo"] = hp
        data["training"]["eval_every_steps"] = eval_every
        data["training"]["checkpoint_every_steps"] = eval_every
        data["training"]["torch_threads"] = 2
        cfg = out / "configs" / f"t{i:02d}.yaml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        jobs += [(data["name"], s, cfg) for s in seeds]

    def run(job: tuple[str, int, Path]) -> None:
        name, seed, cfg = job
        if _finished_run(runs, name, seed):
            log(f"  {name} seed {seed}: already done")
            return
        cmd = [
            sys.executable, "-m", "neurodrive", "train", "ppo", "--config", str(cfg),
            "--seed", str(seed), "--timesteps", str(timesteps), "--runs-dir", str(runs),
            "--validation-tracks", validation_tracks,
            "--validation-episodes", str(validation_episodes),
        ]  # fmt: skip
        logf = out / "logs" / f"{name}_seed{seed}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        with logf.open("w", encoding="utf-8") as fh:
            code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        log(f"  {name} seed {seed}: {'done' if code == 0 else f'FAILED ({code}), see {logf}'}")

    log(f"PPO sweep: {len(params)} trials x {len(seeds)} seeds x {timesteps:,} steps")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, jobs))
    return score_sweep(out, len(params), seeds, timesteps)


def score_run(run_dir: Path, timesteps: int) -> dict[str, float]:
    with (run_dir / "eval_history.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    late = [r for r in rows if int(r["timesteps"]) >= timesteps // 2] or rows
    best = max(rows, key=selection_key)
    lap = best["lap_time_mean"]
    return {
        "late_completion": sum(float(r["completion_rate"]) for r in late) / len(late),
        "best_completion": float(best["completion_rate"]),
        "best_lap": float(lap) if lap not in ("", None) else math.inf,
        "best_step": float(best["timesteps"]),
    }


def score_sweep(out: Path, n_trials: int, seeds: tuple[int, ...], timesteps: int) -> Path:
    runs = out / "runs"
    table = []
    for i in range(n_trials):
        name = f"tune_t{i:02d}"
        hp = yaml.safe_load((out / "configs" / f"t{i:02d}.yaml").read_text(encoding="utf-8"))
        per_seed = {}
        for s in seeds:
            d = _finished_run(runs, name, s)
            if d is not None:
                per_seed[s] = score_run(d, timesteps)
        if len(per_seed) < len(seeds):
            continue
        laps = [v["best_lap"] for v in per_seed.values()]
        table.append(
            {
                "trial": name,
                "stability": min(v["late_completion"] for v in per_seed.values()),
                "mean_late_completion": sum(v["late_completion"] for v in per_seed.values())
                / len(seeds),
                "best_lap": sum(laps) / len(laps),
                "per_seed": per_seed,
                "ppo": hp["ppo"],
            }
        )
    table.sort(key=lambda r: (-r["stability"], r["best_lap"]))
    (out / "trials.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    return out / "trials.json"


def best_trial(out: Path) -> dict[str, Any]:
    table = json.loads((out / "trials.json").read_text(encoding="utf-8"))
    if not table:
        raise ValueError(f"{out}: no finished trials")
    best: dict[str, Any] = table[0]
    return best
