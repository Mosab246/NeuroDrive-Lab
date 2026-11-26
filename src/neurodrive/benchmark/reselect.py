"""Re-pick the best checkpoint of finished training runs on a wider validation set.

During training, a run keeps the checkpoint that scored best on its validation tracks.
With a single validation track that choice can be fooled: in the 5M benchmark, PPO seed 1
was picked at a point where it had lost the ability to drive one of its own training
tracks. Every run saves its checkpoints (PPO every N steps, NEAT every generation's
champion), so a better choice can be made afterwards **without retraining**: score every
checkpoint on more tracks and pick again with the same ranking rule
(:func:`~neurodrive.evaluation.metrics.selection_key`).

Only validation and training tracks are used here; test tracks stay untouched until the
re-selected model's final evaluation.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from neurodrive.agents.model_io import load_controller, meta_path, read_model_meta
from neurodrive.env.config import EnvConfig
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.evaluation.evaluator import evaluate
from neurodrive.evaluation.metrics import selection_key, summarize
from neurodrive.evaluation.scenarios import scenario_seeds
from neurodrive.experiments.splits import check_track_splits

#: The wider validation set: the hand-made validation circuit, three procedural
#: validation tracks, and the three training tracks (so a model that has "forgotten" a
#: track it was trained on cannot be picked).
DEFAULT_VALIDATION = "validation_circuit,pval:3,split:train"
FIELDS = (
    "steps",
    "checkpoint",
    "completion_rate",
    "worst_track_completion",
    "mean_lap_fraction",
    "lap_time_mean",
    "collision_rate",
)


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """``(environment steps, model file)`` for every saved checkpoint of a PPO or NEAT run."""
    out: list[tuple[int, Path]] = []
    for path in sorted((run_dir / "checkpoints").glob("step_*.zip")):  # PPO
        m = re.fullmatch(r"step_(\d+)", path.stem)
        if m:
            out.append((int(m.group(1)), path))
    for path in sorted((run_dir / "champions").glob("gen_*.pkl")):  # NEAT
        out.append((int(read_model_meta(path).get("env_steps_total", 0)), path))
    if not out:
        raise FileNotFoundError(f"{run_dir}: no checkpoints or champions found")
    return sorted(out)


def score_checkpoint(job: tuple[str, tuple[str, ...], int]) -> dict[str, Any]:
    """Validation score of one checkpoint (runs in a worker process)."""
    path_str, tracks, episodes = job
    path = Path(path_str)
    meta = read_model_meta(path)
    base = EnvConfig.model_validate(meta["env_config"])
    env = DrivingEnv(base.model_copy(update={"tracks": tuple(tracks)}))
    records = evaluate(
        env, load_controller(path), scenario_seeds("validation", episodes), tracks=tracks
    )
    s: dict[str, Any] = summarize(records)
    per_track = [
        sum(r.completed for r in records if r.track == t)
        / max(1, sum(r.track == t for r in records))
        for t in tracks
    ]
    return {
        "checkpoint": path_str,
        "completion_rate": s["completion_rate"],
        "worst_track_completion": min(per_track),
        "mean_lap_fraction": s["mean_lap_fraction"],
        "lap_time_mean": s["lap_time_mean"],
        "collision_rate": s["collision_rate"],
    }


def reselect_runs(
    run_dirs: Sequence[Path],
    tracks: Sequence[str],
    episodes: int = 5,
    workers: int = 8,
    log: Callable[[str], None] = print,
) -> dict[Path, dict[str, Any]]:
    """Re-select every run in ``run_dirs``. Writes ``reselection.csv`` and
    ``best_model_v2.<ext>`` (+ metadata) into each run folder; returns the choices."""
    tracks = tuple(tracks)
    jobs: list[tuple[Path, int, Path]] = []
    for run in run_dirs:
        meta = read_model_meta(list_checkpoints(run)[0][1])
        check_track_splits(EnvConfig.model_validate(meta["env_config"]), tracks)
        jobs += [(run, steps, path) for steps, path in list_checkpoints(run)]
    log(f"scoring {len(jobs)} checkpoints from {len(run_dirs)} runs on {len(tracks)} tracks")
    payload = [(str(p), tracks, episodes) for _, _, p in jobs]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            scores = pool.map(score_checkpoint, payload, chunksize=1)
    else:
        scores = [score_checkpoint(j) for j in payload]
    choices: dict[Path, dict[str, Any]] = {}
    for run in run_dirs:
        rows = [
            {"steps": steps} | score
            for (r, steps, _), score in zip(jobs, scores, strict=True)
            if r == run
        ]
        with (run / "reselection.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        best = max(rows, key=selection_key)
        src = Path(best["checkpoint"])
        dst = run / f"best_model_v2{src.suffix}"
        shutil.copyfile(src, dst)
        meta = read_model_meta(src) | {
            "label": f"reselected@{best['steps']}",
            "reselected_from": src.name,
            "reselection": {"tracks": list(tracks), "episodes_per_track": episodes} | best,
        }
        meta_path(dst).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        choices[run] = best
        log(
            f"  {run.name}: picked {src.name} ({best['steps']:,} steps), validation laps "
            f"{100 * best['completion_rate']:.0f}%, worst track "
            f"{100 * best['worst_track_completion']:.0f}%"
        )
    return choices


# ---------------------------------------------------------------- whole benchmark
def _test_job(job: tuple[str, str, str, str]) -> str:
    """Evaluate one model on the benchmark's test scenarios (runs in a worker process)."""
    from neurodrive.benchmark.config import BenchmarkConfig
    from neurodrive.benchmark.runner import Benchmark

    bench_dir, config_json, model, out = job
    bench = Benchmark(
        BenchmarkConfig.model_validate(json.loads(config_json)), root=Path(bench_dir).parent
    )
    out_dir = Path(out)
    if not (out_dir / "summary.json").exists():
        bench._evaluate(load_controller(model), out_dir, read_model_meta(Path(model)))
    return out


def reselect_benchmark(
    bench_dir: Path,
    out_dir: Path,
    tracks: Sequence[str],
    episodes: int = 5,
    workers: int = 8,
    log: Callable[[str], None] = print,
) -> Path:
    """Re-select every run of a finished benchmark, test the new picks on the benchmark's
    own test scenarios, and write a before/after comparison to ``out_dir/reselection.md``."""
    from neurodrive.benchmark.report import _read_csv, _spread, episode_stats

    manifest = json.loads((bench_dir / "manifest.json").read_text(encoding="utf-8"))
    runs = {key: r for key, r in manifest["runs"].items() if r.get("status") == "evaluated"}
    run_dirs = {key: Path(r["run_dir"]) for key, r in runs.items()}
    choices = reselect_runs(list(run_dirs.values()), tracks, episodes, workers, log)
    config_json = json.dumps(manifest["config"])
    jobs = []
    for key, run in run_dirs.items():
        algo, seed = runs[key]["algorithm"], runs[key]["seed"]
        model = next(run.glob("best_model_v2.*[!n]"))  # the model, not its .meta.json
        jobs.append(
            (
                str(bench_dir),
                config_json,
                str(model),
                str(bench_dir / "eval" / f"{algo}_seed{seed}_v2"),
            )
        )
    log(f"testing {len(jobs)} re-selected models on the benchmark's test scenarios")
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        pool.map(_test_job, jobs, chunksize=1)

    rows = []
    for key, run in sorted(
        run_dirs.items(), key=lambda kv: (runs[kv[0]]["algorithm"], runs[kv[0]]["seed"])
    ):
        algo, seed = runs[key]["algorithm"], runs[key]["seed"]
        old = episode_stats(_read_csv(bench_dir / "eval" / f"{algo}_seed{seed}" / "episodes.csv"))
        new = episode_stats(
            _read_csv(bench_dir / "eval" / f"{algo}_seed{seed}_v2" / "episodes.csv")
        )
        old_meta = read_model_meta(next(p for p in run.glob("best_model.*") if p.suffix != ".json"))
        old_step = old_meta.get("timesteps", old_meta.get("env_steps_total"))
        rows.append(
            {
                "algorithm": algo,
                "seed": seed,
                "old_step": old_step,
                "new_step": choices[run]["steps"],
            }
            | {
                f"old_{k}": old[k]
                for k in (
                    "completed",
                    "episodes",
                    "completion_rate",
                    "lap_time_mean",
                    "collision_rate",
                )
            }
            | {
                f"new_{k}": new[k]
                for k in (
                    "completed",
                    "episodes",
                    "completion_rate",
                    "lap_time_mean",
                    "collision_rate",
                )
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "reselection.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    def lap(x: float | None) -> str:
        return "-" if x is None else f"{x:.2f} s"

    lines = [
        f"# Re-selection: {bench_dir.name}",
        "",
        "Generated by `neurodrive benchmark reselect`. Same training runs; only the choice of",
        "checkpoint changed. Old choice: best on the run's own validation track(s) during",
        f"training. New choice: best of all saved checkpoints on `{', '.join(tracks)}`",
        f"({episodes} validation scenarios per track). Both tested on the same benchmark test",
        "scenarios (never used for either choice).",
        "",
        "| Algorithm | Seed | Old pick (steps) | New pick (steps) | Test laps, old | "
        "Test laps, new | Lap time, old | Lap time, new |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['algorithm'].upper()} | {r['seed']} | {r['old_step']:,} | {r['new_step']:,} | "
            f"{r['old_completed']}/{r['old_episodes']} | "
            f"{r['new_completed']}/{r['new_episodes']} | "
            f"{lap(r['old_lap_time_mean'])} | {lap(r['new_lap_time_mean'])} |"
        )
    lines += [
        "",
        "| | Laps completed, old | Laps completed, new | Lap time, old | Lap time, new |",
        "|---|---|---|---|---|",
    ]
    for algo in sorted({r["algorithm"] for r in rows}):
        mine = [r for r in rows if r["algorithm"] == algo]
        cells = []
        for which in ("old", "new"):
            s = _spread([r[f"{which}_completion_rate"] for r in mine])
            mean, std = s["mean"], s["std"] or 0.0
            cells.append(f"{pct(mean)} ± {100 * std:.1f}" if mean is not None else "-")
        for which in ("old", "new"):
            s = _spread([r[f"{which}_lap_time_mean"] for r in mine])
            cells.append(f"{s['mean']:.2f} ± {s['std']:.2f} s" if s["mean"] is not None else "-")
        lines.append(f"| {algo.upper()} | " + " | ".join(cells) + " |")
    lines += ["", "Values across seeds: mean ± standard deviation."]
    (out_dir / "reselection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))
    return out_dir / "reselection.md"
