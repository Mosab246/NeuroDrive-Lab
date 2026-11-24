"""Evolve drivers with NEAT (neat-python) on the same environment PPO uses.

Each generation:

1. every genome (network) drives ``scenarios_per_genome`` episodes. All genomes in a
   generation get the *same* starting scenarios, so nobody wins by luck of the draw;
2. its fitness is computed from those episodes (see :func:`episode_fitness`);
3. neat-python selects, crosses over and mutates to make the next generation.

A run directory ``runs/neat/<name>_seed<seed>_<timestamp>/`` holds:

=========================  =============================================================
``config.yaml``            the experiment config; ``neat.ini``: the genome rules used
``metadata.json``          versions, git commit, machine, command
``generations.csv``        one row per generation: fitness, species, size, env steps, ...
``champions/gen_XXXX.pkl`` best genome of every generation (the "evolution timeline")
``checkpoints/``           full population snapshots (for ``--resume``)
``best_model.pkl``         best generation champion on the validation scenarios
``fittest_genome.pkl``     highest training fitness ever seen
``eval_history.csv``       validation evaluations
``fitness.png``, ``complexity.png``, ``best_model_network.png``: plots
``train_summary.json``     totals: generations, env steps, wall-clock time
=========================  =============================================================
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import shutil
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import neat
import numpy as np
import yaml

from neurodrive.agents.neat_agent import NEATController, genome_size, save_genome
from neurodrive.env.config import EnvConfig
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.evaluation.evaluator import evaluate
from neurodrive.evaluation.metrics import selection_key, summarize
from neurodrive.evaluation.scenarios import scenario_seeds
from neurodrive.experiments.metadata import run_metadata
from neurodrive.experiments.splits import check_track_splits
from neurodrive.training.config import NEATExperiment

GEN_FIELDS = (
    "generation",
    "wall_seconds",
    "env_steps",
    "env_steps_total",
    "episodes_total",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "species",
    "lap_rate",
    "crash_rate",
    "best_lap_time",
    "best_hidden",
    "best_connections",
)


def episode_fitness(info: dict[str, Any], max_seconds: float, speed_weight: float) -> float:
    """Fitness of one episode = its reward total + a bonus for finishing fast.

    Without the bonus every completed lap scores the same (0.1 x length + lap bonus), so
    evolution would have no reason to drive faster once it can finish. PPO doesn't need the
    bonus because it discounts future reward, which already favours getting there sooner.
    """
    fitness = float(info["episode_return"])
    if info.get("termination_reason") == "lap_complete":
        fitness += speed_weight * max(0.0, max_seconds - float(info["lap_times"][0]))
    return fitness


def generation_seeds(seed: int, generation: int, k: int) -> list[int]:
    """Training starts for one generation: shared by all genomes, below the validation
    and test seed ranges (which start at 1,000,000)."""
    return [(seed * 100_003 + generation * k + i) % 999_983 for i in range(k)]


# --------------------------------------------------------------------------- evaluation
_WORKER: dict[str, Any] = {}


def _init_worker(env_config: dict[str, Any], neat_config: neat.Config, speed: float) -> None:
    """Build one environment per worker process, reused for every genome it scores."""
    _WORKER["env"] = DrivingEnv(EnvConfig.model_validate(env_config))
    _WORKER["config"] = neat_config
    _WORKER["speed"] = speed


def _score_genome(
    task: tuple[int, Any, list[tuple[str | None, int]]],
) -> tuple[int, float, int, int, int, float]:
    """Drive every (track, seed) scenario; fitness is the average over them.

    Returns (genome key, fitness, env steps, laps completed, crashes, best lap time)."""
    key, genome, scenarios = task
    env: DrivingEnv = _WORKER["env"]
    net = neat.nn.FeedForwardNetwork.create(genome, _WORKER["config"])
    total, steps, laps, crashes, best_lap = 0.0, 0, 0, 0, float("inf")
    for track, seed in scenarios:
        options: dict[str, Any] = {"perturb": True}
        if track is not None:
            options["track"] = track
        obs, _ = env.reset(seed=seed, options=options)
        while True:
            action = np.clip(np.asarray(net.activate(obs.tolist())), -1.0, 1.0)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        steps += int(info["steps"])
        total += episode_fitness(info, env.config.max_episode_seconds, _WORKER["speed"])
        if info["termination_reason"] == "lap_complete":
            laps += 1
            best_lap = min(best_lap, float(info["lap_times"][0]))
        elif info["termination_reason"] == "collision":
            crashes += 1
    return key, total / len(scenarios), steps, laps, crashes, best_lap


class GenerationEvaluator:
    """The fitness function handed to neat-python, plus all per-generation bookkeeping."""

    def __init__(self, exp: NEATExperiment, run_dir: Path, neat_config: neat.Config) -> None:
        self.exp = exp
        self.run_dir = run_dir
        self.config = neat_config
        self.population: neat.Population | None = None
        self.t0 = time.perf_counter()
        self.steps_total = 0
        self.episodes_total = 0
        self.wall_offset = 0.0
        self.best_key: tuple[float, float, float] | None = None
        self.best_generation: int | None = None
        self.obs_spec: dict[str, Any] = DrivingEnv(exp.env).obs_builder.spec.to_dict()
        self.train_tracks: tuple[str, ...] = tuple(exp.env.tracks or (exp.env.track,))
        self.val_tracks = exp.neat.validation_tracks
        self.val_env = DrivingEnv(
            exp.env.model_copy(update={"tracks": self.val_tracks}) if self.val_tracks else exp.env
        )
        self.val_seeds = scenario_seeds("validation", exp.neat.validation_episodes)
        (run_dir / "champions").mkdir(exist_ok=True)
        self.pool: Any = None
        args = (exp.env.model_dump(mode="json"), neat_config, exp.neat.speed_weight)
        if exp.neat.workers > 0:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(exp.neat.workers, initializer=_init_worker, initargs=args)
        else:
            _init_worker(*args)
        self._restore_totals()

    def _restore_totals(self) -> None:
        """When resuming, continue the cumulative counters from generations.csv."""
        path = self.run_dir / "generations.csv"
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                self.steps_total = int(rows[-1]["env_steps_total"])
                self.episodes_total = int(rows[-1]["episodes_total"])
                self.wall_offset = float(rows[-1]["wall_seconds"])
        hist = self.run_dir / "eval_history.csv"
        if hist.exists():
            with hist.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                best = max(rows, key=selection_key)
                self.best_key = selection_key(best)
                self.best_generation = int(best["generation"])

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def meta(self, label: str, generation: int) -> dict[str, Any]:
        return {
            "label": label,
            "generation": generation,
            "env_steps_total": self.steps_total,
            "seed": self.exp.neat.seed,
            "experiment": self.exp.name,
            "observation": self.obs_spec,
            "env_config": self.exp.env.model_dump(mode="json"),
            "fitness": {"speed_weight": self.exp.neat.speed_weight},
        }

    def __call__(self, genomes: list[tuple[int, Any]], config: neat.Config) -> None:
        assert self.population is not None
        gen = self.population.generation
        seeds = generation_seeds(self.exp.neat.seed, gen, self.exp.neat.scenarios_per_genome)
        # Multi-track: every genome drives every training track, from the same starts.
        tracks: tuple[str | None, ...] = (
            self.train_tracks if len(self.train_tracks) > 1 else (None,)
        )
        scenarios = [(t, s) for t in tracks for s in seeds]
        tasks = [(key, genome, scenarios) for key, genome in genomes]
        if self.pool is not None:
            results = self.pool.map(_score_genome, tasks, chunksize=max(1, len(tasks) // 64))
        else:
            results = [_score_genome(t) for t in tasks]

        by_key = dict(genomes)
        fitness = []
        gen_steps = laps = crashes = 0
        best_lap = float("inf")
        for key, fit, steps, n_laps, n_crash, lap in results:
            by_key[key].fitness = fit
            fitness.append(fit)
            gen_steps += steps
            laps += n_laps
            crashes += n_crash
            best_lap = min(best_lap, lap)
        n_eps = len(results) * len(scenarios)
        self.steps_total += gen_steps
        self.episodes_total += n_eps

        best_key = max(by_key, key=lambda k: by_key[k].fitness)
        champion = by_key[best_key]
        hidden, conns = genome_size(champion, config)
        save_genome(
            self.run_dir / "champions" / f"gen_{gen:04d}",
            champion,
            config,
            self.meta(f"gen{gen}", gen) | {"training_fitness": champion.fitness},
        )
        row = {
            "generation": gen,
            "wall_seconds": round(self.wall_offset + time.perf_counter() - self.t0, 2),
            "env_steps": gen_steps,
            "env_steps_total": self.steps_total,
            "episodes_total": self.episodes_total,
            "best_fitness": round(champion.fitness, 4),
            "mean_fitness": round(float(np.mean(fitness)), 4),
            "std_fitness": round(float(np.std(fitness)), 4),
            "species": len(self.population.species.species),
            "lap_rate": round(laps / n_eps, 4),
            "crash_rate": round(crashes / n_eps, 4),
            "best_lap_time": "" if best_lap == float("inf") else round(best_lap, 3),
            "best_hidden": hidden,
            "best_connections": conns,
        }
        path = self.run_dir / "generations.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(GEN_FIELDS))
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"[gen {gen:>4}] best {champion.fitness:7.2f}  mean {row['mean_fitness']:7.2f}  "
            f"species {row['species']:>2}  laps {100 * row['lap_rate']:5.1f}%  "
            f"net {hidden}h/{conns}c  steps {self.steps_total:,}"
            + (f"  lap {best_lap:.2f} s" if best_lap != float("inf") else ""),
            flush=True,
        )
        if gen % self.exp.neat.validate_every == 0:
            self.validate(champion, gen)

    def validate(self, genome: Any, gen: int) -> dict[str, Any]:
        controller = NEATController(genome, self.config, name=f"neat gen{gen}")
        summary: dict[str, Any] = summarize(
            evaluate(self.val_env, controller, self.val_seeds, tracks=self.val_tracks)
        )
        path = self.run_dir / "eval_history.csv"
        fields = [
            "generation",
            "env_steps_total",
            "completion_rate",
            "collision_rate",
            "mean_lap_fraction",
            "lap_time_mean",
            "lap_time_n",
            "mean_return",
        ]
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow(
                {"generation": gen, "env_steps_total": self.steps_total}
                | {k: summary[k] for k in fields[2:]}
            )
        lap = summary["lap_time_mean"]
        print(
            f"  [validation gen {gen}] laps {summary['completed']}/{summary['episodes']}  "
            f"progress {100 * summary['mean_lap_fraction']:5.1f}%"
            + (f"  lap {lap:.2f} s" if lap is not None else ""),
            flush=True,
        )
        rank = selection_key(summary)
        if self.best_key is None or rank > self.best_key:
            self.best_key = rank
            self.best_generation = gen
            save_genome(
                self.run_dir / "best_model",
                genome,
                self.config,
                self.meta(f"best@gen{gen}", gen) | {"validation": summary},
            )
        return summary


def load_neat_config(exp: NEATExperiment, path: Path) -> neat.Config:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )
    config.pop_size = exp.neat.pop_size
    n_obs = DrivingEnv(exp.env).obs_builder.spec.size
    gc = config.genome_config
    if gc.num_inputs != n_obs or gc.num_outputs != 2:
        raise ValueError(
            f"{path}: num_inputs/num_outputs = {gc.num_inputs}/{gc.num_outputs}, but the "
            f"environment has {n_obs} observations and 2 actions"
        )
    return config


def train_neat(
    exp: NEATExperiment,
    runs_root: Path = Path("runs/neat"),
    resume_from: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    s = exp.neat
    check_track_splits(exp.env, s.validation_tracks)
    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.is_file():
            raise FileNotFoundError(f"checkpoint not found: {resume_from}")
        run_dir = resume_from.parent.parent
        neat_config = load_neat_config(exp, run_dir / "neat.ini")
        population = neat.Checkpointer.restore_checkpoint(str(resume_from), neat_config)
        log(f"resuming {resume_from} at generation {population.generation}")
    else:
        ini = Path(s.neat_config)
        if not ini.is_file():
            raise FileNotFoundError(f"NEAT config not found: {ini}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(runs_root) / f"{exp.name}_seed{s.seed}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(ini, run_dir / "neat.ini")
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(exp.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        (run_dir / "metadata.json").write_text(json.dumps(run_metadata(), indent=2), "utf-8")
        neat_config = load_neat_config(exp, run_dir / "neat.ini")
        population = neat.Population(neat_config, seed=s.seed)

    (run_dir / "checkpoints").mkdir(exist_ok=True)
    population.add_reporter(
        neat.Checkpointer(s.checkpoint_every, filename_prefix=str(run_dir / "checkpoints" / "gen-"))
    )
    evaluator = GenerationEvaluator(exp, run_dir, neat_config)
    evaluator.population = population
    log(
        f"run dir: {run_dir}\nevolving {neat_config.pop_size} genomes for up to "
        f"{s.generations} generations ({s.workers or 'no'} worker processes)"
    )
    t0 = time.perf_counter()
    start_steps = evaluator.steps_total
    fittest = None
    stop_reason = "generations"
    try:
        while population.generation < s.generations:
            if s.max_env_steps is not None and evaluator.steps_total >= s.max_env_steps:
                stop_reason = "max_env_steps"
                break
            fittest = population.run(evaluator, 1)
        # After run(), the population already holds the *next*, unscored generation, so
        # the last evaluated champion is read back from its saved file.
        last_gen = population.generation - 1
        if fittest is not None and last_gen % s.validate_every != 0:
            last = NEATController.load(run_dir / "champions" / f"gen_{last_gen:04d}.pkl")
            evaluator.validate(last.genome, last_gen)
    finally:
        evaluator.close()
    wall = time.perf_counter() - t0
    if fittest is not None:
        save_genome(
            run_dir / "fittest_genome",
            fittest,
            neat_config,
            evaluator.meta("fittest", population.generation - 1)
            | {"training_fitness": fittest.fitness},
        )
    from neurodrive.viz.network import draw_network, plot_neat_run

    plot_neat_run(run_dir)
    best_path = run_dir / "best_model.pkl"
    if best_path.is_file():
        best = NEATController.load(best_path)
        draw_network(
            best.genome,
            best.config,
            run_dir / "best_model_network.png",
            input_names=evaluator.obs_spec["names"],
            title=f"NEAT best model (generation {evaluator.best_generation})",
        )
    summary = {
        "experiment": exp.name,
        "seed": s.seed,
        "generations_completed": population.generation,
        "stopped_because": stop_reason,
        "env_steps_total": evaluator.steps_total,
        "episodes_total": evaluator.episodes_total,
        "wall_seconds_this_session": round(wall, 1),
        "steps_per_second_this_session": round((evaluator.steps_total - start_steps) / wall, 1)
        if wall > 0
        else None,
        "best_model_generation": evaluator.best_generation,
        "best_selection": "completion_rate, then mean_lap_fraction, then lap_time_mean",
        "pop_size": neat_config.pop_size,
        "workers": s.workers,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(
        f"done: {population.generation} generations, {evaluator.steps_total:,} env steps in "
        f"{wall:.0f} s; best validation champion from generation {evaluator.best_generation}"
    )
    return run_dir
