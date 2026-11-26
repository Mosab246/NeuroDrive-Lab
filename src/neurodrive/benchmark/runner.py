"""Run the algorithm benchmark (PPO, SAC, NEAT): train every (algorithm, seed) under the same
budget, then evaluate every chosen model and every baseline on the same test scenarios.

Everything goes into ``runs/benchmark/<name>/``:

* ``manifest.json``: which run folder belongs to which (algorithm, seed), and its status;
* ``runs/ppo/``, ``runs/neat/``: the normal training run folders;
* ``eval/<algo>_seed<seed>/``, ``eval/<baseline>/``: per-episode CSVs and summaries.

It is **resumable**: rerunning skips finished training runs and evaluations, and restarts
any training run that was interrupted (its partial folder is left in place, unused).
Runs execute one at a time, so wall-clock times are not distorted by CPU sharing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neurodrive.agents.model_io import load_controller, read_model_meta
from neurodrive.benchmark.config import BenchmarkConfig
from neurodrive.cli import expand_tracks, make_controller
from neurodrive.env.config import EnvConfig
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.evaluation.evaluator import evaluate, write_results
from neurodrive.evaluation.scenarios import scenario_seeds
from neurodrive.experiments.metadata import run_metadata
from neurodrive.training.config import (
    NEATExperiment,
    PPOExperiment,
    SACExperiment,
    load_neat_config,
    load_ppo_config,
    load_sac_config,
)

MODEL_FILE = {"ppo": "best_model.zip", "sac": "best_model.zip", "neat": "best_model.pkl"}


class Benchmark:
    def __init__(
        self, config: BenchmarkConfig, root: Path = Path("runs/benchmark"), log: Callable = print
    ):
        self.config = config
        self.dir = Path(root) / config.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.manifest_path = self.dir / "manifest.json"
        self.manifest: dict[str, Any] = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.exists()
            else {"config": config.model_dump(mode="json"), "runs": {}, "created": run_metadata()}
        )
        if self.manifest["config"] != config.model_dump(mode="json"):
            raise ValueError(
                f"{self.manifest_path} was created with a different config; use a new "
                "benchmark name instead of changing an existing benchmark"
            )
        self.tracks = expand_tracks(config.eval.tracks)
        self.seeds_eval = scenario_seeds(config.eval.split, config.eval.episodes_per_track)

    def save(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ experiments
    def ppo_experiment(self, seed: int) -> PPOExperiment:
        base = load_ppo_config(self.config.ppo_config)
        data = base.model_dump()
        data["name"] = f"{self.config.name}_ppo"
        data["training"].update(self.config.ppo_overrides)
        data["training"].update({"seed": seed, "total_timesteps": self.config.budget_env_steps})
        return PPOExperiment.model_validate(data)

    def sac_experiment(self, seed: int) -> SACExperiment:
        if self.config.sac_config is None:
            raise ValueError("benchmarking 'sac' needs sac_config in the benchmark YAML")
        base = load_sac_config(self.config.sac_config)
        data = base.model_dump()
        data["name"] = f"{self.config.name}_sac"
        data["training"].update(self.config.sac_overrides)
        data["training"].update({"seed": seed, "total_timesteps": self.config.budget_env_steps})
        return SACExperiment.model_validate(data)

    def neat_experiment(self, seed: int) -> NEATExperiment:
        base = load_neat_config(self.config.neat_config)
        data = base.model_dump()
        data["name"] = f"{self.config.name}_neat"
        data["neat"].update({"generations": 100_000} | self.config.neat_overrides)
        data["neat"].update({"seed": seed, "max_env_steps": self.config.budget_env_steps})
        return NEATExperiment.model_validate(data)

    # ------------------------------------------------------------------ steps
    def train(self, algo: str, seed: int) -> Path:
        key = f"{algo}/{seed}"
        entry = self.manifest["runs"].get(key, {})
        if entry.get("status") in ("trained", "evaluated"):
            return Path(entry["run_dir"])
        self.log(f"=== training {algo} seed {seed} (budget {self.config.budget_env_steps:,} steps)")
        root = self.dir / "runs" / algo
        t0 = time.perf_counter()
        if algo == "ppo":
            from neurodrive.training.ppo import train_ppo

            run_dir = train_ppo(self.ppo_experiment(seed), runs_root=root, log=self.log)
        elif algo == "sac":
            from neurodrive.training.ppo import train_ppo

            run_dir = train_ppo(self.sac_experiment(seed), runs_root=root, log=self.log)
        elif algo == "neat":
            from neurodrive.training.neat_trainer import train_neat

            run_dir = train_neat(self.neat_experiment(seed), runs_root=root, log=self.log)
        else:
            raise ValueError(f"unknown algorithm {algo!r}")
        self.manifest["runs"][key] = {
            "algorithm": algo,
            "seed": seed,
            "run_dir": str(run_dir),
            "status": "trained",
            "wall_seconds_total": round(time.perf_counter() - t0, 1),
        }
        self.save()
        return run_dir

    def eval_dir(self, label: str) -> Path:
        return self.dir / "eval" / label

    def evaluate_model(self, algo: str, seed: int, run_dir: Path) -> None:
        key = f"{algo}/{seed}"
        out = self.eval_dir(f"{algo}_seed{seed}")
        if (out / "summary.json").exists():
            self.manifest["runs"][key]["status"] = "evaluated"
            self.save()
            return
        model = run_dir / MODEL_FILE[algo]
        self.log(f"=== evaluating {algo} seed {seed}: {model}")
        meta = read_model_meta(model)
        self._evaluate(load_controller(model), out, meta)
        self.manifest["runs"][key]["status"] = "evaluated"
        self.save()

    def evaluate_baseline(self, name: str) -> None:
        out = self.eval_dir(name)
        if (out / "summary.json").exists():
            return
        self.log(f"=== evaluating baseline {name}")
        # Same physics, sensors and episode rules the trained models face.
        env = self._env(load_ppo_config(self.config.ppo_config).env)
        self._evaluate(make_controller(name, env, 0), out, None, env)

    def _env(self, base: EnvConfig) -> DrivingEnv:
        return DrivingEnv(base.model_copy(update={"tracks": tuple(self.tracks)}))

    def _evaluate(
        self, controller: Any, out: Path, meta: dict | None, env: DrivingEnv | None = None
    ) -> None:
        if env is None:
            base = EnvConfig.model_validate(meta["env_config"]) if meta else EnvConfig()
            env = self._env(base)
        records = evaluate(
            env,
            controller,
            self.seeds_eval,
            tracks=self.tracks,
            random_start=self.config.eval.random_start,
        )
        write_results(
            records,
            out,
            run_metadata()
            | {
                "tracks": self.tracks,
                "scenario_seeds": [self.seeds_eval[0], self.seeds_eval[-1]],
                "split": self.config.eval.split,
                "random_start": self.config.eval.random_start,
                "model_meta": meta,
            },
        )

    def run(self) -> Path:
        for seed in self.config.seeds:  # interleaved, so partial results stay balanced
            for algo in self.config.algorithms:
                run_dir = self.train(algo, seed)
                self.evaluate_model(algo, seed, run_dir)
        for name in self.config.baselines:
            self.evaluate_baseline(name)
        self.log(f"benchmark complete: {self.dir}")
        return self.dir
