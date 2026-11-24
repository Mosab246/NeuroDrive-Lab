"""Train PPO (Stable-Baselines3) on the driving environment.

A run directory ``runs/ppo/<name>_seed<seed>_<timestamp>/`` holds everything:

==========================  ============================================================
``config.yaml``             the exact experiment config used
``metadata.json``           library versions, git commit, machine, command
``tb/``                     TensorBoard logs (``tensorboard --logdir runs/ppo``)
``checkpoints/``            ``step_00000000.zip`` ... saved every N steps, incl. step 0
                            (the untrained policy) - the "learning timeline"
``best_model.zip``          best validation checkpoint (see :func:`selection_key`)
``final_model.zip``         the model when training stopped
``*.meta.json``             next to every model: what it was trained on (observation spec,
                            env config, step) so it can be loaded and checked later
``eval_history.csv``        one row per periodic validation evaluation
``train_summary.json``      steps, wall-clock time, throughput, best checkpoint
==========================  ============================================================
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, deque
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from neurodrive.agents.ppo_agent import PPOController, write_model_meta
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.evaluation.evaluator import evaluate
from neurodrive.evaluation.metrics import selection_key, summarize
from neurodrive.evaluation.scenarios import scenario_seeds
from neurodrive.experiments.metadata import run_metadata
from neurodrive.experiments.splits import check_track_splits
from neurodrive.training.config import PPOExperiment

EVAL_FIELDS = (
    "timesteps",
    "wall_seconds",
    "completion_rate",
    "collision_rate",
    "mean_lap_fraction",
    "lap_time_mean",
    "lap_time_n",
    "mean_return",
    "mean_speed",
)


def _make_env(exp: PPOExperiment) -> DrivingEnv:
    return DrivingEnv(exp.env)


class EpisodeStatsCallback(BaseCallback):
    """Logs how training episodes end (rolling window of the last 100) to TensorBoard."""

    def __init__(self, window: int = 100) -> None:
        super().__init__()
        self.reasons: deque[str] = deque(maxlen=window)
        self.fractions: deque[float] = deque(maxlen=window)
        self.lap_times: deque[float] = deque(maxlen=window)
        self.episodes = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if not done or "termination_reason" not in info:
                continue
            self.episodes += 1
            reason = info["termination_reason"]
            self.reasons.append(reason)
            self.fractions.append(1.0 if reason == "lap_complete" else info["lap_fraction"])
            if reason == "lap_complete":
                self.lap_times.append(info["lap_times"][0])
        if self.reasons:
            n = len(self.reasons)
            counts = Counter(self.reasons)
            for reason in ("lap_complete", "collision", "stalled", "time_limit"):
                self.logger.record(f"episodes/{reason}_rate", counts[reason] / n)
            self.logger.record("episodes/mean_lap_fraction", float(np.mean(self.fractions)))
            self.logger.record("episodes/total", self.episodes)
            if self.lap_times:
                self.logger.record("episodes/lap_time_mean", float(np.mean(self.lap_times)))
        return True


class EvalAndCheckpointCallback(BaseCallback):
    """Every N steps: save a checkpoint; every M steps: evaluate the deterministic policy
    on the fixed validation scenarios and keep the best model."""

    def __init__(self, exp: PPOExperiment, run_dir: Path, obs_spec: dict[str, Any]) -> None:
        super().__init__()
        self.exp = exp
        self.run_dir = run_dir
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.obs_spec = obs_spec
        self.val_tracks = exp.training.validation_tracks
        self.eval_env = DrivingEnv(
            exp.env.model_copy(update={"tracks": self.val_tracks}) if self.val_tracks else exp.env
        )
        self.seeds = scenario_seeds("validation", exp.training.eval_episodes)
        self.best_key: tuple[float, float, float] | None = None
        self.best_step: int | None = None
        self.t0 = time.perf_counter()
        self.history_path = run_dir / "eval_history.csv"
        self._next_eval = 0
        self._next_ckpt = 0
        self.last_summary: dict[str, Any] | None = None
        self.last_eval_step: int | None = None

    def _meta(self, label: str) -> dict[str, Any]:
        return {
            "algorithm": "ppo",
            "label": label,
            "timesteps": int(self.num_timesteps),
            "seed": self.exp.training.seed,
            "experiment": self.exp.name,
            "observation": self.obs_spec,
            "env_config": self.exp.env.model_dump(mode="json"),
            "ppo": self.exp.ppo.model_dump(mode="json"),
        }

    def save(self, path: Path, label: str) -> None:
        self.model.save(path)
        write_model_meta(path.with_suffix(".zip"), self._meta(label))

    def _on_training_start(self) -> None:
        step = int(self.num_timesteps)
        every_e = self.exp.training.eval_every_steps
        every_c = self.exp.training.checkpoint_every_steps
        # On a fresh run this evaluates/saves the untrained policy at step 0.
        self._next_eval = step if step == 0 else (step // every_e + 1) * every_e
        self._next_ckpt = step if step == 0 else (step // every_c + 1) * every_c
        if self.history_path.exists():
            with self.history_path.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                best = max(rows, key=selection_key)
                self.best_key = selection_key(best)
                self.best_step = int(best["timesteps"])
        self._maybe_act()

    def _on_step(self) -> bool:
        self._maybe_act()
        return True

    def _maybe_act(self) -> None:
        step = int(self.num_timesteps)
        if step >= self._next_ckpt:
            self.save(self.ckpt_dir / f"step_{step:08d}", f"checkpoint@{step}")
            self._next_ckpt += self.exp.training.checkpoint_every_steps
        if step >= self._next_eval:
            self._evaluate(step)
            self._next_eval += self.exp.training.eval_every_steps

    def _evaluate(self, step: int) -> None:
        self.last_eval_step = step
        controller = PPOController(self.model, name=f"ppo@{step}", meta={})
        summary: dict[str, Any] = summarize(
            evaluate(self.eval_env, controller, self.seeds, tracks=self.val_tracks)
        )
        self.last_summary = summary
        row = {k: summary.get(k) for k in EVAL_FIELDS if k in summary}
        row["timesteps"] = step
        row["wall_seconds"] = round(time.perf_counter() - self.t0, 2)
        new_file = not self.history_path.exists()
        with self.history_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(EVAL_FIELDS))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        for key in ("completion_rate", "collision_rate", "mean_lap_fraction", "mean_return"):
            self.logger.record(f"eval/{key}", summary[key])
        if summary["lap_time_mean"] is not None:
            self.logger.record("eval/lap_time_mean", summary["lap_time_mean"])
        lap = summary["lap_time_mean"]
        print(
            f"[eval @ {step:>9,} steps] laps {summary['completed']}/{summary['episodes']}"
            f"  progress {100 * summary['mean_lap_fraction']:5.1f}%"
            f"  return {summary['mean_return']:7.2f}"
            + (f"  lap {lap:.2f} s" if lap is not None else ""),
            flush=True,
        )
        rank = selection_key(summary)
        if self.best_key is None or rank > self.best_key:
            self.best_key = rank
            self.best_step = step
            self.save(self.run_dir / "best_model", f"best@{step}")


def make_run_dir(exp: PPOExperiment, root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{exp.name}_seed{exp.training.seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train_ppo(
    exp: PPOExperiment,
    runs_root: Path = Path("runs/ppo"),
    resume_from: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Train according to ``exp``; returns the run directory."""
    t = exp.training
    check_track_splits(exp.env, t.validation_tracks)
    if t.torch_threads is not None:
        torch.set_num_threads(t.torch_threads)
    set_random_seed(t.seed)

    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.is_file():
            raise FileNotFoundError(f"checkpoint not found: {resume_from}")
        run_dir = (
            resume_from.parent.parent
            if resume_from.parent.name == "checkpoints"
            else (resume_from.parent)
        )
    else:
        run_dir = make_run_dir(exp, runs_root)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(exp.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        (run_dir / "metadata.json").write_text(
            json.dumps(run_metadata() | {"torch_threads": torch.get_num_threads()}, indent=2),
            encoding="utf-8",
        )

    vec_cls = SubprocVecEnv if t.vec_env == "subproc" else DummyVecEnv
    venv = make_vec_env(partial(_make_env, exp), n_envs=t.n_envs, seed=t.seed, vec_env_cls=vec_cls)
    obs_spec = DrivingEnv(exp.env).obs_builder.spec.to_dict()

    if resume_from is not None:
        model = PPO.load(
            resume_from, env=venv, device=t.device, tensorboard_log=str(run_dir / "tb")
        )
        log(f"resuming {resume_from} at {model.num_timesteps:,} steps")
    else:
        p = exp.ppo
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=p.learning_rate,
            n_steps=p.n_steps,
            batch_size=p.batch_size,
            n_epochs=p.n_epochs,
            gamma=p.gamma,
            gae_lambda=p.gae_lambda,
            clip_range=p.clip_range,
            ent_coef=p.ent_coef,
            vf_coef=p.vf_coef,
            max_grad_norm=p.max_grad_norm,
            policy_kwargs={
                "net_arch": {"pi": list(p.net_arch), "vf": list(p.net_arch)},
                "log_std_init": p.log_std_init,
            },
            tensorboard_log=str(run_dir / "tb"),
            seed=t.seed,
            device=t.device,
            verbose=0,
        )

    eval_cb = EvalAndCheckpointCallback(exp, run_dir, obs_spec)
    stats_cb = EpisodeStatsCallback()
    start_steps = int(model.num_timesteps)
    remaining = t.total_timesteps - start_steps
    if remaining <= 0:
        raise ValueError(
            f"model already has {start_steps:,} steps; raise training.total_timesteps to continue"
        )
    log(
        f"run dir: {run_dir}\n"
        f"training PPO for {remaining:,} steps ({t.n_envs} x {t.vec_env} envs, device "
        f"{model.device}, torch threads {torch.get_num_threads()})"
    )
    t0 = time.perf_counter()
    try:
        model.learn(
            total_timesteps=remaining,
            callback=CallbackList([stats_cb, eval_cb]),
            reset_num_timesteps=resume_from is None,
            tb_log_name="ppo",
        )
    finally:
        wall = time.perf_counter() - t0
        venv.close()
    eval_cb.save(run_dir / "final_model", f"final@{model.num_timesteps}")
    if eval_cb.last_eval_step != model.num_timesteps:
        eval_cb._evaluate(int(model.num_timesteps))  # always evaluate the final policy
    steps_done = int(model.num_timesteps) - start_steps
    summary = {
        "experiment": exp.name,
        "seed": t.seed,
        "total_timesteps": int(model.num_timesteps),
        "timesteps_this_session": steps_done,
        "wall_seconds_this_session": round(wall, 1),
        "steps_per_second_this_session": round(steps_done / wall, 1) if wall > 0 else None,
        "training_episodes_this_session": stats_cb.episodes,
        "best_checkpoint_step": eval_cb.best_step,
        "best_selection": "completion_rate, then mean_lap_fraction, then lap_time_mean",
        "final_validation": eval_cb.last_summary,
        "device": str(model.device),
        "n_envs": t.n_envs,
        "vec_env": t.vec_env,
        "torch_threads": torch.get_num_threads(),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(
        f"done: {steps_done:,} steps in {wall:.0f} s ({summary['steps_per_second_this_session']} "
        f"steps/s); best checkpoint (laps, progress, lap time) at step {eval_cb.best_step}"
    )
    return run_dir
