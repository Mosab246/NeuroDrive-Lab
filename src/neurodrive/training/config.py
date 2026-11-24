"""PPO experiment configuration: environment + algorithm hyperparameters + run budget."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from neurodrive.env.config import ConfigError, EnvConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PPOHyperparams(_Strict):
    """Passed to Stable-Baselines3's PPO. Defaults equal SB3's own defaults."""

    learning_rate: float = Field(default=3e-4, gt=0)
    n_steps: int = Field(default=2048, ge=8, description="Steps per environment per rollout.")
    batch_size: int = Field(default=64, ge=2)
    n_epochs: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_range: float = Field(default=0.2, gt=0)
    ent_coef: float = Field(default=0.0, ge=0)
    vf_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    net_arch: tuple[int, ...] = Field(default=(64, 64), min_length=1)
    log_std_init: float = 0.0


class PPOTraining(_Strict):
    total_timesteps: int = Field(default=1_000_000, ge=1)
    n_envs: int = Field(default=8, ge=1)
    vec_env: Literal["dummy", "subproc"] = "subproc"
    seed: int = Field(default=0, ge=0, lt=1_000_000)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    torch_threads: int | None = Field(default=None, ge=1)
    eval_every_steps: int = Field(default=100_000, ge=1)
    eval_episodes: int = Field(default=10, ge=1)
    checkpoint_every_steps: int = Field(default=100_000, ge=1)
    validation_tracks: tuple[str, ...] | None = Field(
        default=None,
        description="Tracks for checkpoint selection (split 'validation' or 'train'). "
        "None = the training tracks, with validation scenario seeds.",
    )


class PPOExperiment(_Strict):
    name: str = Field(default="ppo", pattern=r"^[A-Za-z0-9_.-]+$")
    env: EnvConfig = EnvConfig()
    ppo: PPOHyperparams = PPOHyperparams()
    training: PPOTraining = PPOTraining()

    @model_validator(mode="after")
    def _batch_fits_rollout(self) -> PPOExperiment:
        rollout = self.ppo.n_steps * self.training.n_envs
        if self.ppo.batch_size > rollout:
            raise ValueError(
                f"ppo.batch_size ({self.ppo.batch_size}) is larger than one rollout "
                f"(n_steps * n_envs = {rollout})"
            )
        return self


def load_ppo_config(path: str | Path) -> PPOExperiment:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return PPOExperiment.model_validate(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML ({exc})") from exc
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"{path}: {details}") from exc


class NEATSettings(_Strict):
    neat_config: str = Field(
        default="configs/neat/default.ini", description="neat-python INI file (genome rules)."
    )
    pop_size: int = Field(default=150, ge=2)
    generations: int = Field(default=100, ge=1)
    max_env_steps: int | None = Field(
        default=None, ge=1, description="Stop once this many environment steps were used."
    )
    scenarios_per_genome: int = Field(
        default=2, ge=1, description="Starts each genome is scored on (shared per generation)."
    )
    speed_weight: float = Field(
        default=1.0,
        ge=0,
        description="Fitness bonus per second a completed lap finishes under the time limit.",
    )
    seed: int = Field(default=0, ge=0, lt=1_000_000)
    workers: int = Field(default=0, ge=0, description="0 = evaluate in this process.")
    validate_every: int = Field(default=5, ge=1, description="Generations between validations.")
    validation_episodes: int = Field(default=10, ge=1)
    checkpoint_every: int = Field(default=10, ge=1, description="Generations between saves.")
    validation_tracks: tuple[str, ...] | None = Field(
        default=None,
        description="Tracks for champion selection (split 'validation' or 'train'). "
        "None = the training tracks, with validation scenario seeds.",
    )


class NEATExperiment(_Strict):
    name: str = Field(default="neat", pattern=r"^[A-Za-z0-9_.-]+$")
    env: EnvConfig = EnvConfig()
    neat: NEATSettings = NEATSettings()


def load_neat_config(path: str | Path) -> NEATExperiment:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return NEATExperiment.model_validate(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML ({exc})") from exc
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"{path}: {details}") from exc
