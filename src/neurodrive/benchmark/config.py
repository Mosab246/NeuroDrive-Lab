"""Benchmark configuration: which algorithms, seeds, budget and test scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from neurodrive.env.config import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvalSettings(_Strict):
    tracks: str = Field(description="Track list, as for `evaluate --tracks`.")
    episodes_per_track: int = Field(default=20, ge=1)
    split: str = Field(default="test", pattern=r"^(validation|test)$")
    random_start: bool = True


class BenchmarkConfig(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    budget_env_steps: int = Field(ge=1, description="Training budget per run, for both algorithms.")
    seeds: tuple[int, ...] = Field(min_length=1)
    algorithms: tuple[str, ...] = ("ppo", "neat")
    ppo_config: str
    neat_config: str
    sac_config: str | None = Field(default=None, description="Needed when 'sac' is benchmarked.")
    ppo_overrides: dict[str, Any] = Field(default_factory=dict, description="training.* keys")
    sac_overrides: dict[str, Any] = Field(default_factory=dict, description="training.* keys")
    neat_overrides: dict[str, Any] = Field(default_factory=dict, description="neat.* keys")
    baselines: tuple[str, ...] = ("heuristic", "autopilot", "smoothed_random")
    eval: EvalSettings


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BenchmarkConfig.model_validate(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML ({exc})") from exc
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"{path}: {details}") from exc
