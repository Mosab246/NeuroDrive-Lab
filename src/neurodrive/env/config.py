"""Environment configuration (validated with pydantic, loadable from YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from neurodrive.sim.sensors import DEFAULT_RAY_ANGLES_DEG
from neurodrive.sim.vehicle import VehicleParams


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SensorConfig(_Strict):
    angles_deg: tuple[float, ...] = Field(default=DEFAULT_RAY_ANGLES_DEG, min_length=1)
    max_range: float = Field(default=50.0, gt=0, description="Meters.")


class ObservationConfig(_Strict):
    route_features: bool = Field(
        default=False,
        description=(
            "Append PRIVILEGED simulator-derived features: sin/cos of the heading error to "
            "the local track direction, and the lateral offset from the centerline."
        ),
    )
    motion_features: bool = Field(
        default=False,
        description=(
            "Append the car's sideways speed and rotation rate (what a phone-style motion "
            "sensor measures). Needed to notice a slide with the dynamic physics. Not "
            "privileged: no track knowledge."
        ),
    )


class PhysicsConfig(_Strict):
    model: Literal["kinematic", "dynamic"] = Field(
        default="kinematic",
        description="'kinematic': the car turns where it points, slowed by a grip limit. "
        "'dynamic': tyres slip; the car can understeer, oversteer, slide and spin.",
    )
    tire_friction: float = Field(
        default=1.1, gt=0, description="Dynamic model only: tyre friction coefficient (mu)."
    )

    def vehicle_params(self) -> VehicleParams:
        return VehicleParams(model=self.model, tire_friction=self.tire_friction)


class RewardConfig(_Strict):
    progress_per_meter: float = Field(default=0.1, gt=0)
    lap_bonus: float = Field(default=10.0, ge=0)
    crash_penalty: float = Field(default=10.0, ge=0)
    stall_penalty: float = Field(default=10.0, ge=0)
    stall_seconds: float = Field(default=5.0, gt=0)
    stall_min_progress: float = Field(
        default=2.0, ge=0, description="Meters of net progress required every stall_seconds."
    )


class EnvConfig(_Strict):
    track: str = "beginner_oval"
    tracks: tuple[str, ...] | None = Field(
        default=None,
        min_length=1,
        description="If set, each episode uses one of these tracks (picked by the reset "
        "seed) and `track` is ignored. Procedural tracks: 'proc_<seed>'.",
    )
    random_start: bool = Field(
        default=False,
        description="Start each episode at a random point around the track (seeded).",
    )
    laps_to_finish: int = Field(default=1, ge=1)
    max_episode_seconds: float = Field(default=90.0, gt=0)
    perturb_start: bool = Field(
        default=True, description="Random small start offsets on reset (seeded)."
    )
    physics: PhysicsConfig = PhysicsConfig()
    sensors: SensorConfig = SensorConfig()
    observation: ObservationConfig = ObservationConfig()
    reward: RewardConfig = RewardConfig()


class ConfigError(ValueError):
    """A configuration file is missing or invalid."""


def load_env_config(path: str | Path) -> EnvConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return EnvConfig.model_validate(raw.get("env", raw))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML ({exc})") from exc
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"{path}: {details}") from exc
