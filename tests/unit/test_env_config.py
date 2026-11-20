from pathlib import Path

import pytest

from neurodrive.env.config import ConfigError, EnvConfig, load_env_config

ROOT = Path(__file__).resolve().parents[2]


def test_default_yaml_matches_builtin_defaults():
    assert load_env_config(ROOT / "configs" / "env" / "default.yaml") == EnvConfig()


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("env:\n  reward:\n    progres_per_meter: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="progres_per_meter"):
        load_env_config(p)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_env_config(tmp_path / "nope.yaml")
