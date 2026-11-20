"""What gets recorded next to every result: versions, git commit, time, command."""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PACKAGES = ("neurodrive-lab", "numpy", "gymnasium", "stable-baselines3", "torch", "pygame-ce")


def git_commit(cwd: Path | None = None) -> str | None:
    """Current commit hash, with ``-dirty`` if there are uncommitted changes."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return rev + ("-dirty" if dirty else "")


def run_metadata() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": versions,
    }
