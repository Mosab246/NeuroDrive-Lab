"""Draw a NEAT network and plot how a NEAT run evolved (matplotlib, saved to PNG).

The drawing is technically accurate but makes no claims about *meaning*: a thick blue
line is a large positive weight, nothing more. Individual hidden neurons in an evolved
network rarely have a human-readable role.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no window; safe in headless runs and tests

import matplotlib.pyplot as plt
import neat

OUTPUT_NAMES = ("steering", "throttle/brake")


def _depths(genome: Any, config: neat.Config, used: set[int]) -> dict[int, int]:
    """Longest path from any input to each node: used as the node's column."""
    gc = config.genome_config
    edges = [(a, b) for (a, b), c in genome.connections.items() if c.enabled and b in used]
    depth = {k: 0 for k in gc.input_keys}
    for _ in range(len(used) + 1):  # relax; feed-forward graphs converge quickly
        changed = False
        for a, b in edges:
            if a in depth and depth.get(b, -1) < depth[a] + 1:
                depth[b] = depth[a] + 1
                changed = True
        if not changed:
            break
    return depth


def network_layout(genome: Any, config: neat.Config) -> dict[str, Any]:
    """Node positions (x, y in [0, 1]) and enabled connections of the part of the network
    that can affect the outputs. Inputs on the left, outputs on the right, hidden neurons in
    columns by their depth. Shared by the PNG drawing and the replay viewer's live view."""
    gc = config.genome_config
    used = set(
        neat.graphs.required_for_output(
            gc.input_keys, gc.output_keys, [k for k, c in genome.connections.items() if c.enabled]
        )
    )
    depth = _depths(genome, config, used)
    max_depth = max([depth.get(k, 1) for k in used] + [1])
    hidden = sorted(k for k in used if k not in gc.output_keys)

    pos: dict[int, tuple[float, float]] = {}
    n_in = len(gc.input_keys)
    for i, k in enumerate(gc.input_keys):
        pos[k] = (0.0, 1 - i / max(1, n_in - 1))
    for i, k in enumerate(gc.output_keys):
        pos[k] = (1.0, 0.65 - 0.3 * i)
    columns: dict[int, list[int]] = {}
    for k in hidden:
        columns.setdefault(min(depth.get(k, 1), max_depth - 1), []).append(k)
    for col, keys in columns.items():
        x = col / max_depth if max_depth > 1 else 0.5
        for j, k in enumerate(keys):
            pos[k] = (min(max(x, 0.2), 0.8), 1 - (j + 1) / (len(keys) + 1))
    edges = [
        (int(a), int(b), float(c.weight))
        for (a, b), c in genome.connections.items()
        if c.enabled and b in used and a in pos and b in pos
    ]
    return {
        "inputs": [int(k) for k in gc.input_keys],
        "outputs": [int(k) for k in gc.output_keys],
        "hidden": [int(k) for k in hidden],
        "pos": {int(k): [float(v[0]), float(v[1])] for k, v in pos.items()},
        "edges": edges,
    }


def draw_network(
    genome: Any,
    config: neat.Config,
    path: Path,
    input_names: list[str] | tuple[str, ...] | None = None,
    title: str = "",
) -> Path:
    gc = config.genome_config
    layout = network_layout(genome, config)
    pos = {k: tuple(v) for k, v in layout["pos"].items()}
    hidden = layout["hidden"]
    used = set(hidden) | set(gc.output_keys)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    weights = [abs(c.weight) for c in genome.connections.values() if c.enabled] or [1.0]
    w_max = max(weights)
    for (a, b), c in genome.connections.items():
        if not c.enabled or b not in used or a not in pos or b not in pos:
            continue
        (x1, y1), (x2, y2) = pos[a], pos[b]
        color = "#2b7bba" if c.weight >= 0 else "#d9534f"
        ax.plot([x1, x2], [y1, y2], color=color, lw=0.4 + 3.0 * abs(c.weight) / w_max, alpha=0.7)
    names = list(input_names) if input_names else [str(k) for k in gc.input_keys]
    for k, name in zip(gc.input_keys, names, strict=False):
        x, y = pos[k]
        ax.scatter([x], [y], s=140, color="#9aa5b1", zorder=3)
        ax.text(x - 0.03, y, name, ha="right", va="center", fontsize=9)
    for k in hidden:
        x, y = pos[k]
        ax.scatter([x], [y], s=140, color="#f0ad4e", zorder=3)
    for k, name in zip(gc.output_keys, OUTPUT_NAMES, strict=False):
        x, y = pos[k]
        ax.scatter([x], [y], s=180, color="#5cb85c", zorder=3)
        ax.text(x + 0.03, y, name, ha="left", va="center", fontsize=10)
    n_conn = sum(
        1 for (a, b), c in genome.connections.items() if c.enabled and b in used and a in pos
    )
    ax.set_title(
        (title + "\n" if title else "")
        + f"{len(hidden)} hidden neurons, {n_conn} active connections "
        + "(blue = positive weight, red = negative, width = |weight|)",
        fontsize=10,
    )
    ax.set_xlim(-0.35, 1.3)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_neat_run(run_dir: Path) -> list[Path]:
    """Fitness and network-size curves from ``generations.csv``."""
    with (Path(run_dir) / "generations.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    gen = [int(r["generation"]) for r in rows]
    out = []

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    ax.plot(gen, [float(r["best_fitness"]) for r in rows], label="best this generation")
    ax.plot(gen, [float(r["mean_fitness"]) for r in rows], label="population mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("NEAT fitness")
    fig.tight_layout()
    out.append(Path(run_dir) / "fitness.png")
    fig.savefig(out[-1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    ax.plot(gen, [int(r["best_hidden"]) for r in rows], label="hidden neurons")
    ax.plot(gen, [int(r["best_connections"]) for r in rows], label="active connections")
    ax.set_xlabel("generation")
    ax.set_ylabel("size of the generation's best network")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("How the best network's structure changed")
    fig.tight_layout()
    out.append(Path(run_dir) / "complexity.png")
    fig.savefig(out[-1])
    plt.close(fig)
    return out
