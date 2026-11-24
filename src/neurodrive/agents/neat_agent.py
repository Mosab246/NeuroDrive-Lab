"""An evolved NEAT network behind the common Controller interface.

A saved NEAT model is a pickle of ``(genome, neat_config)`` plus the usual
``.meta.json`` sidecar (see :mod:`neurodrive.agents.model_io`). Pickle files can run code
when loaded, so only load NEAT models you trained yourself.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import neat
import numpy as np

from neurodrive.agents.model_io import ModelError, read_model_meta, write_model_meta


def save_genome(path: Path, genome: Any, config: neat.Config, meta: dict[str, Any]) -> None:
    path = Path(path).with_suffix(".pkl")
    with path.open("wb") as fh:
        pickle.dump((genome, config), fh)
    write_model_meta(path, meta | {"algorithm": "neat"})


class NEATController:
    kind = "neat"

    def __init__(
        self, genome: Any, config: neat.Config, name: str = "neat", meta: dict | None = None
    ):
        self.genome = genome
        self.config = config
        self.name = name
        self.meta = meta or {}
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    @classmethod
    def load(cls, path: str | Path) -> NEATController:
        path = Path(path)
        if not path.is_file():
            raise ModelError(f"model file not found: {path}")
        meta = read_model_meta(path)
        if meta.get("algorithm") != "neat":
            raise ModelError(f"{path.name} is not a NEAT model (algorithm={meta.get('algorithm')})")
        with path.open("rb") as fh:
            genome, config = pickle.load(fh)
        return cls(genome, config, name=f"neat {meta.get('label', path.stem)}", meta=meta)

    def reset(self, seed: int | None = None) -> None:
        pass  # feed-forward network: no memory between steps

    def act(self, obs: np.ndarray) -> np.ndarray:
        out = self.net.activate(obs.tolist())
        return np.clip(np.asarray(out, dtype=np.float64), -1.0, 1.0)

    def size(self) -> tuple[int, int]:
        """(hidden neurons, enabled connections) actually used by the network."""
        return genome_size(self.genome, self.config)


def genome_size(genome: Any, config: neat.Config) -> tuple[int, int]:
    gc = config.genome_config
    used = neat.graphs.required_for_output(
        gc.input_keys,
        gc.output_keys,
        [k for k, c in genome.connections.items() if c.enabled],
    )
    hidden = len([n for n in used if n not in gc.output_keys])
    conns = sum(
        1
        for (a, b), c in genome.connections.items()
        if c.enabled and b in used and (a in used or a in gc.input_keys)
    )
    return hidden, conns
