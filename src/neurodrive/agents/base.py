"""The one interface every driver implements, learned or not.

A controller receives the observation vector and returns ``[steering, throttle/brake]``
in ``[-1, 1]``. Trained PPO models and evolved NEAT networks will be wrapped in this same
interface, so evaluation, replays and the viewer never need to know what kind of driver
they are running.

``kind`` must say honestly what the driver is:

* ``"random"``, ``"heuristic"``: hand-written, not machine learning
* ``"scripted (privileged)"``: hand-written and reads simulator internals the learning
  agents never get
* ``"ppo"``, ``"neat"``: learned (later phases)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Controller(Protocol):
    name: str
    kind: str

    def reset(self, seed: int | None = None) -> None:
        """Called at the start of every episode."""
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return ``[steering, throttle/brake]``."""
        ...
