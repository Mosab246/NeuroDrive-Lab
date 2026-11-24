"""Keep training, validation and test tracks apart, in code rather than by convention.

* Training may only use tracks marked ``split: train``.
* Checkpoint/champion selection may use ``validation`` (or ``train``) tracks, never ``test``.
* ``test`` tracks, including every procedural track, are for final evaluation only.

If a test track were used to train or to pick the best checkpoint, its score would no longer
measure how well the agent handles *unseen* tracks.
"""

from __future__ import annotations

from collections.abc import Sequence

from neurodrive.env.config import EnvConfig
from neurodrive.sim.track import load_any_track


class SplitError(ValueError):
    """A track was used for something its split doesn't allow."""


def training_tracks(env: EnvConfig) -> tuple[str, ...]:
    return tuple(env.tracks or (env.track,))


def check_track_splits(env: EnvConfig, validation_tracks: Sequence[str] | None) -> None:
    for tid in training_tracks(env):
        split = load_any_track(tid).spec.split
        if split != "train":
            raise SplitError(
                f"track '{tid}' is a {split} track and can't be used for training "
                "(only tracks with split: train)"
            )
    for tid in validation_tracks or ():
        split = load_any_track(tid).spec.split
        if split == "test":
            raise SplitError(
                f"track '{tid}' is a test track and can't be used to select checkpoints"
            )
