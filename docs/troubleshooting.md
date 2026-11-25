# Troubleshooting

Problems actually met while building this project, and their fixes.

## Install and Python

**`python -m neurodrive` says "No module named neurodrive", or the wrong Python version.**
The `python` on your PATH is not the project's environment. Always go through uv:
`uv run python -m neurodrive ...`. uv uses (and downloads if needed) Python 3.12 or 3.13.

**`uv` is not recognised.** Install it from <https://docs.astral.sh/uv/>, then open a new
PowerShell window so the PATH change takes effect.

**pygame errors, or two pygames.** This project uses `pygame-ce`. If plain `pygame` is
also installed in the same environment, both provide `import pygame` and they break each
other. `uv sync` installs only `pygame-ce`; if you installed `pygame` by hand, run
`uv pip uninstall pygame` and then `uv sync --reinstall-package pygame-ce`.

**I want to train on my NVIDIA GPU.** The project installs the small CPU build of PyTorch
on purpose (the networks are tiny; see `docs/architecture-review.md` §2). To try CUDA,
delete the `[tool.uv.sources]` and `[[tool.uv.index]]` blocks in `pyproject.toml`, run
`uv sync` (about 2.5 GB), and set `training.device: cuda` in the PPO config. This path
has **not** been tested in this project.

## Windows

**The window is blurry or the wrong size on a laptop screen.** The viewers ask Windows not
to scale them (DPI awareness). If it still looks wrong, pass a smaller size:
`--width 1100 --height 700`.

**A script that trains NEAT starts many copies of itself, or hangs.** NEAT scores genomes
in parallel worker processes. On Windows these start by re-importing your script, so any
script that calls `train_neat` needs the usual guard:

```python
if __name__ == "__main__":
    main()
```

The `neurodrive` command already has it. Or set `workers: 0` in the NEAT config to score
in one process.

## Running without a screen

Tests, servers and CI have no display. Set the SDL dummy driver first:

```powershell
$env:SDL_VIDEODRIVER = "dummy"
uv run pytest
```

Recording replays (`race --no-window --save-dir ...`) and training never open a window.

## Training

**Training is slower than the numbers in `docs/results.md`.** Those were measured on an
i9-13900HX with nothing else running. PPO spends about two thirds of its time on network
updates, which compete with any other heavy program. Two trainings at once roughly halve
each other's speed. `training.torch_threads` caps PyTorch's threads (4 is a good value
when NEAT runs at the same time).

**`SplitError: track '...' is a test track ...`** Test tracks may never be used for
training or for picking the best checkpoint; that would make the test results
meaningless. Train on `split: train` tracks, select on `validation` tracks.

**`ModelError: observation mismatch ...`** The model was trained with a different
observation (for example with route features on). Evaluate and watch it in the
environment it was trained in: `evaluate --model` does that automatically by reading the
model's `.meta.json`; don't pass a different `--config`.

**`... was created with a different config; use a new benchmark name`.** A benchmark
folder belongs to one config. Changing the budget or the seeds would mix incomparable
runs, so give the changed config a new `name:`.

**A run stopped (power cut, closed window).** PPO: `train ppo --resume
runs/ppo/<run>/checkpoints/step_XXXXXXXX.zip`. NEAT: `train neat --resume
runs/neat/<run>/checkpoints/<latest>`. The benchmark: run the same `benchmark run`
command again; it skips finished work.

## Viewer and replays

**`demo` shows only the heuristic and the autopilot.** It looks for trained models in
`runs/benchmark/ppo_vs_neat_1m/`. That folder is not in git (models are generated, not
committed); run the benchmark first (about 2 hours), or use `race --driver
<your model>`.

**`replay --verify` says the replay diverges.** The simulator code changed since the
replay was recorded (each replay stores the git commit it was recorded at, in its
metadata). Record it again with the current code.

**`replays are on different tracks`.** A ghost race needs every replay on the same track.
