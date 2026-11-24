# Training guide

How to train, watch, evaluate and replay agents. Commands are for Windows PowerShell,
run from the repository root.

## PPO

### Train

```powershell
# Pipeline check: a few seconds, learns nothing
uv run python -m neurodrive train ppo --config configs/ppo/smoke.yaml

# The real run: 1 million steps (about 20-25 minutes on the development laptop)
uv run python -m neurodrive train ppo --config configs/ppo/default.yaml

# Same config, another seed (for repeated experiments)
uv run python -m neurodrive train ppo --config configs/ppo/default.yaml --seed 1
```

Every run gets its own folder, `runs/ppo/<name>_seed<seed>_<date-time>/`. The training
code's docstring (`src/neurodrive/training/ppo.py`) lists everything in it. The important
files:

- `checkpoints/step_XXXXXXXX.zip`: the policy every 50,000 steps, **including step 0**
  (the untrained, random policy). These are the "learning timeline".
- `best_model.zip`: the checkpoint with the highest mean return on the validation
  scenarios. `final_model.zip`: where training stopped.
- `eval_history.csv`: one validation evaluation per row.
- `config.yaml` + `metadata.json`: exactly what ran, with which library versions and git
  commit.

`runs/` is git-ignored: checkpoints are never committed.

### Watch it learn

```powershell
uv run tensorboard --logdir runs/ppo
```

Then open http://localhost:6006. What each graph means: [ml-guide.md](ml-guide.md#reading-the-tensorboard-graphs).

### Continue a stopped run

```powershell
uv run python -m neurodrive train ppo --config configs/ppo/default.yaml --timesteps 2000000 --resume runs/ppo/<run>/checkpoints/step_01000000.zip
```

`--timesteps` is the new *total*. The run continues in the same folder.
`eval_history.csv`'s `wall_seconds` restarts from 0 in the resumed session.

### Evaluate and watch a trained model

```powershell
# 100 fixed validation scenarios; uses the model's own training environment config
uv run python -m neurodrive evaluate --model runs/ppo/<run>/best_model.zip --episodes 100

# Drive it in the playground (C switches to other drivers for comparison, V shows rays)
uv run python -m neurodrive play --model runs/ppo/<run>/best_model.zip
```

A model refuses to run in an environment whose observation vector differs from the one
it was trained on (e.g. trained without route features, evaluated with them).

## Choosing settings: what was measured

Hyperparameters in `configs/ppo/default.yaml` are Stable-Baselines3's defaults. Changes
should be made one at a time, for a reason, and recorded here.

**Parallel environments (Phase 3 measurement, details in [results.md](results.md)):**
on this laptop, PPO with the default settings runs at roughly 850-1,100 steps/s whether
the 8 environments share one process (`dummy`) or each get a process (`subproc`), and
whether PyTorch uses 1 or 24 threads. The reason: about two-thirds of the time goes into
*updating the network* (10 epochs of batches of 64 = 2,560 small gradient steps per
16,384-step rollout), not into simulating. More processes only speed up the smaller part.

**The lever that would matter:** a larger `batch_size` (e.g. 256-512) means far fewer,
bigger gradient steps per rollout. It would likely be much faster, but it also changes how
learning behaves, so it's an experiment to run and compare, not a free speed-up.

**GPU:** not measured yet. The project installs the CPU build of PyTorch. Measuring the
GPU needs the CUDA build (~2.5 GB) in a separate environment.
