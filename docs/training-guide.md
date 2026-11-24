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

## NEAT

```powershell
# Pipeline check: a few seconds
uv run python -m neurodrive train neat --config configs/neat/smoke.yaml

# The real run: 150 genomes x 100 generations (about 18 minutes with 12 workers)
uv run python -m neurodrive train neat --config configs/neat/default.yaml

# Continue from a population checkpoint, to a new total number of generations
uv run python -m neurodrive train neat --config configs/neat/default.yaml --generations 150 --resume runs/neat/<run>/checkpoints/gen-100

# Watch, evaluate, draw
uv run python -m neurodrive play --model runs/neat/<run>/best_model.pkl
uv run python -m neurodrive evaluate --model runs/neat/<run>/best_model.pkl --episodes 100
uv run python -m neurodrive show-network runs/neat/<run>/champions/gen_0050.pkl
```

Two config files work together:

- `configs/neat/default.yaml`: the experiment (environment, population size, generations,
  step budget, fitness speed weight, seed, worker processes).
- `configs/neat/default.ini`: neat-python's genome rules (mutation rates, speciation
  threshold, ...). `num_inputs` must equal the observation size (11, or 14 with route
  features); the trainer checks this.

The run folder keeps a copy of the `.ini` it used, every generation's champion
(`champions/`), `generations.csv`, and `fitness.png` / `complexity.png` /
`best_model_network.png`.

**Known tuning issue:** `compatibility_threshold = 3.0` is too high for these small networks:
only one species formed until generation 48. See the Phase 4 report.
