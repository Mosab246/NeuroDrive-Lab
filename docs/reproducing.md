# Reproducing the results

Every run is seeded and the simulator is deterministic, so the same command on the same
code gives the same learning history. Only wall-clock times change between runs and
machines.

## A 1-minute experiment (start here)

Evolve NEAT drivers on the oval for 10 generations, then test the best one on 50 fixed
scenarios:

```powershell
uv run python -m neurodrive train neat --config configs/neat/default.yaml --generations 10 --runs-dir runs/repro
uv run python -m neurodrive evaluate --model runs/repro/<run>/best_model.pkl --split test --episodes 50
uv run python -m neurodrive play --model runs/repro/<run>/best_model.pkl
```

What we got (2026-11-25, i9-13900HX, 12 workers):

- training: `done: 10 generations, 679,015 env steps in 48 s; best validation champion
  from generation 9`
- evaluation: `completed 50/50 (100.0%, 95% CI 92.9-100.0%)`, `lap time: 20.69 s mean`

The training was run twice: the two `generations.csv` files were identical in every
column except `wall_seconds`. If your numbers differ outside the timing column, the code
or a library version differs: compare `metadata.json` in the run folder (it records the
git commit, Python and package versions).

## The full results

| Result | Command | Time on the dev laptop |
|---|---|---|
| Baselines (Phase 2) | `evaluate --controller heuristic --episodes 100` | seconds |
| PPO on the oval (Phase 3) | `train ppo --config configs/ppo/default.yaml` | ~25 min |
| NEAT on the oval (Phase 4) | `train neat --config configs/neat/default.yaml` | ~18 min |
| Multi-track generalization (Phase 5) | `train ppo --config configs/ppo/multitrack.yaml`, `train neat --config configs/neat/multitrack.yaml`, then `evaluate --model ... --tracks "split:test,proc:20" --start random --split test` | PPO ~30 min, NEAT ~11 min (they were run at the same time) |
| PPO vs NEAT benchmark (Phase 6) | `benchmark run --config configs/benchmark/default.yaml` | ~2 h, resumable |

All commands start with `uv run python -m neurodrive`. Exact numbers, dates and caveats
are in [results.md](results.md); the benchmark's generated report is in
[`results/benchmark/ppo_vs_neat_1m/`](../results/benchmark/ppo_vs_neat_1m/report.md)
and can be rebuilt from the run folders with `benchmark report --dir
runs/benchmark/ppo_vs_neat_1m`.

## What can differ on your machine

- **Wall-clock times** (CPU, other programs, number of NEAT workers).
- **PPO with a different number of torch threads or a GPU** may not be bit-identical:
  floating-point sums in a different order round differently. The learning curves should
  look alike but are not guaranteed to match digit for digit.
- **NEAT** results do not depend on the number of workers: 3 generations with `workers: 0`
  and with `workers: 12` gave identical `generations.csv` files (timing column aside).
