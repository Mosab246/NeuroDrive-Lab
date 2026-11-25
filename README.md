# NeuroDrive Lab

[![tests](https://github.com/Mosab246/NeuroDrive-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Mosab246/NeuroDrive-Lab/actions/workflows/ci.yml)

A 2D top-down driving simulator built as a controlled experiment: can a car learn to drive
by itself, and how do **reinforcement learning (PPO)** and **neuroevolution (NEAT)** compare
when they get the same car, sensors, tracks and budget?

> **Status: Phase 7 of 8: interactive lab** (`neurodrive demo` starts a ghost race).
> Benchmark result: trained on three tracks with the same budget (1M environment
> steps, 3 seeds each) and tested on 26 tracks, **PPO drives faster
> (22.30 s vs 23.54 s) but NEAT is more reliable (98.5% vs 92.6% of laps completed)**;
> NEAT also learns in a fraction of the wall-clock time. Both beat the scripted drivers.
> See [results](docs/results.md) and the
> [benchmark report](results/benchmark/ppo_vs_neat_1m/report.md).

![Ghost race: trained PPO and NEAT drivers against scripted ones](docs/images/demo-ghost-race.gif)

*`neurodrive demo`: the six benchmark models (3 PPO, 3 NEAT seeds) and two scripted
drivers on a track none of them trained on, at 2x speed.*

## What exists today

- Deterministic 2D car physics (kinematic bicycle model with steering rate and grip limits)
- Tracks defined as YAML files (centerline + width), validated on load
- Progress, ordered checkpoints, lap counting and lap timing that can't be cheated by
  shortcuts, reversing over the line or rocking over a gate
- Headless simulation and a throughput benchmark
- 9 raycast distance sensors and an 11-number observation vector
- A Gymnasium environment (`NeuroDrive-v0`) that passes the Gymnasium and
  Stable-Baselines3 checkers, with a reward designed against known exploits
- Baseline drivers: random, smoothed random, a sensors-only heuristic, and a
  privileged scripted autopilot
- An evaluator: fixed shared scenarios, one CSV row per episode, summaries with
  confidence intervals
- A pygame-ce playground: keyboard driving, baseline drivers, sensor-ray overlay, pause,
  0.25×–8× speed, follow camera
- PPO training (Stable-Baselines3): configs, seeds, checkpoints from step 0, TensorBoard,
  validation-based checkpoint selection, resume, and trained-model playback
- NEAT neuroevolution (neat-python): parallel scoring, per-generation champions and
  statistics, validation-based selection, reproducible runs, evolved-network drawings
- 6 tracks (train / validation / test, both directions) plus unlimited procedural test
  tracks; random start points; multi-track training; split rules enforced in code
- A resumable PPO-vs-NEAT benchmark: equal env-step budgets, several seeds, identical
  test scenarios, and a generated report with spread across seeds and charts
- Replays (verified to re-simulate exactly), ghost races, a learning timeline, and network
  views in a replay viewer that holds 60 FPS with 8 cars
- 197 automated tests

Not built yet: a live view attached to a running training job (by design: see the
architecture review). See
[docs/architecture-review.md](docs/architecture-review.md).

## Install (Windows PowerShell)

Requires [uv](https://docs.astral.sh/uv/) and git. uv downloads Python 3.12 if needed;
3.12 and 3.13 are tested. About 0.8 GB once installed (PyTorch CPU build included); no GPU
needed.

```powershell
# install uv once (see https://docs.astral.sh/uv/ for other ways)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/Mosab246/NeuroDrive-Lab.git
cd NeuroDrive-Lab
uv sync
uv run python -m neurodrive play     # a window should open: drive with the arrow keys
```

On Linux or macOS the same `uv` commands work. Problems: see
[docs/troubleshooting.md](docs/troubleshooting.md). To check the results yourself, start
with the 1-minute experiment in [docs/reproducing.md](docs/reproducing.md).

## Quick start

```powershell
# Watch the trained PPO and NEAT drivers race the scripted ones (ghost cars)
uv run python -m neurodrive demo

# Drive the car yourself (arrow keys / WASD)
uv run python -m neurodrive play

# Watch a baseline driver (press C in the window to cycle drivers, V for sensor rays)
uv run python -m neurodrive play --driver heuristic

# Evaluate a driver on 100 fixed validation scenarios (writes runs/eval/...)
uv run python -m neurodrive evaluate --controller heuristic --episodes 100

# Train PPO (about 25 minutes), then watch and evaluate it
uv run python -m neurodrive train ppo --config configs/ppo/default.yaml
uv run python -m neurodrive play --model runs/ppo/<run>/best_model.zip
uv run python -m neurodrive evaluate --model runs/ppo/<run>/best_model.zip --episodes 100
uv run tensorboard --logdir runs/ppo

# Evolve NEAT drivers (about 18 minutes), watch the best, draw its network
uv run python -m neurodrive train neat --config configs/neat/default.yaml
uv run python -m neurodrive play --model runs/neat/<run>/best_model.pkl
uv run python -m neurodrive show-network runs/neat/<run>/best_model.pkl

# Train on several tracks, then test on tracks never seen in training
uv run python -m neurodrive train ppo --config configs/ppo/multitrack.yaml
uv run python -m neurodrive evaluate --model runs/ppo/<run>/best_model.zip --tracks "split:test,proc:20" --start random --split test
uv run python -m neurodrive play --track unseen_a --model runs/ppo/<run>/best_model.zip

# Fair benchmark: 3 seeds per algorithm, same budget (about 2 hours on CPU; resumable)
uv run python -m neurodrive benchmark run --config configs/benchmark/default.yaml

# Ghost race: any drivers, same track and start; --save-dir keeps the replays
uv run python -m neurodrive race --benchmark runs/benchmark/ppo_vs_neat_1m --driver autopilot --track unseen_b
uv run python -m neurodrive race --driver runs/ppo/<run>/best_model.zip --driver heuristic --save-dir replays/

# Learning timeline: snapshots of one training run race each other
uv run python -m neurodrive timeline runs/ppo/<run>

# Play back saved replays (--verify re-simulates them and checks every step)
uv run python -m neurodrive replay replays/*.npz --verify

# Measure headless simulation speed
uv run python -m neurodrive bench-sim --steps 100000

# Check a track file
uv run python -m neurodrive validate-track beginner_oval
uv run python -m neurodrive list-tracks

# Tests and lint
uv run pytest
uv run ruff check .
```

### Playground controls

| Key | Action |
|---|---|
| Arrows / WASD | Steer, throttle, brake |
| C | Cycle driver: keyboard → heuristic → smoothed random → autopilot → trained model (only with `--model`) |
| V | Show/hide the distance-sensor rays |
| P | Pause / resume |
| R | Reset to the start line |
| + / − | Simulation speed 0.25× … 8× |
| Z | Camera: whole track ↔ follow the car |
| T | Show/hide the driven path |
| Esc | Quit |

A crash (any corner of the car leaving the track) ends the run; press R.

### Replay viewer controls (`demo`, `race`, `timeline`, `replay`)

| Key | Action |
|---|---|
| Space | Play / pause |
| ← / → | Seek 1 s (hold Shift for 5 s); or click the time bar |
| + / − | Playback speed 0.25× … 8× |
| R / Home | Back to the start |
| Tab | Focus the next car |
| V | Show/hide the focused car's sensor rays |
| N | Show the focused car's network (NEAT: the evolved graph; PPO: its fixed layers) |
| Z | Camera: whole track ↔ follow the focused car |
| T | Show/hide trails |
| Esc | Quit |

Every car is labeled, and the standings panel says whether each driver is learned (PPO,
NEAT) or scripted. Cars are ghosts: they pass through each other.

![Ghost race between the benchmark models and the scripted drivers](docs/images/phase7-ghost-race.png)

## Project layout

```
src/neurodrive/
  sim/          physics, tracks, sensors, progress, simulation (pure NumPy, no graphics)
  tracks/       track YAML files
  env/          Gymnasium environment: config, observations, rewards
  agents/       Controller interface, baselines, PPO and NEAT controllers, model loading
  training/     PPO and NEAT training: configs, run folders
  viz/          network drawings and NEAT plots
  replay/       recording episodes, the replay file format, race/timeline line-ups
  evaluation/   scenarios, evaluator, metrics
  experiments/  run metadata (versions, git commit)
  ui/           pygame-ce playground and replay viewer
  bench.py      headless throughput benchmark
  cli.py        python -m neurodrive ...
configs/        environment, PPO and NEAT experiment configs
scripts/        generate_tracks.py (how the built-in tracks were made)
tests/          unit, integration, smoke
docs/           spec, architecture review, physics, measured results
```

## Documentation

- [docs/SPEC.md](docs/SPEC.md): the original project specification
- [docs/architecture-review.md](docs/architecture-review.md): Phase 0 design review and roadmap
- [docs/physics.md](docs/physics.md): how the car, track, collisions and laps work
- [docs/ml-guide.md](docs/ml-guide.md): the learning problem: observations, actions, reward, endings
- [docs/training-guide.md](docs/training-guide.md): how to train, watch, resume and evaluate
- [docs/results.md](docs/results.md): measured numbers only, with commands
- [docs/reproducing.md](docs/reproducing.md): a 1-minute experiment, and how to rerun everything
- [docs/troubleshooting.md](docs/troubleshooting.md): known problems and fixes
- [results/benchmark/ppo_vs_neat_1m/report.md](results/benchmark/ppo_vs_neat_1m/report.md):
  the generated PPO-vs-NEAT benchmark report
- [docs/phases/phase-1.md](docs/phases/phase-1.md), [docs/phases/phase-2.md](docs/phases/phase-2.md),
  [docs/phases/phase-3.md](docs/phases/phase-3.md),
  [docs/phases/phase-4.md](docs/phases/phase-4.md),
  [docs/phases/phase-5.md](docs/phases/phase-5.md),
  [docs/phases/phase-6.md](docs/phases/phase-6.md),
  [docs/phases/phase-7.md](docs/phases/phase-7.md): phase reports

## License

[MIT](LICENSE). Third-party packages keep their own licenses (installed by uv, not
bundled here).
