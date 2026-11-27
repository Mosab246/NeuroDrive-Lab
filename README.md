<div align="center">

# NeuroDrive Lab

**A 2D driving simulator where cars learn to drive by themselves, built as a controlled
experiment comparing reinforcement learning (PPO) with neuroevolution (NEAT).**

![tests](https://img.shields.io/badge/tests-254%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![A real race: trained PPO and NEAT drivers against scripted ones](docs/images/demo-race.gif)

*A real race (cars see and collide with each other) between six trained drivers (3 PPO,
3 NEAT) and two scripted drivers, on a track none of them saw during training. 2x speed.*

</div>

---

## Overview

NeuroDrive Lab answers one question with care: **given the same car, the same sensors, the
same tracks and the same training budget, how do a gradient-based reinforcement learner
and an evolutionary algorithm compare at learning to drive?**

Everything needed to answer that honestly is built in: a deterministic physics simulator,
a Gymnasium environment, two learning pipelines, scripted baselines, strict
train/validation/test track splits, a multi-seed benchmark with confidence intervals, and
a replay viewer to see what the numbers mean.

| | |
|---|---|
| **Algorithms** | PPO and SAC (Stable-Baselines3), NEAT (neat-python) |
| **Agent inputs** | 9 distance rays, speed, steering angle (no map, no track knowledge) |
| **Agent outputs** | Steering and throttle/brake, continuous in [-1, 1] |
| **Tracks** | 6 hand-built (train / validation / test, both directions) + unlimited procedural |
| **Evaluation** | 26 tracks x 20 fixed scenarios with random starts, 3 seeds per algorithm |
| **Quality** | 254 automated tests (pytest, ruff, mypy), fully reproducible runs |

## Key results

### 1. The winner depends on the training budget

Both algorithms trained on three tracks with identical environment-step budgets, 3 seeds
each, then tested on 26 tracks (22 never seen) with 20 scenarios per track.

| Budget | | PPO | NEAT |
|---|---|---|---|
| **1M steps** | Laps completed | 92.6% ± 9.9 | **98.5% ± 1.8** |
| | Mean lap time | **22.30 s** | 23.54 s |
| **5M steps** | Laps completed | 87.8% ± 12.4 | **100.0% ± 0.0** |
| | Mean lap time | 22.36 s | **21.59 s** |
| | Training time (CPU) | ~2 h | **~7 min** |

<sub>Mean ± standard deviation across seeds. Pooled 95% confidence intervals for laps
completed do not overlap at either budget. Training times are rough (the 5M runs shared
the CPU with other jobs); step counts are exact. Reference drivers on the same scenarios:
scripted autopilot 28.02 s, sensor heuristic 29.56 s (both 100%). Reports:
[1M](results/benchmark/ppo_vs_neat_1m/report.md),
[5M](results/benchmark/ppo_vs_neat_5m/report.md).</sub>

- At 1M steps PPO is faster on all 26 tracks; at 5M steps NEAT is faster on 25 of 26,
  wins all 9 seed pairings and completes every one of its 1,560 test laps.
- PPO plateaued: its best checkpoints came from 0.6M to 4.4M steps and its lap time did
  not improve. NEAT kept improving.
- An earlier comparison with unequal budgets (NEAT ~6x more steps) had suggested NEAT was
  faster at 1M too. Equal budgets and several seeds are what make the comparison meaningful;
  the benchmark runner enforces both.
- How the best checkpoint is chosen matters as much as how long you train. Re-selecting
  each run's checkpoint on 7 validation tracks instead of 1 (no retraining) raised PPO from
  92.6% to **97.6%** of laps at 1M and from 87.8% to **96.9%** at 5M; NEAT stayed at
  98.8% / 100%. Reports: [1M](results/benchmark/ppo_vs_neat_1m/reselection.md),
  [5M](results/benchmark/ppo_vs_neat_5m/reselection.md).

**A third algorithm: SAC.** Soft Actor-Critic, an off-policy learner, under the same rules
(1M steps, 3 seeds, same tests; all three with checkpoints picked on 7 validation tracks):

| At 1M steps | PPO | SAC | NEAT |
|---|---|---|---|
| Laps completed | 97.6% ± 4.1 | 97.8% ± 1.4 | **98.8% ± 2.0** |
| Mean lap time | **22.40 s** | 22.50 s | 24.21 s |
| Steps to first validation lap | 283k | **100k** | 133k |
| Training time per run (CPU) | ~30 min | ~4-9 h | **~2 min** |

SAC learns from the fewest steps, laps as fast as PPO and varies least between seeds, but
costs about 10x PPO's compute per step. [Report](results/benchmark/sac_1m/report.md),
[write-up](docs/phases/phase-13.md).

### 2. Training on one track overfits, in a very specific way

| Trained on | Laps completed, 26 tracks | Counter-clockwise tracks | Clockwise tracks |
|---|---|---|---|
| Oval only (PPO) | 63.7% | 95% | **4%** |
| Oval only (NEAT) | 37.5% | 57% | **0%** |
| 3 varied tracks (PPO / NEAT) | **100% / 100%** | 100% | 100% |

The oval only turns left, so the agents never learned to turn right. Training on three
tracks in both directions fixed it completely, including on 22 tracks never seen in
training.

### 3. Hard physics: fast and reckless vs. slow and careful

With tyres that can slip (understeer, oversteer, spins), 3M steps, 3 seeds:

| | PPO | NEAT | Scripted autopilot | Sensor heuristic |
|---|---|---|---|---|
| Laps completed | 77.2% | **88.1%** | 100% | 30.0% |
| Mean lap time | **27.75 s** | 33.95 s | 38.03 s | 32.04 s |
| Crashes | 22.8% | **1.9%** | 0% | 70.0% |

Hand-written rules that work on the easy physics fail 70% of the time here; both learned
agents adapt. [Report](results/benchmark/ppo_vs_neat_dynamic_3m/report.md).

<p align="center">
  <img src="docs/images/demo-drift.gif" width="60%" alt="A PPO car sliding through a corner, leaving skid marks and tyre smoke">
</p>

### 4. Racing: self-play makes the most reliable racers

In real races (cars see each other with their sensors and collide), how an agent was trained
decides how it races. Tested on 22 tracks with 3 opponents, 3 seeds per model; "learned
opponents" are AI drivers none of the models trained against:

| Driver | Laps completed, scripted opponents | Laps completed, learned opponents |
|---|---|---|
| PPO trained alone | 85.2% | 82.3% |
| PPO trained with traffic | 93.0% | 88.9% |
| **PPO self-play** | **96.1%** | **95.3%** |
| NEAT trained alone | 51.2% | 90.9% |
| NEAT trained with traffic | 96.9% | 95.9% |
| **NEAT self-play** | **98.6%** | **98.1%** |

*Self-play*: trained against a league of earlier learned drivers at full pace, plus scripted
cars. It gives the most reliable driver of each algorithm and the smallest spread between
seeds. NEAT trained alone avoids other cars but swerves into the wall to do so; PPO bumps its
way through. In three 8-car championships (one per seed, rotated grids, F1 scoring), PPO
self-play won on points (507, top two in every seed), ahead of NEAT trained with traffic (483).
[Self-play report](docs/phases/phase-12.md), [standings](results/grand_prix/phase12/summary.md).

**Tuning PPO helps speed, not reliability.** A validation-only hyperparameter search
(`neurodrive tune ppo`) found settings that learn about twice as fast and lap 0.5 s quicker,
but on fresh seeds and the test tracks they completed 94.8% of laps vs. 94.3% for the
defaults. [Search](results/tuning/ppo/report.md).

### 5. Evolution finds tiny solutions

The best NEAT driver of one benchmark seed has **no hidden neurons**: 16 direct
connections from the sensors to the controls, and it laps every test track. The PPO
policy has 5,058 weights.

<p align="center">
  <img src="docs/images/phase7-neat-network.png" width="80%" alt="The evolved NEAT network, shown in the replay viewer">
</p>

An interactive version of these results (filters, hover details, table views, light/dark)
is in [results/dashboard.html](results/dashboard.html): download it and open it in a browser,
or regenerate it with `neurodrive dashboard`.

All numbers come from logged runs with the exact commands, dates and hardware in
[docs/results.md](docs/results.md).

## Features

**Simulation**
- Kinematic bicycle model with steering-rate and grip limits; optional **dynamic
  tyre-slip model** (understeer, oversteer, spins; friction circle; traction control)
- Tracks as YAML centerline + width, validated on load; procedural track generator
- Exact collision (oriented car rectangle vs. track corridor) and cheat-proof lap counting
  (ordered checkpoints; no shortcuts, reversing over the line, or gate rocking)
- Deterministic fixed time step: every run and replay can be reproduced exactly

**Learning**
- Gymnasium environment (`NeuroDrive-v0`) passing the Gymnasium and SB3 environment checkers
- Reward designed against known exploits; `terminated` vs. `truncated` handled correctly
- PPO, SAC and NEAT pipelines with seeds, checkpoints, resume, TensorBoard / statistics, and
  checkpoint selection on separate validation tracks; validation-only hyperparameter
  search for PPO (`tune ppo`)
- Train / validation / test track splits **enforced in code**

**Evaluation**
- Fixed scenario seeds shared by every driver; per-episode CSVs; Wilson confidence intervals
- Resumable multi-seed benchmark with equal environment-step budgets and a generated report

**Visualization and racing**
- Broadcast-style viewer: procedural track art (kerbs, mown grass, checkered line), car
  sprites, skid marks and tyre smoke, F1 start lights, TV camera that follows the closest
  battle, mini-map, live gaps and position changes; MP4 export (`--video`)
- Interactive playground (drive yourself or watch any driver, sensor-ray overlay)
- Replay viewer: ghost races, learning timelines, evolved-network and policy views, 60 FPS
  with 8 cars
- **Real multi-car races**: starting grid, cars see each other with their sensors and
  collide with momentum exchange; championship mode with F1 scoring
- Optional traffic during training, so agents learn to race rather than only to lap
- **Self-play**: opponent cars driven by earlier trained models (PPO or NEAT), in
  training and in tests (`evaluate --opponents`)

## Play it in the browser

[`web/`](web/) is a browser version: race against the trained PPO and NEAT drivers, or
watch them race, with nothing to install. The physics, sensors and both neural networks
are ported to JavaScript and tested against the Python simulator step for step (same
driver, same track: positions agree to within a micrometre). Serve the folder with any
static server (`uv run python -m http.server 8765 --directory web`) and open it.

## Architecture

```mermaid
flowchart LR
    T[Track YAML /<br/>procedural] --> S[Simulator<br/>physics, collisions, laps]
    S --> E[Gymnasium env<br/>rays, reward, endings]
    E --> P[PPO trainer<br/>Stable-Baselines3]
    E --> N[NEAT trainer<br/>neat-python, parallel]
    P --> M[(Models +<br/>metadata)]
    N --> M
    M --> V[Evaluator<br/>fixed scenarios, CI]
    M --> B[Benchmark<br/>seeds, equal budgets]
    M --> R[Recorder / races]
    R --> U[Replay viewer<br/>pygame-ce]
    B --> Rep[Reports + charts]
```

The simulator is pure NumPy and never imports graphics; training runs headless. Every
saved model carries a metadata file (algorithm, observation layout, environment config),
and loading refuses a model into an environment it was not trained for.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and git. Python 3.12 or 3.13 (installed by uv
if missing). No GPU needed.

```powershell
git clone https://github.com/Mosab246/NeuroDrive-Lab.git
cd NeuroDrive-Lab
uv sync
uv run python -m neurodrive play           # drive yourself: arrow keys / WASD
```

### Common commands

| Task | Command (`uv run python -m neurodrive ...`) |
|---|---|
| Watch a ghost race of the benchmark models | `demo` |
| Watch a real race (cars collide) | `demo --real --camera tv` |
| Save a race as a video | `demo --real --camera tv --video race.mp4` |
| Re-pick the best checkpoints of a benchmark | `benchmark reselect --dir runs/benchmark/<name>` |
| Train PPO (~25 min on CPU) | `train ppo --config configs/ppo/default.yaml` |
| Evolve NEAT (~18 min on CPU) | `train neat --config configs/neat/default.yaml` |
| Watch a trained model | `play --model runs/ppo/<run>/best_model.zip` |
| Evaluate on unseen tracks | `evaluate --model <model> --tracks "split:test,proc:20" --start random --split test` |
| Run the full benchmark (~2 h, resumable) | `benchmark run --config configs/benchmark/default.yaml` |
| Race any drivers | `race --real --driver <model or baseline> --driver ...` |
| Race against learned opponents | `evaluate --model <model> --traffic 3 --opponents <folder of models>` |
| Search PPO hyperparameters (resumable) | `tune ppo --trials 8 --seeds 0,1` |
| Championship (rotated grids, F1 points) | `grand-prix --benchmark runs/benchmark/ppo_vs_neat_1m` |
| See a run improve over training | `timeline runs/ppo/<run>` |
| Learning time-lapse video | `timeline runs/neat/<run> --timelapse learning.mp4 --video-speed 4` |
| Draw an evolved network | `show-network runs/neat/<run>/best_model.pkl` |

`demo` uses the benchmark models in `runs/benchmark/` (generated, not committed); without
them it races the scripted drivers. To verify the results yourself, start with the
[1-minute reproducible experiment](docs/reproducing.md).

<details>
<summary><b>Keyboard controls</b></summary>

**Playground (`play`)**

| Key | Action |
|---|---|
| Arrows / WASD | Steer, throttle, brake |
| C | Cycle driver: keyboard, heuristic, smoothed random, autopilot, trained model |
| V | Sensor rays on/off |
| P / R | Pause / reset |
| + / - | Simulation speed 0.25x to 8x |
| Z / T | Follow camera / driven path |

**Replay viewer (`demo`, `race`, `timeline`, `replay`)**

| Key | Action |
|---|---|
| Space | Play / pause |
| Left / Right | Seek 1 s (Shift: 5 s), or click the time bar |
| + / - | Playback speed |
| Tab | Focus the next car |
| V / N | Focused car's sensor rays / neural network |
| Z / T | Follow camera / trails |

</details>

## Project structure

```
src/neurodrive/
  sim/          physics, tracks, sensors, progress, multi-car races (NumPy only)
  env/          Gymnasium environment: config, observations, rewards, traffic
  agents/       controller interface, baselines, PPO and NEAT controllers, model loading
  training/     PPO and NEAT training pipelines
  evaluation/   scenarios, evaluator, metrics
  benchmark/    multi-seed benchmark runner and report generator
  replay/       replay format and recording, races, championships
  ui/           playground and replay viewer (pygame-ce)
configs/        environment, PPO, NEAT and benchmark configurations
results/        committed benchmark reports (CSV, JSON, charts)
tests/          unit, integration and smoke tests
docs/           design, physics, ML guide, results, phase reports
```

## Documentation

| Document | Contents |
|---|---|
| [ML guide](docs/ml-guide.md) | Observations, actions, reward design, PPO and NEAT explained, overfitting, fair comparison |
| [Physics](docs/physics.md) | Vehicle models, tracks, collision, lap counting, races |
| [Results](docs/results.md) | Every measured number, with commands, dates and hardware |
| [Training guide](docs/training-guide.md) | Training, monitoring, resuming, evaluating |
| [Reproducing](docs/reproducing.md) | A 1-minute experiment and how to rerun everything |
| [Architecture review](docs/architecture-review.md) | Design decisions, risks and roadmap |
| [Troubleshooting](docs/troubleshooting.md) | Known problems and fixes |
| [Phase reports](docs/phases/) | What was built and measured at each stage |

## Tech stack

Python · NumPy · Gymnasium · Stable-Baselines3 · PyTorch · neat-python · pygame-ce ·
pydantic · matplotlib · TensorBoard · pytest · ruff · mypy · uv · GitHub Actions

## License

[MIT](LICENSE)
