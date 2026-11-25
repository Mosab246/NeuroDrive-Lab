# Phase 9 report: longer training, hard physics, real racing

Dates: 2026-11-25 to 2026-11-26. Three questions:

1. Does the PPO-vs-NEAT ranking hold with 5x more training?
2. What happens with physics where tyres can slip?
3. Can the agents race other cars, and does training with traffic help?

## What we built

| Area | Files | Status |
|---|---|---|
| Dynamic tyre-slip vehicle model (understeer, oversteer, spins, friction circle, traction control) | `sim/vehicle.py` | implemented, tested |
| `physics:` and `observation.motion_features` (sideways speed, yaw rate) in env configs | `env/config.py`, `env/observations.py` | implemented, tested |
| Multi-car races: grid, cars visible to each other's sensors, exact contact with momentum exchange | `sim/race.py`, `sim/sensors.py` | implemented, tested |
| Traffic during training: scripted opponents at random paces, contact penalty | `env/driving_env.py`, `env/rewards.py` | implemented, tested |
| Car-aware autopilot (keeps a time gap to the car ahead) | `agents/pure_pursuit.py` | implemented, tested |
| `race --real`, `demo --real`, `grand-prix`, `evaluate --traffic N` | `cli.py`, `replay/race.py`, `replay/grand_prix.py` | implemented, tested |
| Configs: 5M benchmark, hard-physics benchmark, traffic training | `configs/` | used for the runs below |

232 tests pass (35 new). Kinematic-physics results are unchanged: baseline lap times are
identical to Phase 2.

## 1. Five times the budget: the ranking reverses

Same setup as Phase 6 (3 training tracks, 3 seeds, 26 test tracks x 20 scenarios), with
5,000,000 environment steps per run. Command:
`neurodrive benchmark run --config configs/benchmark/long.yaml`.

| | PPO 1M | NEAT 1M | PPO 5M | NEAT 5M |
|---|---|---|---|---|
| Laps completed | 92.6% ± 9.9 | 98.5% ± 1.8 | 87.8% ± 12.4 | **100.0% ± 0.0** |
| Lap time | **22.30 s** | 23.54 s | 22.36 s | **21.59 s** |
| Pooled 95% CI, laps | 91.2-93.8% | 97.7-99.0% | 86.1-89.4% | 99.8-100.0% |

With 5M steps NEAT is faster on 25 of 26 tracks and in all 9 seed pairings, and every NEAT
seed completed all 520 test laps. PPO did not get better with more steps: its validation
picked checkpoints from 0.6M, 1.4M and 4.4M steps, and its lap time stayed at 22.4 s.

**What this means:** "which algorithm is better" depends on the budget. At 1M steps PPO is
faster; at 5M steps NEAT wins on both speed and reliability. A single-budget comparison
would have given a confident but incomplete answer.

**A weakness this exposed:** PPO seed 1 failed every scenario on `technical_circuit`, one
of its own *training* tracks. Its checkpoint was chosen on one validation track only, and
that track does not test the tight corners `technical_circuit` has. Selection on a single
validation track can pick a model that has forgotten something; more validation tracks
would catch it.

Report: [results/benchmark/ppo_vs_neat_5m](../../results/benchmark/ppo_vs_neat_5m/report.md).

## 2. Hard physics: tyres can slip

The dynamic model is described in [physics.md](../physics.md). Every track remains
drivable (the autopilot laps all 26), but the hand-written heuristic drops from 100% to
30% of laps. Benchmark: 3M steps, 3 seeds, same tracks and scenarios,
`configs/benchmark/dynamic.yaml`.

| Hard physics | PPO | NEAT | Autopilot | Heuristic |
|---|---|---|---|---|
| Laps completed | 77.2% ± 4.9 | **88.1% ± 13.1** | 100% | 30.0% |
| Lap time | **27.75 s** | 33.95 s | 38.03 s | 32.04 s |
| Crashes | 22.8% | **1.9%** | 0% | 70.0% |

The Phase 6 pattern, stronger: PPO drives about 6 s a lap faster and crashes in almost a
quarter of the scenarios; NEAT is slow and careful. NEAT seed 1's 139 failures are almost
all stalls (138): it gets stuck going too slowly, rather than crashing. Both learned agents beat the privileged autopilot's lap
time by more than 4 s.

Report: [results/benchmark/ppo_vs_neat_dynamic_3m](../../results/benchmark/ppo_vs_neat_dynamic_3m/report.md).

## 3. Racing

### First races: agents trained alone misread other cars

In the first real races, agents trained alone took other cars for walls appearing out of
nowhere and swerved into the real wall: on `unseen_b`, 6 of 8 cars did not finish, some
with zero contacts. Two simulator problems also showed up and were fixed:

- **Deadlock.** Contact first slowed *both* cars. A scripted car that cannot see other
  cars kept hitting the car in front, both crawled to the time limit. Contact now
  exchanges momentum: a car hit from behind is pushed forward.
- **Blind opponents.** The scripted autopilot now keeps a time gap to the car ahead. In one
  test race, its hits dropped from 76 to 0.

### Training with traffic

PPO and NEAT were retrained exactly as in Phase 5 (seed 0; PPO 2M steps, NEAT 60
generations) with 3 opponents on track. Then all four models were tested on the 22 test
tracks x 20 scenarios, alone and with 3 opponents. `results/traffic/summary.csv`.

| Driver | Alone: laps | Alone: lap time | 3 opponents: laps | 3 opponents: lap time | Crashes | Hits / episode |
|---|---|---|---|---|---|---|
| PPO trained alone | 100% | 22.59 s | **99.5%** | 25.77 s | 0.5% | 2.59 |
| PPO trained with traffic | 98.0% | **21.91 s** | 93.6% | **23.21 s** | 6.4% | 1.48 |
| NEAT trained alone | 100% | 21.74 s | 42.5% | 33.15 s | 56.6% | 0.24 |
| NEAT trained with traffic | 100% | 23.02 s | 52.7% | 24.10 s | 47.3% | 0.08 |

- **PPO copes with traffic even untrained** for it: it bumps into cars and carries on.
- **NEAT almost never touches a car, and crashes into the wall instead.** Its small
  networks react to a car in the rays as to a wall, and swerve.
- **Training with traffic made PPO faster and more aggressive** (fewer hits, quicker laps)
  but less safe, and made NEAT 10 points more successful and much faster in traffic.

### Grand Prix

12 races (2 test tracks + 10 procedural), 2 laps, 6 cars, grid rotated every race, F1
points. `neurodrive grand-prix ...`, `results/grand_prix/phase9/standings.md`.

| # | Driver | Points | Wins | Finished | Hits |
|---|---|---|---|---|---|
| 1 | NEAT trained with traffic | 198 | 3 | 10/12 | 1 |
| 2 | PPO trained alone | 196 | 4 | 10/12 | 35 |
| 3 | PPO trained with traffic | 174 | 3 | 9/12 | 30 |
| 4 | Heuristic (scripted) | 140 | 0 | 12/12 | 13 |
| 5 | Autopilot (scripted) | 132 | 0 | 12/12 | 28 |
| 6 | NEAT trained alone | 104 | 2 | 6/12 | 3 |

Training with traffic nearly doubled NEAT's points (104 to 198). The top two are 2 points
apart over 12 races: a tie, not a ranking. PPO races by contact (30-35 hits), NEAT
without it (1 hit).

## Caveats

- Traffic experiments and the Grand Prix use **one seed per model**; the benchmarks use 3.
- Wall-clock times in this phase are rough: up to four jobs shared the CPU, and the
  laptop was shut down and paused during the runs. Environment-step counts are exact.
  The runs interrupted by the shutdown restarted from scratch (benchmarks) or from the
  last saved generation (NEAT with traffic).
- Opponents in traffic training are scripted; agents have not raced against learned
  opponents during training (self-play).
