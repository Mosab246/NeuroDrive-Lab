# Phase 10 report: better checkpoint selection and three seeds for racing

Dates: 2026-11-25 to 2026-11-26. Two questions left open by Phase 9:

1. Can choosing the checkpoint more carefully fix PPO's unreliability, without retraining?
2. Do the racing results (traffic tests, Grand Prix) hold with three seeds instead of one?

## What we built

| Area | Files | Status |
|---|---|---|
| Procedural validation tracks `pval_<n>` (own seed range, never used for training or testing) | `sim/procedural.py`, `sim/track.py` | implemented, tested |
| Re-selection: score every saved checkpoint of a run on a wider validation set, write `best_model_v2.*` | `benchmark/reselect.py` | implemented, tested |
| `neurodrive reselect RUN...`, `benchmark reselect --dir`, `train --validation-tracks` | `cli.py` | implemented, tested |

Validation set (default): `validation_circuit`, `pval_0..2` and the 3 training tracks,
5 scenarios each. Ranking: laps completed, then lap fraction, then lap time. Test tracks
are never used for selection.

## 1. Re-selection on 7 validation tracks

Same trained runs, only the choice of checkpoint changed
(`neurodrive benchmark reselect --dir runs/benchmark/<name>`):

| Benchmark | PPO laps, old pick | PPO laps, new pick | NEAT laps, old | NEAT laps, new |
|---|---|---|---|---|
| 1M steps | 92.6% | **97.6%** | 98.5% | 98.8% |
| 5M steps | 87.8% ± 12.4 | **96.9% ± 4.8** | 100% | 100% |

PPO seed 1 at 5M went from 391/520 to 516/520 test laps: the old pick (chosen on one
validation track) had forgotten `technical_circuit`. Lap times barely changed. Reports:
[1M](../../results/benchmark/ppo_vs_neat_1m/reselection.md),
[5M](../../results/benchmark/ppo_vs_neat_5m/reselection.md),
[hard physics](../../results/benchmark/ppo_vs_neat_dynamic_3m/reselection.md).

## 2. Racing with three seeds

Seeds 1 and 2 of all four racing models were trained exactly like seed 0 in Phase 9
(PPO 2M steps, NEAT 60 generations; alone, or with 3 scripted opponents), with
`--validation-tracks validation_circuit,pval:3,split:train --validation-episodes 5`.
The seed-0 runs were re-selected on the same 7 tracks (`neurodrive reselect ... --episodes 5`),
so all 12 models are chosen the same way.

| Model, seed 0 | Phase 9 pick | Phase 10 pick |
|---|---|---|
| PPO alone | 600,000 steps | 600,000 steps (same) |
| PPO with traffic | 1,900,000 steps | 1,900,000 steps (same) |
| NEAT alone | generation 40 | generation 47 |
| NEAT with traffic | generation 35 | generation 38 |

Seeds 1-2 picks: PPO alone 1.9M / 1.9M, PPO with traffic 1.4M / 1.3M, NEAT generation 59
for all four runs (the last generation; every NEAT run kept improving to the end).

### Traffic tests

All 12 models, 22 test tracks x 20 scenarios, alone and with 3 opponents:
`neurodrive evaluate --model M --tracks split:test,proc:20 --start random --split test --episodes 20 --traffic 0|3`.
Summary: [results/traffic/phase10/summary.csv](../../results/traffic/phase10/summary.csv),
per seed: [per_seed.csv](../../results/traffic/phase10/per_seed.csv).

| Driver | Alone: laps (seed range) | 3 opponents: laps (seed range) | Lap time in traffic | Crashes in traffic | Hits / episode |
|---|---|---|---|---|---|
| PPO trained alone | 93.1% (80.0-100) | 85.2% (72.7-99.5) | 25.50 s | 14.8% | 2.98 |
| PPO trained with traffic | 96.5% (91.6-100) | 93.0% (86.4-99.1) | **23.82 s** | 7.0% | 1.61 |
| NEAT trained alone | **98.6%** (95.9-100) | 51.2% (25.0-66.4) | 31.01 s | 44.3% | 0.48 |
| NEAT trained with traffic | 97.3% (91.8-100) | **96.9%** (93.6-99.3) | 25.83 s | **3.1%** | 0.88 |

Mean over 3 seeds, 1,320 episodes per cell.

- **Training with traffic is what makes NEAT race.** NEAT trained alone loses half its
  laps when other cars appear, in every seed; NEAT trained with traffic barely notices
  them (97.3% alone, 96.9% in traffic) and is the most reliable driver in traffic.
- **The seed-0 "PPO copes with traffic untrained" result from Phase 9 was a lucky seed.**
  Seed 0 completes 99.5% in traffic, seeds 1 and 2 only 83.2% and 72.7%. With traffic
  training PPO is more reliable on average (93.0%) and faster.
- **Phase 9's NEAT-with-traffic result (52.7%) came from the checkpoint, not the training.**
  The same seed-0 run re-selected on 7 tracks completes 97.7% in traffic. Its training
  only validated on one track, alone.
- PPO still drives by contact (1.6-3.0 hits per episode); NEAT almost never touches a car.

### Grand Prix, one championship per seed

Same format as Phase 9: 12 races, 2 laps, 6 cars, grid rotated, F1 points; scripted
drivers identical in all three. [Combined standings](../../results/grand_prix/phase10/summary.md).

| # | Driver | Points (total) | Points per seed | Place per seed | Wins | Finished | Hits |
|---|---|---|---|---|---|---|---|
| 1 | NEAT trained with traffic | **575** | 196 / 184 / 195 | 2 / 2 / 2 | 5 | **33/36** | 25 |
| 2 | PPO trained with traffic | 527 | 172 / 141 / 214 | 3 / 5 / 1 | 11 | 26/36 | 117 |
| 3 | PPO trained alone | 512 | 206 / 198 / 108 | 1 / 1 / 6 | **14** | 24/36 | 133 |
| 4 | Heuristic (scripted) | 431 | 140 / 151 / 140 | 4 / 3 / 4 | 0 | 36/36 | 48 |
| 5 | Autopilot (scripted) | 418 | 132 / 149 / 137 | 5 / 4 / 5 | 0 | 36/36 | 65 |
| 6 | NEAT trained alone | 341 | 110 / 91 / 140 | 6 / 6 / 3 | 6 | 19/36 | 18 |

NEAT trained with traffic is second in every championship and first overall: it rarely
wins but almost always finishes. PPO wins the most races but crashes out of a third of
them, and its place swings from 1st to 6th between seeds. The top three (both PPO models
and NEAT trained with traffic) beat both scripted drivers on points in total.

## Caveats

- Three seeds is the minimum for a spread, not a tight estimate; seed ranges above are
  wide for PPO.
- Opponents during training are scripted. No self-play.
- The Grand Prix uses 12 races per seed; a few crashes move a driver several places.
- Wall-clock times are rough: two training jobs shared the CPU. Step counts are exact.
