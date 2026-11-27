# Phase 14 report: racing with hard physics

Date: 2026-11-27. Question: how do drivers trained with the dynamic tyre-slip model
(understeer, oversteer, spins) cope with other cars on track? No new training: the
hard-physics benchmark models (Phase 9, trained *alone* for 3M steps; 3 seeds each,
`best_model_v2`, re-selected on 7 validation tracks) race against scripted traffic and each
other.

## What we built

| Area | Files | Status |
|---|---|---|
| Scripted drivers in a race use the race's physics (taken from the learned drivers); a race mixing physics models is refused | `replay/race.py`, `replay/recorder.py` | implemented, tested |
| `configs/env/dynamic.yaml`: hard-physics environment for evaluating scripted drivers | `configs/env/` | used below |

255 tests pass (1 new). Before this fix a scripted driver in a hard-physics race would have
driven with the easy (kinematic) physics.

## Traffic test

22 test tracks x 20 scenarios, alone and with 3 scripted opponents (hard physics for
everyone): `evaluate --model M --tracks split:test,proc:20 --start random --split test --episodes 20 --traffic 0|3`;
scripted drivers with `--controller NAME --config configs/env/dynamic.yaml`.
[summary.csv](../../results/hard_racing/summary.csv), [per_seed.csv](../../results/hard_racing/per_seed.csv).

| Driver | Alone: laps (seed range) | Alone: lap time | 3 opponents: laps (seed range) | 3 opponents: lap time | Crashes in traffic | Hits / episode |
|---|---|---|---|---|---|---|
| PPO (3 seeds) | 85.1% (77.5-95.7) | **28.35 s** | 62.8% (59.5-67.5) | **31.49 s** | 37.2% | 2.09 |
| NEAT (3 seeds) | **99.9%** (99.8-100) | 43.02 s | 69.1% (23.9-97.7) | 46.80 s | 29.1% | 0.38 |
| Autopilot (scripted, privileged) | 100% | 38.36 s | **100%** | 47.73 s | 0% | 0.00 |
| Heuristic (scripted, sensors only) | 26.8% | 32.69 s | 46.4% | 44.30 s | 53.6% | 5.05 |

- **Traffic hurts far more with hard physics.** With easy physics PPO trained alone went
  from 93% of laps alone to 85% in traffic (Phase 10); here it falls from 85% to 63%. Braking or swerving for a
  car while the tyres are near their limit is what spins it off.
- **NEAT depends on the seed.** One seed keeps 97.7% in traffic, another collapses to 23.9%
  (it swerves into the wall, as easy-physics NEAT trained alone did). Alone, all three NEAT
  seeds lap almost perfectly but slowly: 15 s a lap slower than PPO.
- **The privileged autopilot is unaffected** (it knows the track and keeps a time gap), so
  the tracks and traffic are drivable; the learned drivers are the limit.
- The heuristic does *better* in traffic (46% vs 27%) because cars ahead slow it down, and
  speed is what makes it crash.

## Grand Prix

One 8-car championship: the 6 hard-physics models plus both scripted drivers, 12 races
(`unseen_a`, `unseen_b`, `proc_0..9`), 2 laps, grid rotated, F1 points (seed 0 = #1).
[Standings](../../results/grand_prix/phase14/standings.md).

| # | Driver | Points | Wins | Finished | Hits |
|---|---|---|---|---|---|
| 1 | NEAT #3 | **194** | 2 | **12/12** | 39 |
| 2 | Autopilot (scripted) | 183 | 1 | **12/12** | 39 |
| 3 | NEAT #1 | 141 | 3 | 7/12 | 9 |
| 4 | PPO #1 | 104 | 2 | 5/12 | 51 |
| 5 | PPO #3 | 68 | 2 | 3/12 | 34 |
| 6 | Heuristic (scripted) | 43 | 1 | 2/12 | 51 |
| 7 | PPO #2 | 40 | 1 | 2/12 | 40 |
| 8 | NEAT #2 | 0 | 0 | 0/12 | 3 |

With 8 cars and slippery tyres, most learned drivers crash out of most races. The only
learned driver to finish every race (NEAT #3) wins; it is also the seed that kept 97.7% of its
laps in the traffic test. NEAT #2, the seed that collapsed in traffic, finishes none.

## What this means for training with traffic

With easy physics, training with traffic fixed exactly this problem (NEAT: 51% to 97% in
traffic, Phase 10). These hard-physics drivers never saw another car in training, and the
gap here (85-100% alone, 63-69% in traffic) is larger than it was with easy physics. Training
them with traffic is the expected fix; it was not run in this phase (about 2 hours with PPO
at tuned settings and NEAT at 30 generations).

## Caveats

- One championship (12 races); the per-seed traffic tests are the more reliable numbers.
- Opponents are scripted in the traffic test.
- No hard-physics SAC model exists yet.
