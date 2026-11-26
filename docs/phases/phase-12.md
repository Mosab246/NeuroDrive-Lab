# Phase 12 report: self-play racing and PPO tuning

Date: 2026-11-26. Two questions:

1. Do agents race better when they train against other *learned* drivers instead of only
   scripted ones (self-play)?
2. Can tuning PPO's hyperparameters make it learn more reliably across seeds?

## What we built

| Area | Files | Status |
|---|---|---|
| Learned opponents in traffic: saved PPO/NEAT models drive opponent cars with their own sensors (`traffic.learned_opponents`, `traffic.learned_share`) | `agents/learned_opponent.py`, `env/config.py`, `env/driving_env.py` | implemented, tested |
| Fast opponent PPO: the policy's mean action in numpy (same numbers as `predict`, checked in tests) | `agents/learned_opponent.py` | implemented, tested |
| `evaluate --opponents FOLDER --learned-share X`: test against learned opponents. Opponents belong to the test, never to the model: a self-play model's training league is not carried into its evaluation | `cli.py` | implemented, tested |
| PPO hyperparameter search, scored on validation only, resumable: `neurodrive tune ppo` | `training/tune.py`, `cli.py` | implemented, smoke-tested |
| Configs: `configs/ppo/tuned.yaml`, `configs/{ppo,neat}/selfplay.yaml`, `configs/benchmark/ppo_{default,tuned}_s345.yaml` | `configs/` | used for the runs below |

251 tests pass (5 new).

## 1. PPO tuning

**Search** (`neurodrive tune ppo --trials 8 --seeds 0,1 --timesteps 1000000`): 8 settings,
2 seeds each, 1M steps. Score = laps completed on the 7 validation tracks, averaged over the
second half of training, for the worse seed. [Report](../../results/tuning/ppo/report.md).

| Setting | Worst-seed late validation laps | Best lap |
|---|---|---|
| Best (tune_t07): learning rate 6.4e-4, rollouts of 512 steps, batch 256, gamma 0.98, GAE 0.9, entropy 0.01, 128x128 network | **90.6%** | **22.40 s** |
| SB3 defaults | 84.5% | 23.64 s |

**Check on the test tracks** with fresh seeds 3-5 (never used in the search), same 1M
budget and 7-track checkpoint selection, 26 tracks x 20 scenarios
(`benchmark run --config configs/benchmark/ppo_{default,tuned}_s345.yaml`):

| | Default | Tuned |
|---|---|---|
| Test laps completed | 94.3% ± 4.4 (90.4-99.0) | 94.8% ± 9.0 (84.4-**100**) |
| Test lap time | 22.28 s | **21.74 s** |
| Steps to first validation lap | 283,000 | **117,000** |
| Steps to lap every validation track | 500,000 | **267,000** |

**What this means:** tuning made PPO learn about **twice as fast** and drive 0.5 s a lap
faster, but it did **not** fix the reliability problem. Two tuned seeds completed every one of
their 520 test laps; the third failed on 5 procedural tracks (84.4%). The seed-to-seed spread
is PPO's real weakness here, and hyperparameters alone did not remove it.
Reports: [default](../../results/benchmark/ppo_default_s345/report.md),
[tuned](../../results/benchmark/ppo_tuned_s345/report.md).

## 2. Self-play

**Training.** PPO (tuned hyperparameters, 2M steps) and NEAT (60 generations), 3 seeds each,
exactly like the traffic runs of Phase 10 except for the opponents: each of the 3 opponents
is, with probability 2/3, a **learned driver at full pace**, drawn from a league of the six
Phase 10 traffic-trained models (PPO and NEAT, seeds 0-2); otherwise a scripted car. Best
checkpoint chosen on the 7 validation tracks. Picks: PPO 0.8M / 1.5M / 1.0M steps, NEAT
generations 45 / 45 / 15.

**Test.** All models, 22 test tracks x 20 scenarios with 3 opponents, two kinds of opponents:

- *scripted*: the Phase 10 test (`evaluate --traffic 3`);
- *learned, held out*: `evaluate --traffic 3 --opponents runs/league/holdout --learned-share 1.0`,
  a league no model trained against (the 6 Phase 10 models trained alone, and the 3 tuned PPO
  benchmark models).

[summary.csv](../../results/selfplay/summary.csv), [per_seed.csv](../../results/selfplay/per_seed.csv).

| Driver (3 seeds each) | Laps, scripted opponents (seed range) | Laps, learned opponents (seed range) | Crashes, learned | Hits / episode, scripted |
|---|---|---|---|---|
| PPO trained alone | 85.2% (72.7-99.5) | 82.3% (74.3-97.0) | 17.7% | 2.98 |
| PPO trained with traffic | 93.0% (86.4-99.1) | 88.9% (77.3-98.9) | 11.0% | 1.61 |
| **PPO self-play** (tuned) | **96.1%** (92.7-99.1) | **95.3%** (90.2-98.9) | 4.6% | 2.26 |
| NEAT trained alone | 51.2% (25.0-66.4) | 90.9% (83.0-96.1) | 7.8% | 0.48 |
| NEAT trained with traffic | 96.9% (93.6-99.3) | 95.9% (91.6-98.4) | 4.1% | 0.88 |
| **NEAT self-play** | **98.6%** (97.7-99.5) | **98.1%** (95.7-99.8) | **1.9%** | 1.02 |

- **Self-play gives the most reliable driver of each algorithm**, against both kinds of
  opponents, and the smallest seed spread. NEAT self-play completes 98% of races against
  drivers it has never met; its worst seed (95.7%) beats the average of every other model.
- **PPO self-play is the first PPO that is reliable in traffic for every seed** (worst seed
  90.2% vs 77.3% for PPO trained with traffic). It still bumps other cars the most.
- **Learned opponents are easier to survive than scripted ones**: at full pace they pull
  away, so there are fewer cars to deal with (about 0.1 hits per episode instead of 1-3).
  That is why NEAT trained alone recovers from 51% to 91%. Slow scripted cars that must be
  overtaken are the harder test.
- NEAT self-play seed 2 was picked at generation 15 and is slower (31.6 s a lap against
  scripted cars); the other two seeds lap in 24-26 s.

### Grand Prix: 8 cars, one championship per seed

12 races per seed (`unseen_a`, `unseen_b`, `proc_0..9`), 2 laps, 8 cars, grid rotated, F1
points; each seed's championship has that seed's six learned models plus the two scripted
drivers. [Standings](../../results/grand_prix/phase12/summary.md).

| # | Driver | Points (total) | Points per seed | Place per seed | Wins | Finished | Hits |
|---|---|---|---|---|---|---|---|
| 1 | **PPO self-play** | **507** | 149 / 163 / 195 | 2 / 2 / 1 | 8 | 27/36 | 80 |
| 2 | NEAT trained with traffic | 483 | 139 / 164 / 180 | 5 / 1 / 2 | 3 | 31/36 | 60 |
| 3 | PPO trained with traffic | 443 | 169 / 95 / 179 | 1 / 8 / 3 | **10** | 23/36 | 68 |
| 4 | NEAT self-play | 397 | 144 / 136 / 117 | 3 / 3 / 5 | 1 | 27/36 | 39 |
| 5 | Heuristic (scripted) | 364 | 114 / 125 / 125 | 6 / 4 / 4 | 0 | **36/36** | 91 |
| 6 | Autopilot (scripted) | 322 | 104 / 111 / 107 | 7 / 5 / 6 | 0 | **36/36** | 88 |
| 7 | PPO trained alone | 294 | 143 / 108 / 43 | 4 / 6 / 8 | 8 | 15/36 | 125 |
| 8 | NEAT trained alone | 266 | 80 / 108 / 78 | 8 / 7 / 7 | 6 | 14/36 | 63 |

PPO self-play wins the championship: top two in every seed, where PPO trained with traffic
swings from 1st to 8th. NEAT self-play, the most reliable driver in the traffic tests, is only
4th here: in 8-car races it is careful and slow. Models trained alone are now last; with 8
cars on track they crash out of more than half their races.

## Reproducing

The leagues are copies of saved models (git-ignored, under `runs/league/`), each with its
`.meta.json` next to it:

- `runs/league/r1/`: the Phase 10 traffic-trained models, `ppo_traffic_s{0,1,2}.zip` and
  `neat_traffic_s{0,1,2}.pkl` (seed 0: `best_model_v2`, seeds 1-2: `best_model`). Used by
  `configs/{ppo,neat}/selfplay.yaml`.
- `runs/league/holdout/`: the Phase 10 models trained alone (`ppo_solo_s*`, `neat_solo_s*`)
  and the tuned PPO benchmark models `ppo_tuned_s{3,4,5}.zip`. Used only for testing.

```
neurodrive tune ppo --config configs/ppo/multitrack.yaml --out runs/tune/ppo --trials 8 --seeds 0,1
neurodrive benchmark run --config configs/benchmark/ppo_tuned_s345.yaml
neurodrive train ppo --config configs/ppo/selfplay.yaml --seed 0 --runs-dir runs/selfplay/ppo
neurodrive train neat --config configs/neat/selfplay.yaml --seed 0 --runs-dir runs/selfplay/neat
neurodrive evaluate --model M --tracks split:test,proc:20 --start random --split test --episodes 20 \
  --traffic 3 --opponents runs/league/holdout --learned-share 1.0
```

## Caveats

- **PPO self-play differs from PPO with traffic in two ways**: its opponents *and* its tuned
  hyperparameters. The tuning check above (same reliability, faster learning) suggests the
  gain in traffic comes mostly from the opponents, but this phase did not train tuned PPO
  with scripted traffic to separate the two.
- One self-play round only: agents trained against the previous generation, not against
  themselves while they learn.
- The training league and the Grand Prix share drivers (the Phase 10 traffic models), so in
  the Grand Prix self-play models race opponents they trained against. The held-out
  traffic test does not have this problem.
- Wall-clock times are rough: up to three jobs shared the CPU. Step counts are exact.
