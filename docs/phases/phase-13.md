# Phase 13 report: SAC, a third learning algorithm

Dates: 2026-11-26 to 2026-11-27. Question: how does an *off-policy* reinforcement learner,
Soft Actor-Critic (SAC), compare with PPO (on-policy) and NEAT (evolution) under the same
rules?

## What we built

| Area | Files | Status |
|---|---|---|
| SAC training (Stable-Baselines3), sharing PPO's pipeline: environments, checkpoints, validation, best-model selection, metadata, resume | `training/ppo.py`, `training/config.py` | implemented, tested |
| `SACController`; SAC models load everywhere a model is accepted (evaluate, play, race, Grand Prix, replays, learned opponents) | `agents/sac_agent.py`, `agents/model_io.py`, `replay/recorder.py` | implemented, tested |
| Benchmark runner and report for any set of algorithms (`algorithms: [sac]`, `sac_config`) | `benchmark/runner.py`, `benchmark/report.py`, `benchmark/config.py` | implemented; existing reports regenerate byte-for-byte |
| `neurodrive train sac --config configs/sac/multitrack.yaml` | `cli.py`, `configs/sac/multitrack.yaml` | used for the runs below |

254 tests pass (3 new). The browser version supports PPO and NEAT only.

## Setup

Same rules as the Phase 6 benchmark: 3 training tracks, random starts, **1,000,000
environment steps** per run, seeds 0-2, tested on 26 tracks x 20 scenarios (22 tracks never
seen in training). Checkpoints chosen on the 7-track validation set (`validation_circuit`,
`pval_0..2`, the training tracks), like the re-selected PPO and NEAT results.

SAC uses Stable-Baselines3's defaults (learning rate 3e-4, batch 256, 256x256 networks,
learned entropy coefficient) with one gradient step per environment step, the standard
setting. 4 environments; the first 10,000 steps are random actions to fill the replay
buffer. The three seeds trained in parallel, then were registered and evaluated with
`neurodrive benchmark run --config configs/benchmark/sac_1m.yaml`.
[Report](../../results/benchmark/sac_1m/report.md).

## Result

| At 1M steps, 3 seeds | PPO | **SAC** | NEAT |
|---|---|---|---|
| Test laps completed | 97.6% ± 4.1 | **97.8% ± 1.4** | **98.8% ± 2.0** |
| Pooled test laps | 1523/1560 | 1526/1560 | 1542/1560 |
| Worst seed | 92.9% | **96.5%** | 96.5% |
| Test lap time | **22.40 s** | 22.50 s | 24.21 s |
| Steps to first validation lap | 283,000 | **100,000** | ~133,000 |
| Training time per run (CPU) | ~20-45 min | **~9 h** (3 runs sharing the CPU) | **~2 min** |

PPO and NEAT: the re-selected 1M benchmark, same seeds and selection
([reselection.md](../../results/benchmark/ppo_vs_neat_1m/reselection.md)); PPO's steps to
first lap: default PPO with 7-track selection, seeds 3-5 (Phase 12).

- **SAC is as fast as PPO and nearly as reliable as NEAT, with the smallest spread between
  seeds.** Its worst seed (96.5%) is better than PPO's worst (92.9%). Every SAC seed
  completed at least 502 of 520 test laps.
- **SAC is the most sample-efficient learner here**: two of three seeds completed validation
  laps after only 50,000 steps. Reusing old experience from a replay buffer is what
  off-policy learning buys.
- **It pays in compute.** One gradient step per environment step makes SAC about 10-14x
  slower than PPO per environment step on this CPU (66 vs. ~900 steps/s, each run alone),
  and more than 100x slower than NEAT.
  Under an equal *step* budget SAC is competitive; under an equal *time* budget it is not.
- **Its validation scores jump between evaluations** (one seed went from 35/35 validation
  laps to 10/35 and back). Checkpoint selection matters even more for SAC than for PPO.
- Failures concentrate on a few tracks: seed 1 completed only 2 of 20 scenarios on
  `proc_3` (the same track tuned PPO seed 4 failed on), otherwise 500 of 500.

## Caveats

- Three seeds per algorithm. Differences of one or two points in laps completed are within
  the seed spread; treat the three as close on reliability.
- SAC's wall-clock time is inflated by running three seeds at once on one CPU; a single run
  alone was measured at about 66 steps/s (~4 h for 1M steps). Step counts are exact.
- SAC was not tuned; PPO's own tuning (Phase 12) made it faster but not more reliable.
- SAC has not been raced or trained with traffic yet.
