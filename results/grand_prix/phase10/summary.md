# Grand Prix over 3 seeds (Phase 10)

Three separate championships, one per training seed. Each: 12 races (`unseen_a`, `unseen_b`, `proc_0`..`proc_9`),
2 laps, 6 cars, grid rotated every race, F1 points (25, 18, 15, 12, 10, 8, 6, 4). The scripted drivers are the same
in all three; the four learned drivers are that seed's models (seed 0: `best_model_v2`, re-selected on 7 tracks;
seeds 1-2: `best_model`, selected on the same 7 tracks during training).

Per-seed standings: [seed 0](seed0/standings.md), [seed 1](seed1/standings.md), [seed 2](seed2/standings.md).

| # | Driver | Type | Points (total) | Points per seed | Place per seed | Wins | Finished | Hits |
|---|---|---|---|---|---|---|---|---|
| 1 | NEAT traffic | learned (NEAT) | 575 | 196 / 184 / 195 | 2 / 2 / 2 | 5 | 33/36 | 25 |
| 2 | PPO traffic | learned (PPO) | 527 | 172 / 141 / 214 | 3 / 5 / 1 | 11 | 26/36 | 117 |
| 3 | PPO solo | learned (PPO) | 512 | 206 / 198 / 108 | 1 / 1 / 6 | 14 | 24/36 | 133 |
| 4 | heuristic | scripted (sensors only) | 431 | 140 / 151 / 140 | 4 / 3 / 4 | 0 | 36/36 | 48 |
| 5 | autopilot | scripted (privileged) | 418 | 132 / 149 / 137 | 5 / 4 / 5 | 0 | 36/36 | 65 |
| 6 | NEAT solo | learned (NEAT) | 341 | 110 / 91 / 140 | 6 / 6 / 3 | 6 | 19/36 | 18 |

Command (per seed, models of that seed):

```
neurodrive grand-prix --driver "PPO solo=<run>/best_model.zip" --driver "PPO traffic=..." \
  --driver "NEAT solo=<run>/best_model.pkl" --driver "NEAT traffic=..." --driver autopilot --driver heuristic \
  --tracks split:test,proc:10 --laps 2 --out results/grand_prix/phase10/seed<N>
```
