# Grand Prix over 3 seeds (Phase 12)

Three separate championships, one per training seed. Each: 12 races (`unseen_a`, `unseen_b`, `proc_0`..`proc_9`),
2 laps, 8 cars, grid rotated every race, F1 points (25, 18, 15, 12, 10, 8, 6, 4). The scripted drivers are the same
in all three; the four learned drivers are that seed's models (seed 0: `best_model_v2`, re-selected on 7 tracks;
seeds 1-2: `best_model`; self-play models: `best_model`; all selected on the same 7 tracks).

Per-seed standings: [seed 0](seed0/standings.md), [seed 1](seed1/standings.md), [seed 2](seed2/standings.md).

| # | Driver | Type | Points (total) | Points per seed | Place per seed | Wins | Finished | Hits |
|---|---|---|---|---|---|---|---|---|
| 1 | PPO self-play | learned (PPO) | 507 | 149 / 163 / 195 | 2 / 2 / 1 | 8 | 27/36 | 80 |
| 2 | NEAT traffic | learned (NEAT) | 483 | 139 / 164 / 180 | 5 / 1 / 2 | 3 | 31/36 | 60 |
| 3 | PPO traffic | learned (PPO) | 443 | 169 / 95 / 179 | 1 / 8 / 3 | 10 | 23/36 | 68 |
| 4 | NEAT self-play | learned (NEAT) | 397 | 144 / 136 / 117 | 3 / 3 / 5 | 1 | 27/36 | 39 |
| 5 | heuristic | scripted (sensors only) | 364 | 114 / 125 / 125 | 6 / 4 / 4 | 0 | 36/36 | 91 |
| 6 | autopilot | scripted (privileged) | 322 | 104 / 111 / 107 | 7 / 5 / 6 | 0 | 36/36 | 88 |
| 7 | PPO solo | learned (PPO) | 294 | 143 / 108 / 43 | 4 / 6 / 8 | 8 | 15/36 | 125 |
| 8 | NEAT solo | learned (NEAT) | 266 | 80 / 108 / 78 | 8 / 7 / 7 | 6 | 14/36 | 63 |

Command (per seed, models of that seed):

```
neurodrive grand-prix --driver "PPO solo=<run>/best_model.zip" --driver "PPO traffic=..." \
  --driver "NEAT solo=<run>/best_model.pkl" --driver "NEAT traffic=..." --driver "PPO self-play=..." --driver "NEAT self-play=..." --driver autopilot --driver heuristic \
  --tracks split:test,proc:10 --laps 2 --out results/grand_prix/phase12/seed<N>
```
