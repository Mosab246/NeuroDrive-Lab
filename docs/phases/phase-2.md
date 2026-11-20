# Phase 2 report: ML environment and baselines

Date: 2026-11-20. Scope: spec §18 Phase 2, acceptance criteria from
[architecture-review.md §7](../architecture-review.md#7-refined-roadmap-with-acceptance-criteria).

## What we built

| Area | Files | Status |
|---|---|---|
| Raycast distance sensors (9 rays, 50 m) | `sim/sensors.py` | implemented, tested against hand-computed distances |
| Observation vector `obs-v1` (11 inputs; optional +3 privileged) | `env/observations.py` | implemented, tested |
| Reward function | `env/rewards.py` | implemented, tested |
| Gymnasium environment `DrivingEnv`, registered as `NeuroDrive-v0` | `env/driving_env.py` | implemented, passes both checkers |
| Validated config (pydantic + YAML) | `env/config.py`, `configs/env/default.yaml` | implemented, tested |
| Controller interface + 4 baselines | `agents/base.py`, `agents/baselines.py` | implemented, tested |
| Evaluator: fixed scenarios, per-episode CSV, summary JSON, run metadata | `evaluation/`, `experiments/metadata.py` | implemented, tested |
| CLI `evaluate`, `play --driver heuristic\|smoothed_random` | `cli.py` | implemented, tested |
| Ray overlay (`V`) and baseline drivers in the playground; `render_mode="rgb_array"` | `ui/` | implemented, headless-tested |
| Dependencies: gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0+cpu | `pyproject.toml` | installed from the CPU wheel index |

## How it works

See [ml-guide.md](../ml-guide.md) for the learner-oriented explanation. In short:

- **Observation:** 9 ray distances, speed and wheel angle, each scaled to 0…1 or -1…1.
  Nothing about the track's layout unless `route_features` is turned on, and then it's
  labeled privileged.
- **Reward:** 0.1 per meter of signed progress, +10 for the lap, −10 for a crash or a stall.
  No time penalty and no checkpoint bonuses, on purpose.
- **Endings:** lap, crash and stall are `terminated`; the 90 s limit is `truncated`.
- **Evaluation:** every driver runs on the same fixed list of seeded starting scenarios;
  validation and test seeds can't overlap.

## ML concepts this phase introduced

- **Observation vs state**, and why the agent gets sensors rather than the map.
- **Normalization**: why every input is scaled to a similar range.
- **The Markov property**: why the wheel angle had to be in the observation.
- **Reward hacking**: six specific loopholes and the rule that closes each.
- **Terminated vs truncated**, and how mixing them up corrupts a value function.
- **Baselines**: a learned agent's result means nothing without "what does no learning
  get?" and "what does simple logic get with the same inputs?"

## What we tested

```powershell
uv run pytest -q          # 129 passed
uv run ruff check .       # All checks passed!
uv run ruff format --check .
uv run mypy               # Success: no issues found in 32 source files
```

Acceptance criteria from the review, and how each was checked:

| Criterion | Result |
|---|---|
| Gymnasium and SB3 env checkers pass | Both pass in `tests/integration/test_env.py`, **with warnings turned into errors** |
| Observations stay valid | 40 random episodes in the tests check finite, `float32` and in bounds every step; plus 1,000 random episodes via `evaluate` with no error raised |
| Random actions run headless; resets repeat | Covered by the above; same seed → identical episode (exact) |
| Metrics and outcomes recorded | `episodes.csv` + `summary.json` with versions, git commit, config, seeds |
| No obvious reward exploit | Tests: progress reward = 0.1 × net distance for any behaviour; standing still = stalled, −10; wrong-way driving loses reward and stalls; crash penalty applied exactly once; lap bonus exactly once; clean lap ≈ 59 |
| Baselines evaluated, real numbers recorded | [results.md](../results.md): heuristic and autopilot 100/100, random and smoothed random 0/100 |

## Problems and limitations

- **I wrote one test that passed for the wrong reason.** The first "wrong-way driving" test
  tried to U-turn the car, but this car can't U-turn on a 12 m track at speed. It just
  crashed, and the test's loose check ("stalled or collision") passed anyway. I inspected
  the episode, saw the heading never went past 40°, and rewrote the test to start the car
  facing backwards. It now checks it really drove >10 m the wrong way and was stalled.
- **The oval is too easy to tell drivers apart.** The sensors-only heuristic finishes every
  lap. PPO will need to be judged against the heuristic, and harder tracks matter more
  than the plan assumed.
- **Start perturbations add almost no variety on the oval** (heuristic lap times span
  26.35–26.36 s over 100 scenarios). Recommendation for Phase 5: also randomize the
  starting point along the track, and use more tracks.
- **The heuristic was tuned on the same track it's evaluated on** (9 settings tried).
  Reported in results.md.
- **Benchmark noise is large** (the same code measured 3,504 / 3,902 / 5,200 steps/s).
  My Phase 1 note claiming "a few percent of noise" was wrong and has been corrected.
- **Observation bounds are enforced by clipping.** By construction values can't leave their
  ranges, so clipping only guards rounding, but it also means the bound check can't
  catch a scaling bug on its own. The hand-computed sensor tests and route-feature tests
  cover the scaling instead.
- **CI still hasn't run** (no GitHub remote). torch adds a ~200 MB download per CI job.

## What comes next: Phase 3, PPO

Train PPO (Stable-Baselines3) on the oval: training config, seeds, checkpoints,
TensorBoard, periodic evaluation on the validation scenarios, best-model saving, a
`PPOController` that loads a saved model, and a comparison against these baselines. Plus
the measured CPU-vs-GPU and `DummyVecEnv`-vs-`SubprocVecEnv` throughput comparisons.
