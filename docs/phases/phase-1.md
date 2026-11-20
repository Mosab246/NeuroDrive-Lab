# Phase 1 report: working driving simulator

Date: 2026-11-20. Scope: spec §18 Phase 1, planned in
[architecture-review.md §10](../architecture-review.md#10-first-implementation-milestone--phase-1-concretely).

## What we built

| Area | Files | Status |
|---|---|---|
| Geometry helpers (spline, resampling, projection, intersections) | `sim/geometry.py` | implemented, tested |
| Track format, loader, validation, Beginner Oval | `sim/track.py`, `tracks/beginner_oval.yaml` | implemented, tested |
| Car physics (kinematic bicycle, steering rate, grip limit) | `sim/vehicle.py` | implemented, tested |
| Progress, ordered checkpoints, laps | `sim/progress.py` | implemented, tested |
| Fixed-step simulation, collisions, lap timing, seeded resets | `sim/simulation.py` | implemented, tested |
| Scripted pure-pursuit autopilot (privileged, not ML) | `agents/pure_pursuit.py` | implemented, tested |
| Headless throughput benchmark | `bench.py` | implemented, measured |
| Playground viewer | `ui/app.py`, `ui/renderer.py`, `ui/clock.py` | implemented, headless-tested; **keyboard driving not verified by a human yet** |
| CLI: `play`, `bench-sim`, `validate-track`, `list-tracks` | `cli.py` | implemented, tested |
| CI (ruff + pytest, Windows & Linux, Python 3.12 & 3.13) | `.github/workflows/ci.yml` | written, **not run yet** (no GitHub remote) |

## How it works

See [physics.md](../physics.md) for the full explanation. The short version:

- The car is a **kinematic bicycle**: speed changes with throttle/brake/drag, and heading
  changes at `speed / wheelbase × tan(wheel angle)`. A stopped car can't turn, the wheels
  take time to turn, and a **grip limit** means the faster you go, the less sharply you
  can turn. So you have to brake for corners.
- A track is a **centerline plus a width**. Progress, collisions, checkpoints and laps all
  come from projecting the car onto that one line.
- The simulation advances in **fixed 1/30 s steps** and never looks at the clock. The
  viewer runs as many fixed steps per frame as the playback speed requires.

## ML concepts this phase sets up

There's no ML yet, but three things built here matter a lot once agents arrive:

1. **Determinism and seeds.** An RL experiment is only reproducible if the environment
   is. Same seed + same actions = bit-identical trajectory, tested with exact equality.
   That's what will let us rerun an experiment and get the same result, and what will
   make "different seed" mean "genuinely different run".
2. **Simulation time ≠ wall time.** Agents will train headless, thousands of steps faster
   than real time. If physics depended on the frame rate, a policy trained headless would
   behave differently in the viewer. The fixed step and the playback-speed test rule
   that out.
3. **Progress that can't be gamed.** In Phase 2 the reward will be "meters of progress".
   Agents are very good at finding loopholes in rewards (*reward hacking*). Making
   progress signed, continuous and checkpoint-ordered now means the obvious loopholes
   (reversing over the line, rocking over a gate, cutting the infield) are already closed
   and tested, before any agent tries them.

## What we tested

Commands run on the development laptop (Windows 11, Python 3.12.13):

```powershell
uv run pytest -q          # 84 passed
uv run ruff check .       # All checks passed!
uv run ruff format --check .
uv run mypy               # Success: no issues found in 17 source files
uv run python -m neurodrive bench-sim --steps 100000
uv run python -m neurodrive validate-track beginner_oval
```

Test groups:

- **Physics** (`test_vehicle.py`): stopped car can't rotate; throttle tops out at max
  speed; braking never goes negative; +steer turns right; steering rate limit; grip limit
  matches its formula; actions clamped; NaN/inf and wrong shapes rejected.
- **Tracks** (`test_track.py`): oval length matches the analytic value within 0.5%;
  start pose; edges exactly half a width away; checkpoints evenly spaced; invalid tracks
  (too-tight turn, figure-eight, bad schema, unknown id, missing file) rejected with
  clear messages.
- **Progress** (`test_progress.py`): one loop = exactly one lap and 10 checkpoints; the
  lap completes on the finish line, not before; continuous across the line; reversing
  never counts; rocking over a gate counts once and nets zero; teleporting can't skip
  progress.
- **Simulation** (`test_simulation.py`): exact determinism; reset; seeded perturbation;
  outer-wall and infield collisions; time limit; stepping after the end raises; a parked
  car stays parked.
- **Integration** (`test_autopilot_laps.py`): the autopilot completes clean laps from
  normal and perturbed starts; 10,000 headless steps; **on/off (keyboard-style) inputs
  can complete a lap**.
- **Smoke** (`test_viewer_headless.py`, `test_cli.py`): the real viewer runs with SDL's
  dummy video driver; its states match a headless run exactly at 0.25× and 8× playback;
  crash and reset render; CLI errors are one line, not a stack trace.

Measured throughput: 3,504 steps/s (simulation alone) after one profiling-driven
optimisation, up from 1,894. Details in [results.md](../results.md).

## Problems and limitations

- **A human hasn't driven it yet.** Phase 1's main criterion ("a human can drive around
  the track") needs you. The digital-input test is a proxy, not a replacement.
- **One bug found in my own test:** the first keyboard-style test ignored steering under
  15%, but the oval's curves need about 13%, so the test car never turned and crashed.
  The car was fine; the test's threshold was wrong. Lowered to 5%.
- **The oval's tightest turn is 32 m, not the designed 40 m**, because a smooth spline
  can't switch instantly from straight to curve. Documented and tested; harmless.
- **Speed:** ~3,500 steps/s before raycasts. Probably fine for PPO, possibly tight for
  NEAT's large populations. Decided by profiling in Phase 2, not guessed now.
- **The autopilot is crude.** It brakes earlier than needed. It exists to prove the track
  is drivable, not to be fast.
- **CI hasn't run** because the repo isn't on GitHub yet.
- **No LICENSE file** until you pick one.
- **Change from the review:** the grip limit is on by default (the review said off),
  because spec §6.2 asks for speed-dependent turning. Explained in physics.md.

## What comes next: Phase 2

Raycast sensors, the 11-number observation vector, the reward function, a Gymnasium
environment that passes the official checkers, random and sensors-only heuristic
baselines, a ray-debug overlay, and the exploit tests listed in the review §7.
