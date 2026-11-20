# Measured results

Only numbers that were actually measured go here, each with the exact command, the
machine, and the date. There are no learned-agent results yet.

## Baseline drivers on the Beginner Oval (Phase 2)

None of these drivers learn. Date: 2026-11-20, commit of Phase 2. Each row is 100
episodes on the same 100 **validation** scenario seeds (1,000,000–1,000,099): one lap
from a standing start, start offset up to 1.5 m sideways and 5° of heading.

```powershell
uv run python -m neurodrive evaluate --controller <name> --episodes 100 --split validation
```

| Driver | Kind | Laps completed (95% CI) | Lap time (completed laps) | Mean lap progress | Endings |
|---|---|---|---|---|---|
| `random` | random | 0/100 (0–3.7%) | none | 0.2% | 100 stalled |
| `smoothed_random` | random | 0/100 (0–3.7%) | none | 2.7% | 60 collision, 40 stalled |
| `heuristic` | hand-written, sensors only | 100/100 (96.3–100%) | 26.36 s mean, 26.35 s best, n = 100 | 100% | 100 lap complete |
| `autopilot` | scripted, **privileged** | 100/100 (96.3–100%) | 25.37 s mean, 25.37 s best, n = 100 | 100% | 100 lap complete |

Also run: `random` on 1,000 validation scenarios (`--episodes 1000`): 0/1000 completed,
1000 stalled, and no invalid observation was produced (the environment raises an error on
any non-finite observation, so a finished run means none occurred).

**What these numbers say:**

- **Uniform random driving goes nowhere.** It averages zero throttle and zero steering, so
  the car shakes in place until the stall rule ends it. Smoothed random at least moves,
  then usually hits a wall.
- **The oval is easy.** A few lines of hand-written logic, seeing only the 9 rays, speed and
  wheel angle, finish every lap. A learned agent on this track must be compared with the
  heuristic, not just with random.
- **The heuristic's gains were tuned on this track** (9 settings of steering gain and top
  target speed tried; all 9 finished every lap; a middle setting was kept as default, not
  the fastest at 22.5 s). This is reported because unequal tuning effort is a known way
  comparisons go wrong.
- **The start perturbation creates almost no variety here.** The heuristic's 100 lap times
  span 26.35–26.36 s, and the autopilot's are identical to two decimals. Deterministic
  drivers recover from a 1.5 m / 5° offset within a few meters, so on this track
  100 scenarios behave almost like one. Real variety will need stronger perturbations
  (e.g. random starting point along the track) and more tracks: Phase 5.
- **The autopilot is slower than its own Phase 1 flying lap** (25.37 s here vs 23.33 s)
  only because these are standing starts.

## Simulator throughput (Phase 1–2)

**Machine:** Lenovo Legion Pro 5i, Intel Core i9-13900HX, 32 GB RAM, Windows 11 Pro,
Python 3.12.13, NumPy 2.5.5. Single process, single thread. Power state was not
controlled or recorded for these runs (to fix in later benchmarks; see the review §2.3).

**Command:**

```powershell
uv run python -m neurodrive bench-sim --steps 100000
```

**Modes:**

- `sim`: `Simulation.step` with seeded random actions; resets whenever the car crashes.
- `sim+autopilot`: the scripted pure-pursuit driver, including its own per-step lookups.
- `env (heuristic)` (added in Phase 2): the full Gymnasium environment, i.e. physics +
  9 raycasts + observation + reward, driven by the heuristic. This is what a learning
  algorithm pays per step, minus its neural network.

One step = 1/30 s of simulated time = 2 physics sub-steps.

| Date | Code state | `sim` | `sim+autopilot` | `env (heuristic)` |
|---|---|---|---|---|
| 2026-11-20 | Phase 1, first version | 1,894 | 1,434 | — |
| 2026-11-20 | Phase 1, after the projection fix below | 3,504 | 2,064 | — |
| 2026-11-20 | Phase 2 (sensors added; sim code unchanged). On AC power (`Win32_Battery` status 2); Windows power mode not recorded | 3,902 | 2,321 | 1,980 |

All values are steps per second from single runs. **The noise is large:** the `sim` code
did not change between the last two rows, yet it measured 3,504 and 3,902, and a
30,000-step run the same day gave 5,200. Laptop boost clocks and thermal state move these
numbers by tens of percent. Treat differences under ~40% as noise until benchmarks are
repeated and averaged (planned for Phase 6).

### Phase 2 profile of the full environment

`cProfile` on 15,000 `env.step` calls with the heuristic driver: raycasting ~25% of the
time, projecting the car onto the centerline ~27%, computing the car's corners ~9%,
the rest spread thin. No single hotspot remains, so no further optimisation was done.

### What the profile showed

`cProfile` on 20,000 steps of the first version: 74% of the time was in
`project_onto_closed_polyline`, about 110 µs per call. The arrays involved are tiny
(5 points × 17 segments), so the cost was NumPy's per-call overhead, not arithmetic.
The fix was to do fewer, larger calls: precompute each segment once, and project all
5 points of the car in one call per sub-step instead of two.

### What this means for training (arithmetic, not measurement)

At ~2,000 environment steps/s in one process, 1 million steps take roughly 8–9 minutes
before the policy network's cost. PPO can run several environments in parallel and NEAT
can evaluate genomes on several cores; whether that helps on Windows gets measured in
Phases 3–4.

## Scripted autopilot lap times (sanity check, not a benchmark)

The pure-pursuit autopilot is a scripted, privileged driver (it reads the track
geometry). Its lap times only show that laps are timed. They are not a performance
reference: its speed control is deliberately simple and brakes earlier than needed.

```powershell
uv run python -c "from neurodrive.sim.track import load_track; from neurodrive.sim.simulation import Simulation, SimConfig; from neurodrive.agents.pure_pursuit import PurePursuitDriver; s=Simulation(load_track('beginner_oval'), config=SimConfig(laps_to_finish=2)); d=PurePursuitDriver(); [s.step(d.act(s)) for _ in iter(lambda: s.done, True)]; print(s.lap_times)"
```

Output on 2026-11-20: `[25.366666666666667, 23.333333333333332]`, i.e. 25.37 s from a
standing start, then 23.33 s for a flying lap. Exact values change whenever the physics
or the autopilot changes; the test suite only checks that both laps complete and the
second isn't slower.
