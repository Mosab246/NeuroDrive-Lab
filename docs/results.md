# Measured results

Only numbers that were actually measured go here, each with the exact command, the
machine, and the date. There are no ML results yet.

## Simulator throughput (Phase 1)

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

One step = 1/30 s of simulated time = 2 physics sub-steps.

| Date | Code state | `sim` steps/s | `sim+autopilot` steps/s |
|---|---|---|---|
| 2026-11-20 | First version: centerline projected separately for progress and collision, segment geometry recomputed on each call | 1,894 | 1,434 |
| 2026-11-20 | Segment geometry precomputed once per track, one projection per sub-step for center + corners | 3,504 | 2,064 |

These are single runs, not averages. Expect a few percent of noise.

### What the profile showed

`cProfile` on 20,000 steps of the first version: 74% of the time was in
`project_onto_closed_polyline`, about 110 µs per call. The arrays involved are tiny
(5 points × 17 segments), so the cost was NumPy's per-call overhead, not arithmetic.
The fix was to do fewer, larger calls: precompute each segment once, and project all
5 points of the car in one call per sub-step instead of two.

### What this means for training (reasoning, not measurement)

At ~3,500 steps/s, one process steps 1 million environment steps in about 5 minutes,
*before* raycasts, observations, rewards and the policy network are added in Phases 2–3.
Raycasts will add cost. Whether more optimisation is needed (more vectorisation, Numba,
parallel environments) will be decided by profiling the full environment in Phase 2,
not now.

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
