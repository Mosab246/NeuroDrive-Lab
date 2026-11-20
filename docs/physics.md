# Simulator physics and track model

How the Phase 1 simulator works, in plain language first and then precisely.
Code: `src/neurodrive/sim/`.

## Units and conventions

| Thing | Convention |
|---|---|
| Distance | meters |
| Time | seconds; one decision step = 1/30 s, split into 2 physics sub-steps of 1/60 s |
| World axes | y points **up** (math convention). The renderer flips it for the screen |
| Angles | radians, counter-clockwise from +x |
| Steering command | **+1 = full right**, -1 = full left (as in the spec) |
| Throttle/brake command | +1 = full throttle, -1 = full brake, 0 = coast |

## The car: a kinematic bicycle

Imagine squashing the car into a bicycle: one rear wheel, one front wheel that steers,
`wheelbase` apart. The car's position is the middle of the rear axle.

```
           front wheel (steers by angle δ)
               ╱
   ●──────────●       heading ψ →
 rear axle    wheelbase L
```

Every physics sub-step (`dt` = 1/60 s):

1. **Steering actuator.** The command `u_s ∈ [-1, 1]` sets a *target* wheel angle
   `u_s · max_steer`. The wheel moves toward it at most `steer_rate · dt` per sub-step.
   So steering takes time, like a real steering rack. This matters for learning:
   jerky, twitchy policies are visibly worse than smooth ones.
2. **Speed.** Throttle adds up to `max_accel`; brake removes up to `max_brake`. Drag
   (`drag · v²`) and rolling resistance always slow a moving car. Speed is clamped to
   `[0, max_speed]`: **no reverse gear**.
3. **Grip limit.** Turning in a circle of radius `R` at speed `v` needs sideways
   acceleration `v² / R`. Tyres can only provide `max_lateral_accel` (~1.2 g). So at
   speed, the usable wheel angle shrinks to `atan(L · a_max / v²)`. A car going too fast
   simply can't turn tightly enough, and runs wide into the wall. **This is what forces
   a driver to brake before corners.** It's the main skill an agent will have to learn.
4. **Heading.** `ψ̇ = −(v / L) · tan(δ)`. The minus sign is because positive steering
   means right, which is clockwise. Because `ψ̇` is proportional to `v`,
   **a stopped car cannot rotate**, as the spec requires.
5. **Position.** Move forward along the new heading at the new speed
   (semi-implicit Euler: velocity first, then position).

Default parameters (`VehicleParams`): wheelbase 2.6 m, body 4.4 × 1.9 m, top speed
35 m/s (126 km/h), 8 m/s² acceleration, 16 m/s² braking, 0.5 rad (~29°) max wheel angle,
2.5 rad/s steering rate, 12 m/s² grip limit.

> **Change from the architecture review:** the review proposed the grip limit as an
> optional flag, off by default. It's **on** by default, because spec §6.2 asks for
> "turning behavior that depends on movement speed", and without it the car could take
> any corner flat out. That would make the driving problem too trivial to be
> interesting. Set `max_lateral_accel = 0` to disable it.

### Why this model and not something fancier

A dynamic model with tyre slip would allow drifting, but it adds many parameters,
is harder to test, and isn't what this project is about (spec §24: no realistic tyre
physics). The kinematic bicycle is the standard starting point in autonomous-driving
courses, has 5 state variables, and every behaviour is easy to check in a unit test.

## The track: a centerline plus a width

A track file (`src/neurodrive/tracks/*.yaml`) lists control points in driving order and
a width. Loading it:

1. fits a smooth closed curve through the points (a *centripetal Catmull-Rom spline*),
2. resamples that curve every 1 m (the **centerline**),
3. offsets the centerline 6 m left and right (for a 12 m track) to get the **edges**.

Everything else comes from that one centerline:

| Concept | How it's computed |
|---|---|
| Progress `s` | Arc length of the car's closest point on the centerline |
| Off-track / collision | A corner of the car is more than half a width from the centerline |
| Checkpoints | 10 gates at evenly spaced arc lengths; the 10th is the finish line |
| Start/finish line | `s = 0`, the first control point |
| Turn radius | Circle through points 3 m apart along the centerline |

**Validation** runs on every load and rejects a track if: an edge crosses itself, the two
edges cross, or a turn is tighter than half the track width (the inner edge would fold
over itself). `python -m neurodrive validate-track <id-or-path>` reports the result.

The Beginner Oval is designed as two 120 m straights joined by 40 m-radius semicircles,
12 m wide, driven counter-clockwise. Its loaded centerline is 491.3 m (analytic value
491.33 m) and stays within 0.11 m of the ideal shape. The tightest measured radius is
32.2 m, not 40 m: a smooth spline can't jump instantly from a straight (zero curvature)
to a 40 m arc, so it tightens briefly at each join.

## Collision: oriented rectangle vs. corridor

The car's hitbox is its real 4.4 × 1.9 m rectangle, rotated with the car. A collision is
any of its 4 corners leaving the drivable corridor. Checking 4 points is exact for a
rectangle on a corridor whose edges curve slowly compared with the car's size, which
holds for any track that passes validation.

A collision ends the run. The spec allowed either bouncing or ending; ending is simpler,
gives a learning agent a clear signal, and avoids "wall-riding" exploits.

**Tunnelling.** If a car moved further than a wall's thickness in one step, it could
jump through it unnoticed. Here the "wall" is a corridor boundary checked every sub-step
at 60 Hz: at top speed (35 m/s) the car moves 0.58 m per sub-step, far less than the
6 m half-width, so it can't skip across the boundary.

## Progress and laps: why shortcuts don't work

Each sub-step, the car's center is projected onto the centerline, but only onto the
**8 segments either side of where it was last time** (±8 m). Two consequences:

* Progress can only change continuously. On a hairpin, the globally closest centerline
  point might be across the infield; the windowed search can't jump there.
* An unwrapped distance counter goes up when driving forward and **down by exactly the
  same amount** when driving backward. Crossing the start line causes no jump (the
  difference is wrapped around the lap length).

Checkpoint `k` of a lap counts the first time that distance reaches `k/10` of the lap.
Checkpoints must be reached in order, and each counts once per lap. Together:

| Attempted cheat | Result |
|---|---|
| Cut across the infield | Corners leave the corridor → collision |
| Reverse over the finish line | Distance goes negative; no checkpoint, no lap |
| Rock back and forth over a gate | Gate counts once; net progress is zero |
| Teleport (e.g. a bug) | Windowed search can't follow; progress moves at most a few meters |

Each row has a unit test in `tests/unit/test_progress.py` or `tests/unit/test_simulation.py`.

## Determinism and the fixed time step

The simulation never reads the wall clock. Simulated time is computed as
`substep_count × dt` (never summed, so no rounding drift). The only randomness is the
optional start perturbation (a sideways shift up to ±1.5 m and a heading change up to
±5°), drawn from the simulation's own seeded generator. So:

* same seed + same actions → **bit-identical** trajectory (tested with exact equality),
* the viewer at 0.25× or 8× playback produces exactly the same states as a headless run
  of the same number of steps (tested), because playback speed only changes how many
  fixed steps run per frame (`ui/clock.py`), never what a step does.
