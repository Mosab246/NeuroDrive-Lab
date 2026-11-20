# ML guide: the environment an agent learns in

This explains the *learning problem* NeuroDrive Lab poses, for someone learning ML.
It grows with each phase. Code: `src/neurodrive/env/`.

## The loop every agent lives in

Reinforcement learning (and our neuroevolution setup) is built around one loop:

```
        observation (11 numbers)
   ┌──────────────────────────────┐
   │                              ▼
 Environment                    Agent (policy)
 (car + track)                  "given what I see, what do I do?"
   ▲                              │
   └──────────────────────────────┘
        action (2 numbers)  ──► reward (1 number) + "is it over?"
```

* **State:** everything about the world: car position, speed, heading, the whole track.
  The agent never sees this directly.
* **Observation:** the part of the state the agent *is* shown. Here: what its sensors
  measure.
* **Action:** what the agent does: steering and throttle/brake.
* **Reward:** a number after every step saying how good that step was.
* **Episode:** one attempt, from reset to an ending (lap done, crash, stall, time out).
* **Policy:** the agent's decision rule, observation → action. *Learning* means improving
  the policy so the total reward per episode goes up.

The environment follows the standard [Gymnasium](https://gymnasium.farama.org) interface
(`reset()` and `step(action)`), so any library that speaks Gymnasium (Stable-Baselines3's
PPO in Phase 3) can train on it unchanged. It's also registered as `NeuroDrive-v0`.

## What the agent sees: the observation vector

11 numbers, each scaled into a small fixed range:

| # | Feature | How it's scaled | Range |
|---|---|---|---|
| 0–8 | Distance along 9 rays at -90°, -60°, -40°, -20°, 0°, +20°, +40°, +60°, +90° (negative = left) | distance ÷ 50 m | 0 … 1 |
| 9 | Speed | ÷ top speed | 0 … 1 |
| 10 | Current front-wheel angle | ÷ max angle | -1 … 1 |

Press `V` in the playground to see the rays.

**Why scale everything?** A neural network starts with small random weights. If one input
ranges 0–35 and another 0–1, the big one dominates the early learning signal just because
its numbers are bigger, not because it matters more. Scaling everything to roughly the
same range lets the network treat inputs fairly from the start.

**Why include the wheel angle?** The wheels turn at a limited rate, so "where are the
wheels pointing right now" is part of the car's state. Without it, two situations that
look identical to the agent could need different actions. RL assumes the observation
tells you enough to act well (the *Markov property*); leaving out the wheel angle would
quietly break that.

**What it does NOT get by default:** its position on the track, the direction of the
road, how far it is from the center line. Those are known to the simulator but not to
any sensor a car would have. They're available as an option
(`observation.route_features: true`, 14 inputs) and are labeled **privileged**. Whether
they help learning, and whether they hurt generalization to new tracks, is one of the
project's experiments.

## What the agent does: the action

Two numbers, each -1 … +1: `[steering, throttle/brake]`. +1 steering = full right;
+1 pedal = full throttle, -1 = full brake. Values outside the range are clamped; NaN or
infinity raises an error rather than being quietly "fixed". PPO and NEAT both produce
actions through this one interface.

## How the agent is scored: the reward

```
reward per step = 0.1 × (meters of progress along the track this step)
                + 10    if the lap was completed this step
                − 10    if the car crashed this step
                − 10    if the car "stalled" (under 2 m of progress in the last 5 s)
```

A clean lap of the Beginner Oval is worth about 0.1 × 491 + 10 ≈ 59. (A test pins this,
so changing the reward scale is always a deliberate, visible decision.)

### Reward design is where RL projects usually go wrong

An RL agent optimizes *exactly* what you reward, not what you meant. If there is a
loophole, it will find it: this is called **reward hacking**. Each rule above exists to
close a specific loophole:

| What the agent might try | Why it doesn't pay |
|---|---|
| Stand still forever (zero reward, zero risk) | After 5 s with under 2 m progress: "stalled", −10, episode over |
| Drive back and forth over the same stretch | Progress is *signed*: going back 1 m costs exactly what going forward 1 m earned. Tested over random episodes: total progress reward always equals 0.1 × net distance |
| Farm a checkpoint by crossing it repeatedly | There are no checkpoint rewards. Checkpoints only validate laps |
| Cut across the infield | That's a crash |
| Turn around and drive the wrong way | Every meter backwards costs reward, then "stalled" ends it |
| Crash early to avoid future penalties | There *are* no per-step penalties to avoid, so crashing only loses future progress |

That last row is a deliberate choice. Many tutorials add "−0.01 every step" to encourage
speed. But if the crash penalty is small compared with a long episode's worth of those
penalties, the cheapest plan becomes "crash immediately". Agents really do learn that.
Progress-based reward already encourages speed (more meters sooner), so we don't need it.

## How an episode ends: terminated vs truncated

| Ending | Gymnasium flag | Meaning |
|---|---|---|
| Lap complete | `terminated` | Success, nothing more to earn |
| Crash | `terminated` | Genuinely over |
| Stalled | `terminated` | Treated as a failure |
| 90 s time limit | `truncated` | *We* stopped it; the car could have kept going |

This distinction feeds into PPO's math. PPO learns a *value function*: "from this
situation, how much more reward will I probably get?" After a crash the honest answer is
zero. After a time cut-off it's not zero, because the car was still driving. If a time
cut-off were reported as `terminated`, the agent would learn that being in perfectly good
situations late in an episode is worthless. Tests check each ending uses the right flag.

## The baselines: what "learning worked" will be compared against

None of these learn. They exist so a trained agent's results mean something.

| Baseline | What it is | Uses |
|---|---|---|
| `random` | New uniformly random action every step | Nothing |
| `smoothed_random` | Random actions that drift smoothly | Nothing |
| `heuristic` | Hand-written: steer toward the side with more room, slow down when the road ahead is short | **Only the same 11 numbers the agents get** |
| `autopilot` | Pure-pursuit steering along the known centerline | **Privileged**: reads the track geometry |

The heuristic is the important one. It answers: "how far does simple logic get with
exactly the information the agent has?" If a learned agent can't beat it, the learning
isn't adding much. Measured numbers are in [results.md](results.md).

## Fair evaluation, in practice

`python -m neurodrive evaluate` runs any controller on a fixed list of **scenario seeds**.
Each seed gives a slightly different start (up to 1.5 m sideways, 5° rotated). Every
controller gets the same list, and validation and test seeds never overlap. Every episode
is written to `episodes.csv`; summaries are computed from those rows, so no number in a
report can exist without episodes behind it. Completion rates come with a 95% confidence
interval (Wilson method), and lap times are always reported with how many laps they
average over.
