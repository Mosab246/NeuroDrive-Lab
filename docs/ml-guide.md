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

## PPO: how the first agent learns (Phase 3)

### The policy is a neural network

The PPO agent's policy is a small neural network: 11 inputs → 64 → 64 → 2 outputs
(steering, pedal). About 5,000 numbers (*weights*) decide what it does. At the start they
are random, so the car does random-ish things. Training adjusts the weights.

During training the network doesn't output one action; it outputs the **center of a bell
curve** (Gaussian) for each action, plus how wide that bell curve is. The actual action is
*sampled* from it. That randomness is **exploration**: trying slightly different things to
find out what works. The width (`std`) starts at 1 and is learned too: as the agent gets
confident, it usually narrows. At evaluation time we use the center directly
(`deterministic=True`), with no randomness.

A second network of the same size, the **value function** (or *critic*), learns to predict
"from this situation, how much reward will I collect from here on?"

### One round of PPO

1. **Collect experience.** Run the current policy in 8 copies of the environment for
   2,048 steps each: 16,384 (observation, action, reward) records.
2. **Score each action.** For every step, compare what actually happened with what the
   critic expected. Better than expected → that action gets a positive **advantage**;
   worse → negative. (The exact recipe, *GAE*, blends one-step and long-run estimates;
   `gamma = 0.99` and `gae_lambda = 0.95` control how far ahead it looks.)
3. **Nudge the policy.** Make actions with positive advantage more likely and negative
   ones less likely, using gradient descent, 10 passes over the data in batches of 64.
4. **But not too far.** This is the "proximal" in PPO. If an update would change the
   probability of an action by more than 20% (`clip_range = 0.2`), the extra push is cut
   off. Without this, one lucky or unlucky batch can wreck a policy that took an hour to
   learn. It's the main reason PPO is popular: it's stable without much tuning.
5. **Throw the data away** and repeat with the updated policy. PPO is *on-policy*: it only
   learns from experience collected by the current policy.

### Reading the TensorBoard graphs

```powershell
uv run tensorboard --logdir runs/ppo
```

| Graph | What it means | Healthy looks like |
|---|---|---|
| `rollout/ep_rew_mean` | Average training-episode reward (with exploration noise) | Rising, then flattening |
| `episodes/lap_complete_rate` etc. | How training episodes end (last 100) | Laps up, collisions/stalls down |
| `eval/completion_rate`, `eval/mean_return` | Deterministic policy on fixed validation scenarios | The honest progress measure |
| `train/entropy_loss` | Minus how random the policy still is | Rises toward 0 slowly as the policy gets more certain |
| `train/std` | Width of the exploration bell curve | Shrinks over training |
| `train/approx_kl` | How much one update changed the policy | Small (~0.01–0.03); large spikes mean unstable updates |
| `train/clip_fraction` | Share of samples where the clip in step 4 kicked in | Roughly 0.05–0.3 |
| `train/value_loss` | How wrong the critic's predictions are | Noisy; can rise when the agent discovers new, bigger rewards |
| `train/explained_variance` | How much of the return variation the critic explains (1 = perfect, 0 = useless) | Climbs toward 0.8–1 |

The "healthy" ranges are rules of thumb, not guarantees.

## NEAT: how the second agent evolves (Phase 4)

### Evolution instead of gradients

PPO improves **one** network by calculating which direction to nudge each weight
(*gradient descent*). NEAT doesn't calculate anything like that. It keeps a **population**
of 150 networks and does what breeders do:

1. **Test everyone.** Each network drives the car; its driving earns a *fitness* score.
2. **Select.** Keep the best performers; the rest are discarded.
3. **Breed.** Make the next generation from the survivors: combine two parents
   (**crossover**) and make small random changes (**mutation**): tweak a weight, add a
   connection, insert a new neuron.
4. Repeat for 100 generations.

No network ever learns during its own drive. Improvement happens *between* generations,
because better drivers have more offspring.

### What makes NEAT special: the network's shape evolves too

Most neuroevolution only evolves the weights of a fixed network. NEAT (*NeuroEvolution of
Augmenting Topologies*, Stanley & Miikkulainen, 2002) also evolves the **structure**:

- Every network starts **minimal**: the 11 inputs wired straight to the 2 outputs, 22
  connections, no hidden neurons.
- Mutations can add connections and insert neurons. Structure only survives if it helps.
- Each new connection or neuron gets an **innovation number**, a historical ID. That's how
  NEAT lines up two different-shaped parents during crossover.
- **Speciation:** a new structure usually performs *worse* at first, before its weights are
  tuned. If it competed with everyone immediately it would be wiped out. NEAT groups
  similar networks into **species** that mostly compete among themselves, giving
  innovations time to mature.

`generations.csv` tracks the best network's hidden neurons and connections every
generation, and `complexity.png` plots it.

### Fitness vs reward

| | PPO | NEAT |
|---|---|---|
| Feedback | A reward after **every step** | One **fitness** number per **whole episode** |
| What it tells the learner | Which actions, in which situations, worked | Only how good the driver was overall |
| Speed incentive | Built in: discounting (γ = 0.99) makes reward sooner worth more | Must be added explicitly |

NEAT's fitness here is:

```
fitness = the episode's total reward (same reward as PPO)
        + 1 point per second the lap finishes under the 90 s limit   (laps only)
```

Why not just the total reward? A completed lap always earns 0.1 × 491 m + 10 ≈ 59,
whatever the lap time, so evolution would stop improving speed as soon as cars can finish.
Why not PPO's discounted reward? With γ = 0.99, reward more than a few seconds ahead counts
almost nothing, so as a score for a whole 25-second episode it would mostly measure the
first few seconds.

**Fair-luck rule:** within one generation, every network drives the *same* 2 starting
scenarios. Otherwise a network could win just by getting an easier start.

### Counting the cost

Every step every network drives counts as an environment step. 150 networks × 2 episodes
× ~700 steps is ~200,000 steps *per generation* once they can finish laps. NEAT typically
uses far more environment steps than PPO, but each step is cheap (no gradient
calculations) and networks are scored in parallel on 12 processes. That's why the
comparison (Phase 6) reports both environment steps and wall-clock time.

## SAC: learning from replayed experience (Phase 13)

PPO is **on-policy**: it learns only from the episodes its current policy just drove, then
throws them away. SAC (Soft Actor-Critic) is **off-policy**: every step it ever took goes
into a *replay buffer* (up to a million), and each update trains on a random batch from it,
so old experience keeps teaching.

- **Two kinds of network.** The *actor* chooses the action, like PPO's policy. Two *critics*
  learn how good an action is in a situation (the expected future reward), and the actor is
  pushed towards actions the critics rate highly. Using the smaller of two critics avoids
  over-optimistic estimates.
- **"Soft": rewarded for staying random.** SAC adds a bonus for keeping its actions varied
  (entropy), tuned automatically. That keeps it exploring instead of locking into the first
  thing that works.
- **The trade-off.** Reusing experience makes SAC sample-efficient: here it lapped after
  ~100k environment steps, where PPO needed ~280k. But it does one gradient update per
  environment step, so on a CPU it is about 10x slower per step than PPO. Which algorithm is
  "faster" depends on whether you count environment steps or wall-clock time.

When judged, SAC (like PPO) drives with its mean action, no randomness:
`deterministic=True`.

## Generalization and overfitting (Phase 5)

An agent can score perfectly on the track it trained on and still not know how to drive.
Our oval-only agents did exactly that: the oval only ever turns left, so they never
learned right turns, and on clockwise tracks they completed 0-4% of laps. That's
**overfitting**: learning the quirks of the training data instead of the general skill.

The defenses, all used here:

- **Varied training data.** Three training tracks in both directions, random start
  points. With that, both agents lapped all 22 tracks they had never seen.
- **A separate validation set** to choose which checkpoint to keep. If you pick the
  checkpoint that does best on the *test* tracks, the test score stops being honest.
- **A test set that is never touched** until the end, enforced in code: training on a
  test track raises an error.

## Comparing two learning methods fairly (Phase 6)

"Which algorithm is better?" is easy to answer wrongly. The traps, and what this project
does about each:

| Trap | Why it misleads | What we do |
|---|---|---|
| Comparing iterations ("100 generations vs 1M steps") | A generation and a gradient update aren't comparable units | Give both the same budget of **environment steps** (car-driving steps) |
| One run each | Random seeds change results a lot; one lucky run proves little | **Several seeds per algorithm**, every seed reported |
| Picking the best model on the test set | The test score then measures your selection, not generalization | Selection on the validation track only |
| Only reporting lap time | A driver that finishes 10% of laps can still have a fast average | Laps completed **and** lap time **and** crashes, with episode counts |
| Only reporting steps | NEAT uses steps cheaply and in parallel; PPO spends time on gradient math | Report **wall-clock time** too, measured with nothing else running |
| Claiming a winner from a small difference | With 3 seeds, randomness can easily explain it | Show the spread across seeds; call differences smaller than that "unresolved" |

What "fair" can't fix: the two methods were set up with standard default settings, not
tuned against each other. A different PPO batch size or a different NEAT population size
could change the result. The benchmark compares *these configurations*, not the
algorithms in the abstract.
