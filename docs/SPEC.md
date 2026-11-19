# NEURODRIVE LAB — Complete Project Specification for Claude Code

> The user's original specification, saved verbatim in content. Sections 1–27.

## 1. Your Role

Act as a senior machine-learning engineer, reinforcement-learning researcher, simulation engineer, Python software architect, and technical mentor.

We are going to design and build a complete, high-quality, open-source machine-learning project called NeuroDrive Lab (working title).

You are responsible for helping me design the architecture, implement the simulation, build the ML training systems, evaluate the algorithms, debug their behavior, create a polished interactive application, and prepare the project for public release.

However, do not attempt to implement the entire project in one enormous change. Work systematically through phases, with explicit milestones, tests, reviews, and working demonstrations. Treat this as a real software-engineering and machine-learning project, not a throwaway prototype or tutorial.

Important: I want to understand how the ML actually works. Do not just generate code and leave me with a black box. Explain the important concepts, algorithms, training decisions, and experimental results as we progress.

## 2. Project Vision

### The Big Idea

Build an interactive, visual AI driving simulator where autonomous agents learn to drive cars around racing circuits through actual machine-learning processes. The agents should initially have no driving knowledge. They must learn driving behavior through training.

The central research question is: **Can an AI learn to drive autonomously, and how do reinforcement learning and neuroevolution compare when solving the same driving problem?**

We will implement at least two independent learning approaches:

1. Reinforcement Learning — PPO (Proximal Policy Optimization)
2. Neuroevolution — NEAT (NeuroEvolution of Augmenting Topologies)

The application should allow us to train, observe, evaluate, replay, and compare these approaches.

The project combines: real machine learning; neural networks; reinforcement learning; evolutionary algorithms; 2D physics simulation; autonomous navigation; experimentation and benchmarking; data visualization; interactive game-like experiences; reproducible scientific evaluation; software engineering and testing. It should feel like an AI experimentation laboratory.

### What Makes This Project Special?

This is not simply a racing game with scripted opponents. We want a simulation in which an AI genuinely learns how to drive. The user should be able to watch a training session where an agent initially crashes, makes mistakes, improves its behavior, and eventually navigates a track successfully. Later, the user should be able to compare two entirely different machine-learning approaches.

Potential demonstrations include:

- Random AI versus trained AI
- Early training versus late training
- PPO versus NEAT
- AI versus human
- Agents trained on one track and evaluated on unfamiliar tracks
- Multiple evolutionary generations competing against each other
- How sensor configurations affect learning
- How reward functions influence behavior
- How changing track geometry affects generalization

Avoid fake learning animations or predefined driving paths masquerading as learned behavior. Every trained agent must use its actual learned policy or evolved neural network to make driving decisions.

## 3. Primary Goals

The project has four equally important goals.

### Goal A — Fun and Interactive

The simulator must be visually interesting and enjoyable to experiment with. I want to be able to:

- Watch an AI learn to drive.
- See cars crashing, recovering, and improving.
- Observe different agents and generations.
- Select tracks and configurations.
- Race manually against trained AI agents.
- Watch saved demonstrations.
- Compare different algorithms.
- Pause, resume, and adjust simulation speed.
- Visualize sensor rays and control decisions.
- Explore training performance visually.

The interface should be modern, clean, responsive, and polished. However, visual polish must never come at the expense of ML correctness or simulation performance.

### Goal B — Actually Understand Machine Learning

The project should teach me through practical implementation. I want to understand:

- What a neural network does.
- How observations become model inputs.
- How a network produces steering and acceleration decisions.
- How a reinforcement-learning environment works.
- What actions, states, observations, rewards, and episodes mean.
- What a policy is.
- What PPO does at a conceptual and implementation level.
- How exploration and exploitation work.
- Why reward design matters.
- How neural networks improve during training.
- What NEAT is.
- How mutation, crossover, and selection work.
- How NEAT evolves network topology.
- How fitness differs from reinforcement-learning rewards.
- Why training can fail.
- How to recognize overfitting.
- How generalization to unfamiliar environments works.
- How to compare algorithms fairly.
- What training stability and reproducibility mean.

Use established ML libraries rather than reimplementing every complex mathematical component unnecessarily. However, explain the underlying principles, and make sure I can inspect what is happening. Include educational diagrams, documentation, examples, and accessible explanations where useful. I should eventually be capable of explaining the project confidently in an interview.

### Goal C — Professional GitHub Portfolio

The repository should demonstrate meaningful engineering skills. It should showcase: clean and maintainable architecture; modular code; object-oriented design where appropriate; typed Python; proper documentation; automated testing; reproducible ML experiments; model training and evaluation; performance profiling; scientific comparisons; visualization and presentation; effective dependency management; good development practices.

A recruiter, software engineer, or ML engineer should be able to inspect the repository and understand what I built, how the models work, and what the experiments demonstrated.

### Goal D — Experimentation Platform

The architecture should allow future expansion without requiring a rewrite. Eventually, I should be able to experiment with: additional learning algorithms; different vehicle physics; new track geometries; curriculum learning; sensor configurations; dynamic obstacles; multiple cars; competitive racing; more sophisticated neural-network architectures; procedural track generation; domain randomization; potential 3D integration.

These are future capabilities, not all mandatory for the initial release.

## 4. Development Environment and Constraints

My development machine:

- OS: Windows 11
- Laptop: Lenovo Legion Pro 5i
- CPU: Intel Core i9, 13th generation
- GPU: NVIDIA RTX 4070 Laptop GPU
- GPU VRAM: 8GB
- RAM: 32GB

The project must run locally. Requirements:

1. Support Windows development without requiring Linux-exclusive tools.
2. Support CPU-only execution when appropriate.
3. Use GPU acceleration when beneficial and supported.
4. Avoid unnecessarily large neural networks.
5. Optimize simulation throughput.
6. Train headlessly without rendering every simulation step.
7. Separate training performance from graphics performance.
8. Make training duration and computational budgets configurable.
9. Avoid paid external services.
10. Do not require cloud infrastructure.
11. Do not depend on Claude or any external LLM API at runtime.
12. Save trained models and results locally.
13. Keep dependencies reasonably lightweight and maintainable.

Do not assume the RTX 4070 is always the performance bottleneck. Small reinforcement-learning workloads may be limited by CPU simulation and environment stepping. Measure performance rather than assuming GPU training is automatically faster.

## 5. Preferred Technology Stack

Start with a Python-first implementation. Suggested technologies:

**Simulation and core application:** Python; Pygame for the initial 2D renderer and interactive viewer; NumPy for mathematical and numerical operations.

**Machine learning:** PyTorch; Gymnasium for the reinforcement-learning environment; Stable-Baselines3 for PPO; NEAT-Python for neuroevolution.

**Experiment tracking and analysis:** TensorBoard; Matplotlib; Pandas when appropriate; CSV and JSON for portable experiment outputs.

**Engineering:** pytest; Ruff or equivalent linting/formatting; Python type annotations; pyproject.toml; Git and GitHub; GitHub Actions for lightweight automated tests.

These are preferred choices, not unquestionable requirements. Before implementing, verify current library compatibility, Python version support, maintenance status, and any known Windows-specific issues. Recommend reasonable alternatives only when there is a concrete technical benefit. Do not add unnecessary frameworks or infrastructure.

Important architectural decision: We should initially prioritize Python and a lightweight simulation framework. Godot or Unity may be considered later if they would materially improve the interactive experience, but should not complicate the initial ML implementation.

## 6. The Driving Simulation

### 6.1 World Design

Create a 2D top-down driving environment. The first environment should be simple but scientifically useful. It needs: a racing circuit; a clearly defined drivable surface; track boundaries; a start and finish line; ordered checkpoints; a vehicle with coherent driving physics; collision detection; track progression tracking; lap counting; a simulation clock; episode termination conditions.

The simulation should operate independently of the renderer. The physics engine should advance using a fixed simulation timestep, initially targeting approximately 30 simulation steps per second. The rendering frame rate must not determine the physics behavior. We should be able to execute thousands of simulation steps without opening a graphical window.

### 6.2 Vehicle Physics

Implement a reasonably believable but computationally efficient 2D car model. A kinematic bicycle model is a suitable starting point.

Vehicle state should include: position X; position Y; heading angle; forward speed; steering command; acceleration/braking command; collision state; track progress; current checkpoint; lap count.

Vehicle behavior should include: steering; acceleration; braking; maximum speed; acceleration limits; steering limits; basic friction/drag; turning behavior that depends on movement speed; collision response or episode termination.

The car should not rotate unrealistically in place at zero speed. Initially, reverse driving is unnecessary. Use a simplified but coherent model that is easy to test, explain, and evolve. Avoid complex tire physics in v1.

### 6.3 Collision Detection

The vehicle must detect contact with track boundaries. For the initial version, a simple collision representation is acceptable if applied consistently. Potential representations include: circle approximation; oriented rectangle; simplified car hitbox. Choose a sensible approach and document the tradeoffs. An agent should not be able to complete laps by driving through walls or skipping required sections of the circuit.

### 6.4 Track System

Do not hardcode the entire simulation around a single track. Design a reusable track representation. Support: track identifiers; track boundaries; centerline or equivalent progress representation; checkpoints; starting position and orientation; track width; lap length; track validation; loading multiple tracks.

Start with a few manually designed tracks. Proposed examples:

1. Beginner Oval — Easy
2. Curved Circuit — Medium
3. Technical Circuit — Hard
4. Unseen Circuit A — Held-out evaluation
5. Unseen Circuit B — Held-out evaluation

An additional validation track may be needed for tuning without using held-out test tracks. Prefer a clear train/validation/test separation. Training tracks should not include every evaluation geometry.

Procedural track generation may be added later. When generating tracks, ensure they are physically navigable, free from invalid self-intersections, and compatible with the progression/checkpoint system.

## 7. The AI's Observation System

The AI must receive numerical observations representing its driving situation. We are not initially training a model directly from raw pixels. Use compact, meaningful observations.

### 7.1 Distance Sensors

Implement simulated raycast distance sensors. Proposed starting configuration: nine sensor rays oriented relative to the car: -90°, -60°, -40°, -20°, 0°, +20°, +40°, +60°, +90°.

Each sensor detects the distance to the nearest obstacle or track boundary, capped at a configurable maximum range. Normalize sensor distances to a consistent numerical range. Visually render these rays when observation visualization is enabled. The user should be able to see what the AI is sensing.

### 7.2 Additional Observations

Candidate inputs: normalized forward speed; heading error relative to a local track direction or upcoming checkpoint; normalized lateral offset from the intended route; angular velocity, if useful.

Define and document the exact observation vector. Keep the initial vector small and interpretable. Do not provide an agent with future information that would not reasonably be available under the defined observation model. If route-aware guidance is included, explicitly state that these are privileged simulator-derived navigation features rather than raw sensor measurements. Both PPO and NEAT must receive equivalent information during fair comparisons.

### 7.3 Observation Design Requirements

- Fixed observation dimensions for a given experiment configuration
- Consistent ordering
- Documented normalization
- No NaN or infinite values
- No hidden target leakage
- Deterministic calculations for a given simulation state
- Tests for boundary conditions
- Visible sensor-debugging mode

## 8. The AI's Action System

Define a shared action interface used by all learning methods. Initial proposal:

**Action 1: Steering.** Continuous value from -1 to +1. -1 = maximum left steering; 0 = neutral; +1 = maximum right steering.

**Action 2: Longitudinal control.** Continuous value from -1 to +1. -1 = maximum braking; 0 = neither accelerating nor braking; +1 = maximum throttle.

No reversing in the initial implementation. The vehicle physics system converts these normalized outputs into actual steering, acceleration, and braking effects. PPO and NEAT should produce actions through the same logical interface, even if their training libraries internally use different policy representations. Add action clamping and validation. Maintain identical vehicle physics between compared algorithms.

## 9. Reinforcement Learning System — PPO

### 9.1 Objective

Train an autonomous-driving agent using reinforcement learning. The agent should improve its driving behavior through interaction with the simulation. It should not receive a hardcoded driving strategy.

### 9.2 Gymnasium Environment

Implement a compliant Gymnasium environment with: observation space; action space; reset(); step(); reward calculation; episode termination; episode truncation; useful diagnostic information; deterministic seeding where possible; optional rendering.

Follow the current Gymnasium API. Use standard RL conventions and compatible data types. Test the environment with the official environment checker when available.

### 9.3 Reward Function

Reward design is one of the most important parts of the project. The objective is to encourage safe, efficient progression around a racing circuit.

Possible reward components:

- **Positive:** forward progress along the track; successfully passing checkpoints; completing a lap; maintaining useful driving behavior.
- **Negative:** hitting track boundaries; moving backward or losing legitimate track progress; remaining stationary for too long; failing to make progress; invalid or exploitative behavior.

Avoid excessive or poorly balanced reward shaping. Specific failure modes to guard against:

- The AI learns to stand still.
- The AI repeatedly drives into walls.
- The AI exploits checkpoint geometry.
- The AI gains reward by oscillating forward and backward.
- The AI cuts across invalid parts of the circuit.
- The AI maximizes progress rewards without completing laps.
- The AI learns to drive indefinitely without finishing.
- The AI learns an unintended exploit of the reward function.

Design progress calculation carefully. Prefer progress derived from ordered track geometry and validated checkpoint transitions. Define how signed progress behaves across the start/finish line. Do not repeatedly reward the same checkpoint. Document reward components and coefficients. Make reward parameters configurable for experiments.

### 9.4 PPO Training Features

Implement: training configuration files; random seed control; total timestep budget; model checkpoints; best-model selection based on validation performance; periodic evaluation; training logs; TensorBoard metrics; resume capability when supported; device selection; clear console output; robust error reporting.

Track at least: episode reward; episode duration; track progress; lap completion rate; collision frequency; mean speed; training timesteps; training wall-clock time; evaluation success rate.

Where useful, expose PPO-specific diagnostics such as entropy, policy loss, value loss, KL divergence, and explained variance. Explain what these diagnostics mean.

### 9.5 Learning Success

An initial goal is to demonstrate that PPO can outperform a random agent on a simple track. Eventually, it should learn to complete laps reliably. This is an experimental objective, not a guaranteed result. If training fails, investigate reward design, observation quality, environment bugs, policy configuration, and hyperparameters. Never manufacture successful training metrics.

## 10. Evolutionary Learning System — NEAT

### 10.1 Objective

Implement a second independent learning approach based on neuroevolution. Instead of optimizing a neural network using PPO, evolve candidate neural networks across generations. Use NEAT or another well-justified neuroevolution method. NEAT is preferred because it can evolve both connection weights and network topology.

### 10.2 Evolutionary Population

Each generation should contain multiple candidate neural networks. Every candidate controls its own simulated car during an evaluation episode. The simulator calculates fitness based on actual driving performance. Candidate behaviors should be evaluated through the same vehicle physics, observation interface, and action interface used by PPO.

### 10.3 Evolution Mechanics

Support: population initialization; fitness evaluation; selection; mutation; crossover; speciation, as provided by NEAT; elitism/champion preservation; generational progression; evolution statistics; checkpointing; saving the best evolved driver.

Do not manually script desirable driving behaviors into the agents. Let driving behavior emerge through optimization.

### 10.4 Fitness Function

Design a fitness function aligned with the driving objective. Possible components: legitimate distance progressed; checkpoints passed; lap completion; efficient driving; collision avoidance; appropriate penalties for stagnation.

The fitness scale does not need to be identical to PPO's reward scale. However, evaluation metrics must remain consistent across algorithms. Avoid allowing the fitness function to reward exploits.

### 10.5 Evolution Visualization

This should be one of the most entertaining parts of NeuroDrive Lab. Eventually, users should be able to view: current generation; population size; best fitness; average fitness; best historical fitness; best-performing driver; candidate failures; evolution progress; best-network topology; champion replay.

If feasible, provide a visual representation of an evolved neural network. Show how its topology changes over generations. Be careful not to render every training vehicle at full fidelity during headless training. A visual batch demonstration can be replayed or sampled separately.

## 11. Baseline Controllers

Before comparing sophisticated algorithms, establish simple baselines. Implement at least:

**Random Driver.** A controller that chooses random or random-but-smoothed actions. Purpose: verify simulation behavior; establish a basic performance baseline; demonstrate the effect of learning.

**Simple Heuristic Driver.** A transparent, hand-coded controller. Examples: track-direction follower; basic steering correction using available observations; simple proportional controller. Purpose: check whether tracks are navigable; provide a meaningful non-ML comparison; detect environment issues; distinguish ML improvements from basic control logic.

Do not pretend heuristic controllers are ML models. Label each controller clearly.

## 12. Fair Algorithm Comparison

A major part of this project is comparing PPO and NEAT. Avoid shallow comparisons such as simply declaring whichever achieved the fastest lap the winner. Both algorithms must be evaluated using a shared environment and common outcome metrics.

### 12.1 Comparison Requirements

For a given benchmark: same vehicle physics; same observation information; same action interface; same track geometry; same collision logic; same lap rules; same episode limits; same evaluation conditions; same evaluation metrics. Use consistent evaluation seeds or scenarios where meaningful. Keep validation and test tracks separate.

### 12.2 Training Budget

PPO and NEAT use different optimization processes. Do not compare their raw training iterations directly. Record: total environment steps consumed; total simulation episodes; training wall-clock time; CPU/GPU utilization when practical; random seeds; relevant algorithm settings.

Compare learning under both a controlled environment-interaction budget and reported computational cost. A perfectly equal comparison is not always possible; document the limitations.

### 12.3 Required Metrics

At minimum:

1. Lap completion rate
2. Average completed-lap time
3. Collision rate
4. Mean track progress
5. Episode length
6. Training environment steps
7. Training wall-clock duration
8. Generalization to held-out tracks
9. Variability across independent training seeds

When comparing lap times, distinguish successful laps from unsuccessful runs. Do not hide failures by averaging only successful outcomes without also showing completion rates. Report the number of evaluation episodes.

### 12.4 Repeated Experiments

Use repeatable seeds and configurations. For serious comparisons, aim for multiple independent training seeds, with three as a reasonable initial target and more when computationally feasible. Evaluate trained agents across repeated scenarios, including randomized start perturbations when appropriate. Report sample sizes and uncertainty. Do not make unsupported claims of statistical superiority.

### 12.5 Research Questions

Investigate:

- Which method learns basic steering more efficiently?
- Which method reaches reliable lap completion first?
- Which method requires more environment interaction?
- Which method is more stable across seeds?
- Which method handles unfamiliar tracks better?
- Does the agent overfit to one circuit?
- Does a larger sensor array improve learning?
- Does a reward-function change improve performance?
- Does either method exploit unintended simulator behavior?

Research results must come from actual experiments.

## 13. Training and Experiment Management

The project should not rely on scattered notebooks or hardcoded scripts. Build a small experiment-management layer. Support: named experiments; configurable training settings; algorithm selection; track selection; seed selection; structured output directories; model checkpoints; training metrics; evaluation summaries; experiment metadata; comparison reports.

Store the exact configuration associated with each training run. Record relevant software versions. Use structured files such as YAML, JSON, and CSV where appropriate. Keep the results inspectable without proprietary software. Avoid creating a database unless file-based experiment management proves insufficient.

### Reproducibility

Someone cloning the repository should be able to:

1. Install dependencies.
2. Run simulation tests.
3. Start the simulator.
4. Train a small PPO experiment.
5. Run an evolutionary experiment.
6. Evaluate saved models.
7. Generate graphs.
8. Reproduce documented experiment configurations.

Exact numerical reproducibility across different hardware is not guaranteed, but control reasonable sources of randomness.

## 14. Interactive Application and UI

After the simulation and training systems work, create a polished interactive application. Do not prioritize interface complexity before the ML foundation is validated. The application should ideally contain several logical modes.

**Mode A — Playground.** The user can: select a track; drive manually; choose an AI controller; reset the simulation; view speed and progress; enable sensor visualization; change simulation speed; pause and resume.

**Mode B — Training Viewer.** Displays useful information about the training process. Potential features: current algorithm; training progress; episode or generation number; best score; recent average score; success rate; current training configuration; learning curves; agent behavior visualization. The training viewer should not block or significantly slow headless training. A viable approach is to inspect checkpoints or replay sampled episodes separately from the actual trainer.

**Mode C — AI Comparison.** Select trained models and evaluate them using identical conditions. Possible pairings: PPO vs. NEAT; trained vs. random; trained vs. heuristic; PPO checkpoint A vs. PPO checkpoint B; NEAT generation A vs. generation B. Show performance metrics and visually compare their driving.

**Mode D — Replay Viewer.** Load and replay saved demonstrations. Features: play/pause; restart; adjustable playback speed; track information; model information; speed indicator; sensor overlays; steering/throttle indicators; progress display. Playback must represent actual recorded simulation behavior or a reproducible model rollout, not fabricated animation.

**UI Design.** Aim for a clean technical aesthetic: part racing simulator, part ML research laboratory. Prioritize: readability; smooth interaction; clear hierarchy; useful visualizations; accessible colors; consistent controls; informative errors; professional presentation. Do not build a huge frontend framework unless there is a compelling reason.

## 15. Visualizations and Educational Features

The application should make ML behavior understandable. Useful visualizations include:

**Real-Time Driving:** vehicle position; sensor rays; sensor distances; current steering; current throttle/brake; track progress; heading; speed; collision events.

**Neural Network:** for selected trained agents, visualize where practical: inputs; hidden layers; outputs; neural-network architecture; connections; active outputs; NEAT topology. A simplified visualization is acceptable if technically accurate. Do not imply that network weights or internal activations have semantic meanings they do not actually possess.

**Training Progress:** graph reward over time; fitness over generations; lap completion over training; collision frequency; progress distributions; evaluation performance.

**Learning Timeline:** a particularly valuable portfolio feature. Allow users to compare saved checkpoints from different stages of learning, for example: initial agent; early training; middle training; later training; best validated model. This should make the learning process visually obvious. Use actual saved models rather than scripted demonstrations.

## 16. Software Architecture

Design the codebase for separation of concerns. The key boundaries should be:

1. Simulation physics
2. Track geometry
3. Observation generation
4. Action interpretation
5. Environment and reward logic
6. Controller interface
7. PPO training
8. NEAT training
9. Evaluation and benchmarking
10. Rendering/UI
11. Experiment configuration
12. Persistence and replay
13. Visualization and reporting

The simulation must not depend on the graphical interface. The training algorithms must not duplicate physics or track logic. All controller types should interact with the same environment contract. A saved model should be loadable independently of its training script.

### Proposed Repository Structure

Adapt this structure if there is an existing codebase or a demonstrably better architecture.

```
neurodrive-lab/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── configs/
│   ├── simulation.yaml
│   ├── ppo.yaml
│   ├── neat.yaml
│   └── experiments/
│
├── src/
│   └── neurodrive/
│       ├── sim/
│       │   ├── vehicle.py
│       │   ├── physics.py
│       │   ├── tracks.py
│       │   ├── collisions.py
│       │   └── sensors.py
│       │
│       ├── env/
│       │   ├── driving_env.py
│       │   ├── observations.py
│       │   └── rewards.py
│       │
│       ├── agents/
│       │   ├── base.py
│       │   ├── random_agent.py
│       │   ├── heuristic_agent.py
│       │   ├── ppo_agent.py
│       │   └── neat_agent.py
│       │
│       ├── training/
│       │   ├── train_ppo.py
│       │   ├── train_neat.py
│       │   └── checkpoints.py
│       │
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── metrics.py
│       │   └── comparison.py
│       │
│       ├── ui/
│       │   ├── application.py
│       │   ├── renderer.py
│       │   └── overlays.py
│       │
│       ├── experiments/
│       │   ├── config.py
│       │   └── tracking.py
│       │
│       ├── replay/
│       │   └── recorder.py
│       │
│       └── cli.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
│
├── docs/
│   ├── architecture.md
│   ├── ml-guide.md
│   ├── training-guide.md
│   ├── experiments.md
│   └── results.md
│
├── assets/
├── examples/
└── results/
```

This is an illustrative layout, not a requirement to create empty files. Avoid unnecessary abstractions and overengineering.

## 17. Command-Line Interface

Provide a consistent and documented CLI. The final interface should support operations conceptually equivalent to:

```
# Open interactive simulator
python -m neurodrive play

# Run a PPO experiment
python -m neurodrive train ppo --config configs/ppo.yaml

# Run a NEAT experiment
python -m neurodrive train neat --config configs/neat.yaml

# Evaluate a trained model
python -m neurodrive evaluate --model PATH_TO_MODEL

# Compare saved experimental runs
python -m neurodrive compare --runs PATH_TO_RUNS

# Replay a trained agent
python -m neurodrive replay --model PATH_TO_MODEL
```

The exact syntax can be refined during architecture design. Provide Windows PowerShell examples in the README. All CLI commands should validate inputs and provide actionable error messages.

## 18. Development Phases

Implement the project incrementally. Each phase must leave the project in a working, testable state.

### PHASE 0 — Discovery and Architecture

Objective: Establish a sound technical foundation before implementation.

Tasks:

1. Inspect the current repository, if one exists.
2. Read CLAUDE.md, README, and existing project conventions.
3. Verify proposed package compatibility.
4. Select the supported Python version.
5. Define the system architecture.
6. Document core interfaces.
7. Specify observations and actions.
8. Specify vehicle physics.
9. Define track representation.
10. Design evaluation metrics.
11. Identify major technical risks.
12. Finalize an incremental implementation plan.

Deliverables: architecture proposal; technology decisions; phased roadmap; initial data/interface definitions; risk assessment; concrete Phase 1 implementation plan.

Gate: Review the design before moving into substantial implementation.

### PHASE 1 — Working Driving Simulator

Objective: Build a playable, deterministic simulation without ML.

Implement: vehicle model; fixed-step physics; one beginner track; collision detection; track progression; checkpoints; lap timing; keyboard driving; basic renderer; headless simulation mode; reset and restart; unit tests.

Success criteria:

- A human can drive around the track.
- The car behaves consistently.
- Collisions work.
- Lap completion is validated.
- The simulator runs headlessly.
- Physics tests pass.

Do not begin ML training before these basics work.

### PHASE 2 — ML Environment and Baselines

Objective: Make the simulation usable for learning agents.

Implement: sensor raycasts; observation vector; shared action interface; Gymnasium integration; reward function; episode termination/truncation; random controller; simple heuristic controller; debug visualization; baseline evaluation; environment validation tests.

Success criteria:

- Standard environment checks pass.
- Observations are stable and valid.
- Random actions can be executed headlessly.
- The agent can reset repeatedly.
- Metrics and episode outcomes are recorded.
- There is no obvious reward exploit.
- Baseline controllers can be evaluated.

### PHASE 3 — PPO Training

Objective: Train the first real AI driver.

Implement: PPO training pipeline; training configurations; checkpoints; TensorBoard; periodic evaluation; model loading; trained-agent inference; training curves; saved demonstrations; basic hyperparameter experiments.

First train on the beginner track. Start with a short smoke test, then perform a meaningful training run.

Success criteria:

- Training executes without environment errors.
- Checkpoints can be loaded.
- Saved policies produce real actions.
- Training metrics are logged.
- Performance can be compared against random and heuristic baselines.
- Improvement, if achieved, is demonstrated with measured results.

If performance fails to improve, diagnose it before proceeding.

### PHASE 4 — NEAT Training

Objective: Train autonomous drivers through neuroevolution.

Implement: population management; fitness evaluation; evolution configuration; NEAT training loop; generation statistics; champion saving; generation checkpoints; champion playback; fitness graphs; optional network topology visualization.

Success criteria:

- Multiple generations execute.
- Fitness is computed correctly.
- Candidate networks control actual vehicles.
- Evolved networks can be saved and loaded.
- Champion behavior can be replayed.
- Training metrics are reproducible within documented limits.

Do not artificially guarantee that evolution improves every generation.

### PHASE 5 — Robust Multi-Track Learning

Objective: Investigate learning beyond a single beginner circuit.

Implement: additional tracks; track loading and validation; multiple training scenarios; validation tracks; held-out evaluation tracks; optional randomized starting conditions; more complex turns; improved evaluation workflows.

Explore whether agents can learn useful driving behavior across multiple track configurations.

Success criteria:

- Track layouts can change without model-specific simulation rewrites.
- Evaluation tracks are excluded from training.
- Generalization results can be measured.
- Overfitting can be identified rather than hidden.

### PHASE 6 — Scientific Benchmarking

Objective: Compare PPO and NEAT with defensible evidence.

Implement: shared evaluation pipeline; matched evaluation conditions; training budget tracking; multiple-seed experiments; aggregate metrics; performance charts; CSV/JSON exports; failure analysis; reproducible comparison reports.

Compare PPO, NEAT, random, and heuristic controllers. Discuss: sample sizes; computational costs; stability; training efficiency; generalization; limitations of the comparison.

Success criteria:

- Results come from actual model evaluations.
- A user can reproduce the experiment configuration.
- Missing experiments are identified.
- Failed runs are not hidden.
- Claims are appropriately supported by data.

### PHASE 7 — Polished Interactive Laboratory

Objective: Turn the engineering project into an impressive user experience.

Implement or improve: main menu; track selector; algorithm selector; interactive driving mode; sensor visualization; model selection; AI replay viewer; learning curves; comparison viewer; NEAT generation viewer; playback controls; training status display; user-friendly error handling.

Success criteria:

- A new user can launch and explore the project.
- The distinction between trained and scripted controllers is clear.
- Training results are understandable.
- Demonstrations run smoothly.
- The UI does not compromise headless training.

### PHASE 8 — GitHub and Portfolio Release

Objective: Prepare a professional public repository.

Complete: high-quality README; installation guide; usage instructions; architecture overview; ML explanation; training guide; reproducibility instructions; experiment results; training graphs; screenshots; demo video or GIF; automated tests; CI configuration; dependency cleanup; license review; example configurations; resume/checkpoint documentation; troubleshooting guide.

Success criteria: A reviewer can understand the project, install it, run a basic simulation, inspect the source code, and reproduce at least a small experiment.

Do not publish fabricated benchmarks or claim more functionality than exists.

## 19. Testing Requirements

Automated testing is mandatory. Test the simulation as ordinary software, not only by manually watching it.

**Unit Tests.** Cover: vehicle movement; steering direction; speed limits; acceleration/braking; fixed-step physics; collision calculations; raycast distances; observation normalization; action validation; track progression; checkpoint crossing; lap counting; reset behavior; reward calculation; episode termination.

**Integration Tests.** Cover: simulation to Gymnasium interface; agent to environment interface; PPO model save/load; NEAT champion save/load; evaluation pipeline; experiment configuration loading; replay correctness where feasible.

**Smoke Tests.** Provide small, fast tests that: run several simulated episodes; perform a very short PPO training run; execute a small NEAT population/generation test; evaluate a saved model; verify output files.

Full training runs should not be part of ordinary continuous integration.

**Reliability Requirements:**

- No hidden global simulation state
- No accidental data leakage between episodes
- No silent exception swallowing
- No fabricated experimental output
- No committing giant checkpoints or generated datasets by default
- No external API keys
- No destructive operations without approval
- No unrelated repository changes

## 20. Performance Requirements

The simulator should support training significantly faster than real-time rendering when feasible. Priorities:

1. Fast headless environment stepping
2. Efficient collision checks
3. Efficient raycasting
4. Avoid excessive allocations
5. Separate rendering from simulation
6. Avoid unnecessary GPU transfers
7. Support multiple environments if benchmarking shows a benefit
8. Profile before optimizing

Record performance such as: environment steps per second; training throughput; CPU usage where practical; GPU usage where useful; memory consumption; rendering frame rate.

Do not invent performance targets or claim speed improvements without measurement. Only add parallelization or advanced optimization when profiling justifies it.

## 21. Learning and Mentoring Requirements

I am building this project partly to learn machine learning. Explain major design and implementation decisions as we work. After every major phase, provide:

- **What We Built:** describe the actual implemented functionality.
- **How It Works:** explain the important technical mechanisms.
- **What ML Concepts I Learned:** connect code to machine-learning theory.
- **What We Tested:** show the actual tests and verification results.
- **Problems and Limitations:** explain failures, uncertainties, and tradeoffs.
- **What Comes Next:** describe the next milestone.

When introducing an ML concept, explain it in practical language first, then provide more technical detail as appropriate. Examples:

- Explain why observations must be normalized.
- Explain what PPO is optimizing.
- Explain what reward hacking looks like.
- Explain why an agent might overfit to one circuit.
- Explain why an evolving population is different from gradient-based learning.
- Explain why repeated seeds are necessary.
- Explain what makes an experiment scientifically meaningful.

Avoid overwhelming me with mathematical derivations unless they are useful, but do not hide important details.

## 22. GitHub Documentation Requirements

The README should eventually include:

1. Project name and concise introduction
2. Demo GIF or video
3. Screenshots
4. Main features
5. ML algorithms
6. Architecture overview
7. Installation instructions
8. Quick-start commands
9. Training instructions
10. Evaluation commands
11. Benchmark methodology
12. Actual experiment results
13. Training graphs
14. Explanation of PPO versus NEAT
15. Technical challenges and lessons learned
16. Project limitations
17. Future roadmap
18. License

Make the repository approachable to both developers and ML learners. Include diagrams where they improve understanding. Avoid generic marketing language. Prioritize evidence and technical clarity.

## 23. Definition of Done

The core project should be considered complete only when:

- The driving simulator works interactively and headlessly.
- Vehicle physics is coherent and tested.
- Multiple usable tracks exist.
- Sensor observations work.
- The Gymnasium environment passes validation.
- A random driving baseline exists.
- A heuristic driving baseline exists.
- PPO can be trained and its checkpoint loaded.
- NEAT can evolve agents and save its champion.
- Both learning approaches use equivalent environment interfaces.
- Learning outcomes are recorded honestly.
- Model behavior can be replayed.
- Held-out track evaluation exists.
- Algorithm comparisons are generated from actual experiments.
- Metrics and experiments are reproducible within documented constraints.
- The project has a usable interactive UI.
- Automated tests pass.
- Documentation explains the system and algorithms.
- The repository is suitable for public GitHub presentation.

If an experiment does not produce successful learning, report that outcome and investigate it. Do not label a model as successfully trained merely because a training command completed.

## 24. Explicit Scope Restrictions

Do not introduce these in the initial implementation:

- Full 3D simulation
- Real-world autonomous vehicle control
- Real camera-based driving
- Realistic traffic simulation
- Multiplayer networking
- Backend accounts and authentication
- Paid AI APIs
- LLM-powered driving decisions
- Cloud infrastructure
- Complex databases
- Unnecessary microservices
- Dozens of training algorithms
- Highly realistic tire physics
- Production robotics integrations

The goal is a focused, polished, educational ML project, not a commercial autonomous-vehicle platform. Future enhancements should be proposed after the foundations work.

## 25. Optional Future Extensions

After the core project is working, we may investigate:

- **Advanced Reinforcement Learning:** SAC; TD3; curriculum learning; different reward functions; domain randomization.
- **Evolutionary Experiments:** alternative fitness functions; population-size experiments; mutation-rate experiments; network complexity analysis; behavioral diversity.
- **Driving Challenges:** procedurally generated tracks; variable road grip; dynamic obstacles; multiple simultaneous AI cars; overtaking; competitive racing; AI versus human; multi-agent learning.
- **Advanced ML:** vision-based observations; convolutional neural networks; transfer learning; imitation learning from human demonstrations; hybrid evolutionary and reinforcement-learning approaches.
- **Advanced Visualization:** neural activation visualization; training timeline comparisons; live experiment dashboards; side-by-side synchronized replays; track heatmaps; failure-case analytics.

Do not implement these until the core release is stable.

## 26. How You Should Work as Claude Code

Follow these rules throughout development:

1. Inspect the repository before changing files.
2. Read existing project instructions.
3. Never invent implementation status.
4. Do not assume something works without testing.
5. Make small, coherent, reviewable changes.
6. Maintain clear separation of concerns.
7. Explain important architectural decisions.
8. Prefer simple solutions over unnecessary abstraction.
9. Use established libraries appropriately.
10. Do not rewrite unrelated code.
11. Run relevant tests after changes.
12. Report the exact commands executed and their results.
13. Clearly distinguish implemented, tested, untested, and planned functionality.
14. Avoid making unsupported performance or ML claims.
15. Preserve user data and existing repository contents.
16. Ask for approval before destructive actions or major architectural deviations.
17. Do not make unrelated improvements without discussing them.
18. Treat actual training results as evidence, not assumptions.
19. Maintain a clear phase-completion checklist.
20. Keep the project usable at the end of each completed phase.

When a phase is finished, provide: files created or changed; features implemented; important design decisions; test results; known limitations; how I can run or inspect it; recommended next phase.

Do not skip verification just to progress faster.

## 27. Your First Task

Do not immediately start implementing all eight phases. First, review this entire specification critically. I want you to act as a technical architect and ML engineer before acting as a coding agent.

Deliver:

1. **Project understanding:** Summarize what we are actually trying to build.
2. **Feasibility analysis:** Evaluate feasibility on my hardware, including training, graphics, and expected computational bottlenecks.
3. **Technology selection:** Confirm or challenge the proposed stack using currently compatible libraries.
4. **Architecture proposal:** Show how physics, Gymnasium, PPO, NEAT, evaluation, and UI connect.
5. **ML design:** Recommend the initial observation vector, action space, reward design, episode termination rules, and baseline controllers.
6. **Evaluation methodology:** Explain how to avoid misleading PPO-versus-NEAT comparisons.
7. **Implementation roadmap:** Refine the phases into actionable milestones and explicit acceptance criteria.
8. **Risks:** Identify likely technical, training, performance, reproducibility, and scope problems.
9. **Repository structure:** Recommend a maintainable project layout.
10. **First implementation milestone:** Define the concrete tasks needed to finish Phase 1.

Challenge weak or impractical parts of this specification. Recommend simplifications where doing so improves the project. Explain tradeoffs rather than blindly agreeing with every proposed feature. Do not invent training results, performance estimates, or completed implementations.

Stop after presenting the reviewed architecture and initial implementation plan. Wait for my approval before starting Phase 1.

Our goal is to produce something genuinely impressive, educational, reproducible, and enjoyable — not just another generic AI-generated demo.
