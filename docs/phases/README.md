# Phase reports

The project was built in phases, each ending in a working, tested state and a short report
of what was built and what was measured. Read them in order for the full story, or jump to
a result.

| Phase | Report | Main result |
|---|---|---|
| 1 | [Working driving simulator](phase-1.md) | Deterministic physics, tracks, collisions, cheat-proof lap counting |
| 2 | [ML environment and baselines](phase-2.md) | Gymnasium environment; scripted and random baseline drivers |
| 3 | [PPO, the first trained driver](phase-3.md) | PPO goes from 0 to 100 of 100 laps on the oval |
| 4 | [NEAT, the evolved driver](phase-4.md) | NEAT evolves a driver that laps the oval |
| 5 | [More tracks and generalization](phase-5.md) | Oval-only training never learns right turns; 3 tracks fix it |
| 6 | [Fair PPO vs NEAT benchmark](phase-6.md) | Equal budgets and 3 seeds: PPO faster, NEAT more reliable at 1M steps |
| 7 | [Interactive lab](phase-7.md) | Replays, ghost races, learning timelines, network views |
| 8 | [Release preparation](phase-8.md) | Demo GIF, install, troubleshooting and reproducibility guides |
| 9 | [Longer training, hard physics, real racing](phase-9.md) | 5M steps reverse the ranking; tyre-slip physics; multi-car races |
| 10 | [Better checkpoint selection, three seeds for racing](phase-10.md) | 7-track selection lifts PPO to 97%; training with traffic makes NEAT race |
| 11 | [Visuals, dashboard and browser version](phase-11.md) | Broadcast-style viewer, interactive dashboard, playable web port |
| 12 | [Self-play racing and PPO tuning](phase-12.md) | Self-play gives the most reliable racers; tuning speeds PPO up, not its reliability |
| 13 | [SAC, a third algorithm](phase-13.md) | SAC: most sample-efficient, ~10x the compute |
| 14 | [Racing with hard physics](phase-14.md) | Drivers trained alone lose a quarter to a third of their laps in traffic |
