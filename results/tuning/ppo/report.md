# PPO hyperparameter search (Phase 12)

Generated from `neurodrive tune ppo --config configs/ppo/multitrack.yaml --out runs/tune/ppo --trials 8 --seeds 0,1 --timesteps 1000000 --workers 4`.
8 settings (trial 0 = Stable-Baselines3 defaults, 7 random draws), 2 seeds each, 1M steps, scored on the 7 validation
tracks only (`validation_circuit`, `pval_0..2`, the 3 training tracks; 5 scenarios each). Test tracks were not used.

Score: laps completed on validation averaged over every evaluation in the second half of training, for the worse of
the two seeds (rewards settings that learn reliably and stay good). Ties: mean best-checkpoint lap time.

| Rank | Trial | Worst-seed late validation laps | Mean late validation laps | Best lap | learning rate | n_steps | batch | epochs | gamma | GAE lambda | clip | entropy | net | log std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tune_t07 | 90.6% | 94.7% | 22.40 s | 0.000636 | 512 | 256 | 10 | 0.98 | 0.9 | 0.2 | 0.01 | 128x128 | 0.0 |
| 2 | tune_t01 | 87.8% | 93.9% | 23.26 s | 0.000699 | 1024 | 64 | 10 | 0.995 | 0.95 | 0.2 | 0.001 | 128x128 | -0.5 |
| 3 | tune_t00 (defaults) | 84.5% | 92.2% | 23.64 s | 0.0003 | 2048 | 64 | 10 | 0.99 | 0.95 | 0.2 | 0.0 | 64x64 | 0.0 |
| 4 | tune_t06 | 84.5% | 90.8% | 30.81 s | 0.000312 | 1024 | 64 | 20 | 0.99 | 0.98 | 0.1 | 0.0 | 64x64 | 0.0 |
| 5 | tune_t05 | 81.2% | 82.7% | 29.37 s | 0.000733 | 1024 | 64 | 20 | 0.98 | 0.9 | 0.3 | 0.001 | 64x64 | -1.0 |
| 6 | tune_t03 | 81.2% | 81.8% | 29.69 s | 0.000961 | 2048 | 64 | 10 | 0.98 | 0.98 | 0.1 | 0.01 | 128x128 | -0.5 |
| 7 | tune_t04 | 74.3% | 82.0% | 24.45 s | 0.000363 | 1024 | 128 | 10 | 0.995 | 0.98 | 0.1 | 0.01 | 128x128 | -0.5 |
| 8 | tune_t02 | 64.9% | 77.3% | 22.92 s | 0.000383 | 512 | 256 | 5 | 0.99 | 0.9 | 0.1 | 0.01 | 128x128 | -1.0 |

The winner (tune_t07) is saved as `configs/ppo/tuned.yaml`. Its check on the test tracks, with fresh seeds 3-5:
[default](../../benchmark/ppo_default_s345/report.md) vs [tuned](../../benchmark/ppo_tuned_s345/report.md).
