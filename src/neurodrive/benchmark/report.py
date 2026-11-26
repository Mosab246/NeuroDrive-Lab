"""Turn a finished benchmark folder into tables, charts and a Markdown report.

Reads only the files the runner wrote (``manifest.json``, each run's ``eval_history.csv``,
``generations.csv``, ``train_summary.json``, and each evaluation's ``episodes.csv``), so the
whole report can be regenerated at any time with ``neurodrive benchmark report``.

Statistics are deliberately simple and honest for a handful of seeds: every seed is shown,
plus mean, standard deviation and range across seeds. Pooled completion rates get a
Wilson 95% interval over episodes. No significance tests: with 3 seeds they would suggest
more certainty than the data has.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neurodrive.evaluation.metrics import wilson_interval
from neurodrive.sim.track import load_any_track

COLORS = {"ppo": "#2b7bba", "sac": "#3a9a5b", "neat": "#e07b39"}
#: Report order. PPO and SAC (Stable-Baselines3) count timesteps; NEAT counts env steps.
ALGORITHMS = ("ppo", "sac", "neat")
SB3 = ("ppo", "sac")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def learning_curve(algo: str, run_dir: Path) -> list[dict[str, float]]:
    """Validation results against env steps and wall-clock seconds, for one run."""
    rows = _read_csv(run_dir / "eval_history.csv")
    out = []
    if algo in SB3:
        for r in rows:
            out.append(
                {
                    "steps": float(r["timesteps"]),
                    "seconds": float(r["wall_seconds"]),
                    "completion": float(r["completion_rate"]),
                    "lap": float(r["lap_time_mean"]) if r["lap_time_mean"] else np.nan,
                }
            )
    else:
        gens = {int(g["generation"]): g for g in _read_csv(run_dir / "generations.csv")}
        for r in rows:
            g = gens[int(r["generation"])]
            out.append(
                {
                    "steps": float(r["env_steps_total"]),
                    "seconds": float(g["wall_seconds"]),
                    "completion": float(r["completion_rate"]),
                    "lap": float(r["lap_time_mean"]) if r["lap_time_mean"] else np.nan,
                }
            )
    return out


def episode_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = len(rows)
    done = [r for r in rows if r["completed"] == "True"]
    laps = np.array([float(r["lap_time"]) for r in done])
    lo, hi = wilson_interval(len(done), n)
    return {
        "episodes": n,
        "completed": len(done),
        "completion_rate": len(done) / n,
        "completion_ci95": [lo, hi],
        "collision_rate": sum(r["termination_reason"] == "collision" for r in rows) / n,
        "mean_lap_fraction": float(np.mean([min(float(r["lap_fraction"]), 1.0) for r in rows])),
        "mean_episode_steps": float(np.mean([int(r["steps"]) for r in rows])),
        "lap_time_mean": float(laps.mean()) if laps.size else None,
        "lap_time_n": int(laps.size),
    }


def _spread(values: list[float]) -> dict[str, float | None]:
    v = np.array([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if v.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.size),
    }


def _fmt(x: float | None, fmt: str = "{:.2f}") -> str:
    return "—" if x is None else fmt.format(x)


def build_report(bench_dir: Path, out_dir: Path) -> Path:
    bench_dir, out_dir = Path(bench_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bench_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    budget = cfg["budget_env_steps"]

    # ---------------------------------------------------------------- per run
    runs: list[dict[str, Any]] = []
    curves: dict[str, list[list[dict[str, float]]]] = {}
    per_track_rows: list[dict[str, Any]] = []
    for entry in manifest["runs"].values():
        if entry.get("status") != "evaluated":
            continue
        algo, seed, run_dir = entry["algorithm"], entry["seed"], Path(entry["run_dir"])
        summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
        curve = learning_curve(algo, run_dir)
        curves.setdefault(algo, []).append(curve)
        first_lap = next((c["steps"] for c in curve if c["completion"] > 0), None)
        all_laps = next((c["steps"] for c in curve if c["completion"] >= 1.0), None)
        episodes = _read_csv(bench_dir / "eval" / f"{algo}_seed{seed}" / "episodes.csv")
        stats = episode_stats(episodes)
        steps_used = summary["total_timesteps"] if algo in SB3 else summary["env_steps_total"]
        best_meta = json.loads((run_dir / ("best_model.meta.json")).read_text(encoding="utf-8"))
        best_at = best_meta.get("timesteps", best_meta.get("env_steps_total"))
        runs.append(
            {
                "algorithm": algo,
                "seed": seed,
                "env_steps_used": steps_used,
                "train_wall_seconds": summary["wall_seconds_this_session"],
                "steps_to_first_val_lap": first_lap,
                "steps_to_all_val_laps": all_laps,
                "best_model_at_steps": best_at,
                **{f"test_{k}": v for k, v in stats.items() if k != "completion_ci95"},
            }
        )
        by_track: dict[str, list[dict[str, str]]] = {}
        for r in episodes:
            by_track.setdefault(r["track"], []).append(r)
        for track, rows in by_track.items():
            t = episode_stats(rows)
            per_track_rows.append(
                {
                    "algorithm": algo,
                    "seed": seed,
                    "track": track,
                    "split": load_any_track(track).spec.split,
                    "completion_rate": t["completion_rate"],
                    "lap_time_mean": t["lap_time_mean"],
                }
            )
    runs.sort(key=lambda r: (r["algorithm"], r["seed"]))
    present = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in runs)]

    baselines: dict[str, dict[str, Any]] = {}
    for name in cfg["baselines"]:
        path = bench_dir / "eval" / name / "episodes.csv"
        if path.exists():
            baselines[name] = episode_stats(_read_csv(path))

    # ---------------------------------------------------------------- aggregate
    aggregate: dict[str, Any] = {}
    for algo in present:
        rs = [r for r in runs if r["algorithm"] == algo]
        aggregate[algo] = {
            "seeds": [r["seed"] for r in rs],
            "test_completion_rate": _spread([r["test_completion_rate"] for r in rs]),
            "test_lap_time_mean": _spread([r["test_lap_time_mean"] for r in rs]),
            "test_collision_rate": _spread([r["test_collision_rate"] for r in rs]),
            "train_wall_seconds": _spread([r["train_wall_seconds"] for r in rs]),
            "env_steps_used": _spread([r["env_steps_used"] for r in rs]),
            "steps_to_first_val_lap": _spread([r["steps_to_first_val_lap"] for r in rs]),
            "steps_to_all_val_laps": _spread([r["steps_to_all_val_laps"] for r in rs]),
            "pooled_completed": sum(r["test_completed"] for r in rs),
            "pooled_episodes": sum(r["test_episodes"] for r in rs),
        }
        lo, hi = wilson_interval(
            aggregate[algo]["pooled_completed"], aggregate[algo]["pooled_episodes"]
        )
        aggregate[algo]["pooled_completion_ci95"] = [lo, hi]

    # Head-to-head lap times per track, seed means.
    tracks = sorted({r["track"] for r in per_track_rows})
    track_means: dict[str, dict[str, float]] = {}
    for algo in present:
        track_means[algo] = {}
        for t in tracks:
            vals = [
                r["lap_time_mean"]
                for r in per_track_rows
                if r["algorithm"] == algo and r["track"] == t and r["lap_time_mean"] is not None
            ]
            track_means[algo][t] = float(np.mean(vals)) if vals else np.nan
    both = "ppo" in present and "neat" in present
    neat_faster = sum(
        1
        for t in tracks
        if both and track_means["neat"].get(t, np.nan) < track_means["ppo"].get(t, np.nan)
    )
    pairings = [
        (p["test_lap_time_mean"], n["test_lap_time_mean"])
        for p in runs
        if p["algorithm"] == "ppo"
        for n in runs
        if n["algorithm"] == "neat"
    ]
    neat_wins_pairs = sum(1 for p, n in pairings if p is not None and n is not None and n < p)

    # ---------------------------------------------------------------- files
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    with (out_dir / "per_track.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_track_rows[0]))
        writer.writeheader()
        writer.writerows(per_track_rows)
    (out_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "config": cfg,
                "aggregate": aggregate,
                "baselines": baselines,
                "neat_faster_on_tracks": neat_faster,
                "tracks": len(tracks),
                "neat_faster_in_seed_pairings": neat_wins_pairs,
                "seed_pairings": len(pairings),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_curves(curves, budget, out_dir)
    _plot_tracks(track_means, tracks, out_dir, present)
    _write_markdown(
        present,
        cfg,
        runs,
        aggregate,
        baselines,
        neat_faster,
        len(tracks),
        neat_wins_pairs,
        len(pairings),
        out_dir,
    )
    return out_dir


def _plot_curves(curves: dict[str, list[list[dict[str, float]]]], budget: int, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    for algo, runs in curves.items():
        for i, curve in enumerate(runs):
            label = algo.upper() if i == 0 else None
            x_steps = [c["steps"] for c in curve]
            x_time = [c["seconds"] / 60 for c in curve]
            comp = [100 * c["completion"] for c in curve]
            lap = [c["lap"] for c in curve]
            kw = {"color": COLORS[algo], "alpha": 0.8, "marker": "o", "ms": 3, "label": label}
            axes[0, 0].plot(x_steps, comp, **kw)
            axes[0, 1].plot(x_time, comp, **kw)
            axes[1, 0].plot(x_steps, lap, **kw)
            axes[1, 1].plot(x_time, lap, **kw)
    axes[0, 0].set_ylabel("validation laps completed (%)")
    axes[1, 0].set_ylabel("validation lap time (s)")
    for ax in axes[:, 0]:
        ax.set_xlabel("environment steps")
        ax.axvline(budget, color="grey", ls="--", lw=1)
    for ax in axes[:, 1]:
        ax.set_xlabel("training wall-clock (minutes)")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning curves on the validation track (one line per seed)")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)


def _plot_tracks(
    means: dict[str, dict[str, float]], tracks: list[str], out: Path, present: list[str]
) -> None:
    order = sorted(tracks, key=lambda t: (not t.startswith("proc_"), t))[::-1]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(order) + 1.5), dpi=100)
    h = 0.7 / len(present)
    for i, algo in enumerate(present):
        ax.barh(
            y + (i - (len(present) - 1) / 2) * h,
            [means[algo][t] for t in order],
            height=h * 0.95,
            color=COLORS[algo],
            label=algo.upper(),
        )
    ax.set_yticks(y, order, fontsize=8)
    ax.set_xlabel("mean lap time over seeds (s); lower is faster")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Test lap time per track")
    fig.tight_layout()
    fig.savefig(out / "test_lap_times.png")
    plt.close(fig)


def _write_markdown(
    present, cfg, runs, agg, baselines, neat_faster, n_tracks, neat_pairs, n_pairs, out
):
    configs = [cfg.get(f"{a}_config") for a in present]
    lines = [
        f"# Benchmark: {cfg['name']}",
        "",
        "Generated by `neurodrive benchmark report`; do not edit by hand.",
        "",
        f"- Budget: **{cfg['budget_env_steps']:,} environment steps** per training run",
        f"- Seeds: {', '.join(str(s) for s in cfg['seeds'])} for each algorithm",
        "- Training configs: " + ", ".join(f"`{c}`" for c in configs if c),
        f"- Test: tracks `{cfg['eval']['tracks']}`, {cfg['eval']['episodes_per_track']} "
        f"{cfg['eval']['split']} scenarios per track, random start "
        f"{cfg['eval']['random_start']}",
        "",
        "## Per run",
        "",
        "| Algorithm | Seed | Env steps used | Train time | First val. lap at | All val. laps at "
        "| Best model at | Test laps | Test lap time | Test collisions |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| {r['algorithm'].upper()} | {r['seed']} | {r['env_steps_used']:,} | "
            f"{r['train_wall_seconds'] / 60:.1f} min | "
            f"{_fmt(r['steps_to_first_val_lap'], '{:,.0f}')} | "
            f"{_fmt(r['steps_to_all_val_laps'], '{:,.0f}')} | "
            f"{_fmt(r['best_model_at_steps'], '{:,.0f}')} | "
            f"{r['test_completed']}/{r['test_episodes']} | "
            f"{_fmt(r['test_lap_time_mean'])} s | {100 * r['test_collision_rate']:.1f}% |"
        )
    lines += [
        "",
        "## Across seeds",
        "",
        "| | " + " | ".join(a.upper() for a in present) + " |",
        "|---|" + "---|" * len(present),
    ]

    def cell(algo: str, key: str, fmt: str, scale: float = 1.0) -> str:
        if algo not in agg:
            return "—"
        s = agg[algo][key]
        if s["mean"] is None:
            return f"never (n = {len(agg[algo]['seeds'])})"
        f = lambda v: fmt.format(v * scale)  # noqa: E731
        return (
            f"{f(s['mean'])} ± {f(s['std'])} (range {f(s['min'])} to {f(s['max'])}, n = {s['n']})"
        )

    rows = [
        ("Test laps completed (%)", "test_completion_rate", "{:.1f}", 100),
        ("Test lap time (s)", "test_lap_time_mean", "{:.2f}", 1),
        ("Test collisions (%)", "test_collision_rate", "{:.1f}", 100),
        ("Steps to first validation lap", "steps_to_first_val_lap", "{:,.0f}", 1),
        ("Steps to all validation laps", "steps_to_all_val_laps", "{:,.0f}", 1),
        ("Env steps used", "env_steps_used", "{:,.0f}", 1),
        ("Training wall-clock (min)", "train_wall_seconds", "{:.1f}", 1 / 60),
    ]
    for label, key, fmt, scale in rows:
        lines.append(f"| {label} | " + " | ".join(cell(a, key, fmt, scale) for a in present) + " |")
    for algo in present:
        lo, hi = agg[algo]["pooled_completion_ci95"]
        lines.append(
            f"| Pooled test laps, {algo.upper()} | {agg[algo]['pooled_completed']}/"
            f"{agg[algo]['pooled_episodes']} (95% CI {100 * lo:.1f} to {100 * hi:.1f}%)"
            + " |"
            * len(present)
        )
    lines += [
        "",
        "Values are mean ± standard deviation across seeds, with the range. With this few "
        "seeds, treat differences smaller than the ranges as unresolved.",
        "",
    ]
    if "ppo" in present and "neat" in present:
        lines += [
            f"- Mean lap time per track (averaged over seeds): NEAT faster on {neat_faster} of "
            f"{n_tracks} tracks.",
            f"- Seed pairings (each PPO seed vs each NEAT seed, overall test lap time): NEAT "
            f"faster in {neat_pairs} of {n_pairs}.",
            "",
        ]
    lines += [
        "## Baselines (same test scenarios)",
        "",
        "| Driver | Test laps | Lap time | Collisions |",
        "|---|---|---|---|",
    ]
    for name, b in baselines.items():
        lines.append(
            f"| {name} | {b['completed']}/{b['episodes']} | {_fmt(b['lap_time_mean'])} s | "
            f"{100 * b['collision_rate']:.1f}% |"
        )
    lines += [
        "",
        "## Charts",
        "",
        "![Learning curves](learning_curves.png)",
        "",
        "![Test lap time per track](test_lap_times.png)",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
