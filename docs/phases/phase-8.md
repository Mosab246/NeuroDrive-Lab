# Phase 8 report: release preparation

Date: 2026-11-25. Scope: spec §18 Phase 8 and
[architecture-review.md §7](../architecture-review.md#7-refined-roadmap-with-acceptance-criteria).

## Done

| Item | Where | Status |
|---|---|---|
| Demo GIF of a real ghost race (rendered by the real viewer, reproducible) | `docs/images/demo-ghost-race.gif`, `scripts/make_demo_gif.py` | done |
| Install guide (PowerShell, uv), first command to check it works | README | done |
| Troubleshooting guide, from problems met during the project | `docs/troubleshooting.md` | done |
| Reproducibility guide with a 1-minute experiment, run twice to confirm identical results | `docs/reproducing.md` | done |
| README: status, real results, every screen's command, controls, doc index | README | done |
| CI: lint, format, type check, tests on Windows + Linux, Python 3.12 + 3.13 | `.github/workflows/ci.yml` | passing on GitHub: all 4 jobs green (first run 2026-11-25, ~2 to 3 min each) |
| MIT license | `LICENSE`, `pyproject.toml` | done |
| No large files in git: models, logs and run folders are git-ignored; largest tracked file ~0.25 MB besides the 2.5 MB GIF | `.gitignore` | checked |

## Verified locally

- Python 3.12: `ruff check`, `ruff format --check`, `mypy`, `pytest` (197 passed).
- Python 3.13 (separate environment, `uv sync --python 3.13`): `pytest`, 197 passed.
- NEAT gives identical results with 0 and 12 worker processes.
- The 1-minute experiment: two runs, identical `generations.csv` apart from timing.

## Not verified

- Linux is tested only by CI (headless); the viewers have not been opened on Linux.
- **GPU training** (CUDA PyTorch) has not been tried.

## Repository

Pushed to a private GitHub repository on 2026-11-25. Making it public is the owner's call.
