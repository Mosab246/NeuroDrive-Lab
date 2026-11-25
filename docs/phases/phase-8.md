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
| CI: lint, format, type check, tests on Windows + Linux, Python 3.12 + 3.13 | `.github/workflows/ci.yml` | configured; its steps pass locally on Windows with 3.12 and 3.13 (see below) |
| No large files in git: models, logs and run folders are git-ignored; largest tracked file ~0.25 MB besides the 2.5 MB GIF | `.gitignore` | checked |

## Verified locally

- Python 3.12: `ruff check`, `ruff format --check`, `mypy`, `pytest` (197 passed).
- Python 3.13 (separate environment, `uv sync --python 3.13`): `pytest`, 197 passed.
- NEAT gives identical results with 0 and 12 worker processes.
- The 1-minute experiment: two runs, identical `generations.csv` apart from timing.

## Not verified

- **CI on GitHub** has never run: the repository has no remote yet. Linux has not been
  tested at all.
- **GPU training** (CUDA PyTorch) has not been tried.

## Waiting on decisions from the owner

- **License.** None chosen; the code is not open source until one is.
- **GitHub repository.** Creating it and pushing is the owner's call; CI then runs on
  every push.
