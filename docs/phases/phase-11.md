# Phase 11 report: visuals, results dashboard and browser version

Date: 2026-11-26. Goal: make the project easy to see and try, without changing any
simulation or training result.

## What we built

| Area | Files | Status |
|---|---|---|
| Track art: asphalt and grass texture, kerbs on corners, chequered finish line | `ui/art.py`, `ui/renderer.py` | implemented, tested |
| Car sprites, skid marks from tyre slip, tyre smoke | `ui/art.py` | implemented, tested |
| Race presentation: start lights, follow / TV camera, mini-map, lap counter, gaps to leader, position arrows | `ui/replay_viewer.py` | implemented, tested |
| Live NEAT network: node activations drawn while the car drives (`E` toggles) | `ui/replay_viewer.py`, `viz/network.py` | implemented, tested |
| MP4 export of replays and races (`--video --camera --fps --video-speed`) via ffmpeg | `ui/video.py`, `cli.py` | implemented, tested |
| Learning time-lapse: one lap per saved checkpoint, as a video (`timeline --timelapse`) | `ui/video.py`, `replay/lab.py` | implemented, tested |
| Demo GIFs for the README (race, drift) | `scripts/make_demo_gif.py`, `docs/images/` | generated |
| Interactive results dashboard (filters, hover details, table views, light/dark) | `benchmark/dashboard.py`, `results/dashboard.html` | implemented, tested |
| Browser version: race the trained PPO and NEAT drivers, JavaScript port of the kinematic simulator | `web_export.py`, `web/` | implemented, tested |

## Notes

- **The browser port is checked against Python.** `tests/smoke/test_web_port.py` runs the
  JavaScript simulator under node and compares positions step by step with the Python
  simulator for the autopilot, a PPO policy and a hand-wired NEAT network. Only the
  default kinematic physics is ported; browser races are ghost races (no car contact).
- **Video export is optional.** It needs ffmpeg on `PATH` or the `video` extra
  (`uv sync --extra video`, installs `imageio-ffmpeg`).
- **Visuals do not touch the simulation.** Skid marks and smoke are drawn from the
  recorded slip, never fed back. All earlier results are unchanged; the full test suite
  (246 tests) passes.

## Commands

| What | Command |
|---|---|
| Race with the TV camera, save an MP4 | `neurodrive race --real ... --camera tv --video race.mp4` |
| Learning time-lapse of a run | `neurodrive timeline RUN --timelapse timelapse.mp4` |
| Regenerate the dashboard | `neurodrive dashboard` |
| Rebuild the browser data | `neurodrive export-web` |
| Play in the browser | `uv run python -m http.server 8765 --directory web`, then open `http://localhost:8765` |
