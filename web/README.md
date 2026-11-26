# NeuroDrive Lab in the browser

Race against the trained PPO and NEAT drivers, or watch them race, with nothing to install.
The kinematic physics, distance sensors, lap counting and both kinds of neural network are
ported to JavaScript (`sim.js`) and checked step for step against the Python simulator
(`tests/smoke/test_web_port.py`: the same driver on the same track stays within 1e-6 m).

```powershell
uv run python -m http.server 8765 --directory web   # then open http://localhost:8765
```

(Browsers do not load JavaScript modules from `file://`, so open it through any static web
server, or host the folder on GitHub Pages.)

`data.js` holds the tracks and the drivers' weights. Regenerate it after training:

```powershell
uv run python -m neurodrive export-web --benchmark runs/benchmark/ppo_vs_neat_5m
```

Only the default (kinematic) physics is ported; cars are ghosts (no contact).
