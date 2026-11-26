// Run the JavaScript port headless for the parity test (tests/smoke/test_web_port.py).
// Usage: node web/test/run.mjs request.json  ->  prints a JSON result.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { Track, Simulation, RaySensor, observe, makeBrain } from "../sim.js";

const req = JSON.parse(readFileSync(process.argv[2], "utf8"));
const { DATA } = await import(pathToFileURL(req.data).href);
const out = {};
if (req.observations) {
  const brain = makeBrain(DATA.models[req.model]);
  out.actions = req.observations.map(o => brain(o));
}
if (req.track) {
  const track = new Track(DATA.tracks.find(t => t.id === req.track));
  const cfg = { ...DATA.sim, max_time: req.max_time, laps_to_finish: 1 };
  const sim = new Simulation(track, DATA.vehicle, cfg);
  const sensor = new RaySensor(track, DATA.sensors.angles_deg, DATA.sensors.max_range);
  const brain = makeBrain(DATA.models[req.model]);
  sim.reset();
  const states = [];
  while (!sim.termination) {
    const { obs } = observe(sim, sensor);
    sim.step(brain(obs));
    const s = sim.state;
    states.push([s.x, s.y, s.heading, s.speed, s.steer]);
  }
  out.states = states;
  out.termination = sim.termination;
  out.lap_times = sim.lapTimes;
}
process.stdout.write(JSON.stringify(out));
