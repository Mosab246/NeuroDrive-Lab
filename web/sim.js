// NeuroDrive Lab simulator, ported from the Python package (kinematic physics only).
// Every function mirrors its Python counterpart line by line so the two agree step for step;
// tests/smoke/test_web_port.py checks that by running both.

const TAU = 2 * Math.PI;
// Python's % for floats: the result has the sign of the divisor.
const pymod = (a, p) => ((a % p) + p) % p;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function wrapDelta(delta, period) {
  return -pymod(-delta + period / 2, period) + period / 2;
}

// ------------------------------------------------------------------ track
export class Track {
  constructor(d) {
    Object.assign(this, d);
    this.n = d.centerline.length;
    this.halfWidthMin = Math.min(...d.half_width);
  }

  // Nearest point on the centerline, searching segments hint-window..hint+window
  // (all segments when hint is null). Returns {segment, s, lateral}.
  project(px, py, hint) {
    const n = this.n, w = this.window;
    let first, count;
    if (hint === null || hint === undefined || 2 * w + 1 >= n) { first = 0; count = n; }
    else { first = pymod(hint, n) - w; count = 2 * w + 1; }
    let best = -1, bestD2 = Infinity, bestT = 0, bestApx = 0, bestApy = 0;
    for (let j = 0; j < count; j++) {
      const i = pymod(first + j, n), k = (i + 1) % n;
      const ax = this.centerline[i][0], ay = this.centerline[i][1];
      const dx = this.centerline[k][0] - ax, dy = this.centerline[k][1] - ay;
      const apx = px - ax, apy = py - ay;
      let t = (apx * dx + apy * dy) / (dx * dx + dy * dy);
      t = clamp(t, 0, 1);
      const ex = apx - t * dx, ey = apy - t * dy, d2 = ex * ex + ey * ey;
      if (d2 < bestD2) { bestD2 = d2; best = i; bestT = t; bestApx = apx; bestApy = apy; }
    }
    const k = (best + 1) % n;
    const dx = this.centerline[k][0] - this.centerline[best][0];
    const dy = this.centerline[k][1] - this.centerline[best][1];
    const len = Math.sqrt(dx * dx + dy * dy);
    let s = this.cum_s[best] + bestT * len;
    if (s >= this.length) s -= this.length;
    const cross = dx * bestApy - dy * bestApx;
    return { segment: best, s, lateral: Math.sign(cross) * Math.sqrt(bestD2) };
  }

  pointAt(s) {
    s = pymod(s, this.length);
    let lo = 0, hi = this.cum_s.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (this.cum_s[mid] <= s) lo = mid; else hi = mid - 1; }
    const i = Math.min(lo, this.n - 1), j = (i + 1) % this.n;
    const t = (s - this.cum_s[i]) / (this.cum_s[i + 1] - this.cum_s[i]);
    const a = this.centerline[i], b = this.centerline[j];
    const dx = b[0] - a[0], dy = b[1] - a[1], norm = Math.hypot(dx, dy);
    return { x: a[0] + t * dx, y: a[1] + t * dy, tx: dx / norm, ty: dy / norm };
  }
}

// ------------------------------------------------------------------ vehicle
export function stepVehicle(st, steering, longitudinal, p, dt) {
  const target = steering * p.max_steer, maxChange = p.steer_rate * dt;
  st.steer += Math.min(maxChange, Math.max(-maxChange, target - st.steer));
  let accel = longitudinal >= 0 ? longitudinal * p.max_accel : longitudinal * p.max_brake;
  let v = st.speed;
  if (v > 0) accel -= p.drag * v * v + p.rolling_resistance;
  v = Math.min(p.max_speed, Math.max(0, v + accel * dt));
  st.speed = v;
  let limit = p.max_steer;
  if (!(p.max_lateral_accel <= 0 || v < 1e-6))
    limit = Math.min(p.max_steer, Math.atan(p.wheelbase * p.max_lateral_accel / (v * v)));
  const effective = Math.min(limit, Math.max(-limit, st.steer));
  st.heading -= v / p.wheelbase * Math.tan(effective) * dt;
  st.heading = pymod(st.heading + Math.PI, TAU) - Math.PI;
  st.x += v * Math.cos(st.heading) * dt;
  st.y += v * Math.sin(st.heading) * dt;
}

export function bodyCenter(st, p) {
  const half = 0.5 * p.wheelbase;
  return [st.x + half * Math.cos(st.heading), st.y + half * Math.sin(st.heading)];
}

// front-left, front-right, rear-right, rear-left (same order as Python)
export function bodyCorners(st, p) {
  const [cx, cy] = bodyCenter(st, p);
  const c = Math.cos(st.heading), s = Math.sin(st.heading);
  const hl = 0.5 * p.length, hw = 0.5 * p.width;
  const fx = c, fy = s, lx = -s, ly = c;
  return [
    [cx + hl * fx + hw * lx, cy + hl * fy + hw * ly],
    [cx + hl * fx - hw * lx, cy + hl * fy - hw * ly],
    [cx - hl * fx - hw * lx, cy - hl * fy - hw * ly],
    [cx - hl * fx + hw * lx, cy - hl * fy + hw * ly],
  ];
}

// ------------------------------------------------------------------ progress
class Progress {
  constructor(track) { this.track = track; }
  reset(px, py, originS = 0) {
    const pr = this.track.project(px, py, null);
    this.segment = pr.segment; this.s = pr.s; this.lateral = pr.lateral;
    this.originS = pymod(originS, this.track.length);
    this.distance = wrapDelta(this.s - this.originS, this.track.length);
    this.lapBase = 0; this.laps = 0; this.nextCheckpoint = 1;
  }
  get lapFraction() { return (this.distance - this.lapBase) / this.track.length; }
  apply(s, segment, lateral) {
    const delta = wrapDelta(s - this.s, this.track.length);
    this.segment = segment; this.s = s; this.lateral = lateral;
    this.distance += delta;
    let passed = 0, lapCompleted = false;
    const cps = this.track.checkpoint_s, K = cps.length;
    while (this.distance - this.lapBase >= cps[this.nextCheckpoint - 1]) {
      passed += 1;
      if (this.nextCheckpoint === K) {
        this.laps += 1; this.lapBase += this.track.length; this.nextCheckpoint = 1;
        lapCompleted = true; break;
      }
      this.nextCheckpoint += 1;
    }
    return { delta, passed, lapCompleted };
  }
}

// ------------------------------------------------------------------ simulation
export class Simulation {
  constructor(track, params, cfg) {
    this.track = track; this.p = params; this.cfg = cfg;
    this.progress = new Progress(track);
  }

  // The car at arc length s (negative = behind the line), lateral meters left of center.
  place(s, lateral = 0) {
    const pt = this.track.pointAt(s);
    const heading = Math.atan2(pt.ty, pt.tx);
    const nx = -Math.sin(heading), ny = Math.cos(heading);
    const cx = pt.x + lateral * nx, cy = pt.y + lateral * ny;
    const half = 0.5 * this.p.wheelbase;
    this.state = { x: cx - half * Math.cos(heading), y: cy - half * Math.sin(heading), heading, speed: 0, steer: 0 };
    this.progress.reset(cx, cy, 0);
    this.substeps = 0; this.lapStart = 0; this.lapTimes = []; this.termination = null;
  }

  // Start line, no perturbation (Python: Simulation.reset(perturb=False)).
  reset() {
    const c = this.track.centerline[0], t = this.track.tangents[0];
    const heading = Math.atan2(t[1], t[0]);
    const half = 0.5 * this.p.wheelbase;
    this.state = { x: c[0] - half * Math.cos(heading), y: c[1] - half * Math.sin(heading), heading, speed: 0, steer: 0 };
    const [bx, by] = bodyCenter(this.state, this.p);
    this.progress.reset(bx, by, 0);
    this.substeps = 0; this.lapStart = 0; this.lapTimes = []; this.termination = null;
  }

  get subDt() { return this.cfg.dt / this.cfg.substeps; }
  get time() { return this.substeps * this.subDt; }
  get currentLapTime() { return (this.substeps - this.lapStart) * this.subDt; }

  step(action) {
    if (this.termination) return;
    const steering = clamp(action[0], -1, 1), longitudinal = clamp(action[1], -1, 1);
    for (let k = 0; k < this.cfg.substeps; k++) {
      stepVehicle(this.state, steering, longitudinal, this.p, this.subDt);
      this.substeps += 1;
      const corners = bodyCorners(this.state, this.p);
      const cx = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
      const cy = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;
      const hint = this.progress.segment;
      const c = this.track.project(cx, cy, hint);
      const up = this.progress.apply(c.s, c.segment, c.lateral);
      if (up.lapCompleted) {
        this.lapTimes.push((this.substeps - this.lapStart) * this.subDt);
        this.lapStart = this.substeps;
        if (this.cfg.laps_to_finish && this.progress.laps >= this.cfg.laps_to_finish) {
          this.termination = "laps_complete"; break;
        }
      }
      for (const [qx, qy] of corners) {
        const q = this.track.project(qx, qy, hint);
        if (Math.abs(q.lateral) > this.track.half_width[q.segment]) { this.termination = "collision"; break; }
      }
      if (this.termination) break;
    }
    if (!this.termination && this.cfg.max_time && this.time >= this.cfg.max_time - 1e-9)
      this.termination = "time_limit";
  }
}

// ------------------------------------------------------------------ sensors and observation
export class RaySensor {
  constructor(track, anglesDeg, maxRange) {
    this.track = track; this.maxRange = maxRange;
    this.offsets = anglesDeg.map(a => -a * Math.PI / 180);
    const n = track.n, spacing = track.length / n;
    this.window = Math.min(Math.floor(n / 2), Math.ceil(1.5 * maxRange / spacing) + 10);
    this.full = 2 * this.window + 1 >= n;
  }
  cast(ox, oy, heading, segment) {
    const n = this.track.n, w = this.window;
    const first = this.full ? 0 : pymod(segment, n) - w, count = this.full ? n : 2 * w + 1;
    const out = [];
    for (const off of this.offsets) {
      const ang = heading + off, dx = Math.cos(ang), dy = Math.sin(ang);
      let best = Infinity;
      for (const edge of [this.track.left_edge, this.track.right_edge]) {
        for (let j = 0; j < count; j++) {
          const i = pymod(first + j, n), k = (i + 1) % n;
          const ax = edge[i][0], ay = edge[i][1], ex = edge[k][0] - ax, ey = edge[k][1] - ay;
          const aox = ax - ox, aoy = ay - oy;
          const denom = dx * ey - dy * ex;
          if (Math.abs(denom) <= 1e-12) continue;
          const t = (aox * ey - aoy * ex) / denom, u = (aox * dy - aoy * dx) / denom;
          if (t >= 0 && u >= 0 && u <= 1 && t < best) best = t;
        }
      }
      out.push(Math.min(best, this.maxRange));
    }
    return out;
  }
}

// The 11-number observation, rounded to float32 like the Python environment's output.
export function observe(sim, sensor) {
  const p = sim.p, st = sim.state;
  const [ox, oy] = bodyCenter(st, p);
  const dist = sensor.cast(ox, oy, st.heading, sim.progress.segment);
  const obs = dist.map(d => clamp(d / sensor.maxRange, 0, 1));
  obs.push(clamp(st.speed / p.max_speed, 0, 1), clamp(st.steer / p.max_steer, -1, 1));
  return { obs: obs.map(Math.fround), dist };
}

// ------------------------------------------------------------------ trained drivers
export function makeBrain(model) {
  if (model.kind === "ppo") {
    return obs => {
      let x = obs;
      model.layers.forEach((layer, li) => {
        const y = layer.b.map((b, r) => layer.w[r].reduce((s, w, c) => s + w * x[c], b));
        x = li < model.layers.length - 1 ? y.map(Math.tanh) : y;
      });
      return x.map(v => clamp(v, -1, 1));
    };
  }
  // NEAT: neat-python's FeedForwardNetwork with tanh(2.5 z) and sum aggregation.
  return obs => {
    const values = new Map();
    model.inputs.forEach((k, i) => values.set(k, obs[i]));
    for (const ev of model.evals) {
      let s = 0;
      for (const [i, w] of ev.links) s += (values.get(i) ?? 0) * w;
      values.set(ev.node, Math.tanh(Math.max(-60, Math.min(60, 2.5 * (ev.bias + ev.response * s)))));
    }
    return model.outputs.map(k => clamp(values.get(k) ?? 0, -1, 1));
  };
}
