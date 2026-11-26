// The browser game: you against trained PPO and NEAT drivers, on the ported simulator.
import { DATA } from "./data.js";
import { Track, Simulation, RaySensor, observe, makeBrain, bodyCenter } from "./sim.js";

const COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948", "#0ca30c"];
const PLAYER_COLOR = "#f2f2f2";
const SCALE = 6; // pixels per meter on the pre-rendered track
const LIGHTS_S = 4.0, LIGHT_STEP = 0.75;

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const $ = id => document.getElementById(id);
const keys = { left: false, right: false, up: false, down: false };
let race = null;

// ------------------------------------------------------------------ menu
const tracks = DATA.tracks.map(t => new Track(t));
$("track").innerHTML = tracks.map((t, i) =>
  `<option value="${i}">${t.name}${t.split === "train" ? " (training track)" : " (never seen by the AI)"}</option>`).join("");
$("track").value = String(Math.max(0, tracks.findIndex(t => t.id === "unseen_a")));
$("drivers").innerHTML = DATA.models.map((m, i) =>
  `<label><input type="checkbox" data-i="${i}" ${i % 3 === 0 ? "checked" : ""}>
   <span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span>${m.label}
   <span style="color:var(--dim);font-size:12px">${m.kind === "ppo" ? "RL" : "evolved"}</span></label>`).join("");
$("race").onclick = () => start(true);
$("watch").onclick = () => start(false);

function start(withPlayer) {
  const track = tracks[+$("track").value];
  const laps = +$("laps").value;
  const chosen = [...document.querySelectorAll("#drivers input:checked")].map(e => +e.dataset.i);
  if (!chosen.length && !withPlayer) return;
  const cfg = { ...DATA.sim, laps_to_finish: laps, max_time: 45 * laps + 30 };
  const cars = chosen.map(i => ({
    label: DATA.models[i].label, kind: DATA.models[i].kind, color: COLORS[i % COLORS.length],
    brain: makeBrain(DATA.models[i]),
  }));
  if (withPlayer) cars.push({ label: "You", kind: "human", color: PLAYER_COLOR, brain: null, player: true });
  const lateral = Math.min(2.6, 0.5 * track.halfWidthMin);
  cars.forEach((c, slot) => {
    c.sim = new Simulation(track, DATA.vehicle, cfg);
    c.sensor = new RaySensor(track, DATA.sensors.angles_deg, DATA.sensors.max_range);
    const row = Math.floor(slot / 2), side = slot % 2;
    c.sim.place(-(row + 0.5) * 9 - 0.5 * side * 9, side === 0 ? lateral : -lateral);
    c.trail = [];
  });
  race = { track, laps, cars, t: -LIGHTS_S, acc: 0, layer: renderTrack(track), done: false, endAt: null };
  $("menu").style.display = "none"; $("results").style.display = "none";
  $("hud").style.display = "block"; $("standings").style.display = "block";
}

addEventListener("keydown", e => {
  const k = keyName(e); if (k) { keys[k] = true; e.preventDefault(); }
  if (e.key === "Escape") toMenu();
  if ((e.key === "r" || e.key === "R") && race) start(race.cars.some(c => c.player));
});
addEventListener("keyup", e => { const k = keyName(e); if (k) keys[k] = false; });
function keyName(e) {
  return { ArrowLeft: "left", a: "left", A: "left", ArrowRight: "right", d: "right", D: "right",
           ArrowUp: "up", w: "up", W: "up", ArrowDown: "down", s: "down", S: "down" }[e.key];
}
if (matchMedia("(pointer: coarse)").matches) {
  $("touch").style.display = "flex";
  for (const b of document.querySelectorAll("#touch button")) {
    b.addEventListener("pointerdown", e => { keys[b.dataset.k] = true; e.preventDefault(); });
    for (const ev of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(ev, () => { keys[b.dataset.k] = false; });
  }
}
function toMenu() {
  race = null;
  for (const id of ["hud", "standings", "banner", "results"]) $(id).style.display = "none";
  $("menu").style.display = "block";
}

// ------------------------------------------------------------------ simulation loop
function stepRace(realDt) {
  const r = race;
  r.t += realDt;
  if (r.t < 0) return;
  r.acc += realDt;
  const dt = DATA.sim.dt;
  let n = 0;
  while (r.acc >= dt && n < 8) {
    r.acc -= dt; n++;
    for (const c of r.cars) {
      if (c.sim.termination) continue;
      let action;
      if (c.player) action = [(keys.right ? 1 : 0) - (keys.left ? 1 : 0), (keys.up ? 1 : 0) - (keys.down ? 1 : 0)];
      else action = c.brain(observe(c.sim, c.sensor).obs);
      c.sim.step(action);
      if (c.sim.termination === "laps_complete" && c.finish == null) c.finish = c.sim.time;
      const [x, y] = bodyCenter(c.sim.state, c.sim.p);
      c.trail.push([x, y]); if (c.trail.length > 90) c.trail.shift();
    }
  }
  const player = r.cars.find(c => c.player);
  const allDone = r.cars.every(c => c.sim.termination);
  if (!r.done && ((player && player.sim.termination) || allDone)) { r.done = true; r.endAt = r.t + (allDone ? 0.5 : 3); }
  if (r.done && r.t >= r.endAt && $("results").style.display !== "block") showResults();
}

function distance(c) { return c.sim.progress.laps + c.sim.progress.lapFraction; }
function standings(r) {
  return [...r.cars].sort((a, b) => {
    if (a.finish != null && b.finish != null) return a.finish - b.finish;
    if (a.finish != null) return -1; if (b.finish != null) return 1;
    const crashedA = a.sim.termination && a.finish == null, crashedB = b.sim.termination && b.finish == null;
    if (crashedA !== crashedB) return crashedA ? 1 : -1;
    return distance(b) - distance(a);
  });
}
const fmt = s => s == null ? "–" : `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, "0")}`;

function showResults() {
  const rows = standings(race).map((c, i) => {
    const result = c.finish != null ? fmt(c.finish) : c.sim.termination === "collision" ? "crashed" : c.sim.termination ? "out of time" : "still racing";
    const kind = c.player ? "human" : c.kind === "ppo" ? "learned (PPO)" : "learned (NEAT)";
    return `<tr><td>${i + 1}</td><td><span class="swatch" style="display:inline-block;background:${c.color}"></span> ${c.label}</td><td>${kind}</td><td>${result}</td></tr>`;
  }).join("");
  $("results").innerHTML = `<h2 style="margin:0">Results</h2><table><tr><th>#</th><th>Driver</th><th>Type</th><th>Time</th></tr>${rows}</table>
    <div class="actions"><button class="primary" id="again">Race again</button><button id="menuBtn">Menu</button></div>`;
  $("results").style.display = "block";
  $("again").onclick = () => start(race.cars.some(c => c.player));
  $("menuBtn").onclick = toMenu;
}

// ------------------------------------------------------------------ drawing
function renderTrack(track) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const e of [track.left_edge, track.right_edge]) for (const [x, y] of e) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const pad = 400;
  const W = Math.ceil((maxX - minX) * SCALE + 2 * pad), H = Math.ceil((maxY - minY) * SCALE + 2 * pad);
  const off = document.createElement("canvas"); off.width = W; off.height = H;
  const g = off.getContext("2d");
  const P = ([x, y]) => [(x - minX) * SCALE + pad, (maxY - y) * SCALE + pad];
  // grass with mowing stripes
  g.fillStyle = "#26482e"; g.fillRect(0, 0, W, H);
  g.fillStyle = "#2c5135";
  const band = 7 * SCALE;
  for (let k = -H; k < W; k += 2 * band) { g.beginPath(); g.moveTo(k, 0); g.lineTo(k + band, 0); g.lineTo(k + band + H, H); g.lineTo(k + H, H); g.fill(); }
  // asphalt: outer edge filled, inner edge cut out (even-odd)
  g.fillStyle = "#40434a";
  g.beginPath();
  for (const e of [track.left_edge, track.right_edge]) { e.forEach((p, i) => { const [x, y] = P(p); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.closePath(); }
  g.fill("evenodd");
  // kerbs on bends
  const n = track.n, bend = track.curvature.map(k => Math.abs(k) > 1 / 70);
  const wide = bend.map((_, i) => { for (let d = -4; d <= 4; d++) if (bend[(i + d + n) % n]) return true; return false; });
  for (const [edge, sign] of [[track.left_edge, -1], [track.right_edge, 1]]) {
    for (let i = 0; i < n; i++) {
      if (!wide[i]) continue;
      const j = (i + 1) % n, inward = (p, k) => [p[0] + sign * track.normals[k][0] * 0.9, p[1] + sign * track.normals[k][1] * 0.9];
      const quad = [edge[i], edge[j], inward(edge[j], j), inward(edge[i], i)].map(P);
      g.fillStyle = Math.floor(track.cum_s[i] / 2) % 2 ? "#c4282c" : "#ececec";
      g.beginPath(); quad.forEach(([x, y], q) => q ? g.lineTo(x, y) : g.moveTo(x, y)); g.fill();
    }
  }
  // edge lines
  g.strokeStyle = "#e2e4e8"; g.lineWidth = Math.max(1, 0.25 * SCALE);
  for (const e of [track.left_edge, track.right_edge]) { g.beginPath(); e.forEach((p, i) => { const [x, y] = P(p); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.closePath(); g.stroke(); }
  // checkered start/finish
  const c0 = track.centerline[0], t0 = track.tangents[0], n0 = track.normals[0], hw = track.half_width[0];
  const cells = Math.max(4, Math.round(2 * hw)), size = 2 * hw / cells;
  for (let row = 0; row < 2; row++) for (let col = 0; col < cells; col++) {
    const pts = [[(row - 1) * size, -hw + col * size], [row * size, -hw + col * size], [row * size, -hw + (col + 1) * size], [(row - 1) * size, -hw + (col + 1) * size]]
      .map(([a, l]) => P([c0[0] + t0[0] * a + n0[0] * l, c0[1] + t0[1] * a + n0[1] * l]));
    g.fillStyle = (row + col) % 2 ? "#18181a" : "#f0f0f0";
    g.beginPath(); pts.forEach(([x, y], q) => q ? g.lineTo(x, y) : g.moveTo(x, y)); g.fill();
  }
  return { canvas: off, P, minX, maxY, pad };
}

function drawCar(c, sx, sy, zoom, number) {
  const p = c.sim.p, st = c.sim.state;
  const L = p.length * zoom, Wd = p.width * zoom;
  const crashed = c.sim.termination === "collision";
  ctx.save(); ctx.translate(sx, sy); ctx.rotate(-st.heading);
  ctx.fillStyle = "#111"; // wheels
  for (const fx of [-0.34, 0.2]) for (const fy of [-0.56, 0.36]) ctx.fillRect(fx * L, fy * Wd, 0.18 * L, 0.2 * Wd);
  ctx.fillStyle = crashed ? "#c84a4a" : c.color;
  roundRect(-L / 2, -Wd / 2 + 0.04 * Wd, L, Wd * 0.92, 0.3 * Wd); ctx.fill();
  ctx.fillStyle = "#1e2634";
  ctx.fillRect(0.06 * L, -0.3 * Wd, 0.14 * L, 0.6 * Wd); ctx.fillRect(-0.34 * L, -0.28 * Wd, 0.08 * L, 0.56 * Wd);
  if (zoom > 4) { ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(-0.1 * L, 0, 0.24 * Wd, 0, 7); ctx.fill();
    ctx.fillStyle = "#111"; ctx.font = `bold ${Math.round(0.34 * Wd)}px system-ui`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.rotate(Math.PI / 2); ctx.fillText(number, 0, 0.1 * L); }
  ctx.restore();
}
function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

function draw() {
  const dpr = devicePixelRatio || 1, W = innerWidth, H = innerHeight;
  if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#24402c"; ctx.fillRect(0, 0, W, H);
  if (!race) return;
  const r = race, L = r.layer;
  const order = standings(r);
  const focus = r.cars.find(c => c.player && !c.sim.termination) || order.find(c => !c.sim.termination) || order[0];
  const zoom = Math.min(SCALE, Math.max(3.2, Math.min(W, H) / 110));
  const k = zoom / SCALE;
  const [fx, fy] = L.P(bodyCenter(focus.sim.state, focus.sim.p));
  ctx.save(); ctx.translate(W / 2 - fx * k, H / 2 - fy * k); ctx.scale(k, k);
  ctx.drawImage(L.canvas, 0, 0);
  for (const c of r.cars) {  // trails
    if (c.trail.length < 2) continue;
    ctx.strokeStyle = c.color; ctx.globalAlpha = 0.45; ctx.lineWidth = 2 / k; ctx.beginPath();
    c.trail.forEach((p, i) => { const [x, y] = L.P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.globalAlpha = 1;
  }
  ctx.restore();
  r.cars.forEach((c, i) => {
    const [x, y] = L.P(bodyCenter(c.sim.state, c.sim.p));
    const sx = W / 2 + (x - fx) * k, sy = H / 2 + (y - fy) * k;
    drawCar(c, sx, sy, zoom, c.player ? "U" : String(i + 1));
    ctx.font = "12px system-ui"; ctx.fillStyle = "rgba(14,16,20,.8)";
    const w = ctx.measureText(c.label).width + 8; ctx.fillRect(sx + 14, sy - 26, w, 17);
    ctx.fillStyle = c.color; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText(c.label, sx + 18, sy - 13);
  });
  drawMinimap(W, H);
  drawLights(W);
  hud(order);
}

function drawMinimap(W, H) {
  const r = race, t = r.track, size = Math.min(190, W * 0.3);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of t.centerline) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
  const s = (size - 24) / Math.max(maxX - minX, maxY - minY), ox = W - size - 12, oy = H - size - 12;
  const M = ([x, y]) => [ox + 12 + (x - minX) * s, oy + 12 + (maxY - y) * s];
  ctx.fillStyle = "rgba(14,16,20,.85)"; ctx.fillRect(ox, oy, size, size);
  ctx.strokeStyle = "#cfd3da"; ctx.lineWidth = 2; ctx.beginPath();
  t.centerline.forEach((p, i) => { const [x, y] = M(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.closePath(); ctx.stroke();
  for (const c of r.cars) { const [x, y] = M(bodyCenter(c.sim.state, c.sim.p)); ctx.fillStyle = c.color; ctx.beginPath(); ctx.arc(x, y, c.player ? 5 : 4, 0, 7); ctx.fill(); }
}

function drawLights(W) {
  const r = race, b = $("banner");
  if (r.t < 0) {
    const lit = Math.min(5, Math.floor((r.t + LIGHTS_S) / LIGHT_STEP) + 1);
    b.style.display = "block";
    b.innerHTML = [0, 1, 2, 3, 4].map(n => `<span style="display:inline-block;width:30px;height:30px;border-radius:50%;margin:0 5px;background:${n < lit ? "#e6201e" : "#3a1616"}"></span>`).join("");
  } else if (r.t < 1) { b.style.display = "block"; b.innerHTML = `<span style="color:#5ae66e">GO</span>`; }
  else b.style.display = "none";
}

function hud(order) {
  const r = race, me = r.cars.find(c => c.player) || order[0];
  const pos = order.indexOf(me) + 1, sim = me.sim;
  const lap = Math.min(r.laps, sim.progress.laps + 1);
  const status = sim.termination === "collision" ? "Crashed: off the track" : sim.termination === "laps_complete" ? "Finished!" : "";
  $("hud").innerHTML = `<div class="dim">${me.player ? "You" : "Leader: " + me.label}</div>
    <div class="big">P${pos}/${r.cars.length} · Lap ${lap}/${r.laps}</div>
    <div class="big">${fmt(Math.max(0, r.t > 0 ? sim.time : 0))}</div>
    <div class="dim">${(sim.state.speed * 3.6).toFixed(0)} km/h ${sim.lapTimes.length ? "· best lap " + fmt(Math.min(...sim.lapTimes)) : ""}</div>
    <div style="color:#f08a8a">${status}</div>`;
  const lead = order[0];
  $("standings").innerHTML = order.map((c, i) => {
    let gap = "";
    if (c.finish != null) gap = fmt(c.finish);
    else if (c.sim.termination) gap = c.sim.termination === "collision" ? "crashed" : "out";
    else if (i > 0) gap = `-${((distance(lead) - distance(c)) * r.track.length).toFixed(0)} m`;
    return `<div><b>${i + 1}</b><span class="swatch" style="background:${c.color}"></span>${c.label}<span class="gap">${gap}</span></div>`;
  }).join("");
}

// ------------------------------------------------------------------ main loop
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (race) stepRace(dt);
  draw();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
