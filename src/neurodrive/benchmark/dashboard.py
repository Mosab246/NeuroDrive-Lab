"""Build one self-contained interactive HTML dashboard from the committed result files.

Reads only files under ``results/`` (benchmark reports, re-selection reports, traffic
summary, Grand Prix standings), so it can be regenerated at any time with
``neurodrive dashboard``. No external scripts: the charts are plain SVG drawn by a small
inline script, with hover tooltips, a table view for each chart, and light/dark modes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

BENCHMARKS = (
    ("ppo_vs_neat_1m", "1M steps"),
    ("ppo_vs_neat_5m", "5M steps"),
    ("ppo_vs_neat_dynamic_3m", "Hard physics, 3M steps"),
)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(x: str | None) -> float | None:
    return float(x) if x not in (None, "", "None") else None


def collect(results: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"benchmarks": []}
    for name, label in BENCHMARKS:
        folder = results / "benchmark" / name
        if not (folder / "runs.csv").is_file():
            continue
        agg = json.loads((folder / "aggregate.json").read_text(encoding="utf-8"))
        runs = [
            {
                "algorithm": r["algorithm"],
                "seed": int(r["seed"]),
                "laps": _num(r["test_completion_rate"]),
                "lap_time": _num(r["test_lap_time_mean"]),
                "collisions": _num(r["test_collision_rate"]),
            }
            for r in _csv(folder / "runs.csv")
        ]
        reselected = None
        if (folder / "reselection.csv").is_file():
            reselected = {
                f"{r['algorithm']}/{r['seed']}": {
                    "laps": _num(r["new_completion_rate"]),
                    "lap_time": _num(r["new_lap_time_mean"]),
                    "collisions": _num(r["new_collision_rate"]),
                }
                for r in _csv(folder / "reselection.csv")
            }
        per_track: dict[str, dict[str, list[float]]] = {}
        splits: dict[str, str] = {}
        for r in _csv(folder / "per_track.csv"):
            splits[r["track"]] = r["split"]
            t = _num(r["lap_time_mean"])
            if t is not None:
                per_track.setdefault(r["track"], {}).setdefault(r["algorithm"], []).append(t)
        tracks = [
            {
                "track": tr,
                "split": splits[tr],
                **{a: sum(v) / len(v) for a, v in algos.items()},
            }
            for tr, algos in per_track.items()
        ]
        baselines = {
            k: {"laps": v.get("completion_rate"), "lap_time": v.get("lap_time_mean")}
            for k, v in agg["baselines"].items()
        }
        data["benchmarks"].append(
            {
                "name": name,
                "label": label,
                "budget": agg["config"]["budget_env_steps"],
                "runs": runs,
                "reselected": reselected,
                "tracks": tracks,
                "baselines": baselines,
            }
        )
    # Prefer the three-seed results (Phase 10); fall back to the one-seed files (Phase 9).
    traffic3 = results / "traffic" / "phase10" / "summary.csv"
    traffic = traffic3 if traffic3.is_file() else results / "traffic" / "summary.csv"
    if traffic.is_file():
        sfx = "_mean" if traffic == traffic3 else ""
        data["traffic"] = [
            {
                "driver": r["driver"],
                "alone": _num(r["alone_rate" + sfx]),
                "traffic": _num(r["traffic3_rate" + sfx]),
                "hits": _num(r["traffic3_hits"]),
                "lap_time_traffic": _num(r["traffic3_lap_time"]),
            }
            for r in _csv(traffic)
        ]
        data["traffic_seeds"] = 3 if traffic == traffic3 else 1
    files = sorted((results / "grand_prix" / "phase10").glob("seed*/standings.json"))
    if not files and (results / "grand_prix" / "phase9" / "standings.json").is_file():
        files = [results / "grand_prix" / "phase9" / "standings.json"]
    if files:
        events = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        keys = ("points", "wins", "finished", "races", "contacts")
        table: dict[str, dict[str, Any]] = {}
        for ev in events:
            for t in ev["standings"]:
                row = table.setdefault(
                    t["driver"], {"driver": t["driver"], "kind": t["kind"]} | dict.fromkeys(keys, 0)
                )
                for k in keys:
                    row[k] += t[k]
        data["grand_prix"] = {
            "races": len(events[0]["tracks"]),
            "laps": events[0]["laps"],
            "seeds": len(events),
            "standings": sorted(table.values(), key=lambda r: -r["points"]),
        }
    return data


def build_dashboard(results: Path, out: Path) -> Path:
    data = collect(results)
    # Plain marker replacement: the page's JavaScript uses ${...} itself.
    html = _TEMPLATE.replace("__DATA__", json.dumps(data))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroDrive Lab results</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f4f2; --surface-1: #fcfcfb; --border: #e2e1dc;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7973;
  --grid: #e8e7e3; --axis: #b9b8b1;
  --ppo: #2a78d6; --neat: #eb6834; --neutral: #8f8e88;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2c;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e88;
    --grid: #2a2a28; --axis: #4a4a47;
    --ppo: #3987e5; --neat: #d95926; --neutral: #7a7973;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2c;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e88;
  --grid: #2a2a28; --axis: #4a4a47;
  --ppo: #3987e5; --neat: #d95926; --neutral: #7a7973;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 40px 0 6px; }
p.lede { color: var(--text-secondary); margin: 0 0 24px; max-width: 760px; }
p.note { color: var(--text-muted); font-size: 13px; margin: 6px 0 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .label { color: var(--text-secondary); font-size: 13px; }
.tile .value { font-size: 30px; font-weight: 650; letter-spacing: -0.02em; margin: 2px 0; }
.tile .sub { color: var(--text-muted); font-size: 13px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-top: 12px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 8px 0 4px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { background: transparent; border: 0; color: var(--text-secondary); padding: 6px 12px; font: inherit; font-size: 13px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--text-primary); color: var(--surface-1); }
.seg button:disabled { opacity: .4; cursor: default; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); margin: 4px 0 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-secondary); font-size: 12px; }
svg .value { fill: var(--text-primary); font-weight: 600; }
.tooltip { position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1);
  border-radius: 8px; padding: 8px 10px; font-size: 13px; line-height: 1.4; max-width: 280px;
  opacity: 0; transition: opacity .08s; z-index: 10; }
.toggle-table { background: none; border: 0; color: var(--text-secondary); font: inherit; font-size: 13px;
  padding: 0; margin-top: 8px; cursor: pointer; text-decoration: underline; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
th { color: var(--text-secondary); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.theme { position: absolute; top: 16px; right: 16px; }
@media (max-width: 640px) { .tile .value { font-size: 24px; } }
</style>
</head>
<body>
<div class="theme seg" role="group" aria-label="Theme">
  <button data-theme-choice="light">Light</button><button data-theme-choice="dark">Dark</button>
</div>
<main>
  <h1>NeuroDrive Lab results</h1>
  <p class="lede">PPO (reinforcement learning) and NEAT (neuroevolution) learning to drive, compared on
  equal training budgets, several seeds and 26 test tracks (22 never seen in training). Every number
  comes from the committed result files and can be regenerated with <code>neurodrive dashboard</code>.</p>
  <div class="tiles" id="tiles"></div>

  <h2>Laps completed per training run</h2>
  <p class="note">Share of 520 test scenarios (26 tracks x 20 random starts) each run finished.</p>
  <div class="card">
    <div class="controls">
      <div class="seg" id="bench-seg" role="group" aria-label="Benchmark"></div>
      <div class="seg" id="pick-seg" role="group" aria-label="Checkpoint choice">
        <button data-pick="training">Picked during training</button><button data-pick="reselected">Re-selected (7 tracks)</button>
      </div>
    </div>
    <div class="legend"><span><i style="background:var(--ppo)"></i>PPO</span><span><i style="background:var(--neat)"></i>NEAT</span><span><i style="background:var(--neutral)"></i>Scripted reference</span></div>
    <div id="runs-chart"></div>
    <button class="toggle-table" data-for="runs">Show table</button><div id="runs-table" hidden></div>
  </div>

  <h2>Lap time on every test track</h2>
  <p class="note">Mean over the seeds' completed laps; lower is faster. Training-time checkpoint choice.</p>
  <div class="card">
    <div class="controls"><div class="seg" id="bench-seg-2" role="group" aria-label="Benchmark"></div></div>
    <div class="legend"><span><i style="background:var(--ppo)"></i>PPO</span><span><i style="background:var(--neat)"></i>NEAT</span></div>
    <div id="tracks-chart"></div>
    <button class="toggle-table" data-for="tracks">Show table</button><div id="tracks-table" hidden></div>
  </div>

  <h2>Racing with traffic</h2>
  <p class="note">22 test tracks x 20 scenarios with 3 opponent cars on track. <span id="traffic-seeds"></span></p>
  <div class="card">
    <div class="legend"><span><i style="background:var(--ppo)"></i>PPO</span><span><i style="background:var(--neat)"></i>NEAT</span></div>
    <div id="traffic-chart"></div>
    <button class="toggle-table" data-for="traffic">Show table</button><div id="traffic-table" hidden></div>
  </div>

  <h2>Grand Prix</h2>
  <p class="note" id="gp-note"></p>
  <div class="card">
    <div class="legend"><span><i style="background:var(--ppo)"></i>PPO</span><span><i style="background:var(--neat)"></i>NEAT</span><span><i style="background:var(--neutral)"></i>Scripted</span></div>
    <div id="gp-chart"></div>
    <button class="toggle-table" data-for="gp">Show table</button><div id="gp-table" hidden></div>
  </div>
</main>
<div class="tooltip" id="tip" role="status"></div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tip");
const pct = x => x == null ? "–" : (100 * x).toFixed(1) + "%";
const sec = x => x == null ? "–" : x.toFixed(2) + " s";
const algoColor = a => a.toLowerCase().includes("ppo") ? "var(--ppo)" : a.toLowerCase().includes("neat") ? "var(--neat)" : "var(--neutral)";

function el(name, attrs, parent) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function showTip(evt, html) {
  tip.innerHTML = html; tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > innerHeight - 8) y = evt.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.style.opacity = 0; }
addEventListener("scroll", hideTip, {passive: true});
function hover(node, html) {
  node.addEventListener("mousemove", e => showTip(e, html));
  node.addEventListener("mouseleave", hideTip);
  node.setAttribute("tabindex", "0");
  node.addEventListener("focus", e => { const r = node.getBoundingClientRect(); showTip({clientX: r.right, clientY: r.top}, html); });
  node.addEventListener("blur", hideTip);
}
function table(container, head, rows, numeric) {
  const t = document.createElement("table");
  t.innerHTML = "<thead><tr>" + head.map((h, i) => `<th class="${numeric[i] ? "num" : ""}">${h}</th>`).join("") + "</tr></thead>"
    + "<tbody>" + rows.map(r => "<tr>" + r.map((c, i) => `<td class="${numeric[i] ? "num" : ""}">${c}</td>`).join("") + "</tr>").join("") + "</tbody>";
  container.replaceChildren(t);
}
// A rounded-top bar anchored to the baseline (4px radius on the data end only).
function bar(parent, x, y, w, h, fill) {
  const r = Math.min(4, w / 2, h);
  const d = `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`;
  return el("path", {d, fill}, parent);
}
function hbar(parent, x, y, w, h, fill) {
  const r = Math.min(4, h / 2, w);
  const d = `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
  return el("path", {d, fill}, parent);
}

// ---------------------------------------------------------------- tiles
function tiles() {
  const box = document.getElementById("tiles");
  const b5 = DATA.benchmarks.find(b => b.name === "ppo_vs_neat_5m");
  const bh = DATA.benchmarks.find(b => b.name === "ppo_vs_neat_dynamic_3m");
  const mean = (runs, a, k) => { const v = runs.filter(r => r.algorithm === a).map(r => r[k]).filter(x => x != null); return v.reduce((s, x) => s + x, 0) / v.length; };
  const items = [];
  if (b5) items.push(["5M steps: NEAT laps completed", pct(mean(b5.runs, "neat", "laps")), "PPO " + pct(mean(b5.runs, "ppo", "laps")) + " · all 3 seeds each"]);
  if (b5) items.push(["5M steps: mean lap time", sec(mean(b5.runs, "neat", "lap_time")), "NEAT, vs PPO " + sec(mean(b5.runs, "ppo", "lap_time"))]);
  if (b5 && b5.reselected) {
    const v = Object.entries(b5.reselected).filter(([k]) => k.startsWith("ppo")).map(([, r]) => r.laps);
    items.push(["PPO with better checkpoint picking", pct(v.reduce((s, x) => s + x, 0) / v.length), "was " + pct(mean(b5.runs, "ppo", "laps")) + " · same training, 5M"]);
  }
  if (bh) items.push(["Hard physics: crashes", pct(mean(bh.runs, "ppo", "collisions")), "PPO, vs NEAT " + pct(mean(bh.runs, "neat", "collisions"))]);
  box.innerHTML = items.map(([l, v, s]) => `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${s}</div></div>`).join("");
}

// ---------------------------------------------------------------- runs chart
let benchIndex = Math.max(0, DATA.benchmarks.findIndex(b => b.name === "ppo_vs_neat_5m"));
let pick = "training";
function segButtons(id, onPick) {
  const seg = document.getElementById(id);
  seg.innerHTML = DATA.benchmarks.map((b, i) => `<button data-i="${i}">${b.label}</button>`).join("");
  seg.addEventListener("click", e => { const i = e.target.dataset.i; if (i !== undefined) onPick(+i); });
}
function syncButtons() {
  for (const id of ["bench-seg", "bench-seg-2"])
    document.querySelectorAll(`#${id} button`).forEach(b => b.setAttribute("aria-pressed", +b.dataset.i === benchIndex));
  const b = DATA.benchmarks[benchIndex];
  document.querySelectorAll("#pick-seg button").forEach(btn => {
    btn.setAttribute("aria-pressed", btn.dataset.pick === pick);
    if (btn.dataset.pick === "reselected") btn.disabled = !b.reselected;
  });
  if (!b.reselected && pick === "reselected") { pick = "training"; syncButtons(); }
}
function runsChart() {
  const b = DATA.benchmarks[benchIndex];
  const rows = b.runs.slice().sort((p, q) => (p.algorithm > q.algorithm ? -1 : 1) || p.seed - q.seed)
    .map(r => {
      const v = pick === "reselected" && b.reselected ? b.reselected[`${r.algorithm}/${r.seed}`] : r;
      return {label: `${r.algorithm.toUpperCase()} seed ${r.seed}`, algo: r.algorithm, laps: v.laps, lap: v.lap_time, col: v.collisions};
    });
  for (const [name, ref] of Object.entries(b.baselines)) if (ref.laps > 0)
    rows.push({label: name, algo: "scripted", laps: ref.laps, lap: ref.lap_time, col: null});
  const W = 1040, H = 300, m = {l: 44, r: 8, t: 22, b: 40};
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Laps completed per run"});
  const iw = W - m.l - m.r, ih = H - m.t - m.b, y = v => m.t + ih * (1 - v);
  for (const v of [0, .25, .5, .75, 1]) {
    el("line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? "var(--axis)" : "var(--grid)"}, svg);
    el("text", {x: m.l - 8, y: y(v) + 4, "text-anchor": "end"}, svg).textContent = (v * 100) + "%";
  }
  const band = iw / rows.length, bw = Math.min(54, band * 0.6);
  rows.forEach((r, i) => {
    const cx = m.l + band * (i + .5), top = y(r.laps);
    const node = bar(svg, cx - bw / 2, top, bw, y(0) - top, algoColor(r.algo));
    el("text", {x: cx, y: top - 6, "text-anchor": "middle", class: "value"}, svg).textContent = (100 * r.laps).toFixed(0) + "%";
    el("text", {x: cx, y: H - m.b + 18, "text-anchor": "middle"}, svg).textContent = r.label;
    const hit = el("rect", {x: cx - band / 2, y: m.t, width: band, height: ih, fill: "transparent"}, svg);
    hover(hit, `<b>${r.label}</b><br>Laps completed: ${pct(r.laps)}<br>Mean lap time: ${sec(r.lap)}` + (r.col != null ? `<br>Crashes: ${pct(r.col)}` : ""));
  });
  document.getElementById("runs-chart").replaceChildren(svg);
  table(document.getElementById("runs-table"), ["Run", "Laps completed", "Mean lap time", "Crashes"],
    rows.map(r => [r.label, pct(r.laps), sec(r.lap), r.col == null ? "–" : pct(r.col)]), [0, 1, 1, 1]);
}

// ---------------------------------------------------------------- per-track dumbbell
function tracksChart() {
  const b = DATA.benchmarks[benchIndex];
  const rows = b.tracks.filter(t => t.ppo != null || t.neat != null)
    .sort((p, q) => Math.min(p.ppo ?? 1e9, p.neat ?? 1e9) - Math.min(q.ppo ?? 1e9, q.neat ?? 1e9));
  const vals = rows.flatMap(r => [r.ppo, r.neat]).filter(x => x != null);
  const lo = Math.floor(Math.min(...vals) - 0.5), hi = Math.ceil(Math.max(...vals) + 0.5);
  const rowH = 22, W = 1040, m = {l: 170, r: 24, t: 10, b: 30}, H = m.t + m.b + rowH * rows.length;
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Lap time per track"});
  const x = v => m.l + (W - m.l - m.r) * (v - lo) / (hi - lo);
  for (let v = lo; v <= hi; v++) {
    el("line", {x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, stroke: "var(--grid)"}, svg);
    if ((v - lo) % 2 === 0) el("text", {x: x(v), y: H - m.b + 18, "text-anchor": "middle"}, svg).textContent = v + " s";
  }
  rows.forEach((r, i) => {
    const cy = m.t + rowH * (i + .5);
    const name = r.track + (r.split === "train" ? "  (trained on)" : r.split === "validation" ? "  (validation)" : "");
    el("text", {x: m.l - 10, y: cy + 4, "text-anchor": "end"}, svg).textContent = name;
    if (r.ppo != null && r.neat != null)
      el("line", {x1: x(r.ppo), x2: x(r.neat), y1: cy, y2: cy, stroke: "var(--axis)", "stroke-width": 2}, svg);
    for (const [k, c] of [["ppo", "var(--ppo)"], ["neat", "var(--neat)"]]) if (r[k] != null)
      el("circle", {cx: x(r[k]), cy, r: 5, fill: c, stroke: "var(--surface-1)", "stroke-width": 2}, svg);
    const hit = el("rect", {x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "transparent"}, svg);
    const faster = r.ppo != null && r.neat != null ? (r.ppo < r.neat ? "PPO" : "NEAT") + " faster by " + Math.abs(r.ppo - r.neat).toFixed(2) + " s" : "";
    hover(hit, `<b>${r.track}</b> (${r.split})<br>PPO: ${sec(r.ppo)}<br>NEAT: ${sec(r.neat)}<br>${faster}`);
  });
  document.getElementById("tracks-chart").replaceChildren(svg);
  table(document.getElementById("tracks-table"), ["Track", "Split", "PPO", "NEAT"],
    rows.map(r => [r.track, r.split, sec(r.ppo), sec(r.neat)]), [0, 0, 1, 1]);
}

// ---------------------------------------------------------------- traffic
function trafficChart() {
  const rows = DATA.traffic || [];
  const ts = document.getElementById("traffic-seeds");
  if (ts) ts.textContent = DATA.traffic_seeds > 1 ? `Mean over ${DATA.traffic_seeds} seeds per model.` : `One seed per model.`;
  if (!rows.length) return;
  const W = 1040, rowH = 46, m = {l: 210, r: 190, t: 24, b: 30}, H = m.t + m.b + rowH * rows.length;
  const hitsX = W - 70;
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Laps completed with traffic"});
  const x = v => m.l + (W - m.l - m.r) * v / 100;
  for (const v of [0, 25, 50, 75, 100]) {
    el("line", {x1: x(v), x2: x(v), y1: m.t - 6, y2: H - m.b, stroke: v === 0 ? "var(--axis)" : "var(--grid)"}, svg);
    el("text", {x: x(v), y: H - m.b + 18, "text-anchor": "middle"}, svg).textContent = v + "%";
  }
  el("text", {x: hitsX, y: m.t - 8}, svg).textContent = "hits / race";
  rows.forEach((r, i) => {
    const cy = m.t + rowH * i + 6, h = 14;
    el("text", {x: m.l - 10, y: cy + 16, "text-anchor": "end"}, svg).textContent = r.driver;
    const ghost = hbar(svg, x(0), cy, x(r.alone) - x(0), h, algoColor(r.driver)); ghost.setAttribute("opacity", ".28");
    hbar(svg, x(0), cy + h + 2, x(r.traffic) - x(0), h, algoColor(r.driver));
    el("text", {x: x(r.traffic) + 6, y: cy + 2 * h, class: "value"}, svg).textContent = r.traffic.toFixed(1) + "%";
    el("text", {x: hitsX, y: cy + 2 * h, class: "value"}, svg).textContent = r.hits.toFixed(2);
    const hit = el("rect", {x: 0, y: cy - 4, width: W, height: rowH, fill: "transparent"}, svg);
    hover(hit, `<b>${r.driver}</b><br>Alone: ${r.alone.toFixed(1)}% laps<br>With 3 opponents: ${r.traffic.toFixed(1)}% laps, ${sec(r.lap_time_traffic)}<br>Hits on other cars: ${r.hits.toFixed(2)} per race`);
  });
  document.getElementById("traffic-chart").replaceChildren(svg);
  const note = document.createElement("p"); note.className = "note";
  note.textContent = "Faint bar: laps completed driving alone. Solid bar: with 3 opponents.";
  document.getElementById("traffic-chart").appendChild(note);
  table(document.getElementById("traffic-table"), ["Driver", "Alone", "With 3 opponents", "Lap time in traffic", "Hits / race"],
    rows.map(r => [r.driver, r.alone.toFixed(1) + "%", r.traffic.toFixed(1) + "%", sec(r.lap_time_traffic), r.hits.toFixed(2)]), [0, 1, 1, 1, 1]);
}

// ---------------------------------------------------------------- grand prix
function gpChart() {
  const g = DATA.grand_prix;
  if (!g) return;
  document.getElementById("gp-note").textContent = `${g.races} races, ${g.laps} laps each, grid rotated every race, F1 points (25, 18, 15, ...). ` + (g.seeds > 1 ? `Totals over ${g.seeds} championships, one per training seed.` : `One seed per model.`);
  const rows = g.standings, max = Math.max(...rows.map(r => r.points));
  const W = 1040, rowH = 30, m = {l: 210, r: 60, t: 8, b: 8}, H = m.t + m.b + rowH * rows.length;
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Grand Prix points"});
  const x = v => m.l + (W - m.l - m.r) * v / max;
  rows.forEach((r, i) => {
    const cy = m.t + rowH * i + 6, h = 18;
    el("text", {x: m.l - 10, y: cy + 13, "text-anchor": "end"}, svg).textContent = `${i + 1}. ${r.driver}`;
    hbar(svg, x(0), cy, Math.max(2, x(r.points) - x(0)), h, algoColor(r.driver));
    el("text", {x: x(r.points) + 6, y: cy + 13, class: "value"}, svg).textContent = r.points;
    const hit = el("rect", {x: 0, y: cy - 6, width: W, height: rowH, fill: "transparent"}, svg);
    hover(hit, `<b>${r.driver}</b> (${r.kind})<br>${r.points} points, ${r.wins} wins<br>Finished ${r.finished}/${r.races} races<br>Hits on other cars: ${r.contacts}`);
  });
  document.getElementById("gp-chart").replaceChildren(svg);
  table(document.getElementById("gp-table"), ["#", "Driver", "Type", "Points", "Wins", "Finished", "Hits"],
    rows.map((r, i) => [i + 1, r.driver, r.kind, r.points, r.wins, `${r.finished}/${r.races}`, r.contacts]), [1, 0, 0, 1, 1, 1, 1]);
}

// ---------------------------------------------------------------- wiring
function renderBench() { syncButtons(); runsChart(); tracksChart(); }
segButtons("bench-seg", i => { benchIndex = i; renderBench(); });
segButtons("bench-seg-2", i => { benchIndex = i; renderBench(); });
document.getElementById("pick-seg").addEventListener("click", e => {
  if (e.target.dataset.pick && !e.target.disabled) { pick = e.target.dataset.pick; renderBench(); }
});
document.querySelectorAll(".toggle-table").forEach(btn => btn.addEventListener("click", () => {
  const t = document.getElementById(btn.dataset.for + "-table"); t.hidden = !t.hidden;
  btn.textContent = t.hidden ? "Show table" : "Hide table";
}));
function setTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  document.querySelectorAll("[data-theme-choice]").forEach(b => b.setAttribute("aria-pressed", b.dataset.themeChoice === t));
  try { localStorage.setItem("nd-theme", t); } catch (e) {}
}
document.querySelectorAll("[data-theme-choice]").forEach(b => b.addEventListener("click", () => setTheme(b.dataset.themeChoice)));
try { const t = localStorage.getItem("nd-theme"); if (t) setTheme(t); } catch (e) {}
tiles(); renderBench(); trafficChart(); gpChart();
</script>
</body>
</html>
"""
