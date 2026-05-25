from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from qubic_lab.rl_deep import DeepRLConfig, train_deep_rl
from qubic_lab.rl_tabular import TabularConfig, train_tabular

app = FastAPI(title="Qubic Lab")

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None
_latest: dict[str, Any] | None = None
_history: list[dict[str, Any]] = []


def _safe_run_dir(run_dir: str) -> Path:
    root = Path("runs").resolve()
    candidate = Path(run_dir).resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=400, detail="run_dir must be under runs/")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return candidate


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qubic Lab</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1214;
      --panel: #171d21;
      --line: #2c3a40;
      --text: #e6ecef;
      --muted: #97a8ae;
      --accent: #67d2a7;
      --warn: #f5c15d;
      --blue: #7aa7ff;
      --loss: #f07c6b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header, main { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; }
    header { padding: 26px 0 18px; display: flex; align-items: end; justify-content: space-between; gap: 20px; }
    h1 { margin: 0; font-size: 30px; line-height: 1.1; }
    .subtle { color: var(--muted); font-size: 14px; }
    .shell { display: grid; grid-template-columns: 310px 1fr; gap: 18px; align-items: start; }
    section, aside {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
    }
    aside { padding: 14px; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; }
    input, select {
      width: 100%;
      min-height: 36px;
      border: 1px solid #3a4a51;
      border-radius: 6px;
      background: #101519;
      color: var(--text);
      padding: 7px 9px;
      font: inherit;
    }
    .actions { display: flex; gap: 10px; margin-top: 14px; }
    .run-picker { display: grid; gap: 8px; margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); }
    button {
      min-height: 38px;
      border: 1px solid #4e656e;
      border-radius: 6px;
      background: #202b31;
      color: var(--text);
      padding: 7px 12px;
      font-weight: 650;
      cursor: pointer;
    }
    button.primary { background: #24483d; border-color: #3e8a70; }
    button:disabled { opacity: .55; cursor: default; }
    .status { margin-top: 14px; display: grid; gap: 8px; }
    .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; padding: 14px; border-bottom: 1px solid var(--line); }
    .metric { min-width: 0; }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong { display: block; margin-top: 3px; font-size: 24px; line-height: 1; }
    .viz { padding: 14px; display: grid; gap: 18px; }
    canvas { width: 100%; height: 360px; background: #101519; border: 1px solid var(--line); border-radius: 8px; }
    #board3d { height: 460px; }
    .board3d-tools { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
    .board3d-tools .subtle { min-width: 180px; }
    .artifacts { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
    .artifact { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #101519; }
    .artifact img { display: block; width: 100%; height: auto; }
    .artifact a { display: block; padding: 9px 11px; color: var(--text); text-decoration: none; border-top: 1px solid var(--line); }
    .layers { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; }
    .layer { display: grid; gap: 8px; }
    .layer-title { color: var(--muted); font-size: 13px; }
    .grid { display: grid; gap: 5px; }
    .cell {
      aspect-ratio: 1;
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 5px;
      display: grid;
      place-items: center;
      color: #07100d;
      font-size: 12px;
      font-weight: 750;
      min-width: 0;
    }
    @media (max-width: 820px) {
      .shell { grid-template-columns: 1fr; }
      header { align-items: start; flex-direction: column; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Qubic Lab</h1>
      <div class="subtle" id="runDir">No run loaded</div>
    </div>
    <div class="subtle" id="status">idle</div>
  </header>
  <main class="shell">
    <aside>
      <div class="controls">
        <label>Method <select id="method">
          <option value="q_learning">Q-learning</option>
          <option value="sarsa">SARSA</option>
          <option value="expected_sarsa">Expected SARSA</option>
          <option value="monte_carlo">Monte Carlo</option>
          <option value="ppo">PPO</option>
          <option value="grpo">GRPO</option>
        </select></label>
        <label>Size <input id="size" type="number" min="2" max="4" value="3"></label>
        <label>Episodes <input id="episodes" type="number" min="1" value="20000"></label>
        <label>Alpha <input id="alpha" type="number" min="0" max="1" step="0.01" value="0.25"></label>
        <label>Gamma <input id="gamma" type="number" min="0" max="1" step="0.01" value="0.98"></label>
        <label>Epsilon <input id="epsilon" type="number" min="0" max="1" step="0.01" value="0.35"></label>
        <label>Seed <input id="seed" type="number" value="0"></label>
        <label>Log every <input id="log_every" type="number" min="1" value="100"></label>
        <label>Batch eps <input id="batch_episodes" type="number" min="1" value="32"></label>
      </div>
      <div class="actions">
        <button class="primary" id="startBtn">Start</button>
        <button id="stopBtn">Stop</button>
      </div>
      <div class="run-picker">
        <label>Saved run <select id="runSelect"></select></label>
        <button id="loadRunBtn">Load</button>
      </div>
      <div class="status">
        <div class="subtle">States: <strong id="states">0</strong></div>
        <div class="subtle">Epsilon: <strong id="eps">0</strong></div>
      </div>
    </aside>
    <section>
      <div class="metric-grid">
        <div class="metric"><span>Episode</span><strong id="episode">0</strong></div>
        <div class="metric"><span>X win</span><strong id="xwin">0%</strong></div>
        <div class="metric"><span>O win</span><strong id="owin">0%</strong></div>
        <div class="metric"><span>Draw</span><strong id="draw">0%</strong></div>
      </div>
      <div class="viz">
        <div class="board3d-tools">
          <div class="subtle" id="board3dLabel">3D value lattice + greedy arrows</div>
          <button id="resetViewBtn">Reset view</button>
        </div>
        <canvas id="board3d" width="1000" height="560"></canvas>
        <canvas id="chart" width="1000" height="560"></canvas>
        <div class="artifacts" id="artifacts"></div>
        <div class="layers" id="layers"></div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const fields = ["size", "episodes", "alpha", "gamma", "epsilon", "seed", "log_every", "batch_episodes"];
    let latestFor3d = null;
    let boardYaw = -0.72;
    let boardPitch = 0.72;
    let dragging3d = false;
    let lastPointer = null;

    function pct(x) { return `${Math.round((x || 0) * 100)}%`; }
    function color(v) {
      const t = Math.max(-1, Math.min(1, Number(v) || 0));
      if (t >= 0) {
        const g = Math.round(78 + 130 * t);
        return `rgb(${Math.round(230 - 130 * t)}, ${g}, ${Math.round(190 - 90 * t)})`;
      }
      const u = -t;
      return `rgb(${Math.round(238 - 30 * u)}, ${Math.round(190 - 90 * u)}, ${Math.round(120 - 60 * u)})`;
    }

    function renderHeatmap(heatmap) {
      const root = $("layers");
      root.innerHTML = "";
      if (!heatmap || !heatmap.length) return;
      const size = heatmap[0].length;
      heatmap.forEach((layer, z) => {
        const wrap = document.createElement("div");
        wrap.className = "layer";
        const title = document.createElement("div");
        title.className = "layer-title";
        title.textContent = `z=${z}`;
        const grid = document.createElement("div");
        grid.className = "grid";
        grid.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
        layer.forEach((row) => row.forEach((value) => {
          const cell = document.createElement("div");
          cell.className = "cell";
          cell.style.background = color(value);
          cell.textContent = Number(value).toFixed(2);
          grid.appendChild(cell);
        }));
        wrap.append(title, grid);
        root.appendChild(wrap);
      });
    }

    function renderArtifacts(latest) {
      const root = $("artifacts");
      root.innerHTML = "";
      if (!latest || !latest.run_dir) return;
      const run = encodeURIComponent(latest.run_dir);
      const artifacts = [
        ["curves.png", "Training curves"],
        ["first_move_heatmap.png", "First-move heatmap"],
      ];
      artifacts.forEach(([file, label]) => {
        const wrap = document.createElement("div");
        wrap.className = "artifact";
        const img = document.createElement("img");
        img.src = `/api/artifact?run_dir=${run}&file=${encodeURIComponent(file)}&t=${Date.now()}`;
        img.alt = label;
        const link = document.createElement("a");
        link.href = img.src;
        link.textContent = label;
        link.target = "_blank";
        wrap.append(img, link);
        root.appendChild(wrap);
      });
    }

    function flattenHeatmap(heatmap) {
      const points = [];
      if (!heatmap || !heatmap.length) return points;
      heatmap.forEach((layer, z) => {
        layer.forEach((row, y) => {
          row.forEach((value, x) => points.push({ x, y, z, value: Number(value) || 0 }));
        });
      });
      return points;
    }

    function project3d(point, size, scale, cx, cy) {
      const ox = point.x - (size - 1) / 2;
      const oy = point.y - (size - 1) / 2;
      const oz = point.z - (size - 1) / 2;
      const cyaw = Math.cos(boardYaw), syaw = Math.sin(boardYaw);
      const cp = Math.cos(boardPitch), sp = Math.sin(boardPitch);
      const x1 = ox * cyaw - oz * syaw;
      const z1 = ox * syaw + oz * cyaw;
      const y1 = oy * cp - z1 * sp;
      const z2 = oy * sp + z1 * cp;
      return {
        x: cx + x1 * scale,
        y: cy + y1 * scale,
        depth: z2,
      };
    }

    function valueColor(value, alpha = 1) {
      const t = Math.max(-1, Math.min(1, value));
      if (t >= 0) {
        const r = Math.round(92 + 40 * (1 - t));
        const g = Math.round(166 + 60 * t);
        const b = Math.round(138 + 20 * (1 - t));
        return `rgba(${r},${g},${b},${alpha})`;
      }
      const u = -t;
      const r = Math.round(245 - 20 * (1 - u));
      const g = Math.round(135 - 40 * u);
      const b = Math.round(105 - 30 * u);
      return `rgba(${r},${g},${b},${alpha})`;
    }

    function drawLine3d(ctx, a, b, size, scale, cx, cy, stroke, alpha = 1) {
      const pa = project3d(a, size, scale, cx, cy);
      const pb = project3d(b, size, scale, cx, cy);
      ctx.strokeStyle = stroke.replace("ALPHA", alpha);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    function neighborValue(heatmap, p) {
      return Number(heatmap[p.z]?.[p.y]?.[p.x]) || 0;
    }

    function bestNeighbor(p, heatmap, size) {
      const candidates = [
        { x: p.x + 1, y: p.y, z: p.z, axis: "+x" },
        { x: p.x - 1, y: p.y, z: p.z, axis: "-x" },
        { x: p.x, y: p.y + 1, z: p.z, axis: "+y" },
        { x: p.x, y: p.y - 1, z: p.z, axis: "-y" },
        { x: p.x, y: p.y, z: p.z + 1, axis: "+z" },
        { x: p.x, y: p.y, z: p.z - 1, axis: "-z" },
      ].filter(q => q.x >= 0 && q.x < size && q.y >= 0 && q.y < size && q.z >= 0 && q.z < size);
      let best = null;
      for (const q of candidates) {
        const value = neighborValue(heatmap, q);
        if (!best || value > best.value) best = { ...q, value };
      }
      return best;
    }

    function drawArrowHead(ctx, from, to, color) {
      const angle = Math.atan2(to.y - from.y, to.x - from.x);
      const len = 10;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(to.x, to.y);
      ctx.lineTo(to.x - len * Math.cos(angle - 0.45), to.y - len * Math.sin(angle - 0.45));
      ctx.lineTo(to.x - len * Math.cos(angle + 0.45), to.y - len * Math.sin(angle + 0.45));
      ctx.closePath();
      ctx.fill();
    }

    function drawArrow3d(ctx, a, b, size, scale, cx, cy, alpha) {
      const pa = project3d(a, size, scale, cx, cy);
      const pb = project3d(b, size, scale, cx, cy);
      const sx = pa.x + (pb.x - pa.x) * 0.28;
      const sy = pa.y + (pb.y - pa.y) * 0.28;
      const ex = pa.x + (pb.x - pa.x) * 0.72;
      const ey = pa.y + (pb.y - pa.y) * 0.72;
      const color = `rgba(245,193,93,${alpha})`;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      drawArrowHead(ctx, { x: sx, y: sy }, { x: ex, y: ey }, color);
    }

    function renderBoard3d(latest) {
      latestFor3d = latest || latestFor3d;
      const canvas = $("board3d");
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, "#101519");
      grad.addColorStop(1, "#151c21");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
      if (!latestFor3d || !latestFor3d.heatmap || !latestFor3d.heatmap.length) {
        ctx.fillStyle = "#97a8ae";
        ctx.font = "18px Inter, sans-serif";
        ctx.fillText("Load or start a run to render the 3D value lattice.", 36, 48);
        return;
      }

      const heatmap = latestFor3d.heatmap;
      const size = heatmap.length;
      const points = flattenHeatmap(heatmap);
      const maxAbs = Math.max(0.001, ...points.map(p => Math.abs(p.value)));
      const scale = Math.min(w, h) / (size <= 3 ? 4.5 : 5.5);
      const cx = w * 0.52;
      const cy = h * 0.55;

      ctx.lineWidth = 1.25;
      for (let y = 0; y < size; y++) {
        for (let z = 0; z < size; z++) {
          drawLine3d(ctx, { x: 0, y, z }, { x: size - 1, y, z }, size, scale, cx, cy, "rgba(111,139,148,ALPHA)", 0.28);
        }
        for (let x = 0; x < size; x++) {
          drawLine3d(ctx, { x, y, z: 0 }, { x, y, z: size - 1 }, size, scale, cx, cy, "rgba(111,139,148,ALPHA)", 0.28);
        }
      }
      for (let x = 0; x < size; x++) {
        for (let z = 0; z < size; z++) {
          drawLine3d(ctx, { x, y: 0, z }, { x, y: size - 1, z }, size, scale, cx, cy, "rgba(111,139,148,ALPHA)", 0.18);
        }
      }

      const arrows = [];
      for (const p of points) {
        const q = bestNeighbor(p, heatmap, size);
        if (!q) continue;
        const delta = q.value - p.value;
        if (delta > maxAbs * 0.035) {
          arrows.push({ from: p, to: q, delta, depth: project3d(p, size, scale, cx, cy).depth });
        }
      }
      arrows
        .sort((a, b) => a.depth - b.depth)
        .forEach(arrow => drawArrow3d(ctx, arrow.from, arrow.to, size, scale, cx, cy, Math.min(0.86, 0.28 + arrow.delta / maxAbs)));

      const sorted = points
        .map(p => ({ ...p, screen: project3d(p, size, scale, cx, cy) }))
        .sort((a, b) => a.screen.depth - b.screen.depth);
      const best = sorted.reduce((acc, p) => p.value > acc.value ? p : acc, sorted[0]);

      for (const p of sorted) {
        const norm = p.value / maxAbs;
        const radius = 10 + 22 * Math.abs(norm);
        ctx.beginPath();
        ctx.arc(p.screen.x, p.screen.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = valueColor(norm, 0.86);
        ctx.fill();
        ctx.lineWidth = p === best ? 4 : 1.4;
        ctx.strokeStyle = p === best ? "#f5c15d" : "rgba(230,236,239,.45)";
        ctx.stroke();
      }

      if (best) {
        ctx.fillStyle = "#e6ecef";
        ctx.font = "16px Inter, sans-serif";
        ctx.fillText(`best move: (${best.x}, ${best.y}, ${best.z}) value=${best.value.toFixed(3)}  arrows=local greedy value gradient`, 28, 36);
        $("board3dLabel").textContent = `${latestFor3d.method || "run"} · ${latestFor3d.run_id || latestFor3d.run_dir || "loaded run"}`;
      }

      const axis = [
        ["x", { x: 0, y: size - 1, z: size - 1 }, { x: size - 1, y: size - 1, z: size - 1 }, "#67d2a7"],
        ["y", { x: 0, y: 0, z: size - 1 }, { x: 0, y: size - 1, z: size - 1 }, "#f5c15d"],
        ["z", { x: 0, y: size - 1, z: 0 }, { x: 0, y: size - 1, z: size - 1 }, "#7aa7ff"],
      ];
      ctx.lineWidth = 3;
      axis.forEach(([label, a, b, stroke]) => {
        const pb = project3d(b, size, scale, cx, cy);
        drawLine3d(ctx, a, b, size, scale, cx, cy, stroke.replace(")", ",ALPHA)").replace("rgb", "rgba"), 1);
        ctx.fillStyle = stroke;
        ctx.font = "18px Inter, sans-serif";
        ctx.fillText(label, pb.x + 8, pb.y - 8);
      });
    }

    function drawAxes(ctx, panel, title, yLabel, yMin, yMax) {
      ctx.fillStyle = "#151c21";
      ctx.fillRect(panel.x, panel.y, panel.w, panel.h);
      ctx.strokeStyle = "#31434a";
      ctx.lineWidth = 1;
      ctx.strokeRect(panel.x, panel.y, panel.w, panel.h);
      ctx.fillStyle = "#dce7e9";
      ctx.font = "15px Inter, sans-serif";
      ctx.fillText(title, panel.x + 10, panel.y + 22);
      ctx.fillStyle = "#97a8ae";
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText(yLabel, panel.x + 10, panel.y + 42);
      ctx.strokeStyle = "#26343a";
      ctx.fillStyle = "#97a8ae";
      for (let i = 0; i <= 4; i++) {
        const frac = i / 4;
        const y = panel.y + panel.h - frac * panel.h;
        const value = yMin + frac * (yMax - yMin);
        ctx.beginPath();
        ctx.moveTo(panel.x, y);
        ctx.lineTo(panel.x + panel.w, y);
        ctx.stroke();
        ctx.fillText(value.toFixed(2), panel.x - 42, y + 4);
      }
    }

    function drawSeries(ctx, panel, history, key, color, yMin, yMax, minX, maxX, transform = x => x) {
      const points = history
        .map(d => ({ episode: d.episode, value: transform(d) }))
        .filter(d => Number.isFinite(d.value));
      if (points.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      points.forEach((d, i) => {
        const x = panel.x + ((d.episode - minX) / Math.max(1, maxX - minX)) * panel.w;
        const y = panel.y + panel.h - ((d.value - yMin) / Math.max(1e-9, yMax - yMin)) * panel.h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const last = points[points.length - 1];
      const lx = panel.x + ((last.episode - minX) / Math.max(1, maxX - minX)) * panel.w;
      const ly = panel.y + panel.h - ((last.value - yMin) / Math.max(1e-9, yMax - yMin)) * panel.h;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(lx, ly, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawLegend(ctx, x, y, items) {
      ctx.font = "13px Inter, sans-serif";
      items.forEach((item, i) => {
        const dx = x + i * 132;
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(dx, y);
        ctx.lineTo(dx + 22, y);
        ctx.stroke();
        ctx.fillStyle = "#dce7e9";
        ctx.fillText(item.label, dx + 30, y + 4);
      });
    }

    function drawChart(history) {
      const canvas = $("chart");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#101519";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#e6ecef";
      ctx.font = "18px Inter, sans-serif";
      ctx.fillText("Real-time training diagnostics", 58, 26);
      if (!history || history.length < 2) {
        ctx.fillStyle = "#97a8ae";
        ctx.font = "15px Inter, sans-serif";
        ctx.fillText("Start or load a run to plot win rates, losses, entropy, and KL.", 58, 58);
        return;
      }
      const xs = history.map(d => d.episode);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const panels = [
        { x: 72, y: 48, w: w - 104, h: 132 },
        { x: 72, y: 218, w: w - 104, h: 116 },
        { x: 72, y: 386, w: w - 104, h: 116 },
      ];
      const losses = history.map(d => Math.abs(Number(d.mean_abs_update || d.loss || 0))).filter(Number.isFinite);
      const maxLoss = Math.max(0.05, ...losses);
      const ent = history.map(d => Number(d.entropy || 0)).filter(Number.isFinite);
      const kl = history.map(d => Number(d.approx_kl || 0)).filter(Number.isFinite);
      const maxAux = Math.max(0.05, ...ent, ...kl);

      drawAxes(ctx, panels[0], "Outcome rates", "rate", 0, 1);
      drawSeries(ctx, panels[0], history, "x_win_rate", "#67d2a7", 0, 1, minX, maxX, d => d.recent?.x_win_rate);
      drawSeries(ctx, panels[0], history, "o_win_rate", "#f07c6b", 0, 1, minX, maxX, d => d.recent?.o_win_rate);
      drawSeries(ctx, panels[0], history, "draw_rate", "#7aa7ff", 0, 1, minX, maxX, d => d.recent?.draw_rate);
      drawLegend(ctx, panels[0].x + 148, panels[0].y + 21, [
        { label: "X win", color: "#67d2a7" },
        { label: "O win", color: "#f07c6b" },
        { label: "draw", color: "#7aa7ff" },
      ]);

      drawAxes(ctx, panels[1], "Optimization", "|update| / loss", 0, maxLoss);
      drawSeries(ctx, panels[1], history, "mean_abs_update", "#f5c15d", 0, maxLoss, minX, maxX, d => Math.abs(Number(d.mean_abs_update || 0)));

      drawAxes(ctx, panels[2], "Policy statistics", "entropy / KL", 0, maxAux);
      drawSeries(ctx, panels[2], history, "entropy", "#b58cff", 0, maxAux, minX, maxX, d => Number(d.entropy || 0));
      drawSeries(ctx, panels[2], history, "approx_kl", "#5cc8ff", 0, maxAux, minX, maxX, d => Number(d.approx_kl || 0));
      drawLegend(ctx, panels[2].x + 148, panels[2].y + 21, [
        { label: "entropy", color: "#b58cff" },
        { label: "approx KL", color: "#5cc8ff" },
      ]);

      ctx.fillStyle = "#97a8ae";
      ctx.font = "13px Inter, sans-serif";
      ctx.fillText(`episode ${minX} to ${maxX}`, panels[2].x + panels[2].w - 132, h - 18);
    }

    async function refresh() {
      const res = await fetch("/api/state");
      const state = await res.json();
      const latest = state.latest;
      $("status").textContent = state.running ? "running" : "idle";
      $("startBtn").disabled = state.running;
      $("stopBtn").disabled = !state.running;
      if (!latest) return;
      $("runDir").textContent = latest.run_dir || "run";
      $("episode").textContent = `${latest.method} ${latest.episode}/${latest.episodes}`;
      $("states").textContent = latest.states;
      $("eps").textContent = latest.epsilon;
      $("xwin").textContent = pct(latest.recent.x_win_rate);
      $("owin").textContent = pct(latest.recent.o_win_rate);
      $("draw").textContent = pct(latest.recent.draw_rate);
      renderHeatmap(latest.heatmap);
      renderArtifacts(latest);
      renderBoard3d(latest);
      drawChart(state.history || []);
    }

    async function refreshRuns() {
      const res = await fetch("/api/runs");
      const data = await res.json();
      const select = $("runSelect");
      const current = select.value;
      select.innerHTML = "";
      (data.runs || []).forEach((run) => {
        const option = document.createElement("option");
        option.value = run.run_dir;
        option.textContent = `${run.method || "run"} · ${run.run_id || run.run_dir} · ${run.episode || 0} ep`;
        select.appendChild(option);
      });
      if (current) select.value = current;
    }

    $("startBtn").addEventListener("click", async () => {
      const payload = {};
      fields.forEach((field) => payload[field] = Number($(field).value));
      payload.method = $("method").value;
      await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      refresh();
    });

    $("stopBtn").addEventListener("click", async () => {
      await fetch("/api/stop", { method: "POST" });
      refresh();
    });

    $("loadRunBtn").addEventListener("click", async () => {
      const runDir = $("runSelect").value;
      if (!runDir) return;
      const res = await fetch(`/api/run?run_dir=${encodeURIComponent(runDir)}`);
      const data = await res.json();
      $("runDir").textContent = data.latest.run_dir || "run";
      $("episode").textContent = `${data.latest.method} ${data.latest.episode}/${data.latest.episodes}`;
      $("states").textContent = data.latest.states;
      $("eps").textContent = data.latest.epsilon;
      $("xwin").textContent = pct(data.latest.recent.x_win_rate);
      $("owin").textContent = pct(data.latest.recent.o_win_rate);
      $("draw").textContent = pct(data.latest.recent.draw_rate);
      renderHeatmap(data.latest.heatmap);
      renderArtifacts(data.latest);
      renderBoard3d(data.latest);
      drawChart(data.history || []);
    });

    $("resetViewBtn").addEventListener("click", () => {
      boardYaw = -0.72;
      boardPitch = 0.72;
      renderBoard3d(latestFor3d);
    });

    $("board3d").addEventListener("pointerdown", (event) => {
      dragging3d = true;
      lastPointer = { x: event.clientX, y: event.clientY };
      $("board3d").setPointerCapture(event.pointerId);
    });
    $("board3d").addEventListener("pointermove", (event) => {
      if (!dragging3d || !lastPointer) return;
      const dx = event.clientX - lastPointer.x;
      const dy = event.clientY - lastPointer.y;
      boardYaw += dx * 0.01;
      boardPitch = Math.max(0.15, Math.min(1.35, boardPitch + dy * 0.01));
      lastPointer = { x: event.clientX, y: event.clientY };
      renderBoard3d(latestFor3d);
    });
    $("board3d").addEventListener("pointerup", () => {
      dragging3d = false;
      lastPointer = null;
    });

    refresh();
    refreshRuns();
    setInterval(refresh, 1000);
    setInterval(refreshRuns, 5000);
  </script>
</body>
</html>
"""


def _trim_history() -> None:
    del _history[:-300]


def _on_snapshot(payload: dict[str, Any]) -> None:
    with _lock:
        global _latest
        _latest = payload
        _history.append(payload)
        _trim_history()


def _worker(cfg: TabularConfig | DeepRLConfig, stop_event: threading.Event) -> None:
    try:
        if cfg.method in {"ppo", "grpo"}:
            train_deep_rl(cfg, callback=_on_snapshot, stop_event=stop_event)
        else:
            train_tabular(cfg, callback=_on_snapshot, stop_event=stop_event)
    finally:
        with _lock:
            global _thread
            _thread = None
            if _latest is not None:
                _latest["running"] = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/state")
def state() -> JSONResponse:
    with _lock:
        return JSONResponse(
            {
                "running": _thread is not None and _thread.is_alive(),
                "latest": _latest,
                "history": _history[-300:],
            }
        )


@app.post("/api/start")
async def start(request: Request) -> JSONResponse:
    payload = await request.json()
    with _lock:
        global _thread, _stop_event, _latest, _history
        if _thread is not None and _thread.is_alive():
            raise HTTPException(status_code=409, detail="run already active")

        method = str(payload.get("method", "q_learning"))
        if method in {"ppo", "grpo"}:
            cfg = DeepRLConfig(
                method=method,
                size=int(payload.get("size", 3)),
                episodes=int(payload.get("episodes", 2_000)),
                batch_episodes=int(payload.get("batch_episodes", 32)),
                gamma=float(payload.get("gamma", 0.99)),
                seed=int(payload.get("seed", 0)),
                log_every=int(payload.get("log_every", 100)),
            )
        else:
            cfg = TabularConfig(
                method=method,
                size=int(payload.get("size", 3)),
                episodes=int(payload.get("episodes", 10_000)),
                alpha=float(payload.get("alpha", 0.25)),
                gamma=float(payload.get("gamma", 0.98)),
                epsilon=float(payload.get("epsilon", 0.35)),
                seed=int(payload.get("seed", 0)),
                log_every=int(payload.get("log_every", 100)),
            )
        _latest = None
        _history = []
        _stop_event = threading.Event()
        _thread = threading.Thread(target=_worker, args=(cfg, _stop_event), daemon=True)
        _thread.start()
    return JSONResponse({"ok": True})


@app.post("/api/stop")
def stop() -> JSONResponse:
    with _lock:
        if _stop_event is not None:
            _stop_event.set()
    return JSONResponse({"ok": True})


@app.get("/api/runs")
def runs() -> JSONResponse:
    items = []
    for latest_path in Path("runs").rglob("latest.json"):
        try:
            latest = json.loads(latest_path.read_text())
            metadata_path = latest_path.parent / "metadata.json"
            if metadata_path.exists():
                latest["metadata"] = json.loads(metadata_path.read_text())
                latest["created_at"] = latest["metadata"].get("created_at")
            items.append(latest)
        except json.JSONDecodeError:
            continue
    items.sort(key=lambda item: item.get("created_at") or item.get("run_dir", ""), reverse=True)
    return JSONResponse({"runs": items})


@app.get("/api/run")
def run(run_dir: str) -> JSONResponse:
    path = _safe_run_dir(run_dir)
    latest_path = path / "latest.json"
    if not latest_path.exists():
        raise HTTPException(status_code=404, detail="latest.json not found")
    latest = json.loads(latest_path.read_text())
    history = []
    metrics_path = path / "metrics.jsonl"
    if metrics_path.exists():
        history = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    return JSONResponse({"latest": latest, "history": history[-300:]})


@app.get("/api/artifact")
def artifact(run_dir: str, file: str) -> FileResponse:
    path = _safe_run_dir(run_dir)
    allowed = {
        "curves.png",
        "first_move_heatmap.png",
        "curves.svg",
        "analysis.md",
        "analysis.json",
        "first_move_policy.json",
    }
    if file not in allowed:
        raise HTTPException(status_code=400, detail="artifact not allowed")
    artifact_path = path / file
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(artifact_path)
