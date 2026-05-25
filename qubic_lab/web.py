from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

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
    canvas { width: 100%; height: 220px; background: #101519; border: 1px solid var(--line); border-radius: 8px; }
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
        </select></label>
        <label>Size <input id="size" type="number" min="2" max="4" value="3"></label>
        <label>Episodes <input id="episodes" type="number" min="1" value="20000"></label>
        <label>Alpha <input id="alpha" type="number" min="0" max="1" step="0.01" value="0.25"></label>
        <label>Gamma <input id="gamma" type="number" min="0" max="1" step="0.01" value="0.98"></label>
        <label>Epsilon <input id="epsilon" type="number" min="0" max="1" step="0.01" value="0.35"></label>
        <label>Seed <input id="seed" type="number" value="0"></label>
        <label>Log every <input id="log_every" type="number" min="1" value="100"></label>
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
        <canvas id="chart" width="900" height="260"></canvas>
        <div class="artifacts" id="artifacts"></div>
        <div class="layers" id="layers"></div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const fields = ["size", "episodes", "alpha", "gamma", "epsilon", "seed", "log_every"];

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

    function drawChart(history) {
      const canvas = $("chart");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#101519";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#2c3a40";
      ctx.lineWidth = 1;
      for (let i = 1; i < 5; i++) {
        const y = 22 + i * ((h - 44) / 5);
        ctx.beginPath(); ctx.moveTo(36, y); ctx.lineTo(w - 16, y); ctx.stroke();
      }
      if (!history || history.length < 2) return;
      const xs = history.map(d => d.episode);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const series = [
        ["x_win_rate", "#67d2a7"],
        ["o_win_rate", "#f07c6b"],
        ["draw_rate", "#7aa7ff"],
      ];
      for (const [key, stroke] of series) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 3;
        ctx.beginPath();
        history.forEach((d, i) => {
          const x = 36 + ((d.episode - minX) / Math.max(1, maxX - minX)) * (w - 56);
          const y = h - 24 - (d.recent[key] || 0) * (h - 48);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
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
      drawChart(data.history || []);
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


def _worker(cfg: TabularConfig, stop_event: threading.Event) -> None:
    try:
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

        cfg = TabularConfig(
            method=str(payload.get("method", "q_learning")),
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
