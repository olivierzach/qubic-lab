from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from qubic_lab.model_api import (
    analyze_position,
    layers_to_state,
    list_models,
    play_game,
    run_tournament,
)
from qubic_lab.rl_deep import DeepRLConfig, train_deep_rl
from qubic_lab.rl_tabular import TabularConfig, train_tabular
from qubic_lab.selfplay import SelfPlayConfig, generate_selfplay_dataset

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


def _empty_layers(size: int) -> list[list[list[int]]]:
    return [[[0 for _ in range(size)] for _ in range(size)] for _ in range(size)]


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
    return JSONResponse({"latest": latest, "history": history[-500:]})


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
        "model.pt",
    }
    if file not in allowed:
        raise HTTPException(status_code=400, detail="artifact not allowed")
    artifact_path = path / file
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(artifact_path)


@app.get("/api/models")
def models() -> JSONResponse:
    return JSONResponse({"models": [record.__dict__ for record in list_models()]})


@app.post("/api/analyze/position")
async def analyze(request: Request) -> JSONResponse:
    payload = await request.json()
    model_id = str(payload.get("model_id", "random"))
    board = payload.get("board") or _empty_layers(int(payload.get("size", 3)))
    player = int(payload.get("player", 1))
    state_obj = layers_to_state(board, player)
    return JSONResponse(analyze_position(model_id, state_obj))


@app.post("/api/play/new")
async def play_new(request: Request) -> JSONResponse:
    payload = await request.json()
    model_id = str(payload.get("model_id", "random"))
    size = int(payload.get("size", 3))
    human_player = int(payload.get("human_player", 1))
    return JSONResponse(play_game(model_id, size, human_player, []))


@app.post("/api/play/move")
async def play_move(request: Request) -> JSONResponse:
    payload = await request.json()
    try:
        return JSONResponse(
            play_game(
                str(payload.get("model_id", "random")),
                int(payload.get("size", 3)),
                int(payload.get("human_player", 1)),
                [int(move) for move in payload.get("moves", [])],
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/eval/tournament")
async def tournament(request: Request) -> JSONResponse:
    payload = await request.json()
    model_ids = [str(model_id) for model_id in payload.get("model_ids", [])]
    return JSONResponse(
        run_tournament(
            model_ids,
            size=int(payload.get("size", 3)),
            games=int(payload.get("games", 10)),
            seed=int(payload.get("seed", 0)),
        )
    )


@app.post("/api/selfplay/generate")
async def selfplay_generate(request: Request) -> JSONResponse:
    payload = await request.json()
    run_dir = generate_selfplay_dataset(
        SelfPlayConfig(
            model_id=str(payload.get("model_id", "random")),
            opponent_id=payload.get("opponent_id"),
            size=int(payload.get("size", 3)),
            games=int(payload.get("games", 100)),
            seed=int(payload.get("seed", 0)),
            greedy=bool(payload.get("greedy", False)),
        )
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return JSONResponse(manifest)


WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


@app.get("/{path:path}", response_class=HTMLResponse)
def spa(path: str = ""):
    index = WEB_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse(
        """
        <html>
          <body style="font-family: system-ui; background: #101418; color: #e6ecef;">
            <main style="max-width: 760px; margin: 80px auto;">
              <h1>Qubic Lab API is running</h1>
              <p>Build the React dashboard with <code>cd web && npm install && npm run build</code>.</p>
            </main>
          </body>
        </html>
        """
    )
