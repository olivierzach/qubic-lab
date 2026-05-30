from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

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

RUN_DEFAULTS: dict[str, dict[str, Any]] = {
    "ppo": {
        "method": "ppo",
        "size": 3,
        "episodes": 5_000,
        "batch_episodes": 64,
        "update_epochs": 4,
        "hidden": 128,
        "lr": 3e-4,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "entropy_coef": 0.02,
        "value_coef": 0.5,
        "max_grad_norm": 1.0,
        "temperature": 1.0,
        "seed": 0,
        "log_every": 100,
        "device": "cpu",
    },
    "grpo": {
        "method": "grpo",
        "size": 3,
        "episodes": 5_000,
        "batch_episodes": 64,
        "update_epochs": 4,
        "hidden": 128,
        "lr": 3e-4,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "entropy_coef": 0.02,
        "value_coef": 0.5,
        "max_grad_norm": 1.0,
        "temperature": 1.0,
        "seed": 0,
        "log_every": 100,
        "device": "cpu",
    },
    "q_learning": {
        "method": "q_learning",
        "size": 3,
        "episodes": 10_000,
        "alpha": 0.25,
        "gamma": 0.98,
        "epsilon": 0.35,
        "epsilon_min": 0.03,
        "epsilon_decay": 0.9995,
        "seed": 0,
        "log_every": 100,
    },
    "sarsa": {
        "method": "sarsa",
        "size": 3,
        "episodes": 10_000,
        "alpha": 0.25,
        "gamma": 0.98,
        "epsilon": 0.35,
        "epsilon_min": 0.03,
        "epsilon_decay": 0.9995,
        "seed": 0,
        "log_every": 100,
    },
    "expected_sarsa": {
        "method": "expected_sarsa",
        "size": 3,
        "episodes": 10_000,
        "alpha": 0.25,
        "gamma": 0.98,
        "epsilon": 0.35,
        "epsilon_min": 0.03,
        "epsilon_decay": 0.9995,
        "seed": 0,
        "log_every": 100,
    },
    "monte_carlo": {
        "method": "monte_carlo",
        "size": 3,
        "episodes": 10_000,
        "alpha": 0.25,
        "gamma": 0.98,
        "epsilon": 0.35,
        "epsilon_min": 0.03,
        "epsilon_decay": 0.9995,
        "seed": 0,
        "log_every": 100,
    },
}


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


def _int_payload(payload: dict[str, Any], key: str, default: int, *, minimum: int, maximum: int) -> int:
    value = int(payload.get(key, default))
    if value < minimum or value > maximum:
        raise HTTPException(status_code=400, detail=f"{key} must be between {minimum} and {maximum}")
    return value


def _float_payload(
    payload: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = float(payload.get(key, default))
    if value < minimum or value > maximum:
        raise HTTPException(status_code=400, detail=f"{key} must be between {minimum:g} and {maximum:g}")
    return value


def _run_payload(payload: dict[str, Any]) -> TabularConfig | DeepRLConfig:
    method = str(payload.get("method", "q_learning")).strip().lower().replace("-", "_")
    if method not in RUN_DEFAULTS:
        raise HTTPException(status_code=400, detail=f"unknown method {method!r}")
    defaults = RUN_DEFAULTS[method]
    max_size = 5 if method in {"ppo", "grpo"} else 4
    common = {
        "method": method,
        "name": payload.get("name") or None,
        "parent_run": payload.get("parent_run") or None,
        "size": _int_payload(payload, "size", int(defaults["size"]), minimum=2, maximum=max_size),
        "episodes": _int_payload(
            payload,
            "episodes",
            int(defaults["episodes"]),
            minimum=1,
            maximum=2_000_000,
        ),
        "seed": _int_payload(payload, "seed", int(defaults["seed"]), minimum=0, maximum=2**31 - 1),
        "log_every": _int_payload(payload, "log_every", int(defaults["log_every"]), minimum=1, maximum=100_000),
    }
    if method in {"ppo", "grpo"}:
        return DeepRLConfig(
            **common,
            batch_episodes=_int_payload(payload, "batch_episodes", int(defaults["batch_episodes"]), minimum=1, maximum=4096),
            update_epochs=_int_payload(payload, "update_epochs", int(defaults["update_epochs"]), minimum=1, maximum=128),
            hidden=_int_payload(payload, "hidden", int(defaults["hidden"]), minimum=8, maximum=4096),
            lr=_float_payload(payload, "lr", float(defaults["lr"]), minimum=1e-7, maximum=1.0),
            gamma=_float_payload(payload, "gamma", float(defaults["gamma"]), minimum=0.0, maximum=1.0),
            clip_eps=_float_payload(payload, "clip_eps", float(defaults["clip_eps"]), minimum=0.01, maximum=1.0),
            entropy_coef=_float_payload(payload, "entropy_coef", float(defaults["entropy_coef"]), minimum=0.0, maximum=1.0),
            value_coef=_float_payload(payload, "value_coef", float(defaults["value_coef"]), minimum=0.0, maximum=10.0),
            max_grad_norm=_float_payload(payload, "max_grad_norm", float(defaults["max_grad_norm"]), minimum=0.01, maximum=100.0),
            temperature=_float_payload(payload, "temperature", float(defaults["temperature"]), minimum=0.01, maximum=10.0),
            device=str(payload.get("device", defaults["device"])),
        )
    return TabularConfig(
        **common,
        alpha=_float_payload(payload, "alpha", float(defaults["alpha"]), minimum=0.0, maximum=1.0),
        gamma=_float_payload(payload, "gamma", float(defaults["gamma"]), minimum=0.0, maximum=1.0),
        epsilon=_float_payload(payload, "epsilon", float(defaults["epsilon"]), minimum=0.0, maximum=1.0),
        epsilon_min=_float_payload(payload, "epsilon_min", float(defaults["epsilon_min"]), minimum=0.0, maximum=1.0),
        epsilon_decay=_float_payload(payload, "epsilon_decay", float(defaults["epsilon_decay"]), minimum=0.0, maximum=1.0),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


ARTIFACT_FILES = {
    "curves.png",
    "first_move_heatmap.png",
    "curves.svg",
    "analysis.md",
    "analysis.json",
    "first_move_policy.json",
    "model.pt",
    "q_table.npz",
    "config.json",
    "metadata.json",
    "metrics.jsonl",
    "latest.json",
    "artifacts.json",
}


def _artifact_manifest(path: Path) -> list[dict[str, Any]]:
    files = []
    declared_path = path / "artifacts.json"
    declared: dict[str, str] = {}
    if declared_path.exists():
        try:
            payload = json.loads(declared_path.read_text())
            declared = {str(label): str(file) for label, file in payload.get("files", {}).items()}
        except json.JSONDecodeError:
            declared = {}

    names = sorted(ARTIFACT_FILES | set(declared.values()))
    for name in names:
        artifact_path = path / name
        if not artifact_path.exists() or not artifact_path.is_file():
            continue
        labels = [label for label, file in declared.items() if file == name]
        files.append(
            {
                "file": name,
                "label": labels[0] if labels else name,
                "bytes": artifact_path.stat().st_size,
                "url": f"/api/artifact?run_dir={quote(path.as_posix(), safe='')}&file={quote(name, safe='')}",
            }
        )
    return files


def _timeline_for_run(path: Path) -> dict[str, Any]:
    latest_path = path / "latest.json"
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else None
    metrics = _read_jsonl(path / "metrics.jsonl")
    snapshots = []
    for item in metrics[-500:]:
        snapshots.append(
            {
                "episode": item.get("episode"),
                "method": item.get("method"),
                "heatmap": item.get("heatmap"),
                "recent": item.get("recent"),
                "value": item.get("value"),
                "policy_loss": item.get("policy_loss"),
                "value_loss": item.get("value_loss"),
                "entropy": item.get("entropy"),
                "approx_kl": item.get("approx_kl"),
                "mean_abs_update": item.get("mean_abs_update"),
            }
        )
    return {
        "run_dir": str(path),
        "latest": latest,
        "snapshots": snapshots,
        "config": latest.get("config", {}) if latest else {},
        "artifacts": _artifact_manifest(path),
    }


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
@app.post("/api/run")
async def start(request: Request) -> JSONResponse:
    payload = await request.json()
    with _lock:
        global _thread, _stop_event, _latest, _history
        if _thread is not None and _thread.is_alive():
            raise HTTPException(status_code=409, detail="run already active")

        cfg = _run_payload(payload)
        _latest = None
        _history = []
        _stop_event = threading.Event()
        _thread = threading.Thread(target=_worker, args=(cfg, _stop_event), daemon=True)
        _thread.start()
    return JSONResponse({"ok": True, "config": asdict(cfg)})


@app.get("/api/run/defaults")
def run_defaults() -> JSONResponse:
    return JSONResponse(
        {
            "methods": list(RUN_DEFAULTS),
            "defaults": RUN_DEFAULTS,
            "groups": {
                "tabular": ["q_learning", "sarsa", "expected_sarsa", "monte_carlo"],
                "policy_gradient": ["ppo", "grpo"],
            },
        }
    )


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
            latest["artifacts"] = _artifact_manifest(latest_path.parent)
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
    history = _read_jsonl(path / "metrics.jsonl")
    return JSONResponse({"latest": latest, "history": history[-500:], "artifacts": _artifact_manifest(path)})


@app.get("/api/model/timeline")
def model_timeline(model_id: str | None = None, run_dir: str | None = None) -> JSONResponse:
    target = run_dir or model_id
    if not target or target == "random":
        return JSONResponse({"run_dir": None, "latest": None, "snapshots": [], "config": {}})
    path = _safe_run_dir(target)
    return JSONResponse(_timeline_for_run(path))


@app.get("/api/artifact")
def artifact(run_dir: str, file: str) -> FileResponse:
    path = _safe_run_dir(run_dir)
    if file not in ARTIFACT_FILES:
        raise HTTPException(status_code=400, detail="artifact not allowed")
    artifact_path = path / file
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(artifact_path)


@app.get("/api/artifacts")
def artifacts(run_dir: str) -> JSONResponse:
    path = _safe_run_dir(run_dir)
    return JSONResponse({"run_dir": str(path), "artifacts": _artifact_manifest(path)})


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
