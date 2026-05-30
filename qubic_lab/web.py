from __future__ import annotations

import json
import random
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from qubic_lab.artifacts import load_metrics, write_plot_artifacts
from qubic_lab.game import State, apply_move, legal_moves, terminal
from qubic_lab.model_api import (
    analyze_position,
    layers_to_state,
    list_models,
    load_model,
    play_game,
    run_tournament,
)
from qubic_lab.neural import PolicyValueNet
from qubic_lab.rl_deep import (
    DeepRLConfig,
    _snapshot as deep_snapshot,
    _write_analysis as write_deep_analysis,
    play_episode,
    train_deep_rl,
    update_policy,
)
from qubic_lab.rl_tabular import (
    QTable,
    TabularConfig,
    choose_action,
    expected_policy_value,
    get_q,
    save_q_table,
    snapshot_payload,
    state_key,
    train_tabular,
    write_analysis,
    write_curves_svg,
)
from qubic_lab.runlog import append_jsonl, run_id, write_json, write_metadata
from qubic_lab.runs import resolve_run_dir
from qubic_lab.selfplay import SelfPlayConfig, generate_selfplay_dataset

app = FastAPI(title="Qubic Lab")

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None
_latest: dict[str, Any] | None = None
_history: list[dict[str, Any]] = []
_step_lock = threading.Lock()
_step_session: "TabularStepSession | DeepStepSession | None" = None

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
        "opponent_mix": "self:0.4,tactical:0.4,random:0.2",
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
        "opponent_mix": "self:0.4,tactical:0.4,random:0.2",
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
            opponent_mix=str(payload.get("opponent_mix", defaults["opponent_mix"])),
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


def _sample_history(history: list[dict[str, Any]], limit: int = 1600) -> list[dict[str, Any]]:
    if len(history) <= limit:
        return history
    last = len(history) - 1
    indexes = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    return [history[index] for index in indexes]


def _model_id_for_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _enrich_latest(path: Path, latest: dict[str, Any]) -> dict[str, Any]:
    payload = dict(latest)
    has_model = (path / "model.pt").exists() or (path / "q_table.npz").exists()
    raw_value = payload.get("value")
    needs_value = raw_value is None or (has_model and isinstance(raw_value, (int, float)) and raw_value == 0)
    needs_top = not payload.get("top_moves")
    if not needs_value and not needs_top:
        return payload

    size = int(payload.get("config", {}).get("size", 3))
    try:
        if has_model:
            analysis = analyze_position(_model_id_for_path(path), State.new(size))
            if needs_value:
                payload["value"] = analysis.get("value")
            if needs_top:
                payload["top_moves"] = analysis.get("top_moves", [])
            return payload
    except Exception:
        pass

    heatmap = payload.get("heatmap")
    if needs_top and heatmap:
        moves = []
        for z, layer in enumerate(heatmap):
            for y, row in enumerate(layer):
                for x, value in enumerate(row):
                    move = z * size * size + y * size + x
                    moves.append(
                        {
                            "move": move,
                            "x": x,
                            "y": y,
                            "z": z,
                            "prob": float(value),
                            "value": float(value),
                        }
                    )
        payload["top_moves"] = sorted(moves, key=lambda item: abs(item["value"]), reverse=True)[:10]
    if needs_value:
        payload["value"] = 0.0
    return payload


def _state_hash(state: State) -> str:
    board = ",".join(str(int(value)) for value in state.board.reshape(-1).tolist())
    return f"{state.player}:{board}"


def _sample_state_space(
    model_id: str,
    *,
    size: int,
    games: int,
    seed: int,
    greedy: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    model = load_model(model_id)
    nodes: dict[str, dict[str, Any]] = {}
    outcomes = {1: 0, -1: 0, 0: 0}
    game_count = max(1, games)

    for game_idx in range(game_count):
        state = State.new(size)
        trajectory: list[tuple[str, int, float, float, int]] = []
        while True:
            probs, value = model.policy_value(state)
            moves = legal_moves(state).astype(int).tolist()
            move_probs = np.array([float(probs[move]) for move in moves], dtype=np.float64)
            total = float(move_probs.sum())
            if total > 0:
                normalized = move_probs / total
                entropy = float(-(normalized * np.log(np.clip(normalized, 1e-12, 1.0))).sum())
            else:
                entropy = 0.0

            key = _state_hash(state)
            trajectory.append((key, len(trajectory), float(value), entropy, int(state.player)))
            move = model.choose_move(state, rng, greedy=greedy)
            state = apply_move(state, move)
            done, winner = terminal(state)
            if done:
                outcome = int(winner or 0)
                outcomes[outcome] += 1
                for key, ply, value, entropy, player in trajectory:
                    signed_return = 0.0 if outcome == 0 else (1.0 if outcome == player else -1.0)
                    node = nodes.setdefault(
                        key,
                        {
                            "id": len(nodes),
                            "ply": ply,
                            "player": player,
                            "visits": 0,
                            "value_sum": 0.0,
                            "return_sum": 0.0,
                            "entropy_sum": 0.0,
                            "wins": 0,
                            "losses": 0,
                            "draws": 0,
                        },
                    )
                    node["visits"] += 1
                    node["value_sum"] += value
                    node["return_sum"] += signed_return
                    node["entropy_sum"] += entropy
                    if signed_return > 0:
                        node["wins"] += 1
                    elif signed_return < 0:
                        node["losses"] += 1
                    else:
                        node["draws"] += 1
                break

    items = []
    for node in nodes.values():
        visits = max(1, int(node["visits"]))
        items.append(
            {
                "id": node["id"],
                "ply": node["ply"],
                "player": node["player"],
                "visits": visits,
                "value": node["value_sum"] / visits,
                "return": node["return_sum"] / visits,
                "entropy": node["entropy_sum"] / visits,
                "wins": node["wins"],
                "losses": node["losses"],
                "draws": node["draws"],
            }
        )
    items.sort(key=lambda item: (item["ply"], -item["visits"], item["id"]))
    return {
        "model_id": model_id,
        "size": size,
        "games": game_count,
        "greedy": greedy,
        "nodes": _sample_history(items, 1200),
        "total_nodes": len(items),
        "outcomes": {
            "x_win_rate": outcomes[1] / game_count,
            "o_win_rate": outcomes[-1] / game_count,
            "draw_rate": outcomes[0] / game_count,
        },
    }


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
    latest = _enrich_latest(path, json.loads(latest_path.read_text())) if latest_path.exists() else None
    metrics = _sample_history(_read_jsonl(path / "metrics.jsonl"))
    snapshots = []
    for item in metrics:
        snapshots.append(
            {
                "episode": item.get("episode"),
                "method": item.get("method"),
                "heatmap": item.get("heatmap"),
                "top_moves": item.get("top_moves"),
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


def _step_signature(cfg: TabularConfig | DeepRLConfig) -> str:
    payload = asdict(cfg)
    payload.pop("run_dir", None)
    return json.dumps(payload, sort_keys=True)


class TabularStepSession:
    def __init__(self, cfg: TabularConfig):
        method = cfg.method.strip().lower().replace("-", "_")
        self.cfg = TabularConfig(**{**asdict(cfg), "method": method})
        self.signature = _step_signature(self.cfg)
        self.run_dir = resolve_run_dir(self.cfg.run_dir)
        self.rng = random.Random(self.cfg.seed)
        self.q: QTable = {}
        self.outcomes: list[int] = []
        self.update_magnitudes: list[float] = []
        self.epsilon = self.cfg.epsilon
        self.episode = 0
        self.latest: dict[str, Any] | None = None

        write_json(self.run_dir / "config.json", asdict(self.cfg))
        write_metadata(
            self.run_dir,
            self.cfg,
            method=self.cfg.method,
            name=self.cfg.name,
            parent_run=self.cfg.parent_run,
        )

    def step(self) -> dict[str, Any]:
        if self.latest is not None and self.episode >= self.cfg.episodes:
            return self.latest

        self.episode += 1
        state = State.new(self.cfg.size)
        moves_played = 0
        episode_updates: list[float] = []

        if self.cfg.method == "monte_carlo":
            trajectory: list[tuple[tuple[int, ...], int, int, int]] = []
            while True:
                key = state_key(state)
                move = choose_action(self.q, state, self.rng, self.epsilon)
                trajectory.append((key, move, state.player, moves_played))
                state = apply_move(state, move)
                moves_played += 1
                done, winner = terminal(state)
                if done:
                    self.outcomes.append(int(winner or 0))
                    horizon = max(1, moves_played - 1)
                    for key_i, move_i, player_i, step_i in trajectory:
                        row = get_q(self.q, key_i, self.cfg.size**3)
                        if winner == 0:
                            target = 0.0
                        else:
                            sign = 1.0 if player_i == winner else -1.0
                            target = sign * (self.cfg.gamma ** (horizon - step_i))
                        delta = target - float(row[move_i])
                        row[move_i] += self.cfg.alpha * delta
                        episode_updates.append(abs(delta))
                    break
        else:
            while True:
                key = state_key(state)
                row = get_q(self.q, key, self.cfg.size**3)
                move = choose_action(self.q, state, self.rng, self.epsilon)
                next_state = apply_move(state, move)
                moves_played += 1
                done, winner = terminal(next_state)

                if done:
                    reward = 0.0 if winner == 0 else 1.0
                    target = reward
                    delta = target - float(row[move])
                    row[move] += self.cfg.alpha * delta
                    episode_updates.append(abs(delta))
                    self.outcomes.append(int(winner or 0))
                    break

                next_row = get_q(self.q, state_key(next_state), self.cfg.size**3)
                next_moves = legal_moves(next_state)
                if self.cfg.method == "q_learning":
                    opponent_best = float(np.max(next_row[next_moves])) if len(next_moves) else 0.0
                elif self.cfg.method == "sarsa":
                    next_move = choose_action(self.q, next_state, self.rng, self.epsilon)
                    opponent_best = float(next_row[next_move])
                else:
                    opponent_best = expected_policy_value(next_row, next_moves, self.epsilon)
                target = -self.cfg.gamma * opponent_best
                delta = target - float(row[move])
                row[move] += self.cfg.alpha * delta
                episode_updates.append(abs(delta))
                state = next_state

        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)
        self.update_magnitudes.extend(episode_updates)
        mean_abs_update = float(np.mean(episode_updates)) if episode_updates else 0.0
        latest = snapshot_payload(
            cfg=self.cfg,
            q=self.q,
            episode=self.episode,
            epsilon=self.epsilon,
            outcomes=self.outcomes,
            run_dir=self.run_dir,
            running=self.episode < self.cfg.episodes,
            mean_abs_update=mean_abs_update,
        )
        latest["last_episode_moves"] = moves_played
        latest["step_mode"] = True
        self.latest = latest
        self._persist(latest)
        return latest

    def _persist(self, latest: dict[str, Any]) -> None:
        append_jsonl(self.run_dir / "metrics.jsonl", latest)
        write_json(self.run_dir / "latest.json", latest)
        save_q_table(self.run_dir / "q_table.npz", self.q)
        metrics = load_metrics(self.run_dir / "metrics.jsonl")
        write_curves_svg(self.run_dir / "curves.svg", metrics)
        write_plot_artifacts(self.run_dir, metrics, latest)
        write_analysis(self.run_dir, self.cfg, metrics, latest)
        write_json(
            self.run_dir / "artifacts.json",
            {
                "run_id": run_id(self.run_dir),
                "files": {
                    "config": "config.json",
                    "metadata": "metadata.json",
                    "metrics": "metrics.jsonl",
                    "latest": "latest.json",
                    "analysis": "analysis.json",
                    "analysis_markdown": "analysis.md",
                    "curves": "curves.svg",
                    "curves_png": "curves.png",
                    "first_move_heatmap": "first_move_heatmap.png",
                    "first_move_policy": "first_move_policy.json",
                    "q_table": "q_table.npz",
                },
            },
        )


class DeepStepSession:
    def __init__(self, cfg: DeepRLConfig):
        method = cfg.method.strip().lower().replace("-", "_")
        self.cfg = DeepRLConfig(**{**asdict(cfg), "method": method})
        self.signature = _step_signature(self.cfg)
        torch.manual_seed(self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.run_dir = resolve_run_dir(self.cfg.run_dir)
        self.model = PolicyValueNet(self.cfg.size, self.cfg.hidden).to(self.cfg.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        self.outcomes: list[int] = []
        self.episode = 0
        self.latest: dict[str, Any] | None = None
        self.losses = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }

        write_json(self.run_dir / "config.json", asdict(self.cfg))
        write_metadata(
            self.run_dir,
            self.cfg,
            method=self.cfg.method,
            name=self.cfg.name,
            parent_run=self.cfg.parent_run,
        )

    def step(self) -> dict[str, Any]:
        if self.latest is not None and self.episode >= self.cfg.episodes:
            return self.latest

        batch_steps = []
        group_ids: list[int] = []
        for group in range(self.cfg.batch_episodes):
            if self.episode >= self.cfg.episodes:
                break
            steps, outcome = play_episode(self.model, self.cfg.size, self.rng, self.cfg)
            batch_steps.extend(steps)
            group_ids.extend([group] * len(steps))
            self.outcomes.append(outcome)
            self.episode += 1

        if batch_steps:
            self.losses = update_policy(self.model, self.optimizer, batch_steps, np.asarray(group_ids), self.cfg)

        latest = deep_snapshot(
            self.cfg,
            self.model,
            self.run_dir,
            self.episode,
            self.outcomes,
            self.losses,
            running=self.episode < self.cfg.episodes,
        )
        latest["step_mode"] = True
        self.latest = latest
        self._persist(latest)
        return latest

    def _persist(self, latest: dict[str, Any]) -> None:
        append_jsonl(self.run_dir / "metrics.jsonl", latest)
        write_json(self.run_dir / "latest.json", latest)
        torch.save({"model": self.model.state_dict(), "config": asdict(self.cfg)}, self.run_dir / "model.pt")
        metrics = load_metrics(self.run_dir / "metrics.jsonl")
        write_plot_artifacts(self.run_dir, metrics, latest)
        write_deep_analysis(self.run_dir, latest)
        write_json(
            self.run_dir / "artifacts.json",
            {
                "run_id": run_id(self.run_dir),
                "files": {
                    "config": "config.json",
                    "metadata": "metadata.json",
                    "metrics": "metrics.jsonl",
                    "latest": "latest.json",
                    "analysis": "analysis.json",
                    "analysis_markdown": "analysis.md",
                    "curves_png": "curves.png",
                    "first_move_heatmap": "first_move_heatmap.png",
                    "first_move_policy": "first_move_policy.json",
                    "model": "model.pt",
                },
            },
        )


def _make_step_session(cfg: TabularConfig | DeepRLConfig) -> TabularStepSession | DeepStepSession:
    if cfg.method in {"ppo", "grpo"}:
        return DeepStepSession(cfg)
    return TabularStepSession(cfg)


@app.get("/api/state")
def state() -> JSONResponse:
    with _lock:
        history = _history[-300:]
        if _latest and _latest.get("run_dir"):
            metrics = _read_jsonl(Path(str(_latest["run_dir"])) / "metrics.jsonl")
            if metrics:
                history = _sample_history(metrics)
            latest = _enrich_latest(Path(str(_latest["run_dir"])), _latest)
        else:
            latest = _latest
        return JSONResponse(
            {
                "running": _thread is not None and _thread.is_alive(),
                "latest": latest,
                "history": history,
                "step_mode": bool(latest and latest.get("step_mode")),
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


@app.post("/api/step")
async def step_run(request: Request) -> JSONResponse:
    payload = await request.json()
    cfg = _run_payload(payload)
    signature = _step_signature(cfg)

    with _step_lock:
        global _step_session, _latest, _history
        with _lock:
            if _thread is not None and _thread.is_alive():
                raise HTTPException(status_code=409, detail="stop the active run before using step mode")
        new_session = _step_session is None or _step_session.signature != signature
        if new_session:
            _step_session = _make_step_session(cfg)
            with _lock:
                _latest = None
                _history = []
        latest = _step_session.step()

    _on_snapshot(latest)
    run_dir = Path(str(latest.get("run_dir")))
    history = _sample_history(_read_jsonl(run_dir / "metrics.jsonl"))
    return JSONResponse(
        {
            "ok": True,
            "new_session": new_session,
            "latest": latest,
            "history": history,
            "run_dir": latest.get("run_dir"),
            "artifacts": _artifact_manifest(run_dir),
            "complete": not bool(latest.get("running")),
        }
    )


@app.post("/api/step/reset")
def reset_step_run() -> JSONResponse:
    with _step_lock:
        global _step_session, _latest, _history
        _step_session = None
        with _lock:
            _latest = None
            _history = []
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
    latest = _enrich_latest(path, json.loads(latest_path.read_text()))
    history = _sample_history(_read_jsonl(path / "metrics.jsonl"))
    return JSONResponse({"latest": latest, "history": history, "artifacts": _artifact_manifest(path)})


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


@app.post("/api/state-space/sample")
async def state_space_sample(request: Request) -> JSONResponse:
    payload = await request.json()
    return JSONResponse(
        _sample_state_space(
            str(payload.get("model_id", "random")),
            size=int(payload.get("size", 3)),
            games=max(1, min(500, int(payload.get("games", 80)))),
            seed=int(payload.get("seed", 0)),
            greedy=bool(payload.get("greedy", False)),
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
