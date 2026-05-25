from __future__ import annotations

import json
import random
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from qubic_lab.artifacts import load_metrics, write_plot_artifacts
from qubic_lab.game import State, apply_move, flatten_board, idx_to_xyz, legal_moves, terminal
from qubic_lab.runs import resolve_run_dir


QTable = dict[tuple[int, ...], np.ndarray]
SnapshotCallback = Callable[[dict], None]


@dataclass(frozen=True)
class TabularConfig:
    method: str = "q_learning"
    name: str | None = None
    parent_run: str | None = None
    size: int = 3
    episodes: int = 10_000
    alpha: float = 0.25
    gamma: float = 0.98
    epsilon: float = 0.35
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.9995
    seed: int = 0
    log_every: int = 100
    run_dir: str | None = None


def state_key(state: State) -> tuple[int, ...]:
    """Encode from the current player's perspective so one table serves X and O."""
    return tuple((flatten_board(state.board) * state.player).astype(np.int8).tolist())


def get_q(q: QTable, key: tuple[int, ...], n_actions: int) -> np.ndarray:
    row = q.get(key)
    if row is None:
        row = np.zeros(n_actions, dtype=np.float32)
        q[key] = row
    return row


def choose_action(
    q: QTable,
    state: State,
    rng: random.Random,
    epsilon: float,
) -> int:
    moves = legal_moves(state).astype(int).tolist()
    if not moves:
        raise ValueError("cannot choose an action from a terminal full board")
    if rng.random() < epsilon:
        return rng.choice(moves)

    row = get_q(q, state_key(state), state.size**3)
    values = row[moves]
    best = float(np.max(values))
    best_moves = [move for move, value in zip(moves, values) if float(value) == best]
    return rng.choice(best_moves)


def expected_policy_value(row: np.ndarray, moves: np.ndarray, epsilon: float) -> float:
    if len(moves) == 0:
        return 0.0
    values = row[moves]
    best = float(np.max(values))
    greedy_count = int(np.sum(values == best))
    random_prob = epsilon / len(moves)
    total = 0.0
    for value in values:
        prob = random_prob
        if float(value) == best:
            prob += (1.0 - epsilon) / greedy_count
        total += prob * float(value)
    return total


def empty_board_heatmap(q: QTable, size: int) -> list[list[list[float]]]:
    empty = state_key(State.new(size))
    row = get_q(q, empty, size**3)
    layers = np.zeros((size, size, size), dtype=np.float32)
    for idx, value in enumerate(row):
        x, y, z = idx_to_xyz(idx, size)
        layers[z, y, x] = float(value)
    return layers.round(4).tolist()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _run_id(run_dir: Path) -> str:
    return run_dir.name


def _append_run_index(run_dir: Path, metadata: dict) -> None:
    index_path = run_dir.parent / "index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            index = {"runs": []}
    else:
        index = {"runs": []}
    index["runs"] = [run for run in index["runs"] if run.get("run_id") != metadata.get("run_id")]
    index["runs"].append(metadata)
    index["runs"].sort(key=lambda item: item.get("created_at", ""))
    _write_json(index_path, index)


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_curves_svg(path: Path, metrics: list[dict]) -> None:
    width, height = 1000, 520
    left, top, right, bottom = 62, 34, 24, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    if not metrics:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n")
        return

    min_ep = min(m["episode"] for m in metrics)
    max_ep = max(m["episode"] for m in metrics)

    def x_for(ep: int) -> float:
        return left + ((ep - min_ep) / max(1, max_ep - min_ep)) * plot_w

    def y_rate(value: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, value))) * plot_h

    losses = [float(m.get("mean_abs_update", 0.0)) for m in metrics]
    max_loss = max(losses) if losses else 1.0

    def y_loss(value: float) -> float:
        scaled = value / max(1e-9, max_loss)
        return top + (1.0 - max(0.0, min(1.0, scaled))) * plot_h

    series = [
        ("X win", "#67d2a7", [(x_for(m["episode"]), y_rate(m["recent"]["x_win_rate"])) for m in metrics]),
        ("O win", "#f07c6b", [(x_for(m["episode"]), y_rate(m["recent"]["o_win_rate"])) for m in metrics]),
        ("Draw", "#7aa7ff", [(x_for(m["episode"]), y_rate(m["recent"]["draw_rate"])) for m in metrics]),
        ("Update", "#f5c15d", [(x_for(m["episode"]), y_loss(float(m.get("mean_abs_update", 0.0)))) for m in metrics]),
    ]
    legend = []
    for i, (label, color, _) in enumerate(series):
        x = left + i * 140
        legend.append(
            f"<g><rect x='{x}' y='{height - 31}' width='18' height='5' fill='{color}'/>"
            f"<text x='{x + 25}' y='{height - 24}' fill='#dce7e9' font-size='15'>{label}</text></g>"
        )
    lines = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='520' viewBox='0 0 1000 520'>",
        "<rect width='1000' height='520' fill='#101418'/>",
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#151c21' stroke='#31434a'/>",
    ]
    for i in range(6):
        y = top + i * plot_h / 5
        lines.append(f"<path d='M{left} {y:.2f}H{left + plot_w}' stroke='#26343a'/>")
    for label, color, points in series:
        lines.append(
            f"<polyline points='{_polyline(points)}' fill='none' stroke='{color}' "
            "stroke-width='3' stroke-linejoin='round' stroke-linecap='round'/>"
        )
    lines.extend(
        [
            f"<text x='{left}' y='24' fill='#dce7e9' font-size='18'>Training curves</text>",
            f"<text x='{left}' y='{height - 10}' fill='#97a8ae' font-size='13'>episode</text>",
            f"<text x='12' y='{top + 16}' fill='#97a8ae' font-size='13'>rate / scaled update</text>",
            *legend,
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_analysis(run_dir: Path, cfg: TabularConfig, metrics: list[dict], latest: dict) -> None:
    recent = latest["recent"]
    first = metrics[0] if metrics else latest
    analysis = {
        "run_id": _run_id(run_dir),
        "method": cfg.method,
        "episodes": latest["episode"],
        "states": latest["states"],
        "final_recent": recent,
        "first_recent": first.get("recent", {}),
        "mean_abs_update_final": latest.get("mean_abs_update", 0.0),
        "notes": [
            "Tabular values are from the current player's perspective.",
            "The heatmap is the learned value of each possible first move from an empty board.",
            "For size=4, tabular methods are diagnostic baselines; neural/self-play methods should take over.",
        ],
    }
    _write_json(run_dir / "analysis.json", analysis)
    markdown = [
        f"# Run {_run_id(run_dir)}",
        "",
        f"- Method: `{cfg.method}`",
        f"- Board: `{cfg.size}x{cfg.size}x{cfg.size}`",
        f"- Episodes: `{latest['episode']}`",
        f"- States visited: `{latest['states']}`",
        f"- Recent X win rate: `{recent['x_win_rate']:.3f}`",
        f"- Recent O win rate: `{recent['o_win_rate']:.3f}`",
        f"- Recent draw rate: `{recent['draw_rate']:.3f}`",
        f"- Final mean absolute update: `{latest.get('mean_abs_update', 0.0):.5f}`",
        "",
        "Artifacts: `config.json`, `metadata.json`, `metrics.jsonl`, `latest.json`, "
        "`analysis.json`, `analysis.md`, `curves.svg`, `q_table.npz`.",
        "",
    ]
    (run_dir / "analysis.md").write_text("\n".join(markdown))


def snapshot_payload(
    *,
    cfg: TabularConfig,
    q: QTable,
    episode: int,
    epsilon: float,
    outcomes: list[int],
    run_dir: Path,
    running: bool,
    mean_abs_update: float,
) -> dict:
    recent = outcomes[-max(1, min(200, len(outcomes))):]
    x_wins = sum(1 for outcome in recent if outcome == 1)
    o_wins = sum(1 for outcome in recent if outcome == -1)
    draws = sum(1 for outcome in recent if outcome == 0)
    denom = max(1, len(recent))
    return {
        "running": running,
        "run_dir": str(run_dir),
        "episode": episode,
        "episodes": cfg.episodes,
        "method": cfg.method,
        "run_id": _run_id(run_dir),
        "epsilon": round(float(epsilon), 6),
        "states": len(q),
        "mean_abs_update": round(float(mean_abs_update), 6),
        "recent": {
            "window": denom,
            "x_win_rate": x_wins / denom,
            "o_win_rate": o_wins / denom,
            "draw_rate": draws / denom,
        },
        "heatmap": empty_board_heatmap(q, cfg.size),
        "config": asdict(cfg),
    }


def save_q_table(path: Path, q: QTable) -> None:
    keys = np.array([" ".join(map(str, key)) for key in q.keys()], dtype=object)
    values = np.stack(list(q.values())).astype(np.float32) if q else np.zeros((0, 0))
    np.savez_compressed(path, keys=keys, values=values)


def train_tabular(
    cfg: TabularConfig,
    *,
    callback: SnapshotCallback | None = None,
    stop_event: threading.Event | None = None,
) -> Path:
    method = cfg.method.strip().lower().replace("-", "_")
    if method not in {"q_learning", "sarsa", "expected_sarsa", "monte_carlo"}:
        raise ValueError(f"unknown method {cfg.method!r}")
    cfg = TabularConfig(**{**asdict(cfg), "method": method})

    if cfg.size < 2:
        raise ValueError("size must be at least 2")
    if cfg.size > 4:
        raise ValueError("tabular runs are intended for size <= 4")

    run_dir = resolve_run_dir(cfg.run_dir)
    metadata = {
        "run_id": _run_id(run_dir),
        "name": cfg.name,
        "method": cfg.method,
        "parent_run": cfg.parent_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "run_dir": str(run_dir),
        "config": asdict(cfg),
    }
    _write_json(run_dir / "config.json", asdict(cfg))
    _write_json(run_dir / "metadata.json", metadata)
    _append_run_index(run_dir, metadata)

    rng = random.Random(cfg.seed)
    q: QTable = {}
    outcomes: list[int] = []
    update_magnitudes: list[float] = []
    epsilon = cfg.epsilon
    latest: dict | None = None

    for episode in range(1, cfg.episodes + 1):
        if stop_event is not None and stop_event.is_set():
            break

        state = State.new(cfg.size)
        moves_played = 0
        episode_updates: list[float] = []

        if cfg.method == "monte_carlo":
            trajectory: list[tuple[tuple[int, ...], int, int, int]] = []
            while True:
                key = state_key(state)
                move = choose_action(q, state, rng, epsilon)
                trajectory.append((key, move, state.player, moves_played))
                state = apply_move(state, move)
                moves_played += 1
                done, winner = terminal(state)
                if done:
                    outcomes.append(int(winner or 0))
                    horizon = max(1, moves_played - 1)
                    for key_i, move_i, player_i, step_i in trajectory:
                        row = get_q(q, key_i, cfg.size**3)
                        if winner == 0:
                            target = 0.0
                        else:
                            sign = 1.0 if player_i == winner else -1.0
                            target = sign * (cfg.gamma ** (horizon - step_i))
                        delta = target - float(row[move_i])
                        row[move_i] += cfg.alpha * delta
                        episode_updates.append(abs(delta))
                    break
            epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)
            update_magnitudes.extend(episode_updates)
            mean_abs_update = float(np.mean(episode_updates)) if episode_updates else 0.0

            if episode == 1 or episode % cfg.log_every == 0 or episode == cfg.episodes:
                latest = snapshot_payload(
                    cfg=cfg,
                    q=q,
                    episode=episode,
                    epsilon=epsilon,
                    outcomes=outcomes,
                    run_dir=run_dir,
                    running=True,
                    mean_abs_update=mean_abs_update,
                )
                latest["last_episode_moves"] = moves_played
                _append_jsonl(run_dir / "metrics.jsonl", latest)
                _write_json(run_dir / "latest.json", latest)
                if callback is not None:
                    callback(latest)
            continue

        while True:
            key = state_key(state)
            row = get_q(q, key, cfg.size**3)
            move = choose_action(q, state, rng, epsilon)
            next_state = apply_move(state, move)
            moves_played += 1
            done, winner = terminal(next_state)

            if done:
                reward = 0.0 if winner == 0 else 1.0
                target = reward
                delta = target - float(row[move])
                row[move] += cfg.alpha * delta
                episode_updates.append(abs(delta))
                outcomes.append(int(winner or 0))
                break

            next_row = get_q(q, state_key(next_state), cfg.size**3)
            next_moves = legal_moves(next_state)
            if cfg.method == "q_learning":
                opponent_best = float(np.max(next_row[next_moves])) if len(next_moves) else 0.0
            elif cfg.method == "sarsa":
                next_move = choose_action(q, next_state, rng, epsilon)
                opponent_best = float(next_row[next_move])
            else:
                opponent_best = expected_policy_value(next_row, next_moves, epsilon)
            target = -cfg.gamma * opponent_best
            delta = target - float(row[move])
            row[move] += cfg.alpha * delta
            episode_updates.append(abs(delta))
            state = next_state

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)
        update_magnitudes.extend(episode_updates)
        mean_abs_update = float(np.mean(episode_updates)) if episode_updates else 0.0

        if episode == 1 or episode % cfg.log_every == 0 or episode == cfg.episodes:
            latest = snapshot_payload(
                cfg=cfg,
                q=q,
                episode=episode,
                epsilon=epsilon,
                outcomes=outcomes,
                run_dir=run_dir,
                running=True,
                mean_abs_update=mean_abs_update,
            )
            latest["last_episode_moves"] = moves_played
            _append_jsonl(run_dir / "metrics.jsonl", latest)
            _write_json(run_dir / "latest.json", latest)
            if callback is not None:
                callback(latest)

    final_episode = len(outcomes)
    latest = snapshot_payload(
        cfg=cfg,
        q=q,
        episode=final_episode,
        epsilon=epsilon,
        outcomes=outcomes,
        run_dir=run_dir,
        running=False,
        mean_abs_update=float(np.mean(update_magnitudes[-1000:])) if update_magnitudes else 0.0,
    )
    _write_json(run_dir / "latest.json", latest)
    save_q_table(run_dir / "q_table.npz", q)
    metrics = load_metrics(run_dir / "metrics.jsonl")
    write_curves_svg(run_dir / "curves.svg", metrics)
    write_plot_artifacts(run_dir, metrics, latest)
    write_analysis(run_dir, cfg, metrics, latest)
    artifacts = {
        "run_id": _run_id(run_dir),
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
    }
    _write_json(run_dir / "artifacts.json", artifacts)
    if callback is not None:
        callback(latest)
    return run_dir
