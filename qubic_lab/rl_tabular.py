from __future__ import annotations

import json
import random
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from qubic_lab.game import State, apply_move, flatten_board, legal_moves, terminal
from qubic_lab.runs import resolve_run_dir


QTable = dict[tuple[int, ...], np.ndarray]
SnapshotCallback = Callable[[dict], None]


@dataclass(frozen=True)
class TabularConfig:
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


def empty_board_heatmap(q: QTable, size: int) -> list[list[list[float]]]:
    empty = state_key(State.new(size))
    row = get_q(q, empty, size**3)
    return row.reshape((size, size, size)).round(4).tolist()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def snapshot_payload(
    *,
    cfg: TabularConfig,
    q: QTable,
    episode: int,
    epsilon: float,
    outcomes: list[int],
    run_dir: Path,
    running: bool,
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
        "epsilon": round(float(epsilon), 6),
        "states": len(q),
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
    if cfg.size < 2:
        raise ValueError("size must be at least 2")
    if cfg.size > 4:
        raise ValueError("tabular runs are intended for size <= 4")

    run_dir = resolve_run_dir(cfg.run_dir)
    _write_json(run_dir / "config.json", asdict(cfg))

    rng = random.Random(cfg.seed)
    q: QTable = {}
    outcomes: list[int] = []
    epsilon = cfg.epsilon
    latest: dict | None = None

    for episode in range(1, cfg.episodes + 1):
        if stop_event is not None and stop_event.is_set():
            break

        state = State.new(cfg.size)
        moves_played = 0

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
                row[move] += cfg.alpha * (target - row[move])
                outcomes.append(int(winner or 0))
                break

            next_row = get_q(q, state_key(next_state), cfg.size**3)
            next_moves = legal_moves(next_state)
            opponent_best = float(np.max(next_row[next_moves])) if len(next_moves) else 0.0
            target = -cfg.gamma * opponent_best
            row[move] += cfg.alpha * (target - row[move])
            state = next_state

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

        if episode == 1 or episode % cfg.log_every == 0 or episode == cfg.episodes:
            latest = snapshot_payload(
                cfg=cfg,
                q=q,
                episode=episode,
                epsilon=epsilon,
                outcomes=outcomes,
                run_dir=run_dir,
                running=True,
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
    )
    _write_json(run_dir / "latest.json", latest)
    save_q_table(run_dir / "q_table.npz", q)
    if callback is not None:
        callback(latest)
    return run_dir
