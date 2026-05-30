from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qubic_lab.game import State, apply_move, flatten_board, idx_to_xyz, legal_moves, terminal, winning_lines
from qubic_lab.neural import PolicyValueNet, legal_mask, obs_from_state
from qubic_lab.rl_tabular import state_key


def board_to_layers(state: State) -> list[list[list[int]]]:
    size = state.size
    return [
        [[int(state.board[x, y, z]) for x in range(size)] for y in range(size)]
        for z in range(size)
    ]


def layers_to_state(layers: list[list[list[int]]], player: int) -> State:
    size = len(layers)
    board = np.zeros((size, size, size), dtype=np.int8)
    for z, layer in enumerate(layers):
        for y, row in enumerate(layer):
            for x, value in enumerate(row):
                board[x, y, z] = int(value)
    return State(board=board, player=int(player))


def vector_to_layers(values: np.ndarray, size: int) -> list[list[list[float]]]:
    layers = np.zeros((size, size, size), dtype=np.float32)
    for idx, value in enumerate(values):
        x, y, z = idx_to_xyz(idx, size)
        layers[z, y, x] = float(value)
    return layers.round(5).tolist()


@dataclass
class ModelRecord:
    id: str
    kind: str
    label: str
    run_dir: str | None
    size: int
    method: str
    score: float | None = None


class LoadedModel:
    def __init__(self, record: ModelRecord):
        self.record = record
        self._neural: PolicyValueNet | None = None
        self._q: dict[tuple[int, ...], np.ndarray] | None = None

    def _load_neural(self) -> PolicyValueNet:
        if self._neural is not None:
            return self._neural
        assert self.record.run_dir is not None
        payload = torch.load(Path(self.record.run_dir) / "model.pt", map_location="cpu")
        cfg = payload["config"]
        model = PolicyValueNet(int(cfg["size"]), int(cfg.get("hidden", 128)))
        model.load_state_dict(payload["model"])
        model.eval()
        self._neural = model
        return model

    def _load_q(self) -> dict[tuple[int, ...], np.ndarray]:
        if self._q is not None:
            return self._q
        assert self.record.run_dir is not None
        payload = np.load(Path(self.record.run_dir) / "q_table.npz", allow_pickle=True)
        keys = payload["keys"]
        values = payload["values"]
        self._q = {
            tuple(int(part) for part in str(key).split()): values[i].astype(np.float32)
            for i, key in enumerate(keys)
        }
        return self._q

    def policy_value(self, state: State) -> tuple[np.ndarray, float]:
        moves = legal_moves(state).astype(int)
        n = state.size**3
        probs = np.zeros(n, dtype=np.float32)
        if len(moves) == 0:
            return probs, 0.0

        if self.record.kind == "random":
            probs[moves] = 1.0 / len(moves)
            return probs, 0.0

        if self.record.kind == "tactical":
            move, score = tactical_move(state, random.Random(0))
            probs[move] = 1.0
            return probs, float(np.tanh(score / 8.0))

        if self.record.kind == "tabular":
            row = self._load_q().get(state_key(state), np.zeros(n, dtype=np.float32))
            masked = np.full(n, -np.inf, dtype=np.float32)
            masked[moves] = row[moves]
            best = np.flatnonzero(masked == np.nanmax(masked)).astype(int)
            probs[best] = 1.0 / len(best)
            return probs, float(np.nanmax(masked[moves]))

        model = self._load_neural()
        obs = obs_from_state(state)
        mask = legal_mask(state)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            logits, value = model(obs_t)
            logits = logits.masked_fill(~mask_t, -1e9)
            probs_t = torch.softmax(logits, dim=-1).squeeze(0)
        return probs_t.cpu().numpy().astype(np.float32), float(value.item())

    def choose_move(self, state: State, rng: random.Random, *, greedy: bool = True) -> int:
        if self.record.kind == "tactical":
            return tactical_move(state, rng)[0]

        probs, _ = self.policy_value(state)
        moves = legal_moves(state).astype(int)
        if greedy:
            legal_probs = probs[moves]
            best = moves[np.flatnonzero(legal_probs == np.max(legal_probs))]
            return int(rng.choice(best.tolist()))
        total = float(probs[moves].sum())
        if total <= 0:
            return int(rng.choice(moves.tolist()))
        weights = (probs[moves] / total).tolist()
        return int(rng.choices(moves.tolist(), weights=weights, k=1)[0])


def _run_latest(path: Path) -> dict[str, Any] | None:
    latest = path / "latest.json"
    if not latest.exists():
        return None
    try:
        return json.loads(latest.read_text())
    except json.JSONDecodeError:
        return None


def _winning_moves(state: State, player: int) -> list[int]:
    wins = []
    test_state = State(board=state.board, player=player)
    for move in legal_moves(state).astype(int).tolist():
        next_state = apply_move(test_state, move)
        done, winner = terminal(next_state)
        if done and winner == player:
            wins.append(int(move))
    return wins


def _move_pressure_score(state: State, move: int) -> float:
    flat = flatten_board(state.board)
    player = int(state.player)
    score = 0.0
    for line in winning_lines(state.size):
        if move not in line:
            continue
        values = flat[list(line)]
        own = int(np.sum(values == player))
        opp = int(np.sum(values == -player))
        empty = int(np.sum(values == 0))
        if opp == 0:
            score += (own + 1) ** 2 + 0.25 * empty
        if own == 0:
            score += 0.7 * (opp + 1) ** 2
    x, y, z = idx_to_xyz(move, state.size)
    center = (state.size - 1) / 2
    score -= 0.08 * (abs(x - center) + abs(y - center) + abs(z - center))
    return score


def tactical_move(state: State, rng: random.Random) -> tuple[int, float]:
    moves = legal_moves(state).astype(int).tolist()
    if not moves:
        return 0, 0.0
    wins = _winning_moves(state, int(state.player))
    if wins:
        return int(rng.choice(wins)), 100.0
    blocks = _winning_moves(state, -int(state.player))
    if blocks:
        return int(rng.choice(blocks)), 60.0
    scored = [(move, _move_pressure_score(state, move)) for move in moves]
    best = max(score for _, score in scored)
    best_moves = [move for move, score in scored if score == best]
    return int(rng.choice(best_moves)), float(best)


def list_models(root: Path = Path("runs")) -> list[ModelRecord]:
    builtins = [
        ModelRecord(id="random", kind="random", label="Random baseline", run_dir=None, size=3, method="random"),
        ModelRecord(id="tactical", kind="tactical", label="Tactical baseline", run_dir=None, size=3, method="tactical"),
    ]
    records = []
    for latest_path in root.rglob("latest.json"):
        run_dir = latest_path.parent
        latest = _run_latest(run_dir)
        if latest is None:
            continue
        cfg = latest.get("config", {})
        size = int(cfg.get("size", 3))
        method = str(latest.get("method", cfg.get("method", "run")))
        label = f"{method} · {latest.get('run_id', run_dir.name)}"
        if (run_dir / "model.pt").exists():
            records.append(
                ModelRecord(
                    id=str(run_dir),
                    kind="neural",
                    label=label,
                    run_dir=str(run_dir),
                    size=size,
                    method=method,
                )
            )
        elif (run_dir / "q_table.npz").exists():
            records.append(
                ModelRecord(
                    id=str(run_dir),
                    kind="tabular",
                    label=label,
                    run_dir=str(run_dir),
                    size=size,
                    method=method,
                )
            )
    records.sort(key=lambda r: (r.method, r.id), reverse=False)
    return [*builtins, *records]


def load_model(model_id: str) -> LoadedModel:
    for record in list_models():
        if record.id == model_id:
            return LoadedModel(record)
    raise KeyError(model_id)


def analyze_position(model_id: str, state: State) -> dict[str, Any]:
    model = load_model(model_id)
    probs, value = model.policy_value(state)
    moves = legal_moves(state).astype(int).tolist()
    top = sorted(
        [
            {
                "move": move,
                "prob": float(probs[move]),
                "x": idx_to_xyz(move, state.size)[0],
                "y": idx_to_xyz(move, state.size)[1],
                "z": idx_to_xyz(move, state.size)[2],
            }
            for move in moves
        ],
        key=lambda item: item["prob"],
        reverse=True,
    )[:10]
    done, winner = terminal(state)
    return {
        "model": model.record.__dict__,
        "board": board_to_layers(state),
        "player": state.player,
        "done": done,
        "winner": winner,
        "legal_moves": moves,
        "value": value,
        "heatmap": vector_to_layers(probs, state.size),
        "top_moves": top,
    }


def append_replay(record: dict[str, Any], root: Path = Path("runs/replay")) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "replay.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def play_game(model_id: str, size: int, human_player: int, moves: list[int]) -> dict[str, Any]:
    rng = random.Random(0)
    state = State.new(size)
    model = load_model(model_id)
    history: list[dict[str, Any]] = []
    if state.player != human_player:
        model_move = model.choose_move(state, rng, greedy=True)
        actor = state.player
        state = apply_move(state, model_move)
        history.append({"player": actor, "move": model_move})
    for move in moves:
        if move not in set(map(int, legal_moves(state))):
            raise ValueError(f"illegal move {move}")
        actor = state.player
        state = apply_move(state, move)
        history.append({"player": actor, "move": move})
        done, _ = terminal(state)
        if done:
            break
        if state.player != human_player:
            model_move = model.choose_move(state, rng, greedy=True)
            actor = state.player
            state = apply_move(state, model_move)
            history.append({"player": actor, "move": model_move})
            done, _ = terminal(state)
            if done:
                break
    done, winner = terminal(state)
    analysis = analyze_position(model_id, state)
    payload = {
        "game_id": str(uuid.uuid4()),
        "model_id": model_id,
        "human_player": human_player,
        "history": history,
        "state": analysis,
        "done": done,
        "winner": winner,
    }
    if done:
        append_replay({"type": "human_play", "created_at": time.time(), **payload})
    return payload


def evaluate_match(a_id: str, b_id: str, *, size: int, games: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    a = load_model(a_id)
    b = load_model(b_id)
    wins = {a_id: 0, b_id: 0, "draw": 0}
    records = []
    for game_idx in range(games):
        state = State.new(size)
        players = {1: a, -1: b} if game_idx % 2 == 0 else {1: b, -1: a}
        while True:
            model = players[state.player]
            move = model.choose_move(state, rng, greedy=True)
            state = apply_move(state, move)
            done, winner = terminal(state)
            if done:
                if winner == 0:
                    wins["draw"] += 1
                    winner_id = "draw"
                else:
                    winner_model = players[int(winner)]
                    wins[winner_model.record.id] += 1
                    winner_id = winner_model.record.id
                records.append({"game": game_idx, "winner": winner_id})
                break
    return {"a": a_id, "b": b_id, "games": games, "wins": wins, "records": records}


def run_tournament(model_ids: list[str], *, size: int = 3, games: int = 20, seed: int = 0) -> dict[str, Any]:
    if not model_ids:
        model_ids = [m.id for m in list_models()[:6]]
    if "random" not in model_ids:
        model_ids = ["random", *model_ids]
    scores = {model_id: 0.0 for model_id in model_ids}
    matches = []
    for i, a_id in enumerate(model_ids):
        for b_id in model_ids[i + 1 :]:
            match = evaluate_match(a_id, b_id, size=size, games=games, seed=seed + i)
            matches.append(match)
            total = max(1, games)
            scores[a_id] += match["wins"].get(a_id, 0) / total
            scores[b_id] += match["wins"].get(b_id, 0) / total
            scores[a_id] += 0.5 * match["wins"].get("draw", 0) / total
            scores[b_id] += 0.5 * match["wins"].get("draw", 0) / total
    leaderboard = sorted(
        [{"model_id": model_id, "score": score} for model_id, score in scores.items()],
        key=lambda item: item["score"],
        reverse=True,
    )
    result = {
        "created_at": time.time(),
        "size": size,
        "games": games,
        "model_ids": model_ids,
        "leaderboard": leaderboard,
        "matches": matches,
    }
    out_dir = Path("runs/evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"tournament_{int(result['created_at'])}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
