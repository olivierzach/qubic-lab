from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qubic_lab.game import State, apply_move, flatten_board, idx_to_xyz, legal_moves, terminal, winning_lines
from qubic_lab.model_api import board_to_layers, load_model


@dataclass(frozen=True)
class ProbeCase:
    id: str
    family: str
    state: State
    expected_moves: tuple[int, ...]
    description: str


def _state_with_marks(size: int, player: int, marks: dict[int, int]) -> State:
    board = np.zeros((size, size, size), dtype=np.int8)
    for move, value in marks.items():
        x, y, z = idx_to_xyz(move, size)
        board[x, y, z] = int(value)
    return State(board=board, player=player)


def _line_cases(size: int, family: str, player: int, mark_player: int, limit: int) -> list[ProbeCase]:
    cases = []
    for idx, line in enumerate(winning_lines(size)):
        marks = {line[0]: mark_player, line[1]: mark_player}
        state = _state_with_marks(size, player, marks)
        expected = int(line[2])
        if expected not in legal_moves(state):
            continue
        cases.append(
            ProbeCase(
                id=f"{family}_{idx}",
                family=family,
                state=state,
                expected_moves=(expected,),
                description=f"{family.replace('_', ' ')} on line {line}",
            )
        )
        if len(cases) >= limit:
            break
    return cases


def _immediate_win_cases(size: int, limit: int) -> list[ProbeCase]:
    return _line_cases(size, "immediate_win", player=1, mark_player=1, limit=limit)


def _immediate_block_cases(size: int, limit: int) -> list[ProbeCase]:
    return _line_cases(size, "immediate_block", player=1, mark_player=-1, limit=limit)


def _winning_moves(state: State, player: int) -> list[int]:
    moves = []
    test_state = State(board=state.board, player=player)
    for move in legal_moves(state).astype(int).tolist():
        done, winner = terminal(apply_move(test_state, move))
        if done and winner == player:
            moves.append(move)
    return moves


def _fork_moves(state: State) -> list[int]:
    moves = []
    for move in legal_moves(state).astype(int).tolist():
        next_state = apply_move(state, move)
        if len(_winning_moves(next_state, state.player)) >= 2:
            moves.append(move)
    return moves


def _random_nonterminal_state(size: int, rng: random.Random, moves_played: int) -> State | None:
    state = State.new(size)
    for _ in range(moves_played):
        moves = legal_moves(state).astype(int).tolist()
        if not moves:
            return None
        state = apply_move(state, int(rng.choice(moves)))
        done, _ = terminal(state)
        if done:
            return None
    return state


def _fork_cases(size: int, family: str, limit: int, seed: int) -> list[ProbeCase]:
    rng = random.Random(seed)
    cases = []
    seen = set()
    attempts = 0
    while len(cases) < limit and attempts < 5000:
        attempts += 1
        state = _random_nonterminal_state(size, rng, rng.randint(3, 7))
        if state is None:
            continue
        moves = tuple(_fork_moves(state))
        if not moves:
            continue
        key = tuple(flatten_board(state.board).astype(int).tolist() + [state.player])
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            ProbeCase(
                id=f"{family}_{len(cases)}",
                family=family,
                state=state,
                expected_moves=moves,
                description=f"{family.replace('_', ' ')} with {len(moves)} valid fork moves",
            )
        )
    return cases


def build_probe_suite(size: int = 3, *, per_family: int = 16, seed: int = 0) -> list[ProbeCase]:
    cases = []
    cases.extend(_immediate_win_cases(size, per_family))
    cases.extend(_immediate_block_cases(size, per_family))
    cases.extend(_fork_cases(size, "fork_create", per_family, seed))
    cases.extend(_fork_cases(size, "fork_prevent", per_family, seed + 1))
    return cases


def evaluate_probes(model_id: str, *, size: int = 3, per_family: int = 16, seed: int = 0) -> dict[str, Any]:
    model = load_model(model_id)
    cases = build_probe_suite(size, per_family=per_family, seed=seed)
    by_family: dict[str, dict[str, Any]] = {}
    failures = []
    for case in cases:
        probs, value = model.policy_value(case.state)
        legal = legal_moves(case.state).astype(int)
        best_prob = probs[legal]
        best_moves = legal[np.flatnonzero(best_prob == np.max(best_prob))].astype(int).tolist()
        passed = any(move in case.expected_moves for move in best_moves)
        family = by_family.setdefault(case.family, {"passed": 0, "total": 0, "pass_rate": 0.0})
        family["total"] += 1
        family["passed"] += int(passed)
        if not passed:
            failures.append(
                {
                    "id": case.id,
                    "family": case.family,
                    "description": case.description,
                    "player": case.state.player,
                    "board": board_to_layers(case.state),
                    "expected_moves": list(case.expected_moves),
                    "best_moves": best_moves,
                    "value": float(value),
                }
            )
    for family in by_family.values():
        family["pass_rate"] = family["passed"] / max(1, family["total"])
    total = sum(item["total"] for item in by_family.values())
    passed = sum(item["passed"] for item in by_family.values())
    return {
        "model_id": model_id,
        "size": size,
        "total": total,
        "passed": passed,
        "pass_rate": passed / max(1, total),
        "families": by_family,
        "failures": failures[:100],
    }


def write_probe_failures_replay(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for failure in report.get("failures", []):
            f.write(json.dumps({"type": "probe_failure", **failure}) + "\n")

