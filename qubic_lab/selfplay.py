from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qubic_lab.game import State, apply_move, legal_moves, terminal
from qubic_lab.model_api import board_to_layers, load_model
from qubic_lab.runs import resolve_run_dir


@dataclass(frozen=True)
class SelfPlayConfig:
    model_id: str = "random"
    opponent_id: str | None = None
    size: int = 3
    games: int = 100
    seed: int = 0
    greedy: bool = False
    run_dir: str | None = None


def _result_for(player: int, winner: int | None) -> float:
    if winner is None or winner == 0:
        return 0.0
    return 1.0 if winner == player else -1.0


def generate_selfplay_dataset(cfg: SelfPlayConfig) -> Path:
    rng = random.Random(cfg.seed)
    run_dir = resolve_run_dir(cfg.run_dir, root=Path("runs/datasets"))
    run_dir.mkdir(parents=True, exist_ok=True)

    x_model = load_model(cfg.model_id)
    o_model = load_model(cfg.opponent_id or cfg.model_id)
    records: list[dict[str, Any]] = []
    outcomes = {1: 0, -1: 0, 0: 0}

    dataset_path = run_dir / "dataset.jsonl"
    with dataset_path.open("w") as f:
        for game_idx in range(cfg.games):
            state = State.new(cfg.size)
            game_steps: list[dict[str, Any]] = []
            while True:
                model = x_model if state.player == 1 else o_model
                probs, value = model.policy_value(state)
                move = model.choose_move(state, rng, greedy=cfg.greedy)
                step = {
                    "game": game_idx,
                    "ply": len(game_steps),
                    "player": state.player,
                    "board": board_to_layers(state),
                    "legal_moves": legal_moves(state).astype(int).tolist(),
                    "action": int(move),
                    "action_prob": float(probs[move]),
                    "value": float(value),
                }
                game_steps.append(step)
                state = apply_move(state, move)
                done, winner = terminal(state)
                if done:
                    winner_key = int(winner or 0)
                    outcomes[winner_key] += 1
                    for item in game_steps:
                        item["winner"] = winner_key
                        item["return"] = _result_for(int(item["player"]), winner)
                        f.write(json.dumps(item) + "\n")
                        records.append(item)
                    break

    manifest = {
        "created_at": time.time(),
        "config": asdict(cfg),
        "run_dir": str(run_dir),
        "dataset": "dataset.jsonl",
        "games": cfg.games,
        "positions": len(records),
        "outcomes": {
            "x_win_rate": outcomes[1] / max(1, cfg.games),
            "o_win_rate": outcomes[-1] / max(1, cfg.games),
            "draw_rate": outcomes[0] / max(1, cfg.games),
        },
        "notes": [
            "Each row stores the pre-move board, acting player, legal mask, selected action, policy probability, value estimate, final winner, and return from the acting player's perspective.",
            "This is an offline imitation/RL dataset artifact. PPO and GRPO training currently still use online self-play rollouts.",
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Self-play dataset {run_dir.name}",
                "",
                f"- Model X: `{cfg.model_id}`",
                f"- Model O: `{cfg.opponent_id or cfg.model_id}`",
                f"- Board: `{cfg.size}x{cfg.size}x{cfg.size}`",
                f"- Games: `{cfg.games}`",
                f"- Positions: `{len(records)}`",
                f"- X win rate: `{manifest['outcomes']['x_win_rate']:.3f}`",
                f"- O win rate: `{manifest['outcomes']['o_win_rate']:.3f}`",
                f"- Draw rate: `{manifest['outcomes']['draw_rate']:.3f}`",
                "",
                "Files: `manifest.json`, `dataset.jsonl`.",
                "",
            ]
        )
    )
    return run_dir
