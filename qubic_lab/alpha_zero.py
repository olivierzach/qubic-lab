from __future__ import annotations

import math
import random
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from qubic_lab.artifacts import load_metrics, write_plot_artifacts
from qubic_lab.game import State, apply_move, terminal
from qubic_lab.model_api import mcts_policy
from qubic_lab.neural import PolicyValueNet, empty_board_policy_analysis, legal_mask, masked_logits, obs_from_state
from qubic_lab.reporting import generate_report_card, write_report_card
from qubic_lab.runlog import append_jsonl, run_id, write_json, write_metadata
from qubic_lab.runs import resolve_run_dir

SnapshotCallback = Callable[[dict], None]


@dataclass(frozen=True)
class AlphaZeroConfig:
    method: str = "alpha_zero"
    name: str | None = None
    parent_run: str | None = None
    size: int = 3
    iterations: int = 10
    games_per_iteration: int = 128
    mcts_simulations: int = 64
    hidden: int = 256
    lr: float = 3e-4
    batch_size: int = 256
    update_epochs: int = 4
    replay_size: int = 50_000
    temperature: float = 1.0
    seed: int = 0
    log_every: int = 1
    run_dir: str | None = None
    device: str = "cpu"


@dataclass
class SearchSample:
    obs: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    player: int
    value: float = 0.0


def _choose_from_policy(policy: np.ndarray, rng: random.Random, temperature: float) -> int:
    legal = np.flatnonzero(policy > 0).astype(int)
    if len(legal) == 0:
        raise ValueError("policy has no legal mass")
    if temperature <= 1e-6:
        best = legal[np.flatnonzero(policy[legal] == np.max(policy[legal]))]
        return int(rng.choice(best.tolist()))
    weights = np.power(np.maximum(policy[legal], 1e-12), 1.0 / temperature)
    weights = weights / weights.sum()
    return int(rng.choices(legal.tolist(), weights=weights.tolist(), k=1)[0])


def self_play_game(model: PolicyValueNet, cfg: AlphaZeroConfig, rng: random.Random) -> tuple[list[SearchSample], int]:
    state = State.new(cfg.size)
    samples: list[SearchSample] = []
    while True:
        policy, _ = mcts_policy(state, simulations=cfg.mcts_simulations, rng=rng)
        mask = legal_mask(state)
        policy = policy * mask
        total = float(policy.sum())
        if total <= 0:
            policy[mask] = 1.0 / max(1, int(mask.sum()))
        else:
            policy = policy / total
        samples.append(SearchSample(obs=obs_from_state(state), mask=mask, policy=policy.astype(np.float32), player=state.player))
        move = _choose_from_policy(policy, rng, cfg.temperature)
        state = apply_move(state, move)
        done, winner = terminal(state)
        if done:
            outcome = int(winner or 0)
            for sample in samples:
                if outcome == 0:
                    sample.value = 0.0
                else:
                    sample.value = 1.0 if sample.player == outcome else -1.0
            return samples, outcome


def _train_epoch(model: PolicyValueNet, optimizer: torch.optim.Optimizer, replay: list[SearchSample], cfg: AlphaZeroConfig, rng: random.Random) -> dict[str, float]:
    if not replay:
        return {"loss": math.nan, "policy_loss": math.nan, "value_loss": math.nan}
    batch = rng.sample(replay, min(cfg.batch_size, len(replay)))
    obs = torch.as_tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=cfg.device)
    mask = torch.as_tensor(np.stack([item.mask for item in batch]), dtype=torch.bool, device=cfg.device)
    targets = torch.as_tensor(np.stack([item.policy for item in batch]), dtype=torch.float32, device=cfg.device)
    values = torch.as_tensor([item.value for item in batch], dtype=torch.float32, device=cfg.device)
    logits, pred_values = model(obs)
    logits = masked_logits(logits, mask)
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(targets * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(pred_values, values)
    loss = policy_loss + value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }


def _snapshot(
    cfg: AlphaZeroConfig,
    model: PolicyValueNet,
    run_dir: Path,
    iteration: int,
    games: int,
    outcomes: list[int],
    replay: list[SearchSample],
    losses: dict[str, float],
    running: bool,
) -> dict:
    recent = outcomes[-max(1, min(200, len(outcomes))):]
    denom = max(1, len(recent))
    opening = empty_board_policy_analysis(model, cfg.size, device=cfg.device)
    return {
        "running": running,
        "run_dir": str(run_dir),
        "episode": games,
        "episodes": cfg.iterations * cfg.games_per_iteration,
        "iteration": iteration,
        "iterations": cfg.iterations,
        "method": cfg.method,
        "run_id": run_id(run_dir),
        "states": len(replay),
        "mean_abs_update": abs(losses.get("loss", 0.0)),
        "policy_loss": losses.get("policy_loss", 0.0),
        "value_loss": losses.get("value_loss", 0.0),
        "recent": {
            "window": denom,
            "x_win_rate": sum(1 for outcome in recent if outcome == 1) / denom,
            "o_win_rate": sum(1 for outcome in recent if outcome == -1) / denom,
            "draw_rate": sum(1 for outcome in recent if outcome == 0) / denom,
        },
        "value": opening["value"],
        "heatmap": opening["heatmap"],
        "top_moves": opening["top_moves"],
        "config": asdict(cfg),
    }


def train_alpha_zero(
    cfg: AlphaZeroConfig,
    *,
    callback: SnapshotCallback | None = None,
    stop_event: threading.Event | None = None,
) -> Path:
    cfg = AlphaZeroConfig(**{**asdict(cfg), "method": "alpha_zero"})
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    run_dir = resolve_run_dir(cfg.run_dir)
    write_json(run_dir / "config.json", asdict(cfg))
    write_metadata(run_dir, cfg, method=cfg.method, name=cfg.name, parent_run=cfg.parent_run)
    model = PolicyValueNet(cfg.size, cfg.hidden).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    replay: list[SearchSample] = []
    outcomes: list[int] = []
    losses = {"loss": math.nan, "policy_loss": math.nan, "value_loss": math.nan}
    games = 0
    latest = _snapshot(cfg, model, run_dir, 0, games, outcomes, replay, losses, running=True)

    for iteration in range(1, cfg.iterations + 1):
        if stop_event is not None and stop_event.is_set():
            break
        for _ in range(cfg.games_per_iteration):
            samples, outcome = self_play_game(model, cfg, rng)
            replay.extend(samples)
            del replay[:-cfg.replay_size]
            outcomes.append(outcome)
            games += 1
        for _ in range(cfg.update_epochs):
            losses = _train_epoch(model, optimizer, replay, cfg, rng)
        if iteration == 1 or iteration % cfg.log_every == 0 or iteration >= cfg.iterations:
            latest = _snapshot(cfg, model, run_dir, iteration, games, outcomes, replay, losses, running=True)
            append_jsonl(run_dir / "metrics.jsonl", latest)
            write_json(run_dir / "latest.json", latest)
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, run_dir / "model.pt")
            if callback is not None:
                callback(latest)

    latest = _snapshot(cfg, model, run_dir, min(cfg.iterations, max(1, latest.get("iteration", 1))), games, outcomes, replay, losses, running=False)
    write_json(run_dir / "latest.json", latest)
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, run_dir / "model.pt")
    metrics = load_metrics(run_dir / "metrics.jsonl")
    write_plot_artifacts(run_dir, metrics, latest)
    report = generate_report_card(str(run_dir), run_dir=run_dir, size=cfg.size, probe_cases_per_family=8, fast=True)
    report_files = write_report_card(report, run_dir)
    write_json(
        run_dir / "artifacts.json",
        {
            "run_id": run_id(run_dir),
            "files": {
                "config": "config.json",
                "metadata": "metadata.json",
                "metrics": "metrics.jsonl",
                "latest": "latest.json",
                "curves_png": "curves.png",
                "first_move_heatmap": "first_move_heatmap.png",
                "first_move_policy": "first_move_policy.json",
                "model": "model.pt",
                **report_files,
            },
        },
    )
    if callback is not None:
        callback(latest)
    return run_dir
