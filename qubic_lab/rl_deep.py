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
from torch import nn

from qubic_lab.artifacts import load_metrics, write_plot_artifacts
from qubic_lab.game import State, apply_move, terminal
from qubic_lab.model_api import load_model
from qubic_lab.neural import (
    PolicyStep,
    PolicyValueNet,
    empty_board_policy_analysis,
    masked_logits,
    select_action,
)
from qubic_lab.runlog import append_jsonl, run_id, write_json, write_metadata
from qubic_lab.runs import resolve_run_dir

SnapshotCallback = Callable[[dict], None]


@dataclass(frozen=True)
class DeepRLConfig:
    method: str = "ppo"
    name: str | None = None
    parent_run: str | None = None
    size: int = 3
    episodes: int = 2_000
    batch_episodes: int = 32
    update_epochs: int = 4
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    opponent_mix: str = "self"
    side_mode: str = "balanced"
    o_target_win_rate: float = 0.5
    o_reward_weight: float = 1.0
    seed: int = 0
    log_every: int = 100
    run_dir: str | None = None
    device: str = "cpu"


def _episode_returns(steps: list[PolicyStep], winner: int | None, gamma: float, reward_weight: float = 1.0) -> np.ndarray:
    returns = []
    horizon = max(1, len(steps) - 1)
    for i, step in enumerate(steps):
        if winner is None or winner == 0:
            ret = 0.0
        else:
            sign = 1.0 if step.player == winner else -1.0
            ret = reward_weight * sign * (gamma ** (horizon - i))
        returns.append(ret)
    return np.asarray(returns, dtype=np.float32)


def _parse_opponent_mix(mix: str) -> list[tuple[str, float]]:
    items = []
    for part in str(mix or "self").split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            name, raw_weight = item.split(":", 1)
            weight = float(raw_weight)
        else:
            name, weight = item, 1.0
        name = name.strip()
        if weight > 0 and name:
            items.append((name, weight))
    return items or [("self", 1.0)]


def _sample_opponent(cfg: DeepRLConfig, rng: np.random.Generator) -> str:
    items = _parse_opponent_mix(cfg.opponent_mix)
    names = [name for name, _ in items]
    weights = np.asarray([weight for _, weight in items], dtype=np.float64)
    weights = weights / weights.sum()
    return names[int(rng.choice(len(names), p=weights))]


def _opponent_move(opponent_id: str, model: PolicyValueNet, state: State, rng: np.random.Generator, cfg: DeepRLConfig) -> int:
    if opponent_id == "self":
        action, *_ = select_action(model, state, rng, temperature=cfg.temperature, device=cfg.device)
        return int(action)
    py_rng = random.Random(int(rng.integers(0, 2**31 - 1)))
    return load_model(opponent_id).choose_move(state, py_rng, greedy=True)


def _recent_o_win_rate(episode_infos: list[dict], window: int = 200) -> float | None:
    recent_o = [
        info for info in episode_infos[-window:]
        if info.get("learner_player") == -1 and info.get("learner_result") is not None
    ]
    if not recent_o:
        return None
    return sum(1 for info in recent_o if info.get("learner_result") == 1) / len(recent_o)


def _learner_player(cfg: DeepRLConfig, rng: np.random.Generator, episode_infos: list[dict] | None = None) -> int:
    mode = cfg.side_mode.strip().lower().replace("-", "_")
    if mode in {"x", "as_x", "x_only"}:
        return 1
    if mode in {"o", "as_o", "o_only"}:
        return -1
    if mode == "adaptive_o":
        o_rate = _recent_o_win_rate(episode_infos or [])
        if o_rate is None:
            p_o = 0.5
        else:
            shortfall = max(0.0, cfg.o_target_win_rate - o_rate)
            p_o = min(0.9, 0.5 + shortfall)
        return -1 if float(rng.random()) < p_o else 1
    return 1 if int(rng.integers(0, 2)) == 0 else -1


def play_episode(
    model: PolicyValueNet,
    size: int,
    rng: np.random.Generator,
    cfg: DeepRLConfig,
    episode_infos: list[dict] | None = None,
) -> tuple[list[PolicyStep], int, dict]:
    state = State.new(size)
    steps: list[PolicyStep] = []
    opponent_id = _sample_opponent(cfg, rng)
    learner_player = _learner_player(cfg, rng, episode_infos)
    reward_weight = cfg.o_reward_weight if opponent_id != "self" and learner_player == -1 else 1.0
    while True:
        if opponent_id != "self" and state.player != learner_player:
            action = _opponent_move(opponent_id, model, state, rng, cfg)
            state = apply_move(state, action)
            done, winner = terminal(state)
            if done:
                returns = _episode_returns(steps, winner, cfg.gamma, reward_weight)
                patched = []
                for step, ret in zip(steps, returns):
                    patched.append(
                        PolicyStep(
                            obs=step.obs,
                            action=step.action,
                            reward=float(ret),
                            done=True,
                            logp=step.logp,
                            value=step.value,
                            mask=step.mask,
                            player=step.player,
                        )
                    )
                outcome = int(winner or 0)
                return patched, outcome, _episode_info(opponent_id, learner_player, outcome, reward_weight)
            continue

        action, logp, value, obs, mask = select_action(model, state, rng, temperature=cfg.temperature, device=cfg.device)
        next_state = apply_move(state, action)
        done, winner = terminal(next_state)
        steps.append(
            PolicyStep(
                obs=obs,
                action=action,
                reward=0.0,
                done=done,
                logp=logp,
                value=value,
                mask=mask,
                player=state.player,
            )
        )
        if done:
            returns = _episode_returns(steps, winner, cfg.gamma, reward_weight)
            patched = []
            for step, ret in zip(steps, returns):
                patched.append(
                    PolicyStep(
                        obs=step.obs,
                        action=step.action,
                        reward=float(ret),
                        done=step.done,
                        logp=step.logp,
                        value=step.value,
                        mask=step.mask,
                        player=step.player,
                    )
                )
            outcome = int(winner or 0)
            return patched, outcome, _episode_info(opponent_id, learner_player, outcome, reward_weight)
        state = next_state


def _episode_info(opponent_id: str, learner_player: int, winner: int, reward_weight: float = 1.0) -> dict:
    if opponent_id == "self":
        return {"opponent_id": opponent_id, "learner_player": None, "learner_result": None, "winner": winner}
    if winner == 0:
        result = 0
    else:
        result = 1 if winner == learner_player else -1
    return {
        "opponent_id": opponent_id,
        "learner_player": int(learner_player),
        "learner_result": result,
        "reward_weight": float(reward_weight),
        "winner": winner,
    }


def _to_tensors(steps: list[PolicyStep], device: str) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.as_tensor(np.stack([s.obs for s in steps]), dtype=torch.float32, device=device),
        "actions": torch.as_tensor([s.action for s in steps], dtype=torch.long, device=device),
        "old_logp": torch.as_tensor([s.logp for s in steps], dtype=torch.float32, device=device),
        "returns": torch.as_tensor([s.reward for s in steps], dtype=torch.float32, device=device),
        "values": torch.as_tensor([s.value for s in steps], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(np.stack([s.mask for s in steps]), dtype=torch.bool, device=device),
    }


def _advantages(batch: dict[str, torch.Tensor], method: str, group_ids: torch.Tensor) -> torch.Tensor:
    returns = batch["returns"]
    if method == "grpo":
        adv = torch.zeros_like(returns)
        for group_id in torch.unique(group_ids):
            idx = group_ids == group_id
            group = returns[idx]
            std = torch.clamp(group.std(unbiased=False), min=1e-6)
            adv[idx] = (group - group.mean()) / std
        return adv
    adv = returns - batch["values"]
    return (adv - adv.mean()) / torch.clamp(adv.std(unbiased=False), min=1e-6)


def update_policy(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    steps: list[PolicyStep],
    group_ids: np.ndarray,
    cfg: DeepRLConfig,
) -> dict[str, float]:
    batch = _to_tensors(steps, cfg.device)
    group_t = torch.as_tensor(group_ids, dtype=torch.long, device=cfg.device)
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []
    approx_kls = []

    for _ in range(cfg.update_epochs):
        logits, values = model(batch["obs"])
        logits = masked_logits(logits, batch["mask"])
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(batch["actions"])
        entropy = dist.entropy().mean()
        adv = _advantages(batch, cfg.method, group_t).detach()
        ratio = torch.exp(logp - batch["old_logp"])
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, batch["returns"])

        if cfg.method == "grpo":
            loss = policy_loss - cfg.entropy_coef * entropy
        else:
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            approx_kl = (batch["old_logp"] - logp).mean().abs()
        losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        entropies.append(float(entropy.item()))
        approx_kls.append(float(approx_kl.item()))

    return {
        "loss": float(np.mean(losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
    }


def _snapshot(
    cfg: DeepRLConfig,
    model: PolicyValueNet,
    run_dir: Path,
    episode: int,
    outcomes: list[int],
    episode_infos: list[dict],
    losses: dict[str, float],
    running: bool,
) -> dict:
    recent = outcomes[-max(1, min(200, len(outcomes))):]
    denom = max(1, len(recent))
    x_wins = sum(1 for outcome in recent if outcome == 1)
    o_wins = sum(1 for outcome in recent if outcome == -1)
    draws = sum(1 for outcome in recent if outcome == 0)
    model_recent = [info for info in episode_infos[-200:] if info.get("learner_result") is not None]
    model_denom = len(model_recent)
    model_wins = sum(1 for info in model_recent if info.get("learner_result") == 1)
    model_losses = sum(1 for info in model_recent if info.get("learner_result") == -1)
    model_draws = sum(1 for info in model_recent if info.get("learner_result") == 0)
    model_x = [info for info in model_recent if info.get("learner_player") == 1]
    model_o = [info for info in model_recent if info.get("learner_player") == -1]
    model_x_wins = sum(1 for info in model_x if info.get("learner_result") == 1)
    model_o_wins = sum(1 for info in model_o if info.get("learner_result") == 1)
    opening = empty_board_policy_analysis(model, cfg.size, device=cfg.device)
    return {
        "running": running,
        "run_dir": str(run_dir),
        "episode": episode,
        "episodes": cfg.episodes,
        "method": cfg.method,
        "run_id": run_id(run_dir),
        "states": episode,
        "mean_abs_update": abs(losses.get("loss", 0.0)),
        "policy_loss": losses.get("policy_loss", 0.0),
        "value_loss": losses.get("value_loss", 0.0),
        "entropy": losses.get("entropy", 0.0),
        "approx_kl": losses.get("approx_kl", 0.0),
        "recent": {
            "window": denom,
            "x_win_rate": x_wins / denom,
            "o_win_rate": o_wins / denom,
            "draw_rate": draws / denom,
        },
        "recent_model": {
            "window": model_denom,
            "win_rate": model_wins / model_denom if model_denom else None,
            "loss_rate": model_losses / model_denom if model_denom else None,
            "draw_rate": model_draws / model_denom if model_denom else None,
            "as_x_games": len(model_x),
            "as_x_win_rate": model_x_wins / len(model_x) if model_x else None,
            "as_o_games": len(model_o),
            "as_o_win_rate": model_o_wins / len(model_o) if model_o else None,
        },
        "value": opening["value"],
        "heatmap": opening["heatmap"],
        "top_moves": opening["top_moves"],
        "config": asdict(cfg),
    }


def _write_analysis(run_dir: Path, latest: dict) -> None:
    recent = latest["recent"]
    model_recent = latest.get("recent_model", {})
    analysis = {
        "run_id": latest["run_id"],
        "method": latest["method"],
        "episodes": latest["episode"],
        "final_recent": recent,
        "final_recent_model": model_recent,
        "policy_loss": latest.get("policy_loss"),
        "value_loss": latest.get("value_loss"),
        "entropy": latest.get("entropy"),
        "approx_kl": latest.get("approx_kl"),
    }
    write_json(run_dir / "analysis.json", analysis)
    (run_dir / "analysis.md").write_text(
        "\n".join(
            [
                f"# Run {latest['run_id']}",
                "",
                f"- Method: `{latest['method']}`",
                f"- Episodes: `{latest['episode']}`",
                f"- Recent X win rate: `{recent['x_win_rate']:.3f}`",
                f"- Recent O win rate: `{recent['o_win_rate']:.3f}`",
                f"- Recent draw rate: `{recent['draw_rate']:.3f}`",
                f"- Recent model win rate: `{model_recent.get('win_rate') if model_recent.get('win_rate') is not None else 'n/a'}`",
                f"- Recent model-as-X win rate: `{model_recent.get('as_x_win_rate') if model_recent.get('as_x_win_rate') is not None else 'n/a'}`",
                f"- Recent model-as-O win rate: `{model_recent.get('as_o_win_rate') if model_recent.get('as_o_win_rate') is not None else 'n/a'}`",
                f"- Policy loss: `{latest.get('policy_loss', 0.0):.5f}`",
                f"- Value loss: `{latest.get('value_loss', 0.0):.5f}`",
                f"- Entropy: `{latest.get('entropy', 0.0):.5f}`",
                f"- Approx KL: `{latest.get('approx_kl', 0.0):.5f}`",
                "",
            ]
        )
    )


def train_deep_rl(
    cfg: DeepRLConfig,
    *,
    callback: SnapshotCallback | None = None,
    stop_event: threading.Event | None = None,
) -> Path:
    method = cfg.method.strip().lower().replace("-", "_")
    if method not in {"ppo", "grpo"}:
        raise ValueError(f"unknown deep RL method {cfg.method!r}")
    cfg = DeepRLConfig(**{**asdict(cfg), "method": method})
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    run_dir = resolve_run_dir(cfg.run_dir)
    write_json(run_dir / "config.json", asdict(cfg))
    write_metadata(run_dir, cfg, method=cfg.method, name=cfg.name, parent_run=cfg.parent_run)

    model = PolicyValueNet(cfg.size, cfg.hidden).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    outcomes: list[int] = []
    episode_infos: list[dict] = []
    episode = 0
    losses = {"loss": math.nan, "policy_loss": math.nan, "value_loss": math.nan, "entropy": math.nan, "approx_kl": math.nan}

    while episode < cfg.episodes:
        if stop_event is not None and stop_event.is_set():
            break

        batch_steps: list[PolicyStep] = []
        group_ids: list[int] = []
        for group in range(cfg.batch_episodes):
            if episode >= cfg.episodes:
                break
            steps, outcome, info = play_episode(model, cfg.size, np_rng, cfg, episode_infos)
            batch_steps.extend(steps)
            group_ids.extend([group] * len(steps))
            outcomes.append(outcome)
            episode_infos.append(info)
            episode += 1

        if batch_steps:
            losses = update_policy(model, optimizer, batch_steps, np.asarray(group_ids), cfg)

        if episode == 1 or episode % cfg.log_every < cfg.batch_episodes or episode >= cfg.episodes:
            latest = _snapshot(cfg, model, run_dir, episode, outcomes, episode_infos, losses, running=True)
            append_jsonl(run_dir / "metrics.jsonl", latest)
            write_json(run_dir / "latest.json", latest)
            if callback is not None:
                callback(latest)

    latest = _snapshot(cfg, model, run_dir, episode, outcomes, episode_infos, losses, running=False)
    write_json(run_dir / "latest.json", latest)
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, run_dir / "model.pt")
    metrics = load_metrics(run_dir / "metrics.jsonl")
    write_plot_artifacts(run_dir, metrics, latest)
    _write_analysis(run_dir, latest)
    write_json(
        run_dir / "artifacts.json",
        {
            "run_id": run_id(run_dir),
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
    if callback is not None:
        callback(latest)
    return run_dir
