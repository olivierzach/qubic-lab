from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from qubic_lab.game import State, flatten_board, idx_to_xyz, legal_moves


def obs_from_state(state: State) -> np.ndarray:
    flat = flatten_board(state.board).astype(np.float32)
    mine = (flat == state.player).astype(np.float32)
    theirs = (flat == -state.player).astype(np.float32)
    empty = (flat == 0).astype(np.float32)
    return np.concatenate([mine, theirs, empty], axis=0)


def legal_mask(state: State) -> np.ndarray:
    mask = np.zeros(state.size**3, dtype=bool)
    mask[legal_moves(state).astype(int)] = True
    return mask


@dataclass(frozen=True)
class PolicyStep:
    obs: np.ndarray
    action: int
    reward: float
    done: bool
    logp: float
    value: float
    mask: np.ndarray
    player: int


class PolicyValueNet(nn.Module):
    def __init__(self, size: int, hidden: int = 128):
        super().__init__()
        n = size**3
        self.size = size
        self.trunk = nn.Sequential(
            nn.Linear(3 * n, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, n)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy(h), self.value(h).squeeze(-1)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask.bool(), -1e9)


def select_action(
    model: PolicyValueNet,
    state: State,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    device: str = "cpu",
) -> tuple[int, float, float, np.ndarray, np.ndarray]:
    obs = obs_from_state(state)
    mask = legal_mask(state)
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        logits, value = model(obs_t)
        logits = masked_logits(logits / max(1e-6, temperature), mask_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    action = int(rng.choice(len(probs), p=probs))
    logp = float(np.log(max(probs[action], 1e-12)))
    return action, logp, float(value.item()), obs, mask


def empty_board_policy_heatmap(model: PolicyValueNet, size: int, *, device: str = "cpu") -> list[list[list[float]]]:
    state = State.new(size)
    obs = obs_from_state(state)
    mask = legal_mask(state)
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        logits, _ = model(obs_t)
        probs = torch.softmax(masked_logits(logits, mask_t), dim=-1).squeeze(0).cpu().numpy()
    layers = np.zeros((size, size, size), dtype=np.float32)
    for idx, value in enumerate(probs):
        x, y, z = idx_to_xyz(idx, size)
        layers[z, y, x] = float(value)
    return layers.round(5).tolist()
