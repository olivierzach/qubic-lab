from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Board = np.ndarray  # shape (4,4,4), dtype int8 with values in {-1,0,+1}


def idx_to_xyz(idx: int) -> tuple[int, int, int]:
    assert 0 <= idx < 64
    z = idx // 16
    rem = idx % 16
    y = rem // 4
    x = rem % 4
    return x, y, z


def xyz_to_idx(x: int, y: int, z: int) -> int:
    assert 0 <= x < 4 and 0 <= y < 4 and 0 <= z < 4
    return z * 16 + y * 4 + x


@lru_cache(maxsize=1)
def winning_lines() -> List[Tuple[int, int, int, int]]:
    """All 4-in-a-row lines on a 4x4x4 grid, as tuples of flattened indices."""

    lines: list[tuple[int, int, int, int]] = []

    # Directions: all vectors (dx,dy,dz) with components in {-1,0,1} excluding 0,0,0,
    # but only take a canonical half to avoid duplicates.
    dirs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                # canonical: first nonzero component must be positive
                v = (dx, dy, dz)
                for c in v:
                    if c != 0:
                        if c > 0:
                            dirs.append(v)
                        break

    def inb(x, y, z):
        return 0 <= x < 4 and 0 <= y < 4 and 0 <= z < 4

    for x in range(4):
        for y in range(4):
            for z in range(4):
                for dx, dy, dz in dirs:
                    xs = [x + i * dx for i in range(4)]
                    ys = [y + i * dy for i in range(4)]
                    zs = [z + i * dz for i in range(4)]
                    if all(inb(xs[i], ys[i], zs[i]) for i in range(4)):
                        line = tuple(xyz_to_idx(xs[i], ys[i], zs[i]) for i in range(4))
                        lines.append(line)

    # Unique
    lines = sorted(set(lines))
    return lines


@dataclass(frozen=True)
class State:
    board: Board
    player: int  # +1 (X) or -1 (O)

    @staticmethod
    def new() -> "State":
        return State(board=np.zeros((4, 4, 4), dtype=np.int8), player=+1)

    def clone(self) -> "State":
        return State(board=self.board.copy(), player=self.player)


def legal_moves(s: State) -> np.ndarray:
    """Return legal moves as 1D array of indices [0..63]."""
    flat = s.board.reshape(-1)
    return np.flatnonzero(flat == 0)


def apply_move(s: State, move: int) -> State:
    x, y, z = idx_to_xyz(move)
    if s.board[x, y, z] != 0:
        raise ValueError(f"illegal move {move} at {(x,y,z)}")
    b = s.board.copy()
    b[x, y, z] = s.player
    return State(board=b, player=-s.player)


def winner(s: State) -> Optional[int]:
    """Return +1 or -1 if that player has won, else None."""
    flat = s.board.reshape(-1)
    for line in winning_lines():
        v = flat[list(line)]
        sm = int(v.sum())
        if sm == 4:
            return +1
        if sm == -4:
            return -1
    return None


def terminal(s: State) -> tuple[bool, Optional[int]]:
    w = winner(s)
    if w is not None:
        return True, w
    if (s.board == 0).sum() == 0:
        return True, 0
    return False, None


def render_layers(s: State) -> str:
    """Pretty-print as 4 layers (z=0..3), each a 4x4 grid."""
    def sym(v: int) -> str:
        return "X" if v == 1 else ("O" if v == -1 else ".")

    out = []
    for z in range(4):
        out.append(f"z={z}")
        for y in range(4):
            row = [sym(int(s.board[x, y, z])) for x in range(4)]
            out.append(" ".join(row))
        out.append("")
    out.append(f"to_move: {'X' if s.player==1 else 'O'}")
    return "\n".join(out)
