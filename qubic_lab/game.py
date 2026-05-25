from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

Board = np.ndarray  # shape (n,n,n), dtype int8 with values in {-1,0,+1}


def flatten_board(board: Board) -> np.ndarray:
    """Flatten with indices matching xyz_to_idx: z-major, then y, then x."""
    return board.transpose(2, 1, 0).reshape(-1)


def idx_to_xyz(idx: int, size: int = 4) -> tuple[int, int, int]:
    assert size >= 2
    assert 0 <= idx < size**3
    z = idx // (size * size)
    rem = idx % (size * size)
    y = rem // size
    x = rem % size
    return x, y, z


def xyz_to_idx(x: int, y: int, z: int, size: int = 4) -> int:
    assert size >= 2
    assert 0 <= x < size and 0 <= y < size and 0 <= z < size
    return z * size * size + y * size + x


@lru_cache(maxsize=None)
def winning_lines(size: int = 4) -> List[Tuple[int, ...]]:
    """All full-length lines on an n x n x n grid, as tuples of flattened indices."""

    if size < 2:
        raise ValueError("size must be at least 2")

    lines: list[tuple[int, ...]] = []

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
        return 0 <= x < size and 0 <= y < size and 0 <= z < size

    for x in range(size):
        for y in range(size):
            for z in range(size):
                for dx, dy, dz in dirs:
                    xs = [x + i * dx for i in range(size)]
                    ys = [y + i * dy for i in range(size)]
                    zs = [z + i * dz for i in range(size)]
                    if all(inb(xs[i], ys[i], zs[i]) for i in range(size)):
                        line = tuple(
                            xyz_to_idx(xs[i], ys[i], zs[i], size=size) for i in range(size)
                        )
                        lines.append(line)

    # Unique
    lines = sorted(set(lines))
    return lines


@dataclass(frozen=True)
class State:
    board: Board
    player: int  # +1 (X) or -1 (O)

    @staticmethod
    def new(size: int = 4) -> "State":
        if size < 2:
            raise ValueError("size must be at least 2")
        return State(board=np.zeros((size, size, size), dtype=np.int8), player=+1)

    @property
    def size(self) -> int:
        return int(self.board.shape[0])

    def clone(self) -> "State":
        return State(board=self.board.copy(), player=self.player)


def legal_moves(s: State) -> np.ndarray:
    """Return legal moves as 1D array of indices [0..63]."""
    flat = flatten_board(s.board)
    return np.flatnonzero(flat == 0)


def apply_move(s: State, move: int) -> State:
    x, y, z = idx_to_xyz(move, size=s.size)
    if s.board[x, y, z] != 0:
        raise ValueError(f"illegal move {move} at {(x,y,z)}")
    b = s.board.copy()
    b[x, y, z] = s.player
    return State(board=b, player=-s.player)


def winner(s: State) -> Optional[int]:
    """Return +1 or -1 if that player has won, else None."""
    flat = flatten_board(s.board)
    target = s.size
    for line in winning_lines(s.size):
        v = flat[list(line)]
        sm = int(v.sum())
        if sm == target:
            return +1
        if sm == -target:
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
    """Pretty-print as n layers, each an n x n grid."""
    def sym(v: int) -> str:
        return "X" if v == 1 else ("O" if v == -1 else ".")

    out = []
    for z in range(s.size):
        out.append(f"z={z}")
        for y in range(s.size):
            row = [sym(int(s.board[x, y, z])) for x in range(s.size)]
            out.append(" ".join(row))
        out.append("")
    out.append(f"to_move: {'X' if s.player==1 else 'O'}")
    return "\n".join(out)
