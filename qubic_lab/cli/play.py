from __future__ import annotations

import argparse
import random

from rich.console import Console

from qubic_lab.game import State, apply_move, legal_moves, render_layers, terminal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", choices=["random"], default="random")
    p.add_argument("--size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    s = State.new(args.size)

    while True:
        console.print(render_layers(s))
        done, w = terminal(s)
        if done:
            if w == 1:
                console.print("X wins")
            elif w == -1:
                console.print("O wins")
            else:
                console.print("Draw")
            return

        if s.player == 1:
            # Human plays X
            lm = set(int(x) for x in legal_moves(s))
            mv = None
            while mv not in lm:
                txt = console.input(f"Your move (0-{args.size**3 - 1}): ")
                try:
                    mv = int(txt)
                except ValueError:
                    mv = None
            s = apply_move(s, mv)
        else:
            # Opponent plays O
            lm = list(map(int, legal_moves(s)))
            mv = random.choice(lm)
            console.print(f"Opponent plays: {mv}")
            s = apply_move(s, mv)


if __name__ == "__main__":
    main()
