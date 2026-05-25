from __future__ import annotations

import argparse

from rich.console import Console

from qubic_lab.selfplay import SelfPlayConfig, generate_selfplay_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qubic self-play dataset artifacts.")
    parser.add_argument("--model-id", default="random")
    parser.add_argument("--opponent-id", default=None)
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = generate_selfplay_dataset(
        SelfPlayConfig(
            model_id=args.model_id,
            opponent_id=args.opponent_id,
            size=args.size,
            games=args.games,
            seed=args.seed,
            greedy=args.greedy,
            run_dir=args.run_dir,
        )
    )
    Console().print(f"wrote self-play dataset to {run_dir}")


if __name__ == "__main__":
    main()
