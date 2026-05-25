from __future__ import annotations

import argparse

from rich.console import Console

from qubic_lab.rl_tabular import TabularConfig, train_tabular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular negamax Q agent for n x n x n Qubic.")
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--epsilon-min", type=float, default=0.03)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TabularConfig(
        size=args.size,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        log_every=args.log_every,
        run_dir=args.run_dir,
    )
    console = Console()

    def on_snapshot(payload: dict) -> None:
        recent = payload["recent"]
        console.print(
            f"ep={payload['episode']}/{payload['episodes']} "
            f"eps={payload['epsilon']:.3f} states={payload['states']} "
            f"X={recent['x_win_rate']:.2f} O={recent['o_win_rate']:.2f} "
            f"D={recent['draw_rate']:.2f}"
        )

    run_dir = train_tabular(cfg, callback=on_snapshot)
    console.print(f"wrote run to {run_dir}")


if __name__ == "__main__":
    main()
