from __future__ import annotations

import argparse

from rich.console import Console

from qubic_lab.alpha_zero import AlphaZeroConfig, train_alpha_zero


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AlphaZero-lite Qubic agent with MCTS policy targets.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--parent-run", default=None)
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=128)
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AlphaZeroConfig(
        name=args.name,
        parent_run=args.parent_run,
        size=args.size,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        mcts_simulations=args.mcts_simulations,
        hidden=args.hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        replay_size=args.replay_size,
        temperature=args.temperature,
        seed=args.seed,
        log_every=args.log_every,
        run_dir=args.run_dir,
        device=args.device,
    )
    console = Console()

    def on_snapshot(payload: dict) -> None:
        recent = payload["recent"]
        console.print(
            f"alpha_zero iter={payload['iteration']}/{payload['iterations']} "
            f"games={payload['episode']}/{payload['episodes']} "
            f"loss={payload.get('mean_abs_update', 0.0):.4f} "
            f"pi={payload.get('policy_loss', 0.0):.4f} "
            f"v={payload.get('value_loss', 0.0):.4f} "
            f"X={recent['x_win_rate']:.2f} O={recent['o_win_rate']:.2f} D={recent['draw_rate']:.2f}"
        )

    run_dir = train_alpha_zero(cfg, callback=on_snapshot)
    console.print(f"wrote run to {run_dir}")


if __name__ == "__main__":
    main()

