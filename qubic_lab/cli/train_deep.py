from __future__ import annotations

import argparse

from rich.console import Console

from qubic_lab.rl_deep import DeepRLConfig, train_deep_rl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural PPO/GRPO-style Qubic agents.")
    parser.add_argument("--method", choices=["ppo", "grpo"], default="ppo")
    parser.add_argument("--name", default=None)
    parser.add_argument("--parent-run", default=None)
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batch-episodes", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--advantage-mode", choices=["gae", "mc"], default="gae")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--opponent-mix", default="self")
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DeepRLConfig(
        method=args.method,
        name=args.name,
        parent_run=args.parent_run,
        size=args.size,
        episodes=args.episodes,
        batch_episodes=args.batch_episodes,
        update_epochs=args.update_epochs,
        hidden=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        temperature=args.temperature,
        advantage_mode=args.advantage_mode,
        gae_lambda=args.gae_lambda,
        opponent_mix=args.opponent_mix,
        mcts_simulations=args.mcts_simulations,
        seed=args.seed,
        log_every=args.log_every,
        run_dir=args.run_dir,
        device=args.device,
    )
    console = Console()

    def on_snapshot(payload: dict) -> None:
        recent = payload["recent"]
        console.print(
            f"{payload['method']} ep={payload['episode']}/{payload['episodes']} "
            f"loss={payload.get('mean_abs_update', 0.0):.4f} "
            f"pi={payload.get('policy_loss', 0.0):.4f} "
            f"v={payload.get('value_loss', 0.0):.4f} "
            f"H={payload.get('entropy', 0.0):.3f} "
            f"X={recent['x_win_rate']:.2f} O={recent['o_win_rate']:.2f} "
            f"D={recent['draw_rate']:.2f}"
        )

    run_dir = train_deep_rl(cfg, callback=on_snapshot)
    console.print(f"wrote run to {run_dir}")


if __name__ == "__main__":
    main()
