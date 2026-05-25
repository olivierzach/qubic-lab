from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from qubic_lab.rl_tabular import TabularConfig, train_tabular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small suite of tabular Qubic RL methods.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["q_learning", "sarsa", "expected_sarsa", "monte_carlo"],
        choices=["q_learning", "sarsa", "expected_sarsa", "monte_carlo"],
    )
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default="runs/suites")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    results = []

    for offset, method in enumerate(args.methods):
        run_dir = root / f"{method}_s{args.size}_seed{args.seed + offset}"
        cfg = TabularConfig(
            method=method,
            name=f"suite-{method}",
            size=args.size,
            episodes=args.episodes,
            seed=args.seed + offset,
            log_every=args.log_every,
            run_dir=str(run_dir),
        )
        console.print(f"starting {method} -> {run_dir}")
        train_tabular(cfg)
        latest = json.loads((run_dir / "latest.json").read_text())
        results.append(latest)

    table = Table(title="Qubic RL suite")
    table.add_column("method")
    table.add_column("run")
    table.add_column("episodes", justify="right")
    table.add_column("states", justify="right")
    table.add_column("X win", justify="right")
    table.add_column("O win", justify="right")
    table.add_column("draw", justify="right")
    table.add_column("update", justify="right")
    for item in results:
        recent = item["recent"]
        table.add_row(
            item["method"],
            item["run_id"],
            str(item["episode"]),
            str(item["states"]),
            f"{recent['x_win_rate']:.3f}",
            f"{recent['o_win_rate']:.3f}",
            f"{recent['draw_rate']:.3f}",
            f"{item.get('mean_abs_update', 0.0):.4f}",
        )
    console.print(table)


if __name__ == "__main__":
    main()
