from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from qubic_lab.reporting import generate_report_card, write_report_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Qubic model report card.")
    parser.add_argument("model_id")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--probe-cases-per-family", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_report_card(
        args.model_id,
        run_dir=args.run_dir,
        size=args.size,
        probe_cases_per_family=args.probe_cases_per_family,
        seed=args.seed,
        fast=args.fast,
    )
    console = Console()
    if args.run_dir:
        files = write_report_card(report, Path(args.run_dir))
        console.print(f"wrote report card to {args.run_dir}: {', '.join(files.values())}")
    else:
        console.print_json(data=report)


if __name__ == "__main__":
    main()

