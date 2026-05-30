from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from qubic_lab.model_api import evaluate_match, load_model
from qubic_lab.probes import evaluate_probes, write_probe_failures_replay
from qubic_lab.runlog import run_id


def _rate(wins: int, games: int) -> float:
    return wins / max(1, games)


def eval_ladder(
    model_id: str,
    *,
    size: int = 3,
    seed: int = 0,
    random_games: int = 200,
    tactical_games: int = 200,
    mcts64_games: int = 100,
    mcts256_games: int = 50,
) -> dict[str, Any]:
    opponents = [
        ("random", random_games),
        ("tactical", tactical_games),
        ("mcts:64", mcts64_games),
        ("mcts:256", mcts256_games),
    ]
    rows = []
    for idx, (opponent_id, games) in enumerate(opponents):
        match = evaluate_match(model_id, opponent_id, size=size, games=games, seed=seed + idx)
        wins = match["wins"]
        side = match.get("side_wins", {}).get(model_id, {})
        rows.append(
            {
                "opponent": opponent_id,
                "games": games,
                "wins": int(wins.get(model_id, 0)),
                "draws": int(wins.get("draw", 0)),
                "losses": int(wins.get(opponent_id, 0)),
                "win_rate": _rate(int(wins.get(model_id, 0)), games),
                "as_x_win_rate": _rate(int(side.get("as_x", 0)), int(side.get("as_x_games", 0))),
                "as_o_win_rate": _rate(int(side.get("as_o", 0)), int(side.get("as_o_games", 0))),
            }
        )
    return {"model_id": model_id, "size": size, "rows": rows}


def _state_coverage(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not rows:
        return {}
    latest = rows[-1]
    return {
        "snapshots": len(rows),
        "episodes": latest.get("episode"),
        "recent": latest.get("recent", {}),
        "recent_model": latest.get("recent_model", {}),
        "entropy": latest.get("entropy"),
        "value_loss": latest.get("value_loss"),
        "policy_loss": latest.get("policy_loss"),
    }


def generate_report_card(
    model_id: str,
    *,
    run_dir: str | Path | None = None,
    size: int | None = None,
    probe_cases_per_family: int = 16,
    seed: int = 0,
    fast: bool = False,
) -> dict[str, Any]:
    model = load_model(model_id)
    resolved_size = int(size or model.record.size or 3)
    ladder_kwargs = {"random_games": 20, "tactical_games": 20, "mcts64_games": 8, "mcts256_games": 4} if fast else {}
    probes = evaluate_probes(model_id, size=resolved_size, per_family=probe_cases_per_family, seed=seed)
    ladder = eval_ladder(model_id, size=resolved_size, seed=seed, **ladder_kwargs)
    path = Path(run_dir or model.record.run_dir) if (run_dir or model.record.run_dir) else None
    return {
        "created_at": time.time(),
        "model": model.record.__dict__,
        "model_id": model_id,
        "run_id": run_id(path) if path else None,
        "size": resolved_size,
        "probes": probes,
        "eval_ladder": ladder,
        "state_coverage": _state_coverage(path) if path else {},
    }


def write_report_card(report: dict[str, Any], run_dir: str | Path) -> dict[str, str]:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "report_card.json"
    md_path = path / "report_card.md"
    failures_path = path / "probe_failures.jsonl"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_probe_failures_replay(report["probes"], failures_path)
    md_path.write_text(_report_markdown(report))
    return {
        "report_card": "report_card.json",
        "report_card_markdown": "report_card.md",
        "probe_failures": "probe_failures.jsonl",
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Report Card {report.get('run_id') or report.get('model_id')}",
        "",
        f"- Model: `{report.get('model_id')}`",
        f"- Probe pass rate: `{report['probes']['pass_rate']:.3f}`",
        "",
        "## Probes",
        "",
    ]
    for name, row in sorted(report["probes"]["families"].items()):
        lines.append(f"- {name}: `{row['passed']}/{row['total']}` (`{row['pass_rate']:.3f}`)")
    lines.extend(["", "## Evaluation", ""])
    for row in report["eval_ladder"]["rows"]:
        lines.append(
            "- "
            f"{row['opponent']}: W `{row['wins']}` D `{row['draws']}` L `{row['losses']}` "
            f"win `{row['win_rate']:.3f}` X `{row['as_x_win_rate']:.3f}` O `{row['as_o_win_rate']:.3f}`"
        )
    lines.append("")
    return "\n".join(lines)

