from pathlib import Path
import json

from qubic_lab.alpha_zero import AlphaZeroConfig, train_alpha_zero
from qubic_lab.probes import build_probe_suite, evaluate_probes
from qubic_lab.reporting import generate_report_card, write_report_card


def test_probe_suite_and_tactical_baseline():
    cases = build_probe_suite(3, per_family=4, seed=0)
    families = {case.family for case in cases}

    assert {"immediate_win", "immediate_block"} <= families
    assert all(case.expected_moves for case in cases)

    report = evaluate_probes("tactical", size=3, per_family=4, seed=0)
    assert report["families"]["immediate_win"]["pass_rate"] == 1.0
    assert report["families"]["immediate_block"]["pass_rate"] == 1.0


def test_report_card_writes_artifacts(tmp_path: Path):
    report = generate_report_card("tactical", run_dir=None, size=3, probe_cases_per_family=2, fast=True)
    files = write_report_card(report, tmp_path)

    assert report["probes"]["total"] > 0
    assert report["eval_ladder"]["rows"]
    assert (tmp_path / files["report_card"]).exists()
    assert (tmp_path / files["report_card_markdown"]).exists()
    assert (tmp_path / files["probe_failures"]).exists()


def test_alpha_zero_smoke_run(tmp_path: Path):
    run_dir = tmp_path / "az"

    train_alpha_zero(
        AlphaZeroConfig(
            size=3,
            iterations=1,
            games_per_iteration=2,
            mcts_simulations=2,
            hidden=32,
            batch_size=8,
            update_epochs=1,
            replay_size=100,
            run_dir=str(run_dir),
        )
    )

    latest = json.loads((run_dir / "latest.json").read_text())
    assert latest["method"] == "alpha_zero"
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "report_card.json").exists()
