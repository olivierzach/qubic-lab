from pathlib import Path

from qubic_lab.rl_tabular import TabularConfig, train_tabular


def test_tabular_smoke_run(tmp_path: Path):
    run_dir = tmp_path / "run"

    train_tabular(TabularConfig(size=3, episodes=20, log_every=5, run_dir=str(run_dir)))

    assert (run_dir / "config.json").exists()
    assert (run_dir / "latest.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "q_table.npz").exists()
