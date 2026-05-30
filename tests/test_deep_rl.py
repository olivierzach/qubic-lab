from pathlib import Path
import json

from qubic_lab.rl_deep import DeepRLConfig, train_deep_rl


def test_deep_ppo_smoke_run(tmp_path: Path):
    run_dir = tmp_path / "ppo"

    train_deep_rl(
        DeepRLConfig(
            method="ppo",
            size=3,
            episodes=8,
            batch_episodes=4,
            update_epochs=1,
            hidden=32,
            log_every=4,
            run_dir=str(run_dir),
        )
    )

    assert (run_dir / "latest.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "curves.png").exists()
    assert (run_dir / "first_move_heatmap.png").exists()
    latest = json.loads((run_dir / "latest.json").read_text())
    assert "value" in latest
    assert len(latest["top_moves"]) == 10
