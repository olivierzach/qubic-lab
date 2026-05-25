from pathlib import Path

from qubic_lab.selfplay import SelfPlayConfig, generate_selfplay_dataset


def test_generate_random_selfplay_dataset(tmp_path: Path):
    run_dir = tmp_path / "dataset"

    generate_selfplay_dataset(
        SelfPlayConfig(model_id="random", size=3, games=2, seed=1, run_dir=str(run_dir))
    )

    dataset = run_dir / "dataset.jsonl"
    assert dataset.exists()
    assert (run_dir / "manifest.json").exists()
    rows = [line for line in dataset.read_text().splitlines() if line.strip()]
    assert rows
