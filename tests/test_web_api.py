import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from qubic_lab.web import analyze, artifacts, model_timeline, models, play_new, run, run_defaults, runs
from qubic_lab.web import selfplay_generate
from qubic_lab.web import reset_step_run, step_run


class JsonRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def _json(response):
    return json.loads(response.body)


def _write_run(root: Path, name: str = "demo") -> Path:
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    latest = {
        "run_id": name,
        "run_dir": str(run_dir),
        "method": "q_learning",
        "episode": 3,
        "config": {"method": "q_learning", "size": 3, "episodes": 10},
    }
    metadata = {"created_at": "2026-05-29T12:00:00Z", "method": "q_learning"}
    (run_dir / "latest.json").write_text(json.dumps(latest) + "\n")
    (run_dir / "metadata.json").write_text(json.dumps(metadata) + "\n")
    (run_dir / "analysis.json").write_text("{}\n")
    (run_dir / "curves.png").write_bytes(b"png")
    metrics = [{"episode": idx, "x_win_rate": idx / 1000} for idx in range(505)]
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n")
    return run_dir


def test_runs_api_lists_metadata_and_inspects_recent_history(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = _write_run(tmp_path)

    listed = _json(runs())["runs"]
    assert len(listed) == 1
    assert listed[0]["run_id"] == "demo"
    assert listed[0]["created_at"] == "2026-05-29T12:00:00Z"
    assert listed[0]["metadata"]["method"] == "q_learning"

    body = _json(run(str(run_dir)))
    assert body["latest"]["config"]["size"] == 3
    assert len(body["history"]) == 500
    assert body["history"][0]["episode"] == 5
    assert body["history"][-1]["episode"] == 504
    assert {item["file"] for item in body["artifacts"]} >= {"analysis.json", "curves.png"}

    artifact_body = _json(artifacts(str(run_dir)))
    assert artifact_body["run_dir"] == str(run_dir)
    assert artifact_body["artifacts"]


def test_run_defaults_and_model_timeline_are_frontend_ready(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = _write_run(tmp_path)

    defaults = _json(run_defaults())
    assert "q_learning" in defaults["groups"]["tabular"]
    assert "ppo" in defaults["groups"]["policy_gradient"]
    assert defaults["defaults"]["q_learning"]["size"] == 3
    assert defaults["defaults"]["ppo"]["batch_episodes"] > 0

    random_timeline = _json(model_timeline(model_id="random"))
    assert random_timeline == {"run_dir": None, "latest": None, "snapshots": [], "config": {}}

    timeline = _json(model_timeline(run_dir=str(run_dir)))
    assert timeline["run_dir"] == str(run_dir)
    assert timeline["config"]["method"] == "q_learning"
    assert len(timeline["snapshots"]) == 500
    assert timeline["snapshots"][0]["episode"] == 5
    assert {item["file"] for item in timeline["artifacts"]} >= {"analysis.json", "curves.png"}


def test_run_api_rejects_paths_outside_runs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(HTTPException) as exc_info:
        run(str(outside))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "run_dir must be under runs/"


def test_model_analysis_and_play_endpoints_return_board_payloads(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert _json(models())["models"][0]["id"] == "random"

    analysis = _json(
        asyncio.run(analyze(JsonRequest({"model_id": "random", "size": 3, "player": 1})))
    )
    assert analysis["model"]["id"] == "random"
    assert len(analysis["legal_moves"]) == 27
    assert len(analysis["heatmap"]) == 3
    assert len(analysis["top_moves"]) == 10

    payload = _json(
        asyncio.run(play_new(JsonRequest({"model_id": "random", "size": 3, "human_player": -1})))
    )
    assert payload["history"][0]["player"] == 1
    assert payload["state"]["player"] == -1


def test_selfplay_generate_endpoint_writes_dataset_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manifest = _json(
        asyncio.run(
            selfplay_generate(
                JsonRequest(
                    {"model_id": "random", "size": 3, "games": 1, "seed": 7, "greedy": True}
                )
            )
        )
    )

    assert manifest["config"]["model_id"] == "random"
    assert manifest["games"] == 1
    assert manifest["positions"] > 0
    assert manifest["dataset"] == "dataset.jsonl"
    assert "offline imitation/RL dataset" in manifest["notes"][1]
    assert (Path(manifest["run_dir"]) / "dataset.jsonl").exists()


def test_step_run_advances_same_tabular_session(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_step_run()
    payload = {
        "method": "q_learning",
        "size": 3,
        "episodes": 3,
        "log_every": 1,
        "seed": 11,
    }

    first = _json(asyncio.run(step_run(JsonRequest(payload))))
    second = _json(asyncio.run(step_run(JsonRequest(payload))))

    assert first["new_session"] is True
    assert second["new_session"] is False
    assert first["latest"]["episode"] == 1
    assert second["latest"]["episode"] == 2
    assert len(second["history"]) == 2
    run_dir = Path(second["run_dir"])
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "q_table.npz").exists()
    assert second["artifacts"]
