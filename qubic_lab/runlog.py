from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def config_dict(config) -> dict:
    return asdict(config) if is_dataclass(config) else dict(config)


def run_id(run_dir: Path) -> str:
    return run_dir.name


def write_metadata(run_dir: Path, config, *, method: str, name: str | None, parent_run: str | None) -> dict:
    metadata = {
        "run_id": run_id(run_dir),
        "name": name,
        "method": method,
        "parent_run": parent_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "run_dir": str(run_dir),
        "config": config_dict(config),
    }
    write_json(run_dir / "metadata.json", metadata)
    append_run_index(run_dir, metadata)
    return metadata


def append_run_index(run_dir: Path, metadata: dict) -> None:
    index_path = run_dir.parent / "index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            index = {"runs": []}
    else:
        index = {"runs": []}
    index["runs"] = [run for run in index["runs"] if run.get("run_id") != metadata.get("run_id")]
    index["runs"].append(metadata)
    index["runs"].sort(key=lambda item: item.get("created_at", ""))
    write_json(index_path, index)
