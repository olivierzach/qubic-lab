from __future__ import annotations

from datetime import datetime
from pathlib import Path


def resolve_run_dir(run_dir: str | None, root: str = "runs") -> Path:
    if run_dir:
        path = Path(run_dir).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(root) / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path
