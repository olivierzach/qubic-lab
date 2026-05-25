from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from qubic_lab.game import idx_to_xyz


def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_plot_artifacts(run_dir: Path, metrics: list[dict], latest: dict) -> None:
    """Write matplotlib/seaborn artifacts for offline inspection.

    Imports live inside the function so non-plotting code remains cheap to import.
    """
    mpl_config = run_dir / ".mplconfig"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="darkgrid")
    if metrics:
        rows = []
        for item in metrics:
            rows.append(
                {
                    "episode": item["episode"],
                    "x_win_rate": item["recent"]["x_win_rate"],
                    "o_win_rate": item["recent"]["o_win_rate"],
                    "draw_rate": item["recent"]["draw_rate"],
                    "mean_abs_update": item.get("mean_abs_update", 0.0),
                    "states": item.get("states", 0),
                }
            )
        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        sns.lineplot(data=df, x="episode", y="x_win_rate", ax=axes[0], label="X win")
        sns.lineplot(data=df, x="episode", y="o_win_rate", ax=axes[0], label="O win")
        sns.lineplot(data=df, x="episode", y="draw_rate", ax=axes[0], label="draw")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("recent rate")

        sns.lineplot(data=df, x="episode", y="mean_abs_update", ax=axes[1], color="#c7902c")
        axes[1].set_ylabel("mean |TD/update|")

        sns.lineplot(data=df, x="episode", y="states", ax=axes[2], color="#6f8fcf")
        axes[2].set_ylabel("states")
        axes[2].set_xlabel("episode")
        fig.suptitle(f"{latest.get('method', 'run')} training curves")
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=150)
        plt.close(fig)

    heatmap = np.array(latest.get("heatmap", []), dtype=float)
    if heatmap.ndim == 3 and heatmap.size:
        size = heatmap.shape[-1]
        fig, axes = plt.subplots(1, heatmap.shape[0], figsize=(4 * heatmap.shape[0], 4))
        if heatmap.shape[0] == 1:
            axes = [axes]
        vmax = max(0.05, float(np.nanmax(np.abs(heatmap))))
        for z, ax in enumerate(axes):
            sns.heatmap(
                heatmap[z],
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="vlag",
                center=0.0,
                vmin=-vmax,
                vmax=vmax,
                cbar=z == heatmap.shape[0] - 1,
                square=True,
            )
            ax.set_title(f"z={z}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        fig.suptitle("Empty-board first-move values")
        fig.tight_layout()
        fig.savefig(run_dir / "first_move_heatmap.png", dpi=150)
        plt.close(fig)

        flat = heatmap.reshape(-1)
        order = np.argsort(flat)[::-1]
        top = []
        for idx in order[: min(12, len(order))]:
            x, y, z = idx_to_xyz(int(idx), size=size)
            top.append({"move": int(idx), "x": x, "y": y, "z": z, "value": float(flat[idx])})
        (run_dir / "first_move_policy.json").write_text(json.dumps({"top_moves": top}, indent=2) + "\n")
