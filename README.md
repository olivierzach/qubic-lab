# qubic-lab (4×4×4 tic-tac-toe)

A playground repo for experimenting with multiple approaches to solving **Qubic** (4×4×4 tic-tac-toe):

- classic search baselines (minimax / MCTS)
- AlphaZero-style (policy+value net + self-play + MCTS)
- other RL baselines (DQN, PPO, etc.)

## Why 4×4×4 is a good learning problem

- Big enough that brute force / shallow search won’t trivially solve it.
- Small enough that you can iterate quickly and *actually understand* what’s happening.
- It forces you to deal with:
  - state/action encoding (64 actions)
  - symmetries (lots of rotational/reflection symmetry)
  - self-play dynamics + evaluation
  - search vs learned policy tradeoffs

## Quickstart

```bash
cd qubic-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m qubic_lab.cli.play --opponent random
```

## Offline RL runs

The first runnable RL path is a family of self-play baselines for generalized
`n x n x n` Qubic boards: Q-learning, SARSA, Expected SARSA, Monte Carlo control,
PPO, and a GRPO-style group-relative policy-gradient trainer.
Start with `--size 3`; `--size 4` is supported by the engine, but tabular learning grows
quickly and should be treated as a diagnostic baseline before neural/self-play agents.

```bash
python -m qubic_lab.cli.train_tabular \
  --method q_learning \
  --size 3 \
  --episodes 20000 \
  --log-every 100 \
  --run-dir runs/q3_baseline
```

Run a comparison suite:

```bash
python -m qubic_lab.cli.run_suite \
  --methods q_learning sarsa expected_sarsa monte_carlo \
  --size 3 \
  --episodes 5000 \
  --root runs/suites/q3_baselines
```

Train neural agents:

```bash
python -m qubic_lab.cli.train_deep \
  --method ppo \
  --size 3 \
  --episodes 2000 \
  --batch-episodes 32 \
  --run-dir runs/deep/q3_ppo

python -m qubic_lab.cli.train_deep \
  --method grpo \
  --size 3 \
  --episodes 2000 \
  --batch-episodes 32 \
  --run-dir runs/deep/q3_grpo
```

The PPO implementation uses a small policy-value MLP, clipped policy ratios, value loss,
and entropy regularization. The GRPO-style trainer uses grouped self-play episode returns
as normalized relative advantages, omitting the value term to keep the method close to the
group-relative idea.

Each run writes:

- `config.json`
- `metadata.json`
- `metrics.jsonl`
- `latest.json`
- `analysis.json` and `analysis.md`
- `curves.png` and `curves.svg`
- `first_move_heatmap.png`
- `first_move_policy.json`
- `q_table.npz`
- `model.pt` for neural PPO/GRPO runs

`metadata.json` carries the method, timestamp, git commit, and optional parent run, so
you can build lineage from coarse sweeps into longer offline runs.

## Optimization dashboard

Build and run the local dashboard:

```bash
cd web
npm install
npm run build
cd ..
uvicorn qubic_lab.web:app --reload --port 8011
```

Open `http://127.0.0.1:8011/lab`, choose a method, board size, and training budget, and
start a run. The lab can also load saved offline runs under `runs/`, showing rolling
X/O/draw rates, generated artifacts, and the learned empty-board policy/value heatmap for
each `z` layer. The live run state keeps the latest snapshot plus a bounded timeline of
recent metrics so the frontend can render progress charts and heatmap-oriented inspection
without reading run files directly.

The React UI direction is split into two focused apps:

- `/lab`: configure short tabular, PPO, or GRPO runs; browse saved runs; inspect
  `latest.json`, recent `metrics.jsonl` history, plots, model artifacts, tournaments, and
  generated self-play dataset artifacts.
- `/play`: play against a selected model on the 3D board with value overlays,
  top-move arrows, and explicit coordinate buttons for every legal move.

The backend API surface used by those apps is intentionally small:

- `GET /api/run/defaults` returns method groups and validated default parameters for
  configurable tabular, PPO, and GRPO runs.
- `POST /api/start` starts a configurable run from JSON parameters such as `method`,
  `size`, `episodes`, `seed`, `log_every`, and method-specific learning settings.
- `GET /api/state` returns whether a run is active, the latest training snapshot, and the
  recent timeline history.
- `GET /api/runs` and `GET /api/run?run_dir=runs/...` list saved runs and inspect one
  run's latest snapshot plus recent metric history.
- `GET /api/model/timeline?run_dir=runs/...` returns a compact model/run timeline for
  heatmap and metric inspection.
- `GET /api/models`, `POST /api/analyze/position`, `POST /api/play/new`, and
  `POST /api/play/move` support model selection, position analysis, heatmap data, and
  interactive play.
- `POST /api/eval/tournament` and `POST /api/selfplay/generate` produce evaluation and
  dataset artifacts for later inspection.

Generate offline self-play data directly:

```bash
python -m qubic_lab.cli.generate_selfplay --model-id random --size 3 --games 100
```

Datasets are written under `runs/datasets/<timestamp>/` with `manifest.json` and
`dataset.jsonl`. Each row stores the pre-move board, acting player, legal moves, chosen
action, model probability/value, final winner, and return from the acting player's
perspective. Current PPO/GRPO runs still train from online self-play rollouts; the dataset
artifact is the foundation for the next offline imitation/RL stage.

## Milestones

1. ✅ Generalized Qubic engine + win-line generation + unit tests
2. ✅ CLI to play (human vs random)
3. ✅ Tabular Q-learning, SARSA, Expected SARSA, and Monte Carlo baselines
4. ✅ Run lineage metadata + analysis artifacts
5. ✅ Local dashboard for live and saved runs with value heatmaps
6. ✅ Neural PPO and GRPO-style self-play trainers
7. ⏳ MCTS baseline (no NN)
8. ⏳ Training loop + evaluation gate + checkpoints
9. ⏳ Web UI to play against trained agents
