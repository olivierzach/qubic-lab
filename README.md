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

The first runnable RL path is a family of tabular self-play baselines for generalized
`n x n x n` Qubic boards: Q-learning, SARSA, Expected SARSA, and Monte Carlo control.
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

`metadata.json` carries the method, timestamp, git commit, and optional parent run, so
you can build lineage from coarse sweeps into longer offline runs.

## Optimization dashboard

Run the local dashboard:

```bash
uvicorn qubic_lab.web:app --reload --port 8011
```

Open `http://127.0.0.1:8011`, choose a method, board size, and training budget, and start
a run. The dashboard can also load saved offline runs under `runs/`, showing rolling
X/O/draw rates, generated artifacts, and the learned empty-board value heatmap for each
`z` layer.

## Milestones

1. ✅ Generalized Qubic engine + win-line generation + unit tests
2. ✅ CLI to play (human vs random)
3. ✅ Tabular Q-learning, SARSA, Expected SARSA, and Monte Carlo baselines
4. ✅ Run lineage metadata + analysis artifacts
5. ✅ Local dashboard for live and saved runs with value heatmaps
6. ⏳ MCTS baseline (no NN)
7. ⏳ Policy+value net + self-play data generation
8. ⏳ Training loop + evaluation gate + checkpoints
9. ⏳ Web UI to play against trained agents
