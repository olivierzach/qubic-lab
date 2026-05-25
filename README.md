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

## Tabular RL baseline

The first runnable RL path is a tabular negamax Q-learning baseline for generalized
`n x n x n` Qubic boards. Start with `--size 3`; `--size 4` is supported by the engine,
but tabular learning grows quickly and should be treated as a baseline rather than the
final solver path.

```bash
python -m qubic_lab.cli.train_tabular \
  --size 3 \
  --episodes 20000 \
  --log-every 100 \
  --run-dir runs/q3_baseline
```

Each run writes:

- `config.json`
- `metrics.jsonl`
- `latest.json`
- `q_table.npz`

## Optimization dashboard

Run the local dashboard:

```bash
uvicorn qubic_lab.web:app --reload --port 8011
```

Open `http://127.0.0.1:8011`, choose a board size and training budget, and start a run.
The dashboard shows the rolling X/O/draw rates and the learned empty-board value heatmap
for each `z` layer.

## Milestones

1. ✅ Generalized Qubic engine + win-line generation + unit tests
2. ✅ CLI to play (human vs random)
3. ✅ Tabular negamax Q-learning baseline + run artifacts
4. ✅ Local dashboard for live runs and value heatmaps
5. ⏳ MCTS baseline (no NN)
6. ⏳ Policy+value net + self-play data generation
7. ⏳ Training loop + evaluation gate + checkpoints
8. ⏳ Web UI to play against trained agents
