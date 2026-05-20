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

## Milestones

1. ✅ Correct game engine + win-line generation + unit tests
2. ✅ CLI to play (human vs random)
3. ⏳ MCTS baseline (no NN)
4. ⏳ Policy+value net + self-play data generation
5. ⏳ Training loop + evaluation gate + checkpoints
6. ⏳ Web UI / server to play against trained agents

