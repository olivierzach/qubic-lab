import numpy as np

from qubic_lab.game import State, idx_to_xyz
from qubic_lab.model_api import analyze_position, board_to_layers, play_game, run_tournament


def test_analyze_random_empty_board():
    result = analyze_position("random", State.new(3))

    assert result["player"] == 1
    assert len(result["legal_moves"]) == 27
    assert len(result["top_moves"]) == 10
    assert result["heatmap"][0][0][0] > 0


def test_analyze_mcts_empty_board():
    result = analyze_position("mcts", State.new(3))

    assert result["model"]["kind"] == "mcts"
    assert len(result["legal_moves"]) == 27
    assert len(result["top_moves"]) == 10
    assert sum(move["prob"] for move in result["top_moves"]) > 0


def test_tactical_baseline_takes_immediate_win():
    board = np.zeros((3, 3, 3), dtype=np.int8)
    for move in [0, 1]:
        x, y, z = idx_to_xyz(move, 3)
        board[x, y, z] = 1

    result = analyze_position("tactical", State(board=board, player=1))

    assert result["top_moves"][0]["move"] == 2
    assert result["top_moves"][0]["prob"] == 1.0


def test_tactical_baseline_blocks_immediate_loss():
    board = np.zeros((3, 3, 3), dtype=np.int8)
    for move in [0, 1]:
        x, y, z = idx_to_xyz(move, 3)
        board[x, y, z] = -1

    result = analyze_position("tactical", State(board=board, player=1))

    assert result["top_moves"][0]["move"] == 2
    assert result["top_moves"][0]["prob"] == 1.0


def test_play_game_model_starts_when_human_is_o():
    result = play_game("random", 3, -1, [])

    assert result["history"]
    assert result["history"][0]["player"] == 1
    assert result["state"]["player"] == -1


def test_tournament_random_smoke():
    result = run_tournament(["random"], size=3, games=2, seed=0)

    assert result["leaderboard"]
    assert result["matches"] == []


def test_tournament_against_tactical_smoke():
    result = run_tournament(["random", "tactical"], size=3, games=2, seed=0)

    assert result["matches"]
    assert result["matches"][0]["wins"]["tactical"] >= 0


def test_tournament_against_mcts_smoke():
    result = run_tournament(["random", "mcts"], size=3, games=1, seed=0)

    assert result["matches"]
    assert "mcts" in result["matches"][0]["wins"]


def test_board_layers_shape():
    layers = board_to_layers(State.new(4))

    assert len(layers) == 4
    assert len(layers[0]) == 4
    assert len(layers[0][0]) == 4
