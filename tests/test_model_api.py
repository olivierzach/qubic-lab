from qubic_lab.game import State
from qubic_lab.model_api import analyze_position, board_to_layers, play_game, run_tournament


def test_analyze_random_empty_board():
    result = analyze_position("random", State.new(3))

    assert result["player"] == 1
    assert len(result["legal_moves"]) == 27
    assert len(result["top_moves"]) == 10
    assert result["heatmap"][0][0][0] > 0


def test_play_game_model_starts_when_human_is_o():
    result = play_game("random", 3, -1, [])

    assert result["history"]
    assert result["history"][0]["player"] == 1
    assert result["state"]["player"] == -1


def test_tournament_random_smoke():
    result = run_tournament(["random"], size=3, games=2, seed=0)

    assert result["leaderboard"]
    assert result["matches"] == []


def test_board_layers_shape():
    layers = board_to_layers(State.new(4))

    assert len(layers) == 4
    assert len(layers[0]) == 4
    assert len(layers[0][0]) == 4
