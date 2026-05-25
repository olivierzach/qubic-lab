import numpy as np

from qubic_lab.game import State, apply_move, legal_moves, terminal, winning_lines, xyz_to_idx


def test_winning_lines_count_reasonable():
    lines = winning_lines()
    # Known value for 4x4x4 qubic is 76 winning lines.
    assert len(lines) == 76


def test_simple_axis_win():
    s = State.new()
    # X plays a line along x at y=0,z=0
    for x in range(4):
        s = apply_move(s, xyz_to_idx(x, 0, 0))  # X
        if x != 3:
            # O plays a dummy move elsewhere
            s = apply_move(s, xyz_to_idx(x, 1, 0))

    done, w = terminal(s)
    assert done
    assert w == 1


def test_legal_moves_use_public_move_indexing():
    s = apply_move(State.new(size=3), xyz_to_idx(2, 1, 0, size=3))

    assert xyz_to_idx(2, 1, 0, size=3) not in set(map(int, legal_moves(s)))
    assert xyz_to_idx(0, 1, 2, size=3) in set(map(int, legal_moves(s)))


def test_three_by_three_diagonal_win():
    s = State.new(size=3)
    for i in range(3):
        s = apply_move(s, xyz_to_idx(i, i, i, size=3))
        if i != 2:
            s = apply_move(s, xyz_to_idx(i, (i + 1) % 3, 0, size=3))

    done, w = terminal(s)
    assert done
    assert w == 1
