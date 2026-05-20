import numpy as np

from qubic_lab.game import State, apply_move, terminal, winning_lines, xyz_to_idx


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
