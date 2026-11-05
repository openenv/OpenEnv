# SPDX-License-Identifier: BSD-3-Clause
"""Check summaries of rejected Sudoku moves."""

import json
import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("row,col", [(10, 1), (1, 10), (0, 1), (1, 0), (-1, 1)])
def test_first_move_outside_board_is_not_a_given_cell(
    tmp_path, monkeypatch, capsys, row, col
):
    game = {
        "puzzle": [[1] * 9 for _ in range(9)],
        "solved": False,
        "filled": 0,
        "empty": 1,
        "mistakes": 1,
        "moves": [{"row": row, "col": col, "valid": False}],
    }
    path = tmp_path / "games.jsonl"
    path.write_text(json.dumps(game) + "\n")
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(path)])

    runpy.run_path(str(ROOT / "examples/sudoku/summarize.py"), run_name="__main__")

    assert "first move on a given cell  0" in capsys.readouterr().out
