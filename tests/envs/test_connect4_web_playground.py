# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Connect4 drawing and buttons in the web playground."""

from connect4_env import Connect4Action
from connect4_env.server.connect4_environment import Connect4Environment


def test_connect4_web_playground():
    env = Connect4Environment()
    obs = env.reset().model_dump()
    assert env.web_actions(obs) == [(f"col {c}", {"column": c}) for c in range(7)]

    obs = env.step(Connect4Action(column=3)).model_dump()
    board = env.render_web(obs)
    assert 'aria-label="Connect4 board"' in board
    assert board.count("var(--color-accent)") == 1  # the disc just dropped
    assert board.count("var(--border-color-primary)") == 41 + 1  # holes + frame
    assert "#" not in board  # theme colours only, so it reads in dark mode
    assert env.render_web({}) is None
