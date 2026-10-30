# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test suite for Minesweeper Environment."""

import socket
import subprocess
import sys
import tempfile
import time
import unittest

import pytest
import requests
from envs.minesweeper_env import (
    GameStatus,
    MinesweeperAction,
    MinesweeperEnv,
    MinesweeperObservation,
)
from envs.minesweeper_env.server.minesweeper_environment import MinesweeperEnvironment


class TestMinesweeperEnv(unittest.IsolatedAsyncioTestCase):
    """Test cases for the Minesweeper environment."""

    server_process = None

    @classmethod
    def setUpClass(cls):
        """Start the server once for all tests."""
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.base_url = f"http://127.0.0.1:{port}"
        server_log = tempfile.TemporaryFile(mode="w+")
        cls.addClassCleanup(server_log.close)
        cls.server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "envs.minesweeper_env.server.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        # unittest skips tearDownClass when setUpClass fails.
        cls.addClassCleanup(cls.tearDownClass)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if cls.server_process.poll() is not None:
                break
            try:
                response = requests.get(cls.base_url + "/health", timeout=0.5)
                if response.status_code == 200:
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.1)

        cls.tearDownClass()
        server_log.seek(0)
        raise RuntimeError(f"Server did not become healthy:\n{server_log.read()}")

    @classmethod
    def tearDownClass(cls):
        """Clean up server after all tests, including failed setup."""
        if cls.server_process is not None:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
                cls.server_process.wait(timeout=5)

    async def test_minesweeper_env_client(self):
        """Test Minesweeper environment client initialization."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            assert isinstance(client, MinesweeperEnv)

    async def test_minesweeper_initial_state(self):
        """Test the initial state after reset."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            result = await client.reset()
            observation = result.observation

            # Check observation type and attributes
            assert isinstance(observation, MinesweeperObservation)
            assert isinstance(observation.board, list)
            assert isinstance(observation.done, bool)
            assert isinstance(observation.reward, float)
            assert isinstance(observation.num_mines, int)
            assert isinstance(observation.flags_placed, int)
            assert isinstance(observation.cells_revealed, int)
            # game_status may be int or GameStatus enum due to HTTP serialization
            assert isinstance(observation.game_status, (GameStatus, int))

            # Check initial state values
            assert observation.done is False
            assert observation.reward == 0.0
            assert observation.flags_placed == 0
            assert observation.cells_revealed == 0
            # Compare with enum value (handles both int and enum)
            assert (
                observation.game_status == GameStatus.ONGOING.value
                or observation.game_status == GameStatus.ONGOING
            )

            # Check board structure (default 5x5)
            assert len(observation.board) == 5  # 5 rows
            assert all(len(row) == 5 for row in observation.board)  # 5 columns

            # Check all cells are initially unrevealed
            assert all(cell == -1 for row in observation.board for cell in row), (
                "All cells should be unrevealed (-1) at start"
            )

    async def test_reveal_action(self):
        """Test revealing a cell."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            await client.reset()

            # Try revealing a cell
            action = MinesweeperAction(row=0, col=0, action_type="reveal")
            result = await client.step(action)
            observation = result.observation

            assert isinstance(observation, MinesweeperObservation)
            assert observation.cells_revealed > 0, (
                "At least one cell should be revealed"
            )

    async def test_flag_action(self):
        """Test placing a flag."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            await client.reset()

            # Place a flag
            action = MinesweeperAction(row=1, col=1, action_type="flag")
            result = await client.step(action)
            observation = result.observation

            assert isinstance(observation, MinesweeperObservation)
            assert observation.flags_placed == 1, "One flag should be placed"
            assert observation.board[1][1] == "F", "Cell should show flag marker"

    async def test_toggle_flag(self):
        """Test toggling a flag on and off."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            await client.reset()

            # Place a flag
            action = MinesweeperAction(row=2, col=2, action_type="flag")
            result = await client.step(action)
            observation = result.observation
            assert observation.flags_placed == 1

            # Remove the flag
            action = MinesweeperAction(row=2, col=2, action_type="flag")
            result = await client.step(action)
            observation = result.observation
            assert observation.flags_placed == 0, "Flag should be removed"

    async def test_invalid_position(self):
        """Test action with invalid position."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            await client.reset()

            # Try invalid row
            action = MinesweeperAction(row=10, col=0, action_type="reveal")
            result = await client.step(action)
            observation = result.observation

            assert observation.reward < 0, (
                "Should receive negative reward for invalid action"
            )

            # Try invalid column
            action = MinesweeperAction(row=0, col=10, action_type="reveal")
            result = await client.step(action)
            observation = result.observation

            assert observation.reward < 0, (
                "Should receive negative reward for invalid action"
            )

    async def test_reveal_already_revealed(self):
        """Test revealing an already revealed cell."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            await client.reset()

            # Find a safe revealed cell to re-target. Mines end the game, so
            # we keep trying until we land on a non-mine revealed cell.
            candidates = [(2, 2), (1, 1), (0, 0), (4, 4), (3, 3)]
            test_row, test_col = None, None
            for r, c in candidates:
                action = MinesweeperAction(row=r, col=c, action_type="reveal")
                result = await client.step(action)
                ongoing = (
                    result.observation.game_status == GameStatus.ONGOING.value
                    or result.observation.game_status == GameStatus.ONGOING
                )
                if ongoing and result.observation.board[r][c] != -1:
                    test_row, test_col = r, c
                    break
                if not ongoing:
                    await client.reset()

            assert test_row is not None, "Could not reveal a safe cell across resets"

            # Re-revealing the same cell should be penalised.
            action = MinesweeperAction(row=test_row, col=test_col, action_type="reveal")
            result = await client.step(action)
            assert result.observation.reward < 0, (
                f"Should receive penalty for revealing already revealed cell, "
                f"got {result.observation.reward}"
            )

    async def test_game_status_ongoing(self):
        """Test that game status remains ONGOING during normal play."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            await client.reset()

            # Make a few safe moves
            action = MinesweeperAction(row=0, col=0, action_type="reveal")
            result = await client.step(action)

            # Game should still be ongoing if we didn't hit a mine or win
            # Handle both int and enum types for game_status
            if (
                result.observation.game_status == GameStatus.ONGOING.value
                or result.observation.game_status == GameStatus.ONGOING
            ):
                assert result.observation.done is False

    async def test_board_cell_values(self):
        """Test that board cells contain valid values."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            await client.reset()

            # Reveal a cell
            action = MinesweeperAction(row=2, col=2, action_type="reveal")
            result = await client.step(action)
            observation = result.observation

            # Check that revealed cells have valid values (0-8 or '*')
            for row in observation.board:
                for cell in row:
                    assert (
                        cell == -1  # Unrevealed
                        or cell == "F"  # Flagged
                        or cell == "*"  # Mine (if revealed)
                        or (
                            isinstance(cell, int) and 0 <= cell <= 8
                        )  # Number of adjacent mines
                    ), f"Invalid cell value: {cell}"

    async def test_metadata_in_observation(self):
        """Test that observations contain metadata."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            result = await client.reset()
            observation = result.observation

            assert hasattr(observation, "metadata"), "Observation should have metadata"
            assert isinstance(observation.metadata, dict), (
                "Metadata should be a dictionary"
            )

    async def test_multiple_steps(self):
        """Test taking multiple steps in the environment."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            await client.reset()

            # Take several actions
            actions = [
                MinesweeperAction(row=0, col=0, action_type="reveal"),
                MinesweeperAction(row=0, col=1, action_type="flag"),
                MinesweeperAction(row=1, col=0, action_type="reveal"),
            ]

            for action in actions:
                result = await client.step(action)
                assert isinstance(result.observation, MinesweeperObservation)

    async def test_reset_clears_state(self):
        """Test that reset properly clears the game state."""
        async with MinesweeperEnv(base_url=self.base_url) as client:
            # First game
            await client.reset()
            action = MinesweeperAction(row=0, col=0, action_type="flag")
            await client.step(action)

            # Reset and check state is cleared
            result = await client.reset()
            observation = result.observation

            assert observation.flags_placed == 0, "Flags should be cleared after reset"
            assert observation.cells_revealed == 0, (
                "Revealed cells should be cleared after reset"
            )
            # Compare with enum value (handles both int and enum)
            assert (
                observation.game_status == GameStatus.ONGOING.value
                or observation.game_status == GameStatus.ONGOING
            ), "Game should be ongoing after reset"


class TestMinesweeperEnvironmentLogic(unittest.TestCase):
    """Direct unit tests against MinesweeperEnvironment (no HTTP server)."""

    def test_hitting_last_mine_does_not_overwrite_lost_with_won(self):
        """Revealing a mine must not be reclassified as a win, even if the
        revealed-cell count happens to equal total_cells - num_mines."""
        # 2x2 with 2 mines: safe = 2, mines = 2.
        # Pre-reveal 1 safe cell. Then reveal a mine — _revealed_cells grows
        # to 2, which equals total(4) - mines(2). The buggy check used to
        # count the mine and overwrite LOST with WON.
        env = MinesweeperEnvironment(height=2, width=2, num_mines=2)
        env.reset()

        env._mine_positions = {(0, 1), (1, 1)}
        env._compute_mine_counts()
        env._revealed_cells = {(0, 0)}
        env._flags_placed = set()
        env._game_status = GameStatus.ONGOING

        obs = env.step(MinesweeperAction(row=0, col=1, action_type="reveal"))

        assert env._game_status == GameStatus.LOST, (
            f"Game must be LOST after revealing a mine, got {env._game_status}"
        )
        assert obs.game_status == GameStatus.LOST
        assert obs.done is True
        assert obs.reward == -10.0

    def test_revealing_last_safe_cell_wins(self):
        """The normal win path still works after the LOST guard."""
        env = MinesweeperEnvironment(height=2, width=2, num_mines=1)
        env.reset()

        env._mine_positions = {(1, 1)}
        env._compute_mine_counts()
        env._revealed_cells = {(0, 0), (0, 1)}
        env._flags_placed = set()
        env._game_status = GameStatus.ONGOING

        obs = env.step(MinesweeperAction(row=1, col=0, action_type="reveal"))

        assert env._game_status == GameStatus.WON
        assert obs.game_status == GameStatus.WON
        assert obs.done is True

    def test_revealing_flagged_cell_returns_error_metadata(self):
        """A reject reason for a no-op action is surfaced in metadata.error."""
        env = MinesweeperEnvironment(height=3, width=3, num_mines=1)
        env.reset()

        env._mine_positions = {(2, 2)}
        env._compute_mine_counts()
        env._revealed_cells = set()
        env._flags_placed = {(0, 0)}
        env._game_status = GameStatus.ONGOING

        obs = env.step(MinesweeperAction(row=0, col=0, action_type="reveal"))

        assert obs.reward == -0.05
        assert obs.metadata.get("error"), (
            "Expected metadata.error explaining the rejection"
        )
        assert "flag" in obs.metadata["error"].lower()

    def test_flagging_revealed_cell_returns_error_metadata(self):
        """Flagging an already-revealed cell is rejected with an error message."""
        env = MinesweeperEnvironment(height=3, width=3, num_mines=1)
        env.reset()

        env._mine_positions = {(2, 2)}
        env._compute_mine_counts()
        env._revealed_cells = {(0, 0)}
        env._flags_placed = set()
        env._game_status = GameStatus.ONGOING

        obs = env.step(MinesweeperAction(row=0, col=0, action_type="flag"))

        assert obs.reward == -0.05
        assert obs.metadata.get("error"), (
            "Expected metadata.error explaining the rejection"
        )


if __name__ == "__main__":
    unittest.main()


def test_seeded_reset_is_reproducible_and_preserves_episode_id():
    env = MinesweeperEnvironment()
    env.reset(seed=42, episode_id="training-episode")
    mines = set(env._mine_positions)
    assert env.state.episode_id == "training-episode"
    env.reset(seed=42)
    assert env._mine_positions == mines


def test_reset_rng_is_independent_of_other_sessions():
    first, second = MinesweeperEnvironment(), MinesweeperEnvironment()
    first.reset(seed=42)
    second.reset(seed=123)
    first.reset()
    mines = set(first._mine_positions)
    reference = MinesweeperEnvironment()
    reference.reset(seed=42)
    reference.reset()
    assert reference._mine_positions == mines


@pytest.mark.parametrize(
    "height,width,num_mines",
    [(0, 5, 1), (5, 0, 1), (-1, 5, 1), (2, 2, -1), (2, 2, 4), (2, 2, 5)],
)
def test_invalid_board_configuration(height, width, num_mines):
    with pytest.raises(ValueError):
        MinesweeperEnvironment(height=height, width=width, num_mines=num_mines)


def test_board_schema_rejects_unknown_cell_values():
    from pydantic import ValidationError

    observation = MinesweeperEnvironment().reset(seed=42)
    payload = observation.model_dump(mode="json")
    payload["board"][0][0] = "unknown"
    with pytest.raises(ValidationError):
        MinesweeperObservation.model_validate(payload)


def test_client_wire_contract_excludes_reward_and_hidden_state():
    from openenv.core.env_server.serialization import serialize_observation

    env = MinesweeperEnvironment(height=2, width=2, num_mines=1)
    env.reset(seed=42)
    row, col = next(iter(env._mine_positions))
    observation = env.step(MinesweeperAction(row=row, col=col, action_type="reveal"))
    payload = serialize_observation(observation)
    assert "reward" not in payload["observation"]
    assert "done" not in payload["observation"]
    assert "mine_locations" not in payload["observation"]
    assert set(env.state.model_dump()) == {"episode_id", "step_count"}
    client = MinesweeperEnv(base_url="http://localhost:8000")
    result = client._parse_result(payload)
    assert result.reward == -10.0
    assert result.done is True
    assert result.observation.game_status == GameStatus.LOST


def test_flag_rewards_are_bounded_per_mine_and_reset_per_episode():
    env = MinesweeperEnvironment(height=3, width=3, num_mines=2)
    for episode in range(2):
        initial = env.reset(seed=42, episode_id=str(episode))
        assert initial.reward == 0.0
        assert initial.flags_placed == 0
        assert initial.done is False
        assert env.state.step_count == 0
        total = 0.0
        for row in range(3):
            for col in range(3):
                action = MinesweeperAction(row=row, col=col, action_type="flag")
                for cycle in range(10):
                    flagged = env.step(action)
                    expected = (
                        0.5 if cycle == 0 and (row, col) in env._mine_positions else 0.0
                    )
                    assert flagged.reward == expected
                    assert flagged.board[row][col] == "F"
                    total += flagged.reward
                    unflagged = env.step(action)
                    assert unflagged.reward == 0.0
                    assert unflagged.board[row][col] == -1
                    assert unflagged.done is False
        assert total == 0.5 * env.num_mines
        # Reset while a rewarded flag remains placed, too.
        row, col = next(iter(env._mine_positions))
        assert (
            env.step(MinesweeperAction(row=row, col=col, action_type="flag")).reward
            == 0.0
        )


@pytest.mark.parametrize("win", [False, True])
def test_terminal_flags_do_not_award_bonus_and_reset_restores_it(win):
    env = MinesweeperEnvironment(height=2, width=2, num_mines=1)
    env.reset(seed=42)
    mine = next(iter(env._mine_positions))
    targets = (
        [(r, c) for r in range(2) for c in range(2) if (r, c) != mine]
        if win
        else [mine]
    )
    for row, col in targets:
        terminal = env.step(MinesweeperAction(row=row, col=col, action_type="reveal"))
    assert terminal.done is True
    assert terminal.game_status == (GameStatus.WON if win else GameStatus.LOST)
    flag = MinesweeperAction(row=mine[0], col=mine[1], action_type="flag")
    for _ in range(4):
        observation = env.step(flag)
        assert observation.reward == 0.0
        assert observation.done is True
        assert observation.board == terminal.board
        assert observation.flags_placed == 0
    env.reset(seed=42)
    assert env.step(flag).reward == 0.5


@pytest.mark.parametrize("height,width", [(1, 2000), (64, 64)])
def test_large_empty_region_reveal(height, width):
    env = MinesweeperEnvironment(height=height, width=width, num_mines=0)
    env.reset(seed=42)

    obs = env.step(MinesweeperAction(row=0, col=0, action_type="reveal"))

    assert obs.board == [[0] * width for _ in range(height)]
    assert obs.cells_revealed == height * width
    assert obs.game_status == GameStatus.WON
    assert obs.done is True
    assert obs.reward == 1.0
    terminal = env.step(MinesweeperAction(row=0, col=0, action_type="reveal"))
    assert terminal.board == obs.board
    assert terminal.done is True
    assert terminal.reward == 0.0


def test_large_flood_fill_preserves_flags_and_numbered_boundary():
    env = MinesweeperEnvironment(height=1, width=2000, num_mines=1)
    env.reset(seed=42)
    env._mine_positions = {(0, 1998)}
    env._compute_mine_counts()
    env.step(MinesweeperAction(row=0, col=1500, action_type="flag"))

    obs = env.step(MinesweeperAction(row=0, col=0, action_type="reveal"))
    assert obs.board == [[0] * 1500 + ["F"] + [-1] * 499]
    assert obs.cells_revealed == 1500
    assert obs.game_status == GameStatus.ONGOING
    assert obs.done is False
    assert obs.reward == 1.0

    env.step(MinesweeperAction(row=0, col=1500, action_type="flag"))
    obs = env.step(MinesweeperAction(row=0, col=1500, action_type="reveal"))
    assert obs.board == [[0] * 1997 + [1, -1, -1]]
    assert obs.cells_revealed == 1998
    assert obs.game_status == GameStatus.ONGOING
    assert obs.done is False
    assert obs.reward == 1.0

    obs = env.step(MinesweeperAction(row=0, col=1999, action_type="reveal"))
    assert obs.game_status == GameStatus.WON
    assert obs.done is True
    assert obs.reward == 1.0


def test_server_fixture_waits_for_delayed_startup(monkeypatch):
    """A healthy server can take longer than three seconds to import."""
    popen = subprocess.Popen

    def delayed_start(args, **kwargs):
        return popen(
            [
                args[0],
                "-c",
                "import runpy, sys, time; time.sleep(4); "
                "sys.argv = sys.argv[1:]; "
                "runpy.run_module(sys.argv[0], run_name='__main__')",
                *args[2:],
            ],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", delayed_start)
    try:
        TestMinesweeperEnv.setUpClass()
        response = requests.get(TestMinesweeperEnv.base_url + "/health", timeout=1)
        assert response.status_code == 200
    finally:
        TestMinesweeperEnv.tearDownClass()
        TestMinesweeperEnv.doClassCleanups()
    assert TestMinesweeperEnv.server_process.poll() is not None


def test_server_fixture_reports_child_failure_and_cleans_up(monkeypatch):
    """Startup failures expose child stderr and do not leak the process."""
    popen = subprocess.Popen

    def failed_start(args, **kwargs):
        return popen(
            [args[0], "-c", "raise RuntimeError('startup regression sentinel')"],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", failed_start)
    try:
        with pytest.raises(RuntimeError, match="startup regression sentinel"):
            TestMinesweeperEnv.setUpClass()
    finally:
        TestMinesweeperEnv.tearDownClass()
        TestMinesweeperEnv.doClassCleanups()
    assert TestMinesweeperEnv.server_process.poll() is not None
