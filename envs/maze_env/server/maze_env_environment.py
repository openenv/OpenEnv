# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Maze Environment Implementation.

A simple gridworld maze with walls, a start cell, and an exit.
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Support both in-repo and standalone imports
try:
    from ..models import MazeAction, MazeObservation, MazeState
except ImportError:
    from models import MazeAction, MazeObservation, MazeState

from .maze import Maze, Render, Status

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "Numpy is not installed. "
        "Please install it following instructions at: "
        "pip install numpy"
    ) from e


DEFAULT_MAZE = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
    ],
    dtype=int,
)


class MazeEnvironment(Environment[MazeAction, MazeObservation, MazeState]):
    """
    A maze environment built on a simple gridworld.

    The agent starts at a configurable start cell and must reach the exit cell.
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        maze_array: Optional[np.ndarray] = None,
        start_cell: Tuple[int, int] = (0, 0),
        exit_cell: Optional[Tuple[int, int]] = None,
    ):
        """Initialize the Maze environment."""
        self._maze_array = np.array(
            maze_array if maze_array is not None else DEFAULT_MAZE, dtype=int
        )
        nrows, ncols = self._maze_array.shape

        self._start_cell = (int(start_cell[0]), int(start_cell[1]))
        self._exit_cell = (
            (ncols - 1, nrows - 1)
            if exit_cell is None
            else (int(exit_cell[0]), int(exit_cell[1]))
        )

        self.env = Maze(
            maze=self._maze_array,
            start_cell=self._start_cell,
            exit_cell=self._exit_cell,
            rendering=Render.NOTHING,
        )

        self._state = MazeState(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._sync_state(done=False)

    def reset(
        self,
        seed: Optional[int] = None,
        start_cell: Optional[Tuple[int, int]] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MazeObservation:
        """
        Reset the environment.

        Args:
            seed: Optional random seed for environment initialization
            start_cell: Optional starting cell (col, row) for the agent
            episode_id: Optional episode ID to set for the new episode

        Returns:
            MazeObservation with initial maze state
        """
        if seed is not None:
            np.random.seed(seed)

        self.env.reset(self._start_cell if start_cell is None else start_cell)
        self._state = MazeState(
            episode_id=episode_id if episode_id is not None else str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1
        self._sync_state(done=False)

        return self._build_observation(reward=0.0, done=False, status=Status.PLAYING)

    def step(
        self,
        action: MazeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MazeObservation:
        """
        Execute a step in the environment.

        Args:
            action: MazeAction containing the action to take
            timeout_s: Optional timeout for the step (unused, kept for interface compatibility)

        Returns:
            MazeObservation with the result of the action
        """
        self._state.step_count += 1

        _, reward, status = self.env.step(int(action.action))
        done = status in (Status.WIN, Status.LOSE)
        self._sync_state(done=done)

        return self._build_observation(reward=reward, done=done, status=status)

    def _build_observation(
        self, reward: float, done: bool, status: Status
    ) -> MazeObservation:
        legal_actions = (
            [int(a) for a in self.env.possible_actions()] if not done else []
        )

        return MazeObservation(
            current_position=[int(x) for x in self.env.current_cell],
            previous_position=[int(x) for x in self.env.previous_cell],
            legal_actions=legal_actions,
            reward=reward,
            done=done,
            metadata={
                "maze": self._maze_array.tolist(),
                "status": status.name.lower(),
                "exit_cell": list(self.env.exit_cell),
                "step": self._state.step_count,
            },
        )

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """The legal moves as buttons, named by direction."""
        names = {0: "up", 1: "down", 2: "left", 3: "right"}
        return [
            (f"{a} · {names[a]}", {"action": a})
            for a in sorted(observation.get("legal_actions") or [])
        ]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the maze: walls, free cells, the exit (outlined) and the agent (dot)."""
        if "current_position" not in observation:
            return None
        maze = observation["metadata"]["maze"]
        cells = []
        for row, values in enumerate(maze):
            for col, value in enumerate(values):
                background = (
                    "var(--body-text-color)" if value else "var(--border-color-primary)"
                )
                style = (
                    "border-radius:4px;display:flex;align-items:center;"
                    f"justify-content:center;background:{background}"
                )
                if [col, row] == observation["metadata"]["exit_cell"]:
                    style += ";box-shadow:inset 0 0 0 3px var(--color-accent)"
                dot = ""
                if [col, row] == observation["current_position"]:
                    dot = '<span style="width:60%;height:60%;border-radius:50%;background:var(--color-accent)"></span>'
                cells.append(f'<span style="{style}">{dot}</span>')
        return (
            '<div role="img" aria-label="Maze board" style="display:inline-grid;'
            f"grid-template-columns:repeat({len(maze[0])},26px);grid-auto-rows:26px;gap:3px;padding:10px;"
            'border:1px solid var(--border-color-primary);border-radius:10px;background:var(--background-fill-secondary)">'
            + "".join(cells)
            + "</div>"
        )

    def _sync_state(self, done: bool) -> None:
        self._state.done = done
        self._state.current_position = [int(x) for x in self.env.current_cell]
        self._state.exit_cell = list(self.env.exit_cell)
        self._state.status = self.env.status.name.lower()

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
