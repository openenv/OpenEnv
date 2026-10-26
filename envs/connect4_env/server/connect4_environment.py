import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openenv.core.env_server import Environment

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import Connect4Action, Connect4Observation, Connect4State
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import Connect4Action, Connect4Observation, Connect4State


class Connect4Environment(Environment):
    ROWS = 6
    COLUMNS = 7
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, opponent=None):
        super().__init__()
        self._opponent = opponent
        self.reset()

    def reset(self):
        self.board = np.zeros((self.ROWS, self.COLUMNS), dtype=np.int8)
        self.next_player = 1
        self.invalid_move_played = False

        self._state = Connect4State(
            board=self.board.copy().tolist(),
            next_player=self.next_player,
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )
        return self._make_observation()

    def step(self, action: Connect4Action):
        col = action.column
        # reward = 0.0
        done = False

        # check action validity
        if col < 0 or col >= self.COLUMNS or self.board[0, col] != 0:
            self.invalid_move_played = True
            reward = -1  # penalty for invalid move
            done = True
        else:
            # drop piece
            for row in range(self.ROWS - 1, -1, -1):
                if self.board[row, col] == 0:
                    self.board[row, col] = self.next_player
                    break

            # check win / full board
            reward, done = self._check_win_or_draw(row, col)

        self.next_player *= -1

        self._state = Connect4State(
            board=self.board.copy().tolist(),
            next_player=self.next_player,
            episode_id=self._state.episode_id,
            step_count=self._state.step_count + 1,
        )

        return self._make_observation(reward, done)

    def _make_observation(self, reward=0.0, done=False):
        legal_actions = [c for c in range(self.COLUMNS) if self.board[0, c] == 0]
        return Connect4Observation(
            board=self.board.copy().tolist(),
            legal_actions=legal_actions,
            reward=reward,
            done=done,
            metadata={"next_player": self.next_player},
        )

    def _check_win_or_draw(self, row, col):
        # Implement 4-in-a-row check (like your Gymnasium code)
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 0
            for step in range(-3, 4):
                r, c = row + step * dr, col + step * dc
                if (
                    0 <= r < self.ROWS
                    and 0 <= c < self.COLUMNS
                    and self.board[r, c] == player
                ):
                    count += 1
                    if count >= 4:
                        return 1.0, True
                else:
                    count = 0
        if np.all(self.board != 0):
            return 0.0, True
        return 0.0, False

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """The legal columns as buttons."""
        return [
            (f"col {c}", {"column": c}) for c in observation.get("legal_actions") or []
        ]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the 6x7 board: player 1 in the accent colour, player -1 in the text colour."""
        if "board" not in observation:
            return None
        colours = {
            1: "var(--color-accent)",
            -1: "var(--body-text-color)",
            0: "var(--border-color-primary)",
        }
        cells = [
            f'<span style="background:{colours[value]};border-radius:50%"></span>'
            for row in observation["board"]
            for value in row
        ]
        cells += [
            f'<span style="text-align:center;font-size:12px;line-height:26px">{c}</span>'
            for c in range(self.COLUMNS)
        ]
        return (
            '<div role="img" aria-label="Connect4 board" style="display:inline-grid;'
            "grid-template-columns:repeat(7,26px);grid-auto-rows:26px;gap:4px;padding:10px;"
            'border:1px solid var(--border-color-primary);border-radius:10px;background:var(--background-fill-secondary)">'
            + "".join(cells)
            + "</div>"
        )

    @property
    def state(self):
        return self._state
