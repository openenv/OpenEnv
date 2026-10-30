"""Data models for the Spatial-Gym OpenEnv environment.

One episode is one SPaRC puzzle. The agent draws the line one move at a time, starting
at the start node, and the episode ends when it reaches the exit, runs out of legal
moves, or hits the step limit.
"""

import re
from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

DIRECTION_WORD_TO_ACTION = {"right": 0, "up": 1, "left": 2, "down": 3}


def parse_action(reply: str) -> int:
    """Parse an action from the model's reply.

    Accepted formats (case-insensitive):
      Final: 2          (digit 0-3)
      Final: left        (direction word)
      Final: LEFT        (direction word, any case)
    Falls back to a bare digit or direction word at the end of the reply.
    """
    # Primary: "Final: <digit>" or "Final: <word>"
    m = re.search(r"Final:\s*([0-3])\b", reply, re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"Final:\s*(right|up|left|down)\b", reply, re.IGNORECASE)
    if m:
        return DIRECTION_WORD_TO_ACTION[m.group(1).lower()]

    # Fallback: standalone digit 0-3 at the end
    m = re.search(r"\b([0-3])\s*$", reply)
    if m:
        return int(m.group(1))

    # Fallback: standalone direction word at the end
    m = re.search(r"\b(right|up|left|down)\s*$", reply, re.IGNORECASE)
    if m:
        return DIRECTION_WORD_TO_ACTION[m.group(1).lower()]

    raise ValueError(
        f"Invalid model output, no 'Final: <0-3|right|up|left|down>' found in response: {reply}"
    )


class SpatialGymAction(Action):
    """One move of the line: 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN."""

    action: int = Field(..., ge=0, le=3, description="0=RIGHT, 1=UP, 2=LEFT, 3=DOWN")

    @classmethod
    def from_reply(cls, reply: str) -> "SpatialGymAction":
        """Read the move from a model reply that ends with `Final: <action>` (paper format).

        Raises ValueError when the reply holds no move.
        """
        return cls(action=parse_action(reply.split("</think>")[-1].strip()))


class SpatialGymObservation(Observation):
    """The board after the last move. `done`, `reward` and `metadata` come from Observation."""

    puzzle_id: str = Field(default="", description="SPaRC puzzle id")
    difficulty: int = Field(
        default=0, description="Difficulty level, 1 (easy) to 5 (hard)"
    )
    step: int = Field(default=0, description="Moves taken so far")
    position: List[int] = Field(
        default_factory=list, description="Current node as [x, y]"
    )
    path: List[List[int]] = Field(
        default_factory=list,
        description="The line so far, as [x, y] nodes from the start",
    )
    legal_actions: List[int] = Field(
        default_factory=list, description="Moves allowed from here"
    )
    grid: List[List[str]] = Field(
        default_factory=list,
        description="Board rows in SPaRC notation; V marks visited nodes and L the current one",
    )
    system_prompt: str = Field(
        default="",
        description="Rules and symbol legend, as in the paper (sent on reset only)",
    )
    prompt: str = Field(default="", description="The paper's message for the next move")
    solved: bool = Field(
        default=False, description="The line matches a stored solution"
    )
    reached_end: bool = Field(
        default=False, description="The line ended on the exit node"
    )


class SpatialGymState(State):
    """Episode state. `episode_id` and `step_count` come from State."""

    puzzle_id: str = ""
    difficulty: int = 0
    split: str = "test"
    backtracking: bool = False
    max_steps: int = 100
    path: List[List[int]] = Field(
        default_factory=list, description="Nodes of the line as [x, y]"
    )
    done: bool = False
    solved: bool = False
    seed: Optional[int] = None
