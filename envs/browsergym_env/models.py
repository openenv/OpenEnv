"""Data models for the BrowserGym environment.

BrowserGym is a unified framework for web-based agent tasks, combining multiple
benchmarks including MiniWoB (training), WebArena (evaluation), VisualWebArena,
and more under a single Gymnasium-compatible API.
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class BrowserGymAction(Action):
    """Action to be executed in the BrowserGym environment.

    BrowserGym supports high-level natural language actions that can be parsed
    into browser operations.

    Example actions:
    - "click('Submit button')"
    - "fill('username', 'john@example.com')"
    - "goto('https://example.com')"
    - "scroll('down')"
    - "send_keys('Enter')"
    """

    action_str: str = Field(
        ..., description="Natural language action string (e.g., \"click('Submit')\")"
    )


class BrowserGymObservation(Observation):
    """Observation returned from the BrowserGym environment.

    Contains multiple observation modalities including text (accessibility tree
    or DOM), visual (screenshot), and page metadata.
    """

    text: str = Field(
        default="",
        description="Text representation of the page (accessibility tree or DOM)",
    )

    url: str = Field(default="", description="Current URL of the page")

    screenshot: Optional[List[List[List[int]]]] = Field(
        default=None,
        description="Screenshot as numpy array [height, width, channels] (if visual observation enabled)",
    )

    goal: str = Field(
        default="", description="Task goal/instruction for the current episode"
    )

    axtree_txt: str = Field(default="", description="Full accessibility tree as text")

    pruned_html: str = Field(
        default="", description="Pruned HTML content (interactive elements only)"
    )

    error: str = Field(
        default="", description="Error message if action execution failed"
    )

    last_action_error: bool = Field(
        default=False, description="Whether the last action resulted in an error"
    )


class BrowserGymState(State):
    """State of the BrowserGym environment.

    Tracks the current benchmark, task, and progress through an episode.
    """

    benchmark: str = Field(
        default="",
        description="Benchmark name (e.g., 'miniwob', 'webarena', 'visualwebarena')",
    )

    task_name: str = Field(
        default="",
        description="Specific task within the benchmark (e.g., 'click-test', 'click-button')",
    )

    task_id: Optional[str] = Field(
        default=None,
        description="Task ID for evaluation benchmarks (e.g., WebArena task number)",
    )

    goal: str = Field(default="", description="Task goal/instruction")

    current_url: str = Field(default="", description="Current URL of the active page")

    max_steps: Optional[int] = Field(
        default=None, description="Maximum steps allowed for this task"
    )

    cum_reward: float = Field(
        default=0.0, description="Cumulative reward for the current episode"
    )
