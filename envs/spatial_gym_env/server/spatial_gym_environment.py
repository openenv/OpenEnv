"""Spatial-Gym served through OpenEnv.

Each episode is one SPaRC puzzle, solved one move per step as in the Spatial-Gym paper.
The reward is sparse: 1.0 when the episode ends with a line that exactly matches one of
the puzzle's stored solutions (`outcome_reward == 1` in `Spatial_Gym.step`), else 0.0.

`reset()` options, all optional:
    puzzle_id     play this SPaRC puzzle
    difficulty    a level (1-5) or a list of levels to draw the puzzle from
    seed          makes the draw repeatable
    split         "test" (default, the 500 benchmark puzzles) or "train"
    backtracking  allow stepping back onto the previous node to undo a move (default False)
    max_steps     the episode ends after this many moves (default 100, as in the paper)
"""

import copy
import json
import random
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

import gymnasium as gym
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

import Spatial_Gym  # noqa: F401  (registers the "Spatial-Gym" gymnasium id)

try:
    from ..models import SpatialGymAction, SpatialGymObservation, SpatialGymState
    from .board_svg import board_svg
    from .prompts import STEP_USER_TEMPLATE, step_backtrack_prompt, step_prompt
except ImportError:
    from models import SpatialGymAction, SpatialGymObservation, SpatialGymState
    from server.board_svg import board_svg
    from server.prompts import STEP_USER_TEMPLATE, step_backtrack_prompt, step_prompt

DATASET = "lkaesberg/SPaRC"
SPLITS = ("test", "train")
ACTION_NAMES = {0: "RIGHT", 1: "UP", 2: "LEFT", 3: "DOWN"}
ARROWS = {0: "→", 1: "↑", 2: "←", 3: "↓"}
MAX_STEPS_LIMIT = 10_000
RESET_OPTIONS = {"puzzle_id", "difficulty", "split", "backtracking", "max_steps"}
PROMPT_FIELDS = ("id", "grid_size", "puzzle_array", "polyshapes", "difficulty_level")

# Building a Spatial-Gym env loads and preprocesses a whole split (about 2 s), and reusing
# one env across episodes leaks visited cells. So, as in the verifiers environment, we keep
# one untouched template per (split, backtracking, max_steps) and deep-copy it per episode.
# Templates are only ever copied, never stepped, so sessions share no mutable state.
_LOCK = threading.Lock()
_TEMPLATES: Dict[Tuple[str, bool, int], gym.Env] = {}
_ROWS: Dict[str, Dict[str, dict]] = {}


def _template(split: str, backtracking: bool, max_steps: int) -> gym.Env:
    key = (split, backtracking, max_steps)
    with _LOCK:
        if key not in _TEMPLATES:
            _TEMPLATES[key] = gym.make(
                "Spatial-Gym",
                df_name=DATASET,
                df_split="all",
                df_set=split,
                render_mode=None,
                observation="SPaRC",
                traceback=backtracking,
                max_steps=max_steps,
            )
        return _TEMPLATES[key]


def _rows(split: str) -> Dict[str, dict]:
    """The raw SPaRC records of a split, by id. The prompts read these fields."""
    with _LOCK:
        if split not in _ROWS:
            from datasets import load_dataset

            # Same dataset version that Spatial_Gym itself loads (no revision pin).
            records = load_dataset(DATASET, "all", split=split)
            _ROWS[split] = {r["id"]: {k: r[k] for k in PROMPT_FIELDS} for r in records}
        return _ROWS[split]


def _plain(value: Any) -> Any:
    """Turn numpy values in Spatial-Gym's info dicts into JSON types."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _levels(difficulty: Union[int, Sequence[int], None]) -> Optional[set]:
    if difficulty is None:
        return None
    levels = (
        {int(difficulty)}
        if isinstance(difficulty, (int, float, str))
        else {int(d) for d in difficulty}
    )
    if not levels or not levels <= {1, 2, 3, 4, 5}:
        raise ValueError(f"difficulty must be levels from 1 to 5, got {difficulty!r}")
    return levels


class SpatialGymEnvironment(
    Environment[SpatialGymAction, SpatialGymObservation, SpatialGymState]
):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._env: Optional[gym.Env] = None
        self._row: Optional[dict] = None
        self._state = SpatialGymState(episode_id=str(uuid4()))

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        puzzle_id: Optional[str] = None,
        difficulty: Union[int, List[int], None] = None,
        split: str = "test",
        backtracking: bool = False,
        max_steps: int = 100,
        **kwargs: Any,
    ) -> SpatialGymObservation:
        if kwargs:
            raise ValueError(
                f"unknown reset options {sorted(kwargs)}; use {sorted(RESET_OPTIONS | {'seed'})}"
            )
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        max_steps = int(max_steps)
        if not 1 <= max_steps <= MAX_STEPS_LIMIT:
            raise ValueError(
                f"max_steps must be between 1 and {MAX_STEPS_LIMIT}, got {max_steps}"
            )
        backtracking = bool(backtracking)

        rows = _rows(split)
        if puzzle_id is not None:
            if puzzle_id not in rows:
                raise ValueError(f"puzzle {puzzle_id!r} is not in the {split} split")
            row = rows[puzzle_id]
        else:
            levels = _levels(difficulty)
            candidates = sorted(
                pid
                for pid, r in rows.items()
                if levels is None or r["difficulty_level"] in levels
            )
            row = rows[random.Random(seed).choice(candidates)]

        env = copy.deepcopy(_template(split, backtracking, max_steps))
        obs, info = env.reset(options={"puzzle_id": row["id"]})
        inner = env.unwrapped
        if inner.puzzles[inner.current_puzzle_index]["id"] != row["id"]:
            raise RuntimeError(f"Spatial-Gym did not load puzzle {row['id']}")
        self._env, self._row = env, row
        self._state = SpatialGymState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            puzzle_id=row["id"],
            difficulty=int(row["difficulty_level"]),
            split=split,
            backtracking=backtracking,
            max_steps=max_steps,
            path=self._path(),
            seed=seed,
        )
        system_prompt = (step_backtrack_prompt if backtracking else step_prompt)(row)[
            "system"
        ]
        return self._observation(obs, info, system_prompt=system_prompt)

    def step(
        self, action: SpatialGymAction, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> SpatialGymObservation:
        if self._env is None:
            raise RuntimeError("call reset() before step()")
        if self._state.done:
            raise RuntimeError("the episode is over; call reset() to start a new one")
        obs, _, terminated, truncated, info = self._env.step(int(action.action))
        self._state.step_count += 1
        self._state.path = self._path()
        self._state.done = bool(terminated or truncated)
        self._state.solved = (
            self._state.done and self._env.unwrapped.outcome_reward == 1
        )
        return self._observation(obs, info)

    @property
    def state(self) -> SpatialGymState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="spatial_gym_env",
            description=(
                "SPaRC grid pathfinding puzzles, solved one move per step (Spatial-Gym). "
                "Reward 1.0 when the line matches a stored solution, else 0.0."
            ),
            version="0.1.0",
            author="Lars Benedikt Kaesberg",
            documentation_url="https://github.com/openenv/OpenEnv/tree/main/envs/spatial_gym_env",
        )

    def close(self) -> None:
        self._env = None

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """The legal moves as buttons, named by direction."""
        return [
            (f"{ARROWS[a]} {ACTION_NAMES[a].lower()}", {"action": a})
            for a in sorted(observation.get("legal_actions") or [])
        ]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw this episode's puzzle with the observation's line on it."""
        if self._row is None:
            return None
        row = self._row
        polyshapes = json.loads(row["polyshapes"]) if row["polyshapes"] else {}
        label = f"SPaRC puzzle {row['id']}, level {row['difficulty_level']}"
        status = f"Level {row['difficulty_level']} · step {observation.get('step', 0)}"
        if observation.get("done"):
            status += " · solved" if observation.get("solved") else " · not solved"
        return (
            '<div style="display:inline-flex;flex-direction:column;gap:6px">'
            + board_svg(
                row["puzzle_array"], observation.get("path") or [], polyshapes, label
            )
            + f'<span style="font-size:13px;opacity:0.8">{status}</span></div>'
        )

    def _path(self) -> List[List[int]]:
        return [[int(x), int(y)] for x, y in self._env.unwrapped.path]

    def _observation(
        self, obs: str, info: dict, system_prompt: str = ""
    ) -> SpatialGymObservation:
        inner = self._env.unwrapped
        x, y = (int(v) for v in info["agent_location"])
        legal = [int(a) for a in info["legal_actions"]]
        grid = json.loads(obs)
        done = self._state.done
        prompt = ""
        if not done:
            prompt = STEP_USER_TEMPLATE.format(
                step=self._state.step_count + 1,
                agent_loc=f"({x}, {y})",
                legal_str=", ".join(f"{a}={ACTION_NAMES[a]}" for a in legal),
                obs_str="\n".join(str(row) for row in grid),
            )
        return SpatialGymObservation(
            puzzle_id=self._state.puzzle_id,
            difficulty=self._state.difficulty,
            step=self._state.step_count,
            position=[x, y],
            path=self._path(),
            legal_actions=legal,
            grid=grid,
            system_prompt=system_prompt,
            prompt=prompt,
            solved=self._state.solved,
            reached_end=(x, y) == tuple(int(v) for v in inner.target_location),
            done=done,
            reward=1.0 if self._state.solved else 0.0,
            metadata={"rule_status": _plain(info.get("rule_status", {}))},
        )
