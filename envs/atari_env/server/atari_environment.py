# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Atari Environment Server Implementation.

This module wraps ALE's ALEInterface and exposes it
via the OpenEnv Environment interface.
"""

import base64
import io
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

from openenv.core.env_server import Action, Environment, Observation
from PIL import Image

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import AtariAction, AtariObservation, AtariState
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import AtariAction, AtariObservation, AtariState

# Import ALE
try:
    import numpy as np
    from ale_py import ALEInterface, roms
except ImportError as e:
    raise ImportError(
        "ALE (Arcade Learning Environment) is not installed. "
        "Please install it with: pip install ale-py"
    ) from e


class AtariEnvironment(Environment):
    """
    Atari Environment wrapper for OpenEnv.

    This environment wraps Atari 2600 games via the Arcade Learning Environment (ALE)
    and provides a clean interface for RL training.

    Supported games include: pong, breakout, space_invaders, and 100+ others.

    Args:
        game_name: Name of the Atari game (e.g., "pong", "breakout").
        obs_type: Observation type - "rgb", "grayscale", or "ram".
        full_action_space: Use full action space (18 actions) vs minimal.
        mode: Game mode (if applicable).
        difficulty: Game difficulty (if applicable).
        repeat_action_probability: Sticky action probability (default 0.0).
        frameskip: Number of frames to skip per action (default 4).

    Example:
        >>> env = AtariEnvironment("pong")
        >>> obs = env.reset()
        >>> print(obs.screen_shape)  # [210, 160, 3]
        >>> obs = env.step(AtariAction(action_id=2))  # UP
        >>> print(obs.reward, obs.done)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        game_name: str = "pong",
        obs_type: Literal["rgb", "grayscale", "ram"] = "rgb",
        full_action_space: bool = False,
        mode: Optional[int] = None,
        difficulty: Optional[int] = None,
        repeat_action_probability: float = 0.0,
        frameskip: int = 4,
    ):
        """Initialize Atari environment."""
        super().__init__()

        self.game_name = game_name
        self.obs_type = obs_type
        self.full_action_space = full_action_space
        self.mode = mode
        self.difficulty = difficulty
        self.repeat_action_probability = repeat_action_probability
        self.frameskip = frameskip

        # Create ALE interface
        self.ale = ALEInterface()

        # Configure ALE
        from ale_py import LoggerMode

        self.ale.setLoggerMode(LoggerMode.Error)  # Error mode only
        self.ale.setFloat("repeat_action_probability", repeat_action_probability)

        # Load ROM
        try:
            rom_path = roms.get_rom_path(game_name)
            self.ale.loadROM(rom_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load Atari game '{game_name}': {e}\n"
                f"Available games can be found via: ale_py.roms.list_roms()"
            ) from e

        # Set mode and difficulty if specified
        if mode is not None:
            self.ale.setMode(mode)
        if difficulty is not None:
            self.ale.setDifficulty(difficulty)

        # Get action set
        if full_action_space:
            self._action_set = self.ale.getLegalActionSet()
        else:
            self._action_set = self.ale.getMinimalActionSet()

        # Get screen dimensions for observation space
        self.screen_height, self.screen_width = self.ale.getScreenDims()
        if obs_type == "rgb":
            self.screen_shape = [self.screen_height, self.screen_width, 3]
        elif obs_type == "grayscale":
            self.screen_shape = [self.screen_height, self.screen_width]
        elif obs_type == "ram":
            self.screen_shape = [self.ale.getRAMSize()]
        else:
            raise ValueError(f"Invalid obs_type: {obs_type}")

        # Initialize state
        self._state = AtariState(
            game_name=game_name,
            obs_type=obs_type,
            full_action_space=full_action_space,
            mode=mode,
            difficulty=difficulty,
            repeat_action_probability=repeat_action_probability,
            frameskip=frameskip,
        )

    def reset(self) -> Observation:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial observation for the agent.
        """
        # Reset ALE
        self.ale.reset_game()

        # Reset state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0

        # Get initial observation
        return self._make_observation()

    def step(self, action: Action) -> Observation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: AtariAction containing the action_id to execute.

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action is not an AtariAction.
        """
        if not isinstance(action, AtariAction):
            raise ValueError(f"Expected AtariAction, got {type(action)}")

        # Validate action_id
        if action.action_id < 0 or action.action_id >= len(self._action_set):
            raise ValueError(
                f"Invalid action_id: {action.action_id}. "
                f"Valid range: [0, {len(self._action_set) - 1}]"
            )

        # Get actual ALE action
        ale_action = self._action_set[action.action_id]

        # Execute action with frameskip
        total_reward = 0.0
        for _ in range(self.frameskip):
            total_reward += self.ale.act(ale_action)
            if self.ale.game_over():
                break

        self._state.step_count += 1

        # Get observation
        obs = self._make_observation()
        obs.reward = total_reward

        return obs

    @property
    def state(self) -> AtariState:
        """Get current environment state."""
        return self._state

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """The legal actions as buttons, named with their ALE meaning (NOOP, FIRE, ...)."""
        return [
            (f"{a} · {self._action_set[a].name}", {"action_id": a})
            for a in observation.get("legal_actions") or []
        ]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the current game frame (RGB or grayscale) as a pixelated PNG."""
        if "screen" not in observation:
            return None
        if self.obs_type == "ram":
            return None
        shape = observation["screen_shape"]
        frame = np.asarray(observation["screen"], dtype=np.uint8).reshape(shape)
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        return (
            '<img alt="Atari frame" '
            f'src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}" '
            f'style="width:{2 * shape[1]}px;max-width:100%;image-rendering:pixelated;'
            'border:1px solid var(--border-color-primary);border-radius:6px">'
        )

    def _make_observation(self) -> AtariObservation:
        """
        Create an AtariObservation from current ALE state.

        Returns:
            AtariObservation for the agent.
        """
        # Get screen observation
        if self.obs_type == "rgb":
            screen = self.ale.getScreenRGB()
        elif self.obs_type == "grayscale":
            screen = self.ale.getScreenGrayscale()
        elif self.obs_type == "ram":
            screen = self.ale.getRAM()
        else:
            raise ValueError(f"Invalid obs_type: {self.obs_type}")

        # Flatten screen for JSON serialization
        # Handle both numpy arrays and lists
        if hasattr(screen, "flatten"):
            screen_flat = screen.flatten().tolist()
        elif hasattr(screen, "tolist"):
            screen_flat = screen.tolist()
        else:
            screen_flat = list(screen)

        # Get game info
        lives = self.ale.lives()
        episode_frame_number = self.ale.getEpisodeFrameNumber()
        frame_number = self.ale.getFrameNumber()
        done = self.ale.game_over()

        # Create legal actions list (indices into action_set)
        legal_actions = list(range(len(self._action_set)))

        # Create observation
        obs = AtariObservation(
            screen=screen_flat,
            screen_shape=self.screen_shape,
            legal_actions=legal_actions,
            lives=lives,
            episode_frame_number=episode_frame_number,
            frame_number=frame_number,
            done=done,
            reward=0.0,  # Will be filled in by step()
            metadata={
                "game_name": self.game_name,
                "action_meanings": [str(a) for a in self._action_set],
            },
        )

        return obs
