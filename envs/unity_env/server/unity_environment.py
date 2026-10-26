# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unity ML-Agents Environment Implementation.

Wraps Unity ML-Agents environments (PushBlock, 3DBall, GridWorld, etc.)
with the OpenEnv interface for standardized reinforcement learning.
"""

import base64
import glob
import hashlib
import io
import os
import socket
from pathlib import Path
from sys import platform
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

# Support multiple import scenarios
try:
    # In-repo imports (when running from OpenEnv repository root)
    from openenv.core.env_server.interfaces import Environment

    from ..models import PUSHBLOCK_ACTIONS, UnityAction, UnityObservation, UnityState
except ImportError:
    # openenv from pip
    from openenv.core.env_server.interfaces import Environment

    try:
        # Direct execution from envs/unity_env/ directory (imports from parent)
        import sys
        from pathlib import Path

        # Add parent directory to path for direct execution
        _parent = str(Path(__file__).parent.parent)
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from models import PUSHBLOCK_ACTIONS, UnityAction, UnityObservation, UnityState
    except ImportError:
        try:
            # Package installed as unity_env
            from unity_env.models import (
                PUSHBLOCK_ACTIONS,
                UnityAction,
                UnityObservation,
                UnityState,
            )
        except ImportError:
            # Running from OpenEnv root with envs prefix
            from envs.unity_env.models import (
                PUSHBLOCK_ACTIONS,
                UnityAction,
                UnityObservation,
                UnityState,
            )


# Persistent cache directory to avoid re-downloading environment binaries
PERSISTENT_CACHE_DIR = os.path.join(str(Path.home()), ".mlagents-cache")


def get_cached_binary_path(cache_dir: str, name: str, url: str) -> Optional[str]:
    """Check if binary is cached and return its path."""
    if platform == "darwin":
        extension = "*.app"
    elif platform in ("linux", "linux2"):
        extension = "*.x86_64"
    elif platform == "win32":
        extension = "*.exe"
    else:
        return None

    bin_dir = os.path.join(cache_dir, "binaries")
    url_hash = "-" + hashlib.md5(url.encode()).hexdigest()
    search_path = os.path.join(bin_dir, name + url_hash, "**", extension)

    candidates = glob.glob(search_path, recursive=True)
    for c in candidates:
        if "UnityCrashHandler64" not in c:
            return c
    return None


class UnityMLAgentsEnvironment(Environment):
    """
    Wraps Unity ML-Agents environments with the OpenEnv interface.

    This environment supports all Unity ML-Agents registry environments
    including PushBlock, 3DBall, GridWorld, and more. Environments are
    automatically downloaded on first use.

    Features:
    - Dynamic environment switching via reset(env_id="...")
    - Support for both discrete and continuous action spaces
    - Optional visual observations (base64-encoded images)
    - Persistent caching to avoid re-downloading binaries
    - Headless mode for faster training (no_graphics=True)

    Example:
        >>> env = UnityMLAgentsEnvironment()
        >>> obs = env.reset()
        >>> print(obs.vector_observations)
        >>>
        >>> # Take a random action
        >>> obs = env.step(UnityAction(discrete_actions=[1]))  # Move forward
        >>> print(obs.reward)

    Example with different environment:
        >>> env = UnityMLAgentsEnvironment(env_id="3DBall")
        >>> obs = env.reset()
        >>>
        >>> # Or switch environment on reset
        >>> obs = env.reset(env_id="PushBlock")
    """

    # Each WebSocket session gets its own environment instance
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(
        self,
        env_id: Optional[str] = None,
        no_graphics: Optional[bool] = None,
        time_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        quality_level: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the Unity ML-Agents environment.

        Configuration can be provided via constructor arguments or environment
        variables. Environment variables are used when constructor arguments
        are not provided (useful for Docker deployment).

        Args:
            env_id: Identifier of the Unity environment to load.
                Available: PushBlock, 3DBall, 3DBallHard, GridWorld, Basic
                Env var: UNITY_ENV_ID (default: PushBlock)
            no_graphics: If True, run in headless mode (faster training).
                Env var: UNITY_NO_GRAPHICS (0 or 1, default: 0 = graphics enabled)
            time_scale: Simulation speed multiplier.
                Env var: UNITY_TIME_SCALE (default: 1.0)
            width: Window width in pixels (when graphics enabled).
                Env var: UNITY_WIDTH (default: 1280)
            height: Window height in pixels (when graphics enabled).
                Env var: UNITY_HEIGHT (default: 720)
            quality_level: Graphics quality 0-5 (when graphics enabled).
                Env var: UNITY_QUALITY_LEVEL (default: 5)
            cache_dir: Directory to cache downloaded environment binaries.
                Env var: UNITY_CACHE_DIR (default: ~/.mlagents-cache)

        `reset()` includes visual observations by default when the env var
        UNITY_INCLUDE_VISUAL is 1 (default: 0).
        """
        # Initialize cleanup-critical attributes first (for __del__ safety)
        self._unity_env = None
        self._behavior_name = None
        self._behavior_spec = None
        self._engine_channel = None

        # Read from environment variables with defaults, allow constructor override
        self._env_id = env_id or os.environ.get("UNITY_ENV_ID", "PushBlock")

        # Handle no_graphics: default is False (graphics enabled)
        if no_graphics is not None:
            self._no_graphics = no_graphics
        else:
            env_no_graphics = os.environ.get("UNITY_NO_GRAPHICS", "0")
            self._no_graphics = env_no_graphics.lower() in ("1", "true", "yes")

        self._time_scale = (
            time_scale
            if time_scale is not None
            else float(os.environ.get("UNITY_TIME_SCALE", "1.0"))
        )
        self._width = (
            width if width is not None else int(os.environ.get("UNITY_WIDTH", "1280"))
        )
        self._height = (
            height if height is not None else int(os.environ.get("UNITY_HEIGHT", "720"))
        )
        self._quality_level = (
            quality_level
            if quality_level is not None
            else int(os.environ.get("UNITY_QUALITY_LEVEL", "5"))
        )
        self._cache_dir = cache_dir or os.environ.get(
            "UNITY_CACHE_DIR", PERSISTENT_CACHE_DIR
        )
        env_include_visual = os.environ.get("UNITY_INCLUDE_VISUAL", "0")
        self._default_include_visual = env_include_visual.lower() in (
            "1",
            "true",
            "yes",
        )
        self._include_visual = self._default_include_visual

        # State tracking
        self._state = UnityState(
            episode_id=str(uuid4()),
            step_count=0,
            env_id=self._env_id,
        )

        # Ensure cache directory exists
        os.makedirs(self._cache_dir, exist_ok=True)

    def _load_environment(self, env_id: str) -> None:
        """Load or switch to a Unity environment."""
        # Close existing environment if any
        if self._unity_env is not None:
            try:
                self._unity_env.close()
            except Exception:
                pass

        # Import ML-Agents components
        try:
            from mlagents_envs.base_env import ActionTuple
            from mlagents_envs.registry import default_registry
            from mlagents_envs.registry.remote_registry_entry import RemoteRegistryEntry
            from mlagents_envs.side_channel.engine_configuration_channel import (
                EngineConfigurationChannel,
            )
        except ImportError as e:
            raise ImportError(
                "mlagents-envs is required. Install with: pip install mlagents-envs"
            ) from e

        # Create engine configuration channel
        self._engine_channel = EngineConfigurationChannel()

        # Check if environment is in registry
        if env_id not in default_registry:
            available = list(default_registry.keys())
            raise ValueError(
                f"Environment '{env_id}' not found. Available: {available}"
            )

        # Get registry entry and create with persistent cache
        entry = default_registry[env_id]

        # Create a new entry with our persistent cache directory
        persistent_entry = RemoteRegistryEntry(
            identifier=entry.identifier,
            expected_reward=entry.expected_reward,
            description=entry.description,
            linux_url=getattr(entry, "_linux_url", None),
            darwin_url=getattr(entry, "_darwin_url", None),
            win_url=getattr(entry, "_win_url", None),
            additional_args=getattr(entry, "_add_args", []),
            tmp_dir=self._cache_dir,
        )

        # Each instance needs its own communication port: ask the OS for a free one
        with socket.socket() as sock:
            sock.bind(("localhost", 0))
            port = sock.getsockname()[1]

        # Create the environment
        self._unity_env = persistent_entry.make(
            base_port=port,
            no_graphics=self._no_graphics,
            side_channels=[self._engine_channel],
        )

        # Configure engine settings
        if not self._no_graphics:
            self._engine_channel.set_configuration_parameters(
                width=self._width,
                height=self._height,
                quality_level=self._quality_level,
                time_scale=self._time_scale,
            )
        else:
            self._engine_channel.set_configuration_parameters(
                time_scale=self._time_scale
            )

        # Get behavior info
        if not self._unity_env.behavior_specs:
            self._unity_env.step()

        self._behavior_name = list(self._unity_env.behavior_specs.keys())[0]
        self._behavior_spec = self._unity_env.behavior_specs[self._behavior_name]

        # Update state
        self._env_id = env_id
        self._state.env_id = env_id
        self._state.behavior_name = self._behavior_name
        self._state.action_spec = self._get_action_spec_info()
        self._state.observation_spec = self._get_observation_spec_info()
        self._state.available_envs = list(default_registry.keys())

    def _get_action_spec_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        spec = self._behavior_spec.action_spec
        return {
            "is_discrete": spec.is_discrete(),
            "is_continuous": spec.is_continuous(),
            "discrete_size": spec.discrete_size,
            "discrete_branches": list(spec.discrete_branches)
            if spec.is_discrete()
            else [],
            "continuous_size": spec.continuous_size,
        }

    def _get_observation_spec_info(self) -> Dict[str, Any]:
        """Get information about the observation space."""
        specs = self._behavior_spec.observation_specs
        obs_info = []
        for i, spec in enumerate(specs):
            obs_info.append(
                {
                    "index": i,
                    "shape": list(spec.shape),
                    "dimension_property": str(spec.dimension_property),
                    "observation_type": str(spec.observation_type),
                }
            )
        return {"observations": obs_info, "count": len(specs)}

    def _get_observation(
        self,
        decision_steps=None,
        terminal_steps=None,
        reward: float = 0.0,
        done: bool = False,
    ) -> UnityObservation:
        """Convert Unity observation to UnityObservation."""
        vector_obs = []
        visual_obs = []

        # Determine which steps to use
        if terminal_steps is not None and len(terminal_steps) > 0:
            steps = terminal_steps
            done = True
            # Get reward from terminal step
            if len(terminal_steps.agent_id) > 0:
                reward = float(terminal_steps[terminal_steps.agent_id[0]].reward)
        elif decision_steps is not None and len(decision_steps) > 0:
            steps = decision_steps
            # Get reward from decision step
            if len(decision_steps.agent_id) > 0:
                reward = float(decision_steps[decision_steps.agent_id[0]].reward)
        else:
            # No agents, return empty observation
            return UnityObservation(
                vector_observations=[],
                visual_observations=None,
                behavior_name=self._behavior_name or "",
                done=done,
                reward=reward,
                action_spec_info=self._state.action_spec,
                observation_spec_info=self._state.observation_spec,
            )

        # Process observations from first agent
        for obs in steps.obs:
            if len(obs.shape) == 2:
                # Vector observation (agents, features)
                vector_obs.extend(obs[0].tolist())
            elif len(obs.shape) == 4 and self._include_visual:
                # Visual observation (agents, channels, height, width)
                img_array = (np.moveaxis(obs[0], 0, -1) * 255).astype(np.uint8)
                # Encode as base64 PNG
                try:
                    from PIL import Image

                    img = Image.fromarray(img_array)
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    visual_obs.append(img_b64)
                except ImportError:
                    # PIL not available, skip visual observations
                    pass

        return UnityObservation(
            vector_observations=vector_obs,
            visual_observations=visual_obs if visual_obs else None,
            behavior_name=self._behavior_name or "",
            done=done,
            reward=reward,
            action_spec_info=self._state.action_spec,
            observation_spec_info=self._state.observation_spec,
        )

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """One button per option of a single discrete action branch (PushBlock moves are named)."""
        branches = observation["action_spec_info"].get("discrete_branches", [])
        if len(branches) != 1:
            return []
        pushblock = observation["behavior_name"].startswith("PushBlock")
        return [
            (
                f"{i} · {PUSHBLOCK_ACTIONS[i]}" if pushblock else f"action {i}",
                {"discrete_actions": [i]},
            )
            for i in range(branches[0])
        ]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the agent's camera image, when the observation carries one."""
        images = observation.get("visual_observations")
        if not images:
            return None
        return (
            '<img alt="Unity camera" '
            f'src="data:image/png;base64,{images[0]}" '
            'style="width:336px;max-width:100%;image-rendering:pixelated;'
            'border:1px solid var(--border-color-primary);border-radius:6px">'
        )

    def reset(
        self,
        env_id: Optional[str] = None,
        seed: Optional[int] = None,
        include_visual: Optional[bool] = None,
        **kwargs,
    ) -> UnityObservation:
        """
        Reset the environment and return initial observation.

        Args:
            env_id: Optionally switch to a different Unity environment.
            seed: Random seed (not fully supported by Unity ML-Agents).
            include_visual: If True, include visual observations in output.
                Defaults to UNITY_INCLUDE_VISUAL.
            **kwargs: Additional arguments (ignored).

        Returns:
            UnityObservation with initial state.
        """
        self._include_visual = (
            self._default_include_visual if include_visual is None else include_visual
        )

        # Load or switch environment if needed
        target_env = env_id or self._env_id
        if self._unity_env is None or target_env != self._env_id:
            self._load_environment(target_env)

        # Reset the environment
        self._unity_env.reset()

        # Update state
        self._state = UnityState(
            episode_id=str(uuid4()),
            step_count=0,
            env_id=self._env_id,
            behavior_name=self._behavior_name,
            action_spec=self._state.action_spec,
            observation_spec=self._state.observation_spec,
            available_envs=self._state.available_envs,
        )

        # Get initial observation
        decision_steps, terminal_steps = self._unity_env.get_steps(self._behavior_name)

        return self._get_observation(
            decision_steps=decision_steps,
            terminal_steps=terminal_steps,
            reward=0.0,
            done=False,
        )

    def step(self, action: UnityAction) -> UnityObservation:
        """
        Execute one step in the environment.

        Args:
            action: UnityAction with discrete and/or continuous actions.

        Returns:
            UnityObservation with new state, reward, and done flag.
        """
        if self._unity_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        from mlagents_envs.base_env import ActionTuple

        # Get current decision steps to know how many agents
        decision_steps, terminal_steps = self._unity_env.get_steps(self._behavior_name)

        # Check if episode already ended
        if len(terminal_steps) > 0:
            return self._get_observation(
                decision_steps=decision_steps,
                terminal_steps=terminal_steps,
                done=True,
            )

        n_agents = len(decision_steps)
        if n_agents == 0:
            # No agents need decisions, just step
            self._unity_env.step()
            self._state.step_count += 1
            decision_steps, terminal_steps = self._unity_env.get_steps(
                self._behavior_name
            )
            return self._get_observation(
                decision_steps=decision_steps,
                terminal_steps=terminal_steps,
            )

        # Build action tuple
        action_tuple = ActionTuple()

        # Handle discrete actions
        if action.discrete_actions is not None:
            discrete = np.array([action.discrete_actions] * n_agents, dtype=np.int32)
            # Ensure correct shape (n_agents, n_branches)
            if discrete.ndim == 1:
                discrete = discrete.reshape(n_agents, -1)
            action_tuple.add_discrete(discrete)
        elif self._behavior_spec.action_spec.is_discrete():
            # Default to no-op (action 0)
            n_branches = self._behavior_spec.action_spec.discrete_size
            discrete = np.zeros((n_agents, n_branches), dtype=np.int32)
            action_tuple.add_discrete(discrete)

        # Handle continuous actions
        if action.continuous_actions is not None:
            continuous = np.array(
                [action.continuous_actions] * n_agents, dtype=np.float32
            )
            if continuous.ndim == 1:
                continuous = continuous.reshape(n_agents, -1)
            action_tuple.add_continuous(continuous)
        elif self._behavior_spec.action_spec.is_continuous():
            # Default to zero actions
            n_continuous = self._behavior_spec.action_spec.continuous_size
            continuous = np.zeros((n_agents, n_continuous), dtype=np.float32)
            action_tuple.add_continuous(continuous)

        # Set actions and step
        self._unity_env.set_actions(self._behavior_name, action_tuple)
        self._unity_env.step()
        self._state.step_count += 1

        # Get new observation
        decision_steps, terminal_steps = self._unity_env.get_steps(self._behavior_name)

        return self._get_observation(
            decision_steps=decision_steps,
            terminal_steps=terminal_steps,
        )

    async def reset_async(
        self,
        env_id: Optional[str] = None,
        seed: Optional[int] = None,
        include_visual: Optional[bool] = None,
        **kwargs,
    ) -> UnityObservation:
        """
        Async version of reset - runs in a thread to avoid blocking the event loop.

        Unity ML-Agents environments can take 10-60+ seconds to initialize.
        Running in a thread allows the event loop to continue processing
        WebSocket keepalive pings during this time.
        """
        import asyncio

        return await asyncio.to_thread(
            self.reset,
            env_id=env_id,
            seed=seed,
            include_visual=include_visual,
            **kwargs,
        )

    async def step_async(self, action: UnityAction) -> UnityObservation:
        """
        Async version of step - runs in a thread to avoid blocking the event loop.

        Although step() is usually fast, running in a thread ensures
        the event loop remains responsive.
        """
        import asyncio

        return await asyncio.to_thread(self.step, action)

    @property
    def state(self) -> UnityState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Close the Unity environment."""
        unity_env = getattr(self, "_unity_env", None)
        if unity_env is not None:
            try:
                unity_env.close()
            except Exception:
                pass
            self._unity_env = None

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
