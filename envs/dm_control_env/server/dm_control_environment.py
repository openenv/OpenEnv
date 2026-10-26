# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
dm_control Environment Implementation.

Wraps dm_control.suite environments (cartpole, walker, humanoid, etc.)
with the OpenEnv interface for standardized reinforcement learning.
"""

import base64
import io
import os
import sys
from typing import Any, Dict, Optional
from uuid import uuid4

# Configure MuJoCo rendering backend before importing dm_control
# On macOS, we don't set MUJOCO_GL - use default (glfw) which works
# when running synchronously in the main thread (see reset_async/step_async)
# On Linux, use egl for headless rendering
if "MUJOCO_GL" not in os.environ and sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

try:
    from openenv.core.env_server.interfaces import Environment

    from ..models import DMControlAction, DMControlObservation, DMControlState
except ImportError:
    from openenv.core.env_server.interfaces import Environment

    try:
        import sys
        from pathlib import Path

        _parent = str(Path(__file__).parent.parent)
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from models import DMControlAction, DMControlObservation, DMControlState
    except ImportError:
        try:
            from dm_control_env.models import (
                DMControlAction,
                DMControlObservation,
                DMControlState,
            )
        except ImportError:
            from envs.dm_control_env.models import (
                DMControlAction,
                DMControlObservation,
                DMControlState,
            )


class DMControlEnvironment(Environment):
    """
    Wraps dm_control.suite environments with the OpenEnv interface.

    This environment supports all dm_control.suite domains and tasks including
    cartpole, walker, humanoid, cheetah, and more.

    Features:
    - Dynamic environment switching via reset(domain_name="...", task_name="...")
    - Support for all continuous control tasks
    - Optional visual observations (base64-encoded images)
    - Configurable via constructor or environment variables

    Example:
        >>> env = DMControlEnvironment()
        >>> obs = env.reset()  # Default: cartpole/balance
        >>> print(obs.observations)
        >>>
        >>> # Take an action
        >>> obs = env.step(DMControlAction(values=[0.5]))  # Push cart right
        >>> print(obs.reward)

    Example with different environment:
        >>> env = DMControlEnvironment(domain_name="walker", task_name="walk")
        >>> obs = env.reset()
        >>>
        >>> # Or switch environment on reset
        >>> obs = env.reset(domain_name="cheetah", task_name="run")
    """

    # dm_control environments are isolated and thread-safe
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        domain_name: Optional[str] = None,
        task_name: Optional[str] = None,
        render_height: Optional[int] = None,
        render_width: Optional[int] = None,
    ):
        """
        Initialize the dm_control environment.

        Args:
            domain_name: The dm_control domain to load.
                Env var: DMCONTROL_DOMAIN (default: cartpole)
            task_name: The task within the domain.
                Env var: DMCONTROL_TASK (default: balance)
            render_height: Height of rendered images (when render=True).
                Env var: DMCONTROL_RENDER_HEIGHT (default: 480)
            render_width: Width of rendered images (when render=True).
                Env var: DMCONTROL_RENDER_WIDTH (default: 640)
        """
        self._env = None

        self._domain_name = domain_name or os.environ.get(
            "DMCONTROL_DOMAIN", "cartpole"
        )
        self._task_name = task_name or os.environ.get("DMCONTROL_TASK", "balance")
        self._render_height = (
            render_height
            if render_height is not None
            else int(os.environ.get("DMCONTROL_RENDER_HEIGHT", "480"))
        )
        self._render_width = (
            render_width
            if render_width is not None
            else int(os.environ.get("DMCONTROL_RENDER_WIDTH", "640"))
        )
        self._include_pixels = False

        self._state = DMControlState(
            episode_id=str(uuid4()),
            step_count=0,
            domain_name=self._domain_name,
            task_name=self._task_name,
        )

    def _load_environment(self, domain_name: str, task_name: str) -> None:
        """Load or switch to a dm_control environment."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

        try:
            from dm_control import suite
        except ImportError as e:
            raise ImportError(
                "dm_control is required. Install with: pip install dm_control"
            ) from e
        except Exception as e:
            # MuJoCo/OpenGL initialization can fail on macOS
            error_msg = str(e)
            if sys.platform == "darwin":
                raise RuntimeError(
                    f"Failed to import dm_control (MuJoCo error): {error_msg}\n\n"
                    "On macOS, try one of these solutions:\n"
                    "1. Install osmesa: brew install mesa\n"
                    "2. Run with MUJOCO_GL=glfw (requires display)\n"
                    "3. Run with MUJOCO_GL=egl (if EGL is available)"
                ) from e
            raise

        try:
            self._env = suite.load(domain_name=domain_name, task_name=task_name)
        except Exception as e:
            error_msg = str(e).lower()
            # Check for MuJoCo/OpenGL errors
            if "gl" in error_msg or "render" in error_msg or "display" in error_msg:
                if sys.platform == "darwin":
                    raise RuntimeError(
                        f"MuJoCo initialization failed: {e}\n\n"
                        "On macOS, try one of these solutions:\n"
                        "1. Install osmesa: brew install mesa\n"
                        "2. Run with MUJOCO_GL=glfw (requires display)\n"
                        "3. Set PYOPENGL_PLATFORM=osmesa"
                    ) from e
            # Check if it's an invalid environment error
            try:
                available = [(d, t) for d, t in suite.BENCHMARKING]
                raise ValueError(
                    f"Failed to load {domain_name}/{task_name}. "
                    f"Available environments: {available[:10]}... "
                    f"(use dm_control.suite.BENCHMARKING for full list)"
                ) from e
            except Exception:
                raise

        self._domain_name = domain_name
        self._task_name = task_name

        self._state.domain_name = domain_name
        self._state.task_name = task_name
        self._state.action_spec = self._get_action_spec_info()
        self._state.observation_spec = self._get_observation_spec_info()
        self._state.physics_timestep = self._env.physics.timestep()
        self._state.control_timestep = self._env.control_timestep()

    def _get_action_spec_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        spec = self._env.action_spec()
        return {
            "shape": list(spec.shape),
            "dtype": str(spec.dtype),
            "minimum": spec.minimum.tolist(),
            "maximum": spec.maximum.tolist(),
            "name": spec.name,
        }

    def _get_observation_spec_info(self) -> Dict[str, Any]:
        """Get information about the observation space."""
        specs = self._env.observation_spec()
        obs_info = {}
        for name, spec in specs.items():
            obs_info[name] = {
                "shape": list(spec.shape),
                "dtype": str(spec.dtype),
            }
        return obs_info

    def _get_observation(
        self,
        time_step,
        include_pixels: bool = False,
    ) -> DMControlObservation:
        """Convert dm_control TimeStep to DMControlObservation."""
        import dm_env

        observations = {}
        for name, value in time_step.observation.items():
            observations[name] = np.asarray(value).flatten().tolist()

        pixels = None
        if include_pixels:
            try:
                frame = self._env.physics.render(
                    height=self._render_height,
                    width=self._render_width,
                    camera_id=0,
                )
                from PIL import Image

                img = Image.fromarray(frame)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                pixels = base64.b64encode(buffer.getvalue()).decode("utf-8")
            except Exception:
                pass

        done = time_step.step_type == dm_env.StepType.LAST
        reward = float(time_step.reward) if time_step.reward is not None else 0.0

        return DMControlObservation(
            observations=observations,
            pixels=pixels,
            reward=reward,
            done=done,
        )

    def reset(
        self,
        domain_name: Optional[str] = None,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        render: bool = False,
        **kwargs,
    ) -> DMControlObservation:
        """
        Reset the environment and return initial observation.

        Args:
            domain_name: Optionally switch to a different domain.
            task_name: Optionally switch to a different task.
            seed: Random seed for reproducibility.
            render: If True, include pixel observations.
            **kwargs: Additional arguments (ignored).

        Returns:
            DMControlObservation with initial state.
        """
        self._include_pixels = render

        target_domain = domain_name or self._domain_name
        target_task = task_name or self._task_name

        if (
            self._env is None
            or target_domain != self._domain_name
            or target_task != self._task_name
        ):
            self._load_environment(target_domain, target_task)

        if seed is not None:
            np.random.seed(seed)

        time_step = self._env.reset()

        self._state = DMControlState(
            episode_id=str(uuid4()),
            step_count=0,
            domain_name=self._domain_name,
            task_name=self._task_name,
            action_spec=self._state.action_spec,
            observation_spec=self._state.observation_spec,
            physics_timestep=self._state.physics_timestep,
            control_timestep=self._state.control_timestep,
        )

        return self._get_observation(time_step, include_pixels=render)

    def step(
        self,
        action: DMControlAction,
        render: bool = False,
        **kwargs,
    ) -> DMControlObservation:
        """
        Execute one step in the environment.

        Args:
            action: DMControlAction with continuous action values.
            render: If True, include pixel observations.

        Returns:
            DMControlObservation with new state, reward, and done flag.
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action_array = np.array(action.values, dtype=np.float64)

        action_spec = self._env.action_spec()
        expected_shape = action_spec.shape
        if action_array.shape != expected_shape:
            if action_array.size == np.prod(expected_shape):
                action_array = action_array.reshape(expected_shape)
            else:
                raise ValueError(
                    f"Action shape {action_array.shape} doesn't match "
                    f"expected shape {expected_shape}"
                )

        action_array = np.clip(action_array, action_spec.minimum, action_spec.maximum)

        time_step = self._env.step(action_array)
        self._state.step_count += 1

        return self._get_observation(
            time_step, include_pixels=render or self._include_pixels
        )

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the physics scene from the first camera (rendering does not step the simulation)."""
        if "observations" not in observation:
            return None
        try:
            frame = self._env.physics.render(height=240, width=320, camera_id=0)
        except Exception:
            return None
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=85)
        return (
            '<img alt="dm_control frame" '
            f'src="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}" '
            'style="width:320px;max-width:100%;border-radius:10px;'
            'border:1px solid var(--border-color-primary)">'
        )

    async def reset_async(
        self,
        domain_name: Optional[str] = None,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        render: bool = False,
        **kwargs,
    ) -> DMControlObservation:
        """Async version of reset.

        On macOS, runs synchronously to avoid MuJoCo threading crashes.
        On other platforms, runs in a thread pool.
        """
        if sys.platform == "darwin":
            # On macOS, MuJoCo crashes when run in a background thread
            # Run synchronously (blocks event loop but avoids crash)
            return self.reset(
                domain_name=domain_name,
                task_name=task_name,
                seed=seed,
                render=render,
                **kwargs,
            )
        else:
            import asyncio

            return await asyncio.to_thread(
                self.reset,
                domain_name=domain_name,
                task_name=task_name,
                seed=seed,
                render=render,
                **kwargs,
            )

    async def step_async(
        self,
        action: DMControlAction,
        render: bool = False,
        **kwargs,
    ) -> DMControlObservation:
        """Async version of step.

        On macOS, runs synchronously to avoid MuJoCo threading crashes.
        On other platforms, runs in a thread pool.
        """
        if sys.platform == "darwin":
            # On macOS, MuJoCo crashes when run in a background thread
            # Run synchronously (blocks event loop but avoids crash)
            return self.step(action, render=render, **kwargs)
        else:
            import asyncio

            return await asyncio.to_thread(self.step, action, render=render, **kwargs)

    @property
    def state(self) -> DMControlState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Close the dm_control environment."""
        env = getattr(self, "_env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            self._env = None

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
