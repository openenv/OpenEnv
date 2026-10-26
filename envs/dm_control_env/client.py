# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
dm_control Environment Client.

This module provides the client for connecting to a dm_control
Environment server via WebSocket for persistent sessions.
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import (
        AVAILABLE_ENVIRONMENTS,
        DMControlAction,
        DMControlObservation,
        DMControlState,
    )
except ImportError:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    try:
        from models import (
            AVAILABLE_ENVIRONMENTS,
            DMControlAction,
            DMControlObservation,
            DMControlState,
        )
    except ImportError:
        try:
            from dm_control_env.models import (
                AVAILABLE_ENVIRONMENTS,
                DMControlAction,
                DMControlObservation,
                DMControlState,
            )
        except ImportError:
            from envs.dm_control_env.models import (
                AVAILABLE_ENVIRONMENTS,
                DMControlAction,
                DMControlObservation,
                DMControlState,
            )


class DMControlEnv(EnvClient[DMControlAction, DMControlObservation, DMControlState]):
    """
    Client for dm_control.suite environments.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.

    Supported Environments (via dm_control.suite):
    - cartpole: balance, swingup, swingup_sparse
    - walker: stand, walk, run
    - humanoid: stand, walk, run
    - cheetah: run
    - hopper: stand, hop
    - reacher: easy, hard
    - And many more...

    Example:
        >>> # Connect to a running server
        >>> with DMControlEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(f"Observations: {result.observation.observations.keys()}")
        ...
        ...     # Take action (cartpole: push right)
        ...     result = client.step(DMControlAction(values=[0.5]))
        ...     print(f"Reward: {result.reward}")

    Example switching environments:
        >>> client = DMControlEnv(base_url="http://localhost:8000")
        >>> # Start with cartpole balance
        >>> result = client.reset(domain_name="cartpole", task_name="balance")
        >>> # ... train on cartpole ...
        >>> # Switch to walker walk
        >>> result = client.reset(domain_name="walker", task_name="walk")
        >>> # ... train on walker ...
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        provider: Optional[Any] = None,
    ):
        """
        Initialize dm_control environment client.

        Args:
            base_url: Base URL of the environment server (http:// or ws://).
            connect_timeout_s: Timeout for establishing WebSocket connection.
            message_timeout_s: Timeout for receiving responses.
            provider: Optional container/runtime provider for lifecycle management.
        """
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            provider=provider,
        )

    def _step_payload(self, action: DMControlAction) -> Dict:
        """
        Convert DMControlAction to JSON payload for step request.

        Args:
            action: DMControlAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, Any] = {"values": action.values}

        if action.metadata:
            payload["metadata"] = action.metadata

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[DMControlObservation]:
        """
        Parse server response into StepResult[DMControlObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with DMControlObservation
        """
        obs_data = payload.get("observation", {})

        observation = DMControlObservation(
            observations=obs_data.get("observations", {}),
            pixels=obs_data.get("pixels"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> DMControlState:
        """
        Parse server response into DMControlState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            DMControlState object with environment information
        """
        return DMControlState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            domain_name=payload.get("domain_name", ""),
            task_name=payload.get("task_name", ""),
            action_spec=payload.get("action_spec", {}),
            observation_spec=payload.get("observation_spec", {}),
            physics_timestep=payload.get("physics_timestep", 0.002),
            control_timestep=payload.get("control_timestep", 0.02),
        )

    def reset(
        self,
        domain_name: Optional[str] = None,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        render: bool = False,
        **kwargs,
    ) -> StepResult[DMControlObservation]:
        """
        Reset the environment.

        Args:
            domain_name: Optionally switch to a different domain.
            task_name: Optionally switch to a different task.
            seed: Random seed for reproducibility.
            render: If True, include pixel observations in response.
            **kwargs: Additional arguments passed to server.

        Returns:
            StepResult with initial observation.
        """
        reset_kwargs = dict(kwargs)
        if domain_name is not None:
            reset_kwargs["domain_name"] = domain_name
        if task_name is not None:
            reset_kwargs["task_name"] = task_name
        if seed is not None:
            reset_kwargs["seed"] = seed
        reset_kwargs["render"] = render

        return super().reset(**reset_kwargs)

    def step(
        self,
        action: DMControlAction,
        render: bool = False,
        **kwargs,
    ) -> StepResult[DMControlObservation]:
        """
        Execute one step in the environment.

        Args:
            action: DMControlAction with continuous action values.
            render: If True, include pixel observations in response.
            **kwargs: Additional arguments passed to server.

        Returns:
            StepResult with new observation, reward, and done flag.
        """
        # Note: render flag needs to be passed differently
        # For now, the server remembers the render setting from reset
        return super().step(action, **kwargs)

    @staticmethod
    def available_environments() -> List[Tuple[str, str]]:
        """
        List available dm_control environments.

        Returns:
            List of (domain_name, task_name) tuples.
        """
        return AVAILABLE_ENVIRONMENTS

    @classmethod
    def from_direct(
        cls,
        domain_name: str = "cartpole",
        task_name: str = "balance",
        render_height: int = 480,
        render_width: int = 640,
        port: int = 8765,
    ) -> "DMControlEnv":
        """
        Create a dm_control environment client with an embedded local server.

        This method starts a local uvicorn server in a subprocess and returns
        a client connected to it.

        Args:
            domain_name: Default domain to use.
            task_name: Default task to use.
            render_height: Height of rendered images.
            render_width: Width of rendered images.
            port: Port for the local server.

        Returns:
            DMControlEnv client connected to the local server.

        Example:
            >>> client = DMControlEnv.from_direct(domain_name="walker", task_name="walk")
            >>> try:
            ...     result = client.reset()
            ...     for _ in range(100):
            ...         result = client.step(DMControlAction(values=[0.0] * 6))
            ... finally:
            ...     client.close()
        """
        import os
        import subprocess
        import sys
        import time

        import requests

        try:
            from pathlib import Path

            client_dir = Path(__file__).parent
            server_app = "envs.dm_control_env.server.app:app"
            cwd = client_dir.parent.parent

            if not (cwd / "envs" / "dm_control_env" / "server" / "app.py").exists():
                if (client_dir / "server" / "app.py").exists():
                    server_app = "server.app:app"
                    cwd = client_dir
        except Exception:
            server_app = "envs.dm_control_env.server.app:app"
            cwd = None

        env = {
            **os.environ,
            "DMCONTROL_DOMAIN": domain_name,
            "DMCONTROL_TASK": task_name,
            "DMCONTROL_RENDER_HEIGHT": str(render_height),
            "DMCONTROL_RENDER_WIDTH": str(render_width),
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        }

        if cwd:
            src_path = str(cwd / "src")
            existing_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{src_path}:{cwd}:{existing_path}"
                if existing_path
                else f"{src_path}:{cwd}"
            )

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            server_app,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

        server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
        )

        base_url = f"http://127.0.0.1:{port}"
        healthy = False
        for _ in range(30):
            try:
                response = requests.get(
                    f"{base_url}/health",
                    timeout=2,
                    proxies={"http": None, "https": None},
                )
                if response.status_code == 200:
                    healthy = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        if not healthy:
            server_process.kill()
            raise RuntimeError(
                f"Failed to start local dm_control server on port {port}. "
                "Check that the port is available and dependencies are installed."
            )

        class DirectModeProvider:
            """Provider that manages the embedded server subprocess."""

            def __init__(self, process: subprocess.Popen):
                self._process = process

            def stop(self):
                """Stop the embedded server."""
                if self._process:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                    self._process = None

        provider = DirectModeProvider(server_process)
        client = cls(base_url=base_url, provider=provider)
        return client
