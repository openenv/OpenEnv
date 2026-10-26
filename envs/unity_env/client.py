# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unity ML-Agents Environment Client.

This module provides the client for connecting to a Unity ML-Agents
Environment server via WebSocket for persistent sessions.
"""

from typing import Any, Dict, List, Optional

# Support multiple import scenarios
try:
    # In-repo imports (when running from OpenEnv repository root)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import UnityAction, UnityObservation, UnityState
except ImportError:
    # openenv from pip
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    try:
        # Direct execution from envs/unity_env/ directory
        from models import UnityAction, UnityObservation, UnityState
    except ImportError:
        try:
            # Package installed as unity_env
            from unity_env.models import UnityAction, UnityObservation, UnityState
        except ImportError:
            # Running from OpenEnv root with envs prefix
            from envs.unity_env.models import UnityAction, UnityObservation, UnityState


class UnityEnv(EnvClient[UnityAction, UnityObservation, UnityState]):
    """
    Client for Unity ML-Agents environments.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Note: Unity environments can take 30-60+ seconds to initialize on first reset
    (downloading binaries, starting Unity process). The client is configured with
    longer ping timeouts to handle this.

    Supported Unity Environments:
    - PushBlock: Push a block to a goal (discrete actions: 7)
    - 3DBall: Balance a ball on a platform (continuous actions: 2)
    - 3DBallHard: Harder version of 3DBall
    - GridWorld: Navigate a grid to find goals
    - Basic: Simple movement task
    - And more from the ML-Agents registry

    Example:
        >>> # Connect to a running server
        >>> with UnityEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(f"Vector obs: {len(result.observation.vector_observations)} dims")
        ...
        ...     # Take action (PushBlock: 1=forward)
        ...     result = client.step(UnityAction(discrete_actions=[1]))
        ...     print(f"Reward: {result.reward}")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = UnityEnv.from_docker_image("unity-env:latest")
        >>> try:
        ...     result = client.reset(env_id="3DBall")
        ...     result = client.step(UnityAction(continuous_actions=[0.5, -0.3]))
        ... finally:
        ...     client.close()

    Example switching environments:
        >>> client = UnityEnv(base_url="http://localhost:8000")
        >>> # Start with PushBlock
        >>> result = client.reset(env_id="PushBlock")
        >>> # ... train on PushBlock ...
        >>> # Switch to 3DBall
        >>> result = client.reset(env_id="3DBall")
        >>> # ... train on 3DBall ...
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 180.0,  # 3 minutes for slow Unity initialization
        provider: Optional[Any] = None,
    ):
        """
        Initialize Unity environment client.

        Uses longer default timeouts than the base EnvClient because Unity
        environments can take 30-60+ seconds to initialize on first reset.

        Args:
            base_url: Base URL of the environment server (http:// or ws://).
            connect_timeout_s: Timeout for establishing WebSocket connection
            message_timeout_s: Timeout for receiving responses (default 3 min for Unity)
            provider: Optional container/runtime provider for lifecycle management.
        """
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            provider=provider,
        )

    def connect(self) -> "UnityEnv":
        """
        Establish WebSocket connection to the server.

        Overrides the default connection to use longer ping timeouts,
        since Unity environments can take 30-60+ seconds to initialize.

        Returns:
            self for method chaining

        Raises:
            ConnectionError: If connection cannot be established
        """
        from websockets.sync.client import connect as ws_connect

        if self._ws is not None:
            return self

        try:
            # Use longer ping_timeout for Unity (60s) since environment
            # initialization can block the server for a while
            self._ws = ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                ping_timeout=120,  # 2 minutes for slow Unity initialization
                ping_interval=30,  # Send pings every 30 seconds
                close_timeout=30,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e

        return self

    def _step_payload(self, action: UnityAction) -> Dict:
        """
        Convert UnityAction to JSON payload for step request.

        Args:
            action: UnityAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, Any] = {}

        if action.discrete_actions is not None:
            payload["discrete_actions"] = action.discrete_actions

        if action.continuous_actions is not None:
            payload["continuous_actions"] = action.continuous_actions

        if action.metadata:
            payload["metadata"] = action.metadata

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[UnityObservation]:
        """
        Parse server response into StepResult[UnityObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with UnityObservation
        """
        obs_data = payload.get("observation", {})

        observation = UnityObservation(
            vector_observations=obs_data.get("vector_observations", []),
            visual_observations=obs_data.get("visual_observations"),
            behavior_name=obs_data.get("behavior_name", ""),
            action_spec_info=obs_data.get("action_spec_info", {}),
            observation_spec_info=obs_data.get("observation_spec_info", {}),
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

    def _parse_state(self, payload: Dict) -> UnityState:
        """
        Parse server response into UnityState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            UnityState object with environment information
        """
        return UnityState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            env_id=payload.get("env_id", ""),
            behavior_name=payload.get("behavior_name", ""),
            action_spec=payload.get("action_spec", {}),
            observation_spec=payload.get("observation_spec", {}),
            available_envs=payload.get("available_envs", []),
        )

    def reset(
        self,
        env_id: Optional[str] = None,
        include_visual: bool = False,
        **kwargs,
    ) -> StepResult[UnityObservation]:
        """
        Reset the environment.

        Args:
            env_id: Optionally switch to a different Unity environment.
                Available: PushBlock, 3DBall, 3DBallHard, GridWorld, Basic
            include_visual: If True, include visual observations in response.
            **kwargs: Additional arguments passed to server.

        Returns:
            StepResult with initial observation.
        """
        reset_kwargs = dict(kwargs)
        if env_id is not None:
            reset_kwargs["env_id"] = env_id
        reset_kwargs["include_visual"] = include_visual

        return super().reset(**reset_kwargs)

    @staticmethod
    def available_environments() -> List[str]:
        """
        List commonly available Unity environments.

        Note: The actual list may vary based on the ML-Agents registry version.
        Use state.available_envs after connecting for the authoritative list.

        Returns:
            List of environment identifiers.
        """
        return [
            "PushBlock",
            "3DBall",
            "3DBallHard",
            "GridWorld",
            "Basic",
            "VisualPushBlock",
        ]

    @classmethod
    def from_direct(
        cls,
        env_id: str = "PushBlock",
        no_graphics: bool = False,
        width: int = 1280,
        height: int = 720,
        time_scale: float = 1.0,
        quality_level: int = 5,
        port: int = 8765,
    ) -> "UnityEnv":
        """
        Create a Unity environment client with an embedded local server.

        This method starts a local uvicorn server in a subprocess and returns
        a client connected to it. This provides the convenience of direct mode
        while maintaining the client-server separation.

        Note: The first call will download Unity binaries (~500MB) which may
        take several minutes. Binaries are cached for subsequent runs.

        Args:
            env_id: Default Unity environment to use (PushBlock, 3DBall, etc.)
            no_graphics: If True, run Unity in headless mode (faster for training)
            width: Window width in pixels (default: 1280)
            height: Window height in pixels (default: 720)
            time_scale: Simulation speed multiplier (default: 1.0, use 20.0 for fast training)
            quality_level: Graphics quality 0-5 (default: 5)
            port: Port for the local server (default: 8765)

        Returns:
            UnityEnv client connected to the local server

        Example:
            >>> # Quick start with direct mode
            >>> client = UnityEnv.from_direct(no_graphics=True, time_scale=20)
            >>> try:
            ...     result = client.reset(env_id="PushBlock")
            ...     for _ in range(100):
            ...         result = client.step(UnityAction(discrete_actions=[1]))
            ... finally:
            ...     client.close()

            >>> # With custom settings
            >>> client = UnityEnv.from_direct(
            ...     env_id="3DBall",
            ...     no_graphics=True,
            ...     time_scale=20,
            ...     port=9000
            ... )
        """
        import os
        import subprocess
        import sys
        import time

        import requests

        # Find the project root and server module
        # Try to locate the server module
        try:
            from pathlib import Path

            # Get the directory containing this file
            client_dir = Path(__file__).parent
            server_app = "envs.unity_env.server.app:app"
            cwd = client_dir.parent.parent  # OpenEnv root

            # Check if we're in the envs/unity_env directory structure
            if not (cwd / "envs" / "unity_env" / "server" / "app.py").exists():
                # Try alternative paths
                if (client_dir / "server" / "app.py").exists():
                    server_app = "server.app:app"
                    cwd = client_dir
        except Exception:
            server_app = "envs.unity_env.server.app:app"
            cwd = None

        # Set up environment variables for Unity configuration
        env = {
            **os.environ,
            "UNITY_ENV_ID": env_id,
            "UNITY_NO_GRAPHICS": "1" if no_graphics else "0",
            "UNITY_WIDTH": str(width),
            "UNITY_HEIGHT": str(height),
            "UNITY_TIME_SCALE": str(time_scale),
            "UNITY_QUALITY_LEVEL": str(quality_level),
            # Bypass proxy for localhost
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        }

        # Add src to PYTHONPATH if needed
        if cwd:
            src_path = str(cwd / "src")
            existing_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{src_path}:{cwd}:{existing_path}"
                if existing_path
                else f"{src_path}:{cwd}"
            )

        # Start the server
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

        # Wait for server to become healthy
        base_url = f"http://127.0.0.1:{port}"
        healthy = False
        for _ in range(30):  # Wait up to 30 seconds
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
                f"Failed to start local Unity server on port {port}. "
                "Check that the port is available and dependencies are installed."
            )

        # Create a provider to manage the subprocess lifecycle
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

        # Create and return the client
        client = cls(base_url=base_url, provider=provider)
        return client
