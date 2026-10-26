# SPDX-License-Identifier: BSD-3-Clause

"""
Container provider abstractions for running environment servers.

This module provides a pluggable architecture for different container providers
(local Docker, Kubernetes, cloud providers, etc.) to be used with EnvClient.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, TypeVar

_ContainerProviderT = TypeVar("_ContainerProviderT", bound="ContainerProvider")


class ContainerProvider(ABC):
    """
    Abstract base class for container providers.

    Providers implement this interface to support different container platforms:
    - LocalDockerProvider: Runs containers on local Docker daemon
    - KubernetesProvider: Runs containers in Kubernetes cluster
    - FargateProvider: Runs containers on AWS Fargate
    - CloudRunProvider: Runs containers on Google Cloud Run

    The provider manages a single container lifecycle and provides the base URL
    for connecting to it.

    Examples:

        ```python
        provider = LocalDockerProvider()
        base_url = provider.start_container("echo-env:latest")
        print(base_url)  # http://localhost:8000
        # Use the environment via base_url
        provider.stop_container()
        ```
    """

    @abstractmethod
    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Start a container from the specified image.

        Args:
            image (`str`):
                Provider-specific container *source* identifier. For
                container-based providers this is a registry image name (e.g.
                `"echo-env:latest"`); other providers may map it to a
                provider-specific source (see the provider's documentation).
            port (`int`, *optional*):
                Port to expose. If `None`, the provider chooses.
            env_vars (`dict`, *optional*):
                Environment variables to pass to container.
            **kwargs:
                Provider-specific options.

        Returns:
            `str`: Base URL to connect to the container (e.g., `"http://localhost:8000"`).

        Raises:
            RuntimeError: If container fails to start.
        """
        pass

    @abstractmethod
    def stop_container(self) -> None:
        """
        Stop and remove the running container.

        This cleans up the container that was started by start_container().
        """
        pass

    @abstractmethod
    def wait_for_ready(self, base_url: str, timeout_s: float = 30.0) -> None:
        """
        Wait for the container to be ready to accept requests.

        This typically polls the /health endpoint until it returns 200.

        Args:
            base_url (`str`):
                Base URL of the container.
            timeout_s (`float`, *optional*, defaults to `30.0`):
                Maximum time to wait in seconds.

        Raises:
            TimeoutError: If container doesn't become ready in time.
        """
        pass

    def close(self) -> None:
        """
        Release provider-held resources (e.g. SDK clients, connections).

        Defaults to a no-op so existing providers are unaffected. Providers that
        hold external resources beyond the container itself (such as a cloud SDK
        client) override this to release them; it is also invoked on context-
        manager exit. Lightweight providers need not override it.
        """
        pass

    def __enter__(self: _ContainerProviderT) -> _ContainerProviderT:
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self.close()
        return None


class LocalDockerProvider(ContainerProvider):
    """
    Container provider for local Docker daemon.

    This provider runs containers on the local machine using Docker.
    Useful for development and testing.

    Examples:

        ```python
        provider = LocalDockerProvider()
        base_url = provider.start_container("echo-env:latest")
        # Container running on http://localhost:<random-port>
        provider.stop_container()
        ```
    """

    def __init__(self):
        """Initialize the local Docker provider."""
        self._container_id: Optional[str] = None
        self._container_name: Optional[str] = None

        # Check if Docker is available
        import subprocess

        try:
            subprocess.run(
                ["docker", "version"],
                check=True,
                capture_output=True,
                timeout=5,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            raise RuntimeError(
                "Docker is not available. Please install Docker Desktop or Docker Engine."
            )

    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Start a Docker container locally.

        Args:
            image (`str`):
                Docker image name.
            port (`int`, *optional*):
                Port to expose. If `None`, finds an available port.
            env_vars (`dict`, *optional*):
                Environment variables for the container.
            **kwargs:
                Additional Docker run options.

        Returns:
            `str`: Base URL to connect to the container.
        """
        import subprocess
        import time

        # Find available port if not specified
        if port is None:
            port = self._find_available_port()

        # Generate container name
        self._container_name = self._generate_container_name(image)

        # Build docker run command
        cmd = [
            "docker",
            "run",
            "-d",  # Detached
            "--name",
            self._container_name,
            "-p",
            f"{port}:8000",  # Map port
        ]

        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Add image
        cmd.append(image)

        # Run container
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._container_id = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to start Docker container.\nCommand: {' '.join(cmd)}\nExit code: {e.returncode}\nStderr: {e.stderr}\nStdout: {e.stdout}"
            raise RuntimeError(error_msg) from e

        # Wait a moment for container to start
        time.sleep(1)

        base_url = f"http://localhost:{port}"
        return base_url

    def stop_container(self) -> None:
        """
        Stop and remove the Docker container.
        """
        if self._container_id is None:
            return

        import subprocess

        try:
            # Stop container
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
                check=True,
                timeout=10,
            )

            # Remove container
            subprocess.run(
                ["docker", "rm", self._container_id],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except subprocess.CalledProcessError:
            # Container might already be stopped/removed
            pass
        finally:
            self._container_id = None
            self._container_name = None

    def wait_for_ready(self, base_url: str, timeout_s: float = 30.0) -> None:
        """
        Wait for container to be ready by polling /health endpoint.

        Args:
            base_url (`str`):
                Base URL of the container.
            timeout_s (`float`, *optional*, defaults to `30.0`):
                Maximum time to wait in seconds.

        Raises:
            TimeoutError: If container doesn't become ready.
        """
        import time

        import requests

        start_time = time.time()
        health_url = f"{base_url}/health"

        # Bypass proxy for localhost to avoid proxy issues
        proxies = {"http": None, "https": None}

        while time.time() - start_time < timeout_s:
            try:
                response = requests.get(health_url, timeout=2.0, proxies=proxies)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            time.sleep(0.5)

        raise TimeoutError(
            f"Container at {base_url} did not become ready within {timeout_s}s"
        )

    def _find_available_port(self) -> int:
        """
        Find an available port on localhost.

        Returns:
            `int`: An available port number.
        """
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _generate_container_name(self, image: str) -> str:
        """
        Generate a unique container name based on image name and timestamp.

        Args:
            image (`str`):
                Docker image name.

        Returns:
            `str`: A unique container name.
        """
        import time

        clean_image = image.split("/")[-1].split(":")[0]
        timestamp = int(time.time() * 1000)
        return f"{clean_image}-{timestamp}"


class DockerSwarmProvider(ContainerProvider):
    """
    Container provider that uses Docker Swarm services for local concurrency.

    This provider creates a replicated Swarm service backed by the local Docker
    engine. The built-in load-balancer fans requests across the replicas,
    allowing multiple container instances to run concurrently on the developer
    workstation (mirroring the workflow described in the Docker stack docs).
    """

    def __init__(
        self,
        *,
        auto_init_swarm: bool = True,
        overlay_network: Optional[str] = None,
    ):
        """
        Args:
            auto_init_swarm (`bool`, *optional*, defaults to `True`):
                Whether to call `docker swarm init` when Swarm is not active.
                Otherwise, the user must manually initialize Swarm.
            overlay_network (`str`, *optional*):
                Overlay network name for the service. When provided, the network
                is created with `docker network create --driver overlay --attachable`
                if it does not already exist.
        """
        self._service_name: Optional[str] = None
        self._service_id: Optional[str] = None
        self._published_port: Optional[int] = None
        self._overlay_network = overlay_network
        self._auto_init_swarm = auto_init_swarm

        self._ensure_docker_available()
        self._ensure_swarm_initialized()
        if self._overlay_network:
            self._ensure_overlay_network(self._overlay_network)

    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Start (or scale) a Swarm service for the given image.

        Args:
            image (`str`):
                Docker image name.
            port (`int`, *optional*):
                Port to expose. If `None`, finds an available port.
            env_vars (`dict`, *optional*):
                Environment variables for the container.
            replicas (`int`, *optional*, defaults to `2`):
                Number of container replicas.
            cpu_limit (`float` or `str`, *optional*):
                CPU limit passed to `--limit-cpu`.
            memory_limit (`str`, *optional*):
                Memory limit passed to `--limit-memory`.
            constraints (`Sequence[str]`, *optional*):
                Placement constraints.
            labels (`dict`, *optional*):
                Service labels.
            command (`Sequence[str]` or `str`, *optional*):
                Override container command.

        Returns:
            `str`: Base URL to connect to the service.
        """
        import shlex
        import subprocess
        import time

        allowed_kwargs = {
            "replicas",
            "cpu_limit",
            "memory_limit",
            "constraints",
            "labels",
            "command",
        }
        unknown = set(kwargs) - allowed_kwargs
        if unknown:
            raise ValueError(f"Unsupported kwargs for DockerSwarmProvider: {unknown}")

        replicas = int(kwargs.get("replicas", 2))
        cpu_limit = kwargs.get("cpu_limit")
        memory_limit = kwargs.get("memory_limit")
        constraints: Optional[Sequence[str]] = kwargs.get("constraints")
        labels: Optional[Dict[str, str]] = kwargs.get("labels")
        command_override = kwargs.get("command")

        if port is None:
            port = self._find_available_port()

        self._service_name = self._generate_service_name(image)
        self._published_port = port

        cmd = [
            "docker",
            "service",
            "create",
            "--detach",
            "--name",
            self._service_name,
            "--replicas",
            str(max(1, replicas)),
            "--publish",
            f"{port}:8000",
        ]

        if self._overlay_network:
            cmd.extend(["--network", self._overlay_network])

        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["--env", f"{key}={value}"])

        if cpu_limit is not None:
            cmd.extend(["--limit-cpu", str(cpu_limit)])

        if memory_limit is not None:
            cmd.extend(["--limit-memory", str(memory_limit)])

        if constraints:
            for constraint in constraints:
                cmd.extend(["--constraint", constraint])

        if labels:
            for key, value in labels.items():
                cmd.extend(["--label", f"{key}={value}"])

        cmd.append(image)

        if command_override:
            if isinstance(command_override, str):
                cmd.extend(shlex.split(command_override))
            else:
                cmd.extend(command_override)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            self._service_id = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = (
                "Failed to start Docker Swarm service.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {e.returncode}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            )
            raise RuntimeError(error_msg) from e

        # Give Swarm a brief moment to schedule the tasks.
        time.sleep(1.0)

        return f"http://localhost:{port}"

    def stop_container(self) -> None:
        """
        Remove the Swarm service (and keep the Swarm manager running).
        """
        if not self._service_name:
            return

        import subprocess

        try:
            subprocess.run(
                ["docker", "service", "rm", self._service_name],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except subprocess.CalledProcessError:
            # Service may already be gone; ignore.
            pass
        finally:
            self._service_name = None
            self._service_id = None
            self._published_port = None

    def wait_for_ready(self, base_url: str, timeout_s: float = 30.0) -> None:
        """
        Wait for at least one replica to become healthy by polling /health.

        With Swarm's load balancer, requests round-robin across replicas,
        so this only verifies that at least one replica is responding. Some
        replicas may still be starting when this returns.
        """
        import time

        import requests

        deadline = time.time() + timeout_s
        health_url = f"{base_url}/health"

        # Bypass proxy for localhost to avoid proxy issues
        proxies = {"http": None, "https": None}

        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=2.0, proxies=proxies)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            time.sleep(0.5)

        raise TimeoutError(
            f"Swarm service at {base_url} did not become ready within {timeout_s}s"
        )

    def _ensure_docker_available(self) -> None:
        import subprocess

        try:
            subprocess.run(
                ["docker", "version"],
                check=True,
                capture_output=True,
                timeout=5,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise RuntimeError(
                "Docker is not available. Please install Docker Desktop or Docker Engine."
            ) from exc

    def _ensure_swarm_initialized(self) -> None:
        import subprocess

        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.Swarm.LocalNodeState}}"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            state = result.stdout.strip().lower()
            if state == "active":
                return
        except subprocess.CalledProcessError:
            state = "unknown"

        if not self._auto_init_swarm:
            raise RuntimeError(
                f"Docker Swarm is not active (state={state}). Enable Swarm manually or pass auto_init_swarm=True."
            )

        try:
            subprocess.run(
                ["docker", "swarm", "init"],
                check=True,
                capture_output=True,
                timeout=10,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Failed to initialize Docker Swarm") from e

    def _ensure_overlay_network(self, network: str) -> None:
        import subprocess

        inspect = subprocess.run(
            ["docker", "network", "inspect", network],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspect.returncode == 0:
            return

        try:
            subprocess.run(
                [
                    "docker",
                    "network",
                    "create",
                    "--driver",
                    "overlay",
                    "--attachable",
                    network,
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create overlay network '{network}'") from e

    def _find_available_port(self) -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _generate_service_name(self, image: str) -> str:
        import time

        clean_image = image.split("/")[-1].split(":")[0]
        timestamp = int(time.time() * 1000)
        return f"{clean_image}-swarm-{timestamp}"


class KubernetesProvider(ContainerProvider):
    """
    Planned container provider for Kubernetes clusters.

    Not yet implemented: this is a placeholder for the planned Kubernetes
    backend and does not implement the abstract `ContainerProvider` methods, so
    it cannot be instantiated. Use `LocalDockerProvider`, `DockerSwarmProvider`,
    `DaytonaProvider`, or `ACASandboxProvider` instead.
    """

    pass


class RuntimeProvider(ABC):
    """
    Abstract base class for runtime providers that are not container providers.
    Providers implement this interface to support different runtime platforms:
    - UVProvider: Runs environments via `uv run`

    The provider manages a single runtime lifecycle and provides the base URL
    for connecting to it.

    Examples:

        ```python
        provider = UVProvider(project_path="/path/to/env")
        base_url = provider.start()
        print(base_url)  # http://localhost:8000
        provider.stop()
        ```
    """

    @abstractmethod
    def start(
        self,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Start the runtime.

        Args:
            port (`int`, *optional*):
                Port to expose. If `None`, the provider chooses.
            env_vars (`dict`, *optional*):
                Environment variables for the runtime.
            **kwargs:
                Additional runtime options.

        Returns:
            `str`: Base URL to connect to the runtime.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the runtime.
        """
        pass

    @abstractmethod
    def wait_for_ready(self, timeout_s: float = 30.0) -> None:
        """
        Wait for the runtime to be ready to accept requests.
        """
        pass

    def __enter__(self) -> "RuntimeProvider":
        """
        Enter the runtime provider.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Exit the runtime provider.
        """
        self.stop()
        return False
