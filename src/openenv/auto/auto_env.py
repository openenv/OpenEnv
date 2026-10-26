# SPDX-License-Identifier: BSD-3-Clause

"""
AutoEnv - Automatic Environment Selection
==========================================

AutoEnv provides a HuggingFace-style API for automatically selecting and
instantiating the correct environment client from installed packages or
HuggingFace Hub.

This module simplifies environment creation by automatically detecting the
environment type from the name and instantiating the appropriate client class.

Example:
    >>> from openenv import AutoEnv, AutoAction
    >>>
    >>> # From installed package
    >>> env = AutoEnv.from_env("coding-env")
    >>>
    >>> # From HuggingFace Hub
    >>> env = AutoEnv.from_env("openenv/coding_env")
    >>>
    >>> # With configuration
    >>> env = AutoEnv.from_env("coding", env_vars={"DEBUG": "1"})
"""

from __future__ import annotations

import importlib
import ipaddress
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, TYPE_CHECKING
from urllib.parse import urlparse

import requests
from openenv.core.utils import hf_space_subdomain, run_async_safely

from ._discovery import _is_hub_url, get_discovery


if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider
    from openenv.core.env_client import EnvClient

logger = logging.getLogger(__name__)

# Cache for repo ID → env_name mapping to avoid redundant downloads
_hub_env_name_cache: Dict[str, str] = {}

# Environment variable to skip user confirmation for remote installs
OPENENV_TRUST_REMOTE_CODE = "OPENENV_TRUST_REMOTE_CODE"


def _has_uv() -> bool:
    """Check if uv is available in the system."""
    return shutil.which("uv") is not None


def _get_pip_command() -> list[str]:
    """
    Get the appropriate pip command (uv pip or pip).

    Returns:
        List of command parts for pip installation
    """
    if _has_uv():
        return ["uv", "pip"]
    return [sys.executable, "-m", "pip"]


def _confirm_remote_install(repo_id: str) -> bool:
    """
    Ask user for confirmation before installing remote code.

    This is a security measure since we're executing code from the internet.

    Args:
        repo_id: The HuggingFace repo ID being installed

    Returns:
        True if user confirms, False otherwise
    """
    # Check environment variable for automated/CI environments
    if os.environ.get(OPENENV_TRUST_REMOTE_CODE, "").lower() in ("1", "true", "yes"):
        logger.info("Skipping confirmation (OPENENV_TRUST_REMOTE_CODE is set)")
        return True

    # Check if we're in an interactive terminal
    if not sys.stdin.isatty():
        logger.warning(
            "Cannot prompt for confirmation in non-interactive mode. "
            "Set OPENENV_TRUST_REMOTE_CODE=1 to allow remote installs."
        )
        return False

    print(f"\n{'=' * 60}")
    print("⚠️  SECURITY WARNING: Remote Code Installation")
    print(f"{'=' * 60}")
    print("You are about to install code from a remote repository:")
    print(f"  Repository: {repo_id}")
    print(f"  Source: https://huggingface.co/spaces/{repo_id}")
    print("\nThis will execute code from the internet on your machine.")
    print("Only proceed if you trust the source.")
    print(f"{'=' * 60}\n")

    try:
        response = input("Do you want to proceed? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print("\nInstallation cancelled.")
        return False


class AutoEnv:
    """
    AutoEnv automatically selects and instantiates the correct environment client
    based on environment names or HuggingFace Hub repositories.

    This class follows the HuggingFace AutoModel pattern, making it easy to work
    with different environments without needing to import specific client classes.

    The class provides factory methods that:
    1. Check if name is a HuggingFace Hub URL/repo ID
    2. If Hub: download and install the environment package
    3. If local: look up the installed openenv-* package
    4. Import and instantiate the client class

    Example:
        >>> # From installed package
        >>> env = AutoEnv.from_env("coding-env")
        >>>
        >>> # From HuggingFace Hub
        >>> env = AutoEnv.from_env("openenv/coding_env")
        >>>
        >>> # List available environments
        >>> AutoEnv.list_environments()

    Note:
        AutoEnv is not meant to be instantiated directly. Use the class method
        from_env() instead.
    """

    def __init__(self):
        """AutoEnv should not be instantiated directly. Use class methods instead."""
        raise TypeError(
            "AutoEnv is a factory class and should not be instantiated directly. "
            "Use AutoEnv.from_hub() or AutoEnv.from_env() instead."
        )

    @classmethod
    def _resolve_space_url(cls, repo_id: str) -> str:
        """
        Resolve HuggingFace Space repo ID to Space URL.

        Args:
            repo_id: HuggingFace repo ID (e.g., "wukaixingxp/coding-env-test")

        Returns:
            Space URL (e.g., "https://wukaixingxp-coding-env-test.hf.space")

        Examples:
            >>> AutoEnv._resolve_space_url("wukaixingxp/coding-env-test")
            'https://wukaixingxp-coding-env-test.hf.space'
        """
        # Clean up repo_id if it's a full URL
        if "huggingface.co" in repo_id:
            # Extract org/repo from URL
            # https://huggingface.co/wukaixingxp/coding-env-test -> wukaixingxp/coding-env-test
            parts = repo_id.split("/")
            if len(parts) >= 2:
                repo_id = f"{parts[-2]}/{parts[-1]}"

        # Convert user/space_name to user-space-name.hf.space
        return f"https://{hf_space_subdomain(repo_id)}.hf.space"

    @classmethod
    def _is_local_url(cls, url: str) -> bool:
        """
        Check if a URL points to a local server.

        Args:
            url: URL to check

        Returns:
            True if URL is localhost or 127.0.0.1, False otherwise

        Examples:
            >>> AutoEnv._is_local_url("http://localhost:8000")
            True
            >>> AutoEnv._is_local_url("http://127.0.0.1:8000")
            True
            >>> AutoEnv._is_local_url("https://example.com")
            False
        """
        try:
            hostname = urlparse(url).hostname or ""
        except Exception:
            return False
        if not hostname:
            return False
        if hostname in ("localhost", "0.0.0.0"):
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    @classmethod
    def _check_server_availability(cls, base_url: str, timeout: float = 2.0) -> bool:
        """
        Check if a server at the given URL is running and accessible.

        Args:
            base_url: Server base URL to check
            timeout: Request timeout in seconds

        Returns:
            True if server is accessible, False otherwise

        Examples:
            >>> AutoEnv._check_server_availability("http://localhost:8000")
            True  # if server is running
        """
        try:
            # Bypass proxy for localhost to avoid proxy issues
            proxies = None
            if cls._is_local_url(base_url):
                proxies = {"http": None, "https": None}

            # Try to access the health endpoint
            response = requests.get(
                f"{base_url}/health", timeout=timeout, proxies=proxies
            )
            if response.status_code == 200:
                return True

            # If health endpoint doesn't exist, try root endpoint
            response = requests.get(base_url, timeout=timeout, proxies=proxies)
            return response.status_code == 200
        except (requests.RequestException, Exception) as e:
            logger.debug(f"Server {base_url} not accessible: {e}")
            return False

    @classmethod
    def _check_space_availability(cls, space_url: str, timeout: float = 5.0) -> bool:
        """
        Check if HuggingFace Space is running and accessible.

        Args:
            space_url: Space URL to check
            timeout: Request timeout in seconds

        Returns:
            True if Space is accessible, False otherwise

        Examples:
            >>> AutoEnv._check_space_availability("https://wukaixingxp-coding-env-test.hf.space")
            True
        """
        try:
            # Try to access the health endpoint
            response = requests.get(f"{space_url}/health", timeout=timeout)
            if response.status_code == 200:
                return True

            # If health endpoint doesn't exist, try root endpoint
            response = requests.get(space_url, timeout=timeout)
            return response.status_code == 200
        except (requests.RequestException, Exception) as e:
            logger.debug(f"Space {space_url} not accessible: {e}")
            return False

    @classmethod
    def _get_hub_git_url(cls, repo_id: str) -> str:
        """
        Get the git URL for a HuggingFace Space.

        Args:
            repo_id: HuggingFace repo ID (e.g., "wukaixingxp/coding-env-test")

        Returns:
            Git URL for pip installation (e.g., "git+https://huggingface.co/spaces/wukaixingxp/coding-env-test")
        """
        # Clean up repo_id if it's a full URL
        if "huggingface.co" in repo_id:
            parts = repo_id.split("/")
            if len(parts) >= 2:
                repo_id = f"{parts[-2]}/{parts[-1]}"

        return f"git+https://huggingface.co/spaces/{repo_id}"

    @classmethod
    def _install_from_hub(cls, repo_id: str, trust_remote_code: bool = False) -> str:
        """
        Install environment package directly from HuggingFace Hub using git+.

        This is the preferred method as it avoids downloading the entire repo
        and uses pip/uv's native git support.

        Args:
            repo_id: HuggingFace repo ID (e.g., "wukaixingxp/coding-env-test")
            trust_remote_code: If True, skip user confirmation

        Returns:
            Package name that was installed

        Raises:
            ValueError: If installation fails or user declines
        """
        # Security check - confirm with user before installing remote code
        if not trust_remote_code and not _confirm_remote_install(repo_id):
            raise ValueError(
                "Installation cancelled by user.\n"
                "To allow remote installs without prompting, set OPENENV_TRUST_REMOTE_CODE=1"
            )

        git_url = cls._get_hub_git_url(repo_id)
        pip_cmd = _get_pip_command()
        pip_name = "uv pip" if pip_cmd[0] == "uv" else "pip"

        logger.info(f"Installing from HuggingFace Space using {pip_name}: {repo_id}")
        logger.info(f"Command: {' '.join(pip_cmd)} install {git_url}")

        try:
            result = subprocess.run(
                [*pip_cmd, "install", git_url],
                check=True,
                capture_output=True,
                text=True,
            )

            # Try to extract package name from pip output
            # Look for "Successfully installed <package-name>-<version>"
            for line in result.stdout.split("\n"):
                if "Successfully installed" in line:
                    # Parse package name from the line
                    parts = line.replace("Successfully installed", "").strip().split()
                    for part in parts:
                        if part.startswith("openenv-"):
                            # Remove version suffix (e.g., "openenv-coding_env-0.1.0" -> "openenv-coding_env")
                            # Check if last segment looks like a version number
                            last_segment = part.rsplit("-", 1)[-1]
                            if last_segment.replace(".", "").isdigit():
                                package_name = "-".join(part.rsplit("-", 1)[:-1])
                            else:
                                package_name = part
                            logger.info(f"Successfully installed: {package_name}")
                            return package_name

            # Fallback: try to determine package name from repo_id
            # Convention: repo name like "coding-env-test" -> package "openenv-coding_env"
            env_name = repo_id.split("/")[-1]  # Get repo name from "user/repo"
            env_name = env_name.replace("-", "_")
            if not env_name.endswith("_env"):
                env_name = f"{env_name}_env"
            package_name = f"openenv-{env_name}"

            logger.info(f"Installed (inferred package name): {package_name}")
            return package_name

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or str(e)
            raise ValueError(
                f"Failed to install environment from HuggingFace Space: {repo_id}\n"
                f"Command: {' '.join(pip_cmd)} install {git_url}\n"
                f"Error: {error_msg}\n"
                f"Make sure the repository exists and contains a valid Python package."
            ) from e

    @classmethod
    def _is_package_installed(cls, package_name: str) -> bool:
        """
        Check if a package is already installed.

        Args:
            package_name: Package name (e.g., "openenv-coding_env")

        Returns:
            True if installed, False otherwise
        """
        try:
            import importlib.metadata

            importlib.metadata.distribution(package_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    @classmethod
    def _ensure_package_from_hub(
        cls, name: str, trust_remote_code: bool = False
    ) -> str:
        """
        Ensure package from HuggingFace Hub is installed.

        Uses git+ URLs for direct installation without downloading the entire repo.
        Prompts user for confirmation before installing remote code.

        Args:
            name: HuggingFace repo ID (e.g., "wukaixingxp/coding-env-test")
            trust_remote_code: If True, skip user confirmation

        Returns:
            Environment name (e.g., "coding_env")
        """
        global _hub_env_name_cache

        # Check if we already resolved this repo ID
        if name in _hub_env_name_cache:
            env_name = _hub_env_name_cache[name]
            logger.debug(f"Using cached env name for {name}: {env_name}")
            return env_name

        # Try to infer expected package name from repo ID
        # Convention: repo "user/coding-env" -> package "openenv-coding_env"
        repo_name = name.split("/")[-1] if "/" in name else name
        expected_env_name = repo_name.replace("-", "_")
        if not expected_env_name.endswith("_env"):
            expected_env_name = f"{expected_env_name}_env"
        expected_package_name = f"openenv-{expected_env_name}"

        # Check if already installed
        if cls._is_package_installed(expected_package_name):
            logger.info(f"Package already installed: {expected_package_name}")
            # Clear and refresh discovery cache to make sure it's detected
            get_discovery().clear_cache()
            get_discovery().discover(use_cache=False)
            # Cache the result
            _hub_env_name_cache[name] = expected_env_name
            return expected_env_name

        # Not installed, install using git+ URL
        logger.info(f"Package not found locally, installing from Hub: {name}")

        # Track existing packages before installation
        get_discovery().clear_cache()
        existing_envs = set(get_discovery().discover(use_cache=False).keys())

        # Install the package
        cls._install_from_hub(name, trust_remote_code=trust_remote_code)

        # Clear discovery cache to pick up the newly installed package
        try:
            importlib.invalidate_caches()
        except Exception:
            pass
        get_discovery().clear_cache()
        discovered_envs = get_discovery().discover(use_cache=False)

        # Find the newly installed environment by comparing before/after
        new_envs = set(discovered_envs.keys()) - existing_envs

        if new_envs:
            # Use the first newly discovered environment
            env_name = next(iter(new_envs))
            logger.info(f"Found newly installed environment: '{env_name}'")
        else:
            # Fallback: try to find by matching module patterns
            # Look for any env that might match the repo name pattern
            repo_name = name.split("/")[-1] if "/" in name else name
            repo_base = (
                repo_name.replace("-", "_").replace("_env", "").replace("_test", "")
            )

            env_name = None
            for env_key, env_info in discovered_envs.items():
                # Check if env_key is a prefix/substring match
                if env_key in repo_base or repo_base.startswith(env_key):
                    env_name = env_key
                    logger.info(
                        f"Found matching environment '{env_name}' for repo '{name}'"
                    )
                    break

            if env_name is None:
                # Last resort: use inferred name from repo
                env_name = repo_name.replace("-", "_")
                if not env_name.endswith("_env"):
                    env_name = f"{env_name}_env"
                # Strip to get env_key
                env_name = env_name.replace("_env", "")
                logger.warning(
                    f"Could not find newly installed environment for repo '{name}', "
                    f"using inferred name: {env_name}"
                )

        # Cache the result to avoid redundant installs
        _hub_env_name_cache[name] = env_name

        return env_name

    @classmethod
    def from_env(
        cls,
        name: str,
        base_url: Optional[str] = None,
        docker_image: Optional[str] = None,
        container_provider: Optional[ContainerProvider] = None,
        wait_timeout: float = 30.0,
        env_vars: Optional[Dict[str, str]] = None,
        trust_remote_code: bool = False,
        skip_install: bool = False,
        **kwargs: Any,
    ) -> "EnvClient":
        """
        Create an environment client from a name or HuggingFace Hub repository.

        This method automatically:
        1. Checks if the name is a HuggingFace Hub URL/repo ID
        2. If Hub: installs the environment package using git+ URL
        3. If local: looks up the installed openenv-* package
        4. Imports the client class and instantiates it

        Args:
            name: Environment name or HuggingFace Hub repo ID
                  Examples:
                  - "coding" / "coding-env" / "coding_env"
                  - "openenv/coding_env" (Hub repo ID)
                  - "https://huggingface.co/openenv/coding_env" (Hub URL)
            base_url: Optional base URL for HTTP connection
            docker_image: Optional Docker image name (overrides default)
            container_provider: Optional container provider
            wait_timeout: Timeout for container startup (seconds)
            env_vars: Optional environment variables for the container
            trust_remote_code: If True, skip user confirmation when installing
                from HuggingFace Hub. Can also be set via OPENENV_TRUST_REMOTE_CODE
                environment variable.
            skip_install: If True, skip package installation and return a
                GenericEnvClient for remote environments. Useful when you only
                want to connect to a running server without installing any
                remote code. When True:
                - If base_url is provided: connects directly using GenericEnvClient
                - If HF Space is running: connects to Space using GenericEnvClient
                - If HF Space is not running: uses Docker from HF registry
            **kwargs: Additional arguments passed to the client class

        Returns:
            Instance of the environment client class

        Raises:
            ValueError: If environment not found or cannot be loaded
            ImportError: If environment package is not installed

        Examples:
            >>> # From installed package
            >>> env = AutoEnv.from_env("coding-env")
            >>>
            >>> # From HuggingFace Hub
            >>> env = AutoEnv.from_env("openenv/coding_env")
            >>>
            >>> # With custom Docker image
            >>> env = AutoEnv.from_env("coding", docker_image="my-coding-env:v2")
            >>>
            >>> # With environment variables
            >>> env = AutoEnv.from_env(
            ...     "dipg",
            ...     env_vars={"DIPG_DATASET_PATH": "/data/dipg"}
            ... )
            >>>
            >>> # Skip package installation, use GenericEnvClient
            >>> env = AutoEnv.from_env(
            ...     "user/my-env",
            ...     skip_install=True
            ... )
        """
        from openenv.core import GenericEnvClient

        # Handle skip_install mode - return GenericEnvClient without package installation
        if skip_install:
            # If base_url is provided, connect directly
            if base_url:
                if cls._check_server_availability(base_url):
                    logger.info(
                        f"Using GenericEnvClient for {base_url} (skip_install=True)"
                    )
                    return GenericEnvClient(base_url=base_url, **kwargs)
                else:
                    raise ConnectionError(
                        f"Server not available at {base_url}. "
                        f"Please ensure the server is running."
                    )

            # If it's a Hub URL, try to connect to Space or use Docker
            if _is_hub_url(name):
                space_url = cls._resolve_space_url(name)
                logger.info(f"Checking if HuggingFace Space is accessible: {space_url}")

                if cls._check_space_availability(space_url):
                    logger.info(
                        f"Using GenericEnvClient for Space {space_url} (skip_install=True)"
                    )
                    return GenericEnvClient(base_url=space_url, **kwargs)
                else:
                    # Space not running, use Docker from HF registry
                    logger.info(
                        f"Space not running at {space_url}, "
                        f"using GenericEnvClient with HF Docker registry"
                    )
                    return run_async_safely(
                        GenericEnvClient.from_env(
                            name,
                            use_docker=True,
                            provider=container_provider,
                            env_vars=env_vars or {},
                            **kwargs,
                        )
                    )

            # For local environments with skip_install, we need docker_image
            if docker_image:
                logger.info(
                    f"Using GenericEnvClient with Docker image {docker_image} "
                    f"(skip_install=True)"
                )
                return run_async_safely(
                    GenericEnvClient.from_docker_image(
                        image=docker_image,
                        provider=container_provider,
                        wait_timeout=wait_timeout,
                        env_vars=env_vars or {},
                        **kwargs,
                    )
                )
            else:
                raise ValueError(
                    f"Cannot use skip_install=True for local environment '{name}' "
                    f"without providing base_url or docker_image. "
                    f"For local environments, either:\n"
                    f"  1. Provide base_url to connect to a running server\n"
                    f"  2. Provide docker_image to start a container\n"
                    f"  3. Set skip_install=False to use the installed package"
                )

        # Check if it's a HuggingFace Hub URL or repo ID
        if _is_hub_url(name):
            # Try to connect to Space directly first
            space_url = cls._resolve_space_url(name)
            logger.info(f"Checking if HuggingFace Space is accessible: {space_url}")

            space_is_available = cls._check_space_availability(space_url)

            if space_is_available and base_url is None:
                # Space is accessible! We'll connect directly without Docker
                logger.info(f"Space is accessible at: {space_url}")
                logger.info("Installing package for client code (no Docker needed)...")

                # Ensure package is installed (uses git+ URL)
                env_name = cls._ensure_package_from_hub(
                    name, trust_remote_code=trust_remote_code
                )

                # Set base_url to connect to remote Space
                base_url = space_url
                logger.info("Will connect to remote Space (no local Docker)")
            else:
                # Space not accessible or user provided explicit base_url
                if not space_is_available:
                    logger.info(f"Space not accessible at {space_url}")
                    logger.info("Falling back to local Docker mode...")

                # Ensure package is installed (uses git+ URL)
                env_name = cls._ensure_package_from_hub(
                    name, trust_remote_code=trust_remote_code
                )
        else:
            env_name = name

        # Get environment info from discovery
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(env_name)

        if not env_info:
            # Environment not found - provide helpful error message
            available_envs = discovery.discover()

            if not available_envs:
                raise ValueError(
                    "No OpenEnv environments found.\n"
                    "Install an environment with: pip install openenv-<env-name>\n"
                    "Or specify a HuggingFace Hub repository: AutoEnv.from_env('openenv/echo_env')"
                )

            # Try to suggest similar environment names
            from difflib import get_close_matches

            env_keys = list(available_envs.keys())
            suggestions = get_close_matches(env_name, env_keys, n=3, cutoff=0.6)

            error_msg = f"Unknown environment '{env_name}'.\n"
            if suggestions:
                error_msg += f"Did you mean: {', '.join(suggestions)}?\n"
            error_msg += f"Available environments: {', '.join(sorted(env_keys))}"

            raise ValueError(error_msg)

        # Get the client class
        try:
            client_class = env_info.get_client_class()
        except ImportError as e:
            raise ImportError(
                f"Failed to import environment client for '{env_name}'.\n"
                f"Package '{env_info.package_name}' appears to be installed but the module cannot be imported.\n"
                f"Try reinstalling: pip install --force-reinstall {env_info.package_name}\n"
                f"Original error: {e}"
            ) from e

        # Determine Docker image to use
        if docker_image is None:
            docker_image = env_info.default_image

        # Create client instance
        try:
            if base_url:
                # Check if the server at base_url is available
                is_local = cls._is_local_url(base_url)
                server_available = cls._check_server_availability(base_url)

                if server_available:
                    # Server is running, connect directly
                    logger.info(
                        f"✅ Server available at {base_url}, connecting directly"
                    )
                    return client_class(base_url=base_url, provider=None, **kwargs)
                elif is_local:
                    # Local server not running, auto-start Docker container
                    logger.info(f"❌ Server not available at {base_url}")
                    logger.info(f"🐳 Auto-starting Docker container: {docker_image}")
                    return run_async_safely(
                        client_class.from_docker_image(
                            image=docker_image,
                            provider=container_provider,
                            wait_timeout=wait_timeout,
                            env_vars=env_vars or {},
                            **kwargs,
                        )
                    )
                else:
                    # Remote server not available, cannot auto-start
                    raise ConnectionError(
                        f"Remote server not available at {base_url}. "
                        f"Please ensure the server is running."
                    )
            else:
                # No base_url provided, start new Docker container
                return run_async_safely(
                    client_class.from_docker_image(
                        image=docker_image,
                        provider=container_provider,
                        wait_timeout=wait_timeout,
                        env_vars=env_vars or {},
                        **kwargs,
                    )
                )
        except Exception as e:
            raise ValueError(
                f"Failed to create environment client for '{env_name}'.\n"
                f"Client class: {client_class.__name__}\n"
                f"Docker image: {docker_image}\n"
                f"Error: {e}"
            ) from e

    @classmethod
    def from_hub(
        cls,
        name: str,
        base_url: Optional[str] = None,
        docker_image: Optional[str] = None,
        container_provider: Optional["ContainerProvider"] = None,
        wait_timeout: float = 30.0,
        env_vars: Optional[Dict[str, str]] = None,
        trust_remote_code: bool = False,
        skip_install: bool = False,
        **kwargs: Any,
    ) -> "EnvClient":
        """
        Create an environment client from a name or HuggingFace Hub repository.

        This is an alias for from_env() for backward compatibility.

        Args:
            name: Environment name or HuggingFace Hub repo ID
            base_url: Optional base URL for HTTP connection
            docker_image: Optional Docker image name (overrides default)
            container_provider: Optional container provider
            wait_timeout: Timeout for container startup (seconds)
            env_vars: Optional environment variables for the container
            trust_remote_code: If True, skip user confirmation when installing
                from HuggingFace Hub
            skip_install: If True, skip package installation and return a
                GenericEnvClient for remote environments
            **kwargs: Additional arguments passed to the client class

        Returns:
            Instance of the environment client class

        Examples:
            >>> env = AutoEnv.from_hub("coding-env")
            >>> env = AutoEnv.from_hub("openenv/coding_env")
        """
        return cls.from_env(
            name=name,
            base_url=base_url,
            docker_image=docker_image,
            container_provider=container_provider,
            wait_timeout=wait_timeout,
            env_vars=env_vars,
            trust_remote_code=trust_remote_code,
            skip_install=skip_install,
            **kwargs,
        )

    @classmethod
    def get_env_class(cls, name: str):
        """
        Get the environment client class without instantiating it.

        Args:
            name: Environment name

        Returns:
            The environment client class

        Raises:
            ValueError: If environment not found

        Examples:
            >>> CodingEnv = AutoEnv.get_env_class("coding")
            >>> # Now you can instantiate it yourself
            >>> env = CodingEnv(base_url="http://localhost:8000")
        """
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(name)

        if not env_info:
            raise ValueError(f"Unknown environment: {name}")

        return env_info.get_client_class()

    @classmethod
    def get_env_info(cls, name: str) -> Dict[str, Any]:
        """
        Get detailed information about an environment.

        Args:
            name: Environment name

        Returns:
            Dictionary with environment metadata

        Raises:
            ValueError: If environment not found

        Examples:
            >>> info = AutoEnv.get_env_info("coding")
            >>> print(info['description'])
            'Coding environment for OpenEnv'
            >>> print(info['default_image'])
            'coding-env:latest'
        """
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(name)

        if not env_info:
            raise ValueError(f"Unknown environment: {name}")

        return {
            "env_key": env_info.env_key,
            "name": env_info.name,
            "package": env_info.package_name,
            "version": env_info.version,
            "description": env_info.description,
            "env_class": env_info.client_class_name,
            "action_class": env_info.action_class_name,
            "observation_class": env_info.observation_class_name,
            "module": env_info.client_module_path,
            "default_image": env_info.default_image,
            "spec_version": env_info.spec_version,
        }

    @classmethod
    def list_environments(cls) -> None:
        """
        Print a formatted list of all available environments.

        This discovers all installed openenv-* packages and displays
        their metadata in a user-friendly format.

        Examples:
            >>> AutoEnv.list_environments()
            Available OpenEnv Environments:
            ----------------------------------------------------------------------
              echo           : Echo Environment (v0.1.0)
                               Package: openenv-echo-env
              coding         : Coding Environment (v0.1.0)
                               Package: openenv-coding_env
            ----------------------------------------------------------------------
            Total: 2 environments
        """
        discovery = get_discovery()
        discovery.list_environments()
