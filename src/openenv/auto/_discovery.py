# SPDX-License-Identifier: BSD-3-Clause

"""
Environment Auto-Discovery System
==================================

This module provides automatic discovery of OpenEnv environments by:
1. Discovering installed openenv-* packages using importlib.metadata
2. Loading manifests (openenv.yaml) from package resources
3. Caching results for performance
4. Supporting HuggingFace Hub downloads

This enables AutoEnv to work without coupling to src/envs/ directory.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import json
import logging
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Type

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """
    Rich information about a discovered environment.

    Attributes:
        env_key: Environment key (e.g., "echo", "coding")
        name: Full environment name (e.g., "echo_env")
        package_name: Package name (e.g., "openenv-echo_env")
        version: Version string
        description: Human-readable description
        client_module_path: Full module path to client (e.g., "echo_env.client")
        client_class_name: Client class name (e.g., "EchoEnv")
        action_class_name: Action class name (e.g., "EchoAction")
        observation_class_name: Observation class name (e.g., "EchoObservation")
        default_image: Default Docker image name (e.g., "echo-env:latest")
        spec_version: OpenEnv spec version (from openenv.yaml)
        manifest: Original manifest data
    """

    env_key: str
    name: str
    package_name: str
    version: str
    description: str
    client_module_path: str
    client_class_name: str
    action_class_name: str
    observation_class_name: str
    default_image: str
    spec_version: int | None = None
    manifest: dict[str, Any] | None = None

    def get_client_class(self) -> Type[Any]:
        """
        Dynamically import and return the client class.

        Returns:
            Client class (e.g., EchoEnv)

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.client_module_path)
            return getattr(module, self.client_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.client_class_name} from {self.client_module_path}: {e}\n"
                f"Make sure the package '{self.package_name}' is installed: "
                f"pip install {self.package_name}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class {self.client_class_name} not found in {self.client_module_path}: {e}"
            ) from e

    def get_action_class(self) -> Type[Any]:
        """
        Dynamically import and return the action class.

        Returns:
            Action class (e.g., EchoAction)

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.client_module_path)
            return getattr(module, self.action_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.action_class_name} from {self.client_module_path}: {e}\n"
                f"Make sure the package '{self.package_name}' is installed: "
                f"pip install {self.package_name}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class {self.action_class_name} not found in {self.client_module_path}: {e}"
            ) from e

    def get_observation_class(self) -> Type[Any]:
        """
        Dynamically import and return the observation class.

        Returns:
            Observation class (e.g., EchoObservation)

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.client_module_path)
            return getattr(module, self.observation_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.observation_class_name} from {self.client_module_path}: {e}\n"
                f"Make sure the package '{self.package_name}' is installed: "
                f"pip install {self.package_name}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class {self.observation_class_name} not found in {self.client_module_path}: {e}"
            ) from e


def _normalize_env_name(name: str) -> str:
    """
    Normalize environment name to standard format.

    Args:
        name: Input name (e.g., "echo", "echo-env", "echo_env")

    Returns:
        Normalized name (e.g., "echo_env")

    Examples:
        >>> _normalize_env_name("echo")
        'echo_env'
        >>> _normalize_env_name("echo-env")
        'echo_env'
        >>> _normalize_env_name("echo_env")
        'echo_env'
    """
    # Remove common suffixes
    name = re.sub(r"[-_]env$", "", name)
    # Convert hyphens to underscores
    name = name.replace("-", "_")
    # Add _env suffix if not present
    if not name.endswith("_env"):
        name = f"{name}_env"
    return name


def _is_hub_url(name: str) -> bool:
    """
    Check if name is a HuggingFace Hub URL or repo ID.

    Args:
        name: Input name

    Returns:
        True if it looks like a Hub URL

    Examples:
        >>> _is_hub_url("openenv/echo_env")
        True
        >>> _is_hub_url("https://huggingface.co/openenv/echo_env")
        True
        >>> _is_hub_url("echo")
        False
    """
    # Contains org/repo pattern or huggingface.co domain
    return "/" in name or "huggingface.co" in name


def _infer_class_name(env_name: str, class_type: str) -> str:
    """
    Infer class name from environment name using simple conventions.

    Args:
        env_name: Environment name (e.g., "echo_env")
        class_type: Type of class ("client", "action", "observation")

    Returns:
        Inferred class name

    Examples:
        >>> _infer_class_name("echo_env", "client")
        'EchoEnv'
        >>> _infer_class_name("echo_env", "action")
        'EchoAction'
    """
    # Remove _env suffix for base name
    base_name = env_name.replace("_env", "")

    # Convert to PascalCase
    pascal_name = "".join(word.capitalize() for word in base_name.split("_"))

    # Add suffix based on type
    if class_type == "client":
        return f"{pascal_name}Env"
    elif class_type == "action":
        return f"{pascal_name}Action"
    elif class_type == "observation":
        return f"{pascal_name}Observation"
    else:
        raise ValueError(f"Unknown class type: {class_type}")


def _load_manifest_from_package(
    package_name: str, module_name: str
) -> dict[str, Any] | None:
    """
    Load openenv.yaml manifest from an installed package.

    Args:
        package_name: Package name (e.g., "openenv-echo_env")
        module_name: Module name (e.g., "echo_env")

    Returns:
        Parsed manifest dictionary, or None if not found

    """
    try:
        # Try to read openenv.yaml from package
        if hasattr(importlib.resources, "files"):
            # Python 3.9+
            package_files = importlib.resources.files(module_name)
            if (package_files / "openenv.yaml").is_file():
                manifest_text = (package_files / "openenv.yaml").read_text()
                return yaml.safe_load(manifest_text)
        else:
            # Python 3.7-3.8 fallback
            with importlib.resources.open_text(module_name, "openenv.yaml") as f:
                return yaml.safe_load(f)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        logger.debug(f"No openenv.yaml found in {module_name}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load openenv.yaml from {module_name}: {e}")
        return None


def _create_env_info_from_package(
    package_name: str, module_name: str, version: str
) -> EnvironmentInfo | None:
    """
    Create EnvironmentInfo from an installed package.

    Args:
        package_name: Package name (e.g., "openenv-echo_env")
        module_name: Module name (e.g., "echo_env")
        version: Package version

    Returns:
        EnvironmentInfo instance, or None if invalid
    """
    # Load manifest
    manifest = _load_manifest_from_package(package_name, module_name)

    # Get environment name
    if manifest and "name" in manifest:
        env_name = manifest["name"]
    else:
        # Infer from module name
        env_name = module_name

    # Normalize to ensure _env suffix
    if not env_name.endswith("_env"):
        env_name = f"{env_name}_env"

    # Determine env_key (e.g., "echo_env" → "echo")
    env_key = env_name.replace("_env", "") if env_name.endswith("_env") else env_name

    # Get description
    description = (
        manifest.get("description", f"{env_name} environment")
        if manifest
        else f"{env_name} environment"
    )

    # Get spec version
    spec_version = manifest.get("spec_version") if manifest else None

    # Determine class names
    # Check if manifest has custom class names (custom format)
    if manifest and "action" in manifest and "observation" in manifest:
        # Custom format (like coding_env)
        client_class_name = _infer_class_name(env_name, "client")
        action_class_name = manifest.get(
            "action", _infer_class_name(env_name, "action")
        )
        observation_class_name = manifest.get(
            "observation", _infer_class_name(env_name, "observation")
        )
    else:
        # Use conventions
        client_class_name = _infer_class_name(env_name, "client")
        action_class_name = _infer_class_name(env_name, "action")
        observation_class_name = _infer_class_name(env_name, "observation")

    # Module path is just module_name.client
    client_module_path = f"{module_name}.client"

    # Determine default Docker image name
    image_name = env_name.replace("_", "-")
    default_image = f"{image_name}:latest"

    return EnvironmentInfo(
        env_key=env_key,
        name=env_name,
        package_name=package_name,
        version=version,
        description=description,
        client_module_path=client_module_path,
        client_class_name=client_class_name,
        action_class_name=action_class_name,
        observation_class_name=observation_class_name,
        default_image=default_image,
        spec_version=spec_version,
        manifest=manifest,
    )


def _default_cache_file() -> Path:
    """
    Return the per-user discovery cache file path.

    Uses a per-user cache directory (``$XDG_CACHE_HOME`` or ``~/.cache``) rather than a
    shared, world-writable temporary directory. A fixed path under the shared temp dir
    lets another local user pre-create the cache file and redirect discovery to
    attacker-controlled import paths (`import_module` on a cached `client_module_path`).
    Per the XDG Base Directory specification, relative `XDG_CACHE_HOME` values are
    ignored so an untrusted working tree cannot supply a victim-owned cache file.
    The fallback home must itself be absolute; otherwise discovery fails closed.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    if base and Path(base).is_absolute():
        root = Path(base)
    else:
        home = Path.home()
        if not home.is_absolute():
            raise RuntimeError(
                "discovery cache requires an absolute home directory when "
                "XDG_CACHE_HOME is unset or relative"
            )
        root = home / ".cache"
    return root / "openenv" / "discovery_cache.json"


def _is_trusted_stat(info: os.stat_result) -> bool:
    """
    Return whether the file *info* describes is safe to load.

    On POSIX the file must be owned by the current user and not writable by group or
    others, so a cache file planted by another user is ignored rather than trusted.

    Args:
        info (`os.stat_result`):
            Metadata for the file being considered.

    Returns:
        `bool`: `True` if the file is owned by the current user and not
        group/other-writable (always `True` on non-POSIX platforms).
    """
    if os.name != "posix":
        return True
    return info.st_uid == os.getuid() and not (
        info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    )


def _is_trusted_cache_file(path: Path) -> bool:
    """
    Return whether *path* is safe to load.

    Prefer `_open_trusted_cache`, which checks the descriptor it hands back. This
    variant resolves the path a second time, so on its own it cannot promise that
    the file inspected is the file later read.

    Args:
        path (`Path`):
            The cache file to check.

    Returns:
        `bool`: `True` if the file is owned by the current user and not
        group/other-writable (always `True` on non-POSIX platforms).
    """
    if os.name != "posix":
        return True
    try:
        return _is_trusted_stat(path.stat())
    except OSError:
        return False


def _open_trusted_cache(path: Path) -> int | None:
    """
    Open *path* for reading and return the descriptor only if it is trustworthy.

    Checking a path and then opening it are two separate resolutions, and an
    attacker who can write in the cache directory can swap the file for a symlink
    in between, so the file that was vetted is not the file that gets read.
    ``O_NOFOLLOW`` refuses a symlink outright and ``fstat`` inspects the descriptor
    itself, which makes the ownership check and the read the same object.

    Args:
        path (`Path`):
            The cache file to open.

    Returns:
        `int` or `None`: an open read-only descriptor the caller must close, or
        `None` if the file is missing, is a symlink, is not a regular file, or
        is not owned by the current user.
    """
    # O_NONBLOCK matters before the descriptor can be inspected: opening a FIFO
    # read-only otherwise waits for a writer that an attacker simply never
    # provides, so discovery hangs before any check runs. It has no effect on a
    # regular file, which is the only thing accepted here anyway.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        # Missing, unreadable, or a symlink (ELOOP under O_NOFOLLOW).
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not _is_trusted_stat(info):
            os.close(fd)
            return None
    except OSError:
        os.close(fd)
        return None
    return fd


class EnvironmentDiscovery:
    """
    Auto-discovery system for OpenEnv environments using installed packages.

    This class discovers installed openenv-* packages and loads their metadata.
    """

    def __init__(self):
        """Initialize discovery system."""
        self._cache: dict[str, EnvironmentInfo] | None = None
        self._cache_file = _default_cache_file()

    def _discover_installed_packages(self) -> dict[str, EnvironmentInfo]:
        """
        Discover all installed openenv-* packages.

        Returns:
            Dictionary mapping env_key to EnvironmentInfo
        """
        environments: dict[str, EnvironmentInfo] = {}

        # Invalidate import caches to ensure we pick up newly installed packages
        importlib.invalidate_caches()

        # Get all installed packages
        try:
            distributions = importlib.metadata.distributions()
        except Exception as e:
            logger.warning(f"Failed to get installed packages: {e}")
            return environments

        # Filter for openenv-* environment packages; ignore the legacy core wheel.
        for dist in distributions:
            package_name = dist.metadata["Name"]

            if not package_name.startswith("openenv-"):
                continue

            if package_name == "openenv-core":
                continue

            # Get module name (e.g., "openenv-echo_env" → "echo_env")
            module_name = package_name.replace("openenv-", "").replace("-", "_")

            # Get version
            version = dist.version

            try:
                # Create environment info
                env_info = _create_env_info_from_package(
                    package_name, module_name, version
                )

                if env_info:
                    environments[env_info.env_key] = env_info
                    logger.debug(
                        f"Discovered environment: {env_info.env_key} ({package_name})"
                    )

            except Exception as e:
                logger.warning(f"Failed to load environment from {package_name}: {e}")
                continue

        return environments

    def _load_cache(self) -> dict[str, EnvironmentInfo] | None:
        """
        Load cached discovery results.

        Returns:
            Dictionary of env_key -> EnvironmentInfo, or None if cache invalid
        """
        # Only trust a cache file owned by the current user. This prevents another
        # local user from planting a file that would redirect discovery (and the
        # subsequent import_module) to attacker-controlled modules/classes. The
        # descriptor is what gets vetted and then read, so the file cannot be
        # swapped for a symlink after the check.
        fd = _open_trusted_cache(self._cache_file)
        if fd is None:
            if self._cache_file.exists():
                logger.warning(
                    f"Ignoring discovery cache {self._cache_file}: not owned by "
                    "the current user, writable by group/others, or a symlink."
                )
            return None

        try:
            with os.fdopen(fd, "r") as f:
                cache_data = json.load(f)

            # Reconstruct EnvironmentInfo objects
            cache: dict[str, EnvironmentInfo] = {}
            for env_key, env_data in cache_data.items():
                cache[env_key] = EnvironmentInfo(**env_data)

            return cache
        except Exception as e:
            logger.warning(f"Failed to load discovery cache: {e}")
            return None

    def _save_cache(self, environments: dict[str, EnvironmentInfo]) -> None:
        """
        Save discovery results to cache.

        Args:
            environments: Dictionary of env_key -> EnvironmentInfo
        """
        try:
            cache_data = {}
            for env_key, env_info in environments.items():
                cache_data[env_key] = asdict(env_info)

            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Write a new file and move it into place rather than writing
            # through whatever already sits at the destination. A mode passed to
            # `os.open` only applies when it creates the file, so truncating an
            # existing group/world-writable cache would keep that mode and
            # publish what was just written; and `os.replace` is atomic, so a
            # reader never observes a half-written cache when serialization
            # fails part way. `O_EXCL` requires the temporary name to be new,
            # and 0600 is applied at creation so the umask cannot widen it.
            tmp_path = self._cache_file.with_name(
                f"{self._cache_file.name}.{os.getpid()}.tmp"
            )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(tmp_path, flags, 0o600)
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(cache_data, f, indent=2)
                os.replace(tmp_path, self._cache_file)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            logger.warning(f"Failed to save discovery cache: {e}")

    def discover(self, use_cache: bool = True) -> dict[str, EnvironmentInfo]:
        """
        Discover all installed OpenEnv environments.

        Args:
            use_cache: If True, try to load from cache first

        Returns:
            Dictionary mapping env_key to EnvironmentInfo

        Examples:
            >>> discovery = EnvironmentDiscovery()
            >>> envs = discovery.discover()
            >>> print(envs.keys())
            dict_keys(['echo', 'coding', ...])
        """
        # Try to load from memory cache first
        if use_cache and self._cache is not None:
            return self._cache

        # Try to load from file cache
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                self._cache = cached
                return self._cache

        # Discover from installed packages
        environments = self._discover_installed_packages()

        # Save to cache
        self._save_cache(environments)
        self._cache = environments

        return environments

    def get_environment(self, env_key: str) -> EnvironmentInfo | None:
        """
        Get information about a specific environment.

        Args:
            env_key: Environment key (e.g., "echo", "coding")

        Returns:
            EnvironmentInfo if found, None otherwise

        Examples:
            >>> discovery = EnvironmentDiscovery()
            >>> env = discovery.get_environment("echo")
            >>> print(env.client_class_name)
            'EchoEnv'
        """
        environments = self.discover()
        return environments.get(env_key)

    def get_environment_by_name(self, name: str) -> EnvironmentInfo | None:
        """
        Get environment info by flexible name matching.

        Args:
            name: Environment name (e.g., "echo", "echo-env", "echo_env")

        Returns:
            EnvironmentInfo if found, None otherwise
        """
        # Normalize name to env_key
        normalized = _normalize_env_name(name)
        env_key = normalized.replace("_env", "")

        return self.get_environment(env_key)

    def list_environments(self) -> None:
        """
        Print a formatted list of all discovered environments.

        Examples:
            >>> discovery = EnvironmentDiscovery()
            >>> discovery.list_environments()
            Available OpenEnv Environments:
            ----------------------------------------------------------------------
              echo           : Echo Environment (v0.1.0) - openenv-echo_env
              coding         : Coding Environment (v0.1.0) - openenv-coding_env
              ...
        """
        environments = self.discover()

        print("Available OpenEnv Environments:")
        print("-" * 70)

        if not environments:
            print("  No OpenEnv environments found.")
            print("  Install environments with: pip install openenv-<env-name>")
        else:
            for env_key in sorted(environments.keys()):
                env = environments[env_key]
                print(f"  {env_key:<15}: {env.description} (v{env.version})")
                print(f"                   Package: {env.package_name}")

        print("-" * 70)
        print(f"Total: {len(environments)} environments")

    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        if self._cache_file.exists():
            self._cache_file.unlink()
        self._cache = None


# Global discovery instance
_global_discovery: EnvironmentDiscovery | None = None


def get_discovery() -> EnvironmentDiscovery:
    """
    Get or create the global discovery instance.

    Returns:
        Global EnvironmentDiscovery instance

    Examples:
        >>> discovery = get_discovery()
        >>> envs = discovery.discover()
    """
    global _global_discovery

    if _global_discovery is None:
        _global_discovery = EnvironmentDiscovery()

    return _global_discovery


def reset_discovery() -> None:
    """Drop the global discovery singleton (useful for testing).

    Does **not** delete the on-disk cache. The cache is now a persistent
    per-user file under `$XDG_CACHE_HOME`; wiping it belongs on an explicit
    [`EnvironmentDiscovery.clear_cache`][] call so test fixtures that only
    need a fresh singleton cannot clobber a developer's real cache.
    """
    global _global_discovery
    _global_discovery = None
