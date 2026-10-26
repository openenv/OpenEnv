# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for Package-Based Environment Discovery
===================================================

Tests cover:
1. Package discovery using importlib.metadata
2. Manifest loading from package resources
3. Class name inference
4. Cache management
5. Helper functions (_normalize_env_name, _is_hub_url, etc.)
"""

import json
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import openenv.auto._discovery as _discovery_module
import pytest
from openenv.auto._discovery import (
    _create_env_info_from_package,
    _default_cache_file,
    _infer_class_name,
    _is_hub_url,
    _is_trusted_cache_file,
    _normalize_env_name,
    _open_trusted_cache,
    EnvironmentDiscovery,
    EnvironmentInfo,
    get_discovery,
    reset_discovery,
)


@pytest.fixture(autouse=True)
def _isolate_discovery_cache(tmp_path, monkeypatch):
    """Keep every test off the real per-user discovery cache.

    The cache used to live in the shared temp directory, so tests that
    exercised `_save_cache`/`clear_cache` against a default-constructed
    `EnvironmentDiscovery` only ever disturbed a throwaway file. Now that it
    lives under `$XDG_CACHE_HOME`, the same tests would write to (and
    `clear_cache()` would delete) the cache belonging to whoever runs the
    suite. Redirecting the default path keeps that entirely inside `tmp_path`.
    """
    monkeypatch.setattr(
        _discovery_module,
        "_default_cache_file",
        lambda: tmp_path / "openenv" / "discovery_cache.json",
    )


class TestEnvironmentInfo:
    """Test EnvironmentInfo dataclass and methods."""

    def test_environment_info_creation(self):
        """Test creating EnvironmentInfo instance."""
        env_info = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            package_name="openenv-echo-env",
            version="0.1.0",
            description="Echo environment",
            client_module_path="echo_env.client",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest",
        )

        assert env_info.env_key == "echo"
        assert env_info.name == "echo_env"
        assert env_info.package_name == "openenv-echo-env"
        assert env_info.client_class_name == "EchoEnv"
        assert env_info.default_image == "echo-env:latest"


class TestHelperFunctions:
    """Test helper functions."""

    def test_normalize_env_name_simple(self):
        """Test normalizing simple names."""
        assert _normalize_env_name("echo") == "echo_env"
        assert _normalize_env_name("coding") == "coding_env"

    def test_normalize_env_name_with_suffix(self):
        """Test normalizing names with -env suffix."""
        assert _normalize_env_name("echo-env") == "echo_env"
        assert _normalize_env_name("coding-env") == "coding_env"

    def test_normalize_env_name_with_underscore(self):
        """Test normalizing names with _env suffix."""
        assert _normalize_env_name("echo_env") == "echo_env"
        assert _normalize_env_name("coding_env") == "coding_env"

    def test_is_hub_url_with_slash(self):
        """Test Hub URL detection with org/repo pattern."""
        assert _is_hub_url("meta-pytorch/coding-env")
        assert _is_hub_url("myorg/myenv")

    def test_is_hub_url_with_domain(self):
        """Test Hub URL detection with full URL."""
        assert _is_hub_url("https://huggingface.co/meta-pytorch/coding-env")
        assert _is_hub_url("huggingface.co/spaces/myenv")

    def test_is_hub_url_local(self):
        """Test that local names are not detected as Hub URLs."""
        assert not _is_hub_url("echo")
        assert not _is_hub_url("coding-env")
        assert not _is_hub_url("echo_env")

    def test_infer_class_name_client(self):
        """Test inferring client class names."""
        assert _infer_class_name("echo_env", "client") == "EchoEnv"
        assert _infer_class_name("coding_env", "client") == "CodingEnv"
        assert _infer_class_name("browser_gym_env", "client") == "BrowserGymEnv"

    def test_infer_class_name_action(self):
        """Test inferring action class names."""
        assert _infer_class_name("echo_env", "action") == "EchoAction"
        assert _infer_class_name("coding_env", "action") == "CodingAction"

    def test_infer_class_name_observation(self):
        """Test inferring observation class names."""
        assert _infer_class_name("echo_env", "observation") == "EchoObservation"
        assert _infer_class_name("coding_env", "observation") == "CodingObservation"


class TestCreateEnvInfoFromPackage:
    """Test creating EnvironmentInfo from package data."""

    @patch("openenv.auto._discovery._load_manifest_from_package")
    def test_create_env_info_with_manifest(self, mock_load_manifest):
        """Test creating env info when manifest exists."""
        # Mock manifest data
        mock_load_manifest.return_value = {
            "name": "echo_env",
            "version": "0.1.0",
            "description": "Echo environment for OpenEnv",
            "spec_version": 1,
        }

        env_info = _create_env_info_from_package(
            package_name="openenv-echo-env", module_name="echo_env", version="0.1.0"
        )

        assert env_info is not None
        assert env_info.env_key == "echo"
        assert env_info.name == "echo_env"
        assert env_info.package_name == "openenv-echo-env"
        assert env_info.version == "0.1.0"
        assert env_info.client_class_name == "EchoEnv"
        assert env_info.action_class_name == "EchoAction"

    @patch("openenv.auto._discovery._load_manifest_from_package")
    def test_create_env_info_with_custom_class_names(self, mock_load_manifest):
        """Test creating env info with custom class names from manifest."""
        # Mock manifest with custom class names
        mock_load_manifest.return_value = {
            "name": "coding_env",
            "version": "0.1.0",
            "description": "Coding environment",
            "action": "CodeAction",  # Custom name
            "observation": "CodeObservation",  # Custom name
        }

        env_info = _create_env_info_from_package(
            package_name="openenv-coding_env", module_name="coding_env", version="0.1.0"
        )

        assert env_info.action_class_name == "CodeAction"
        assert env_info.observation_class_name == "CodeObservation"

    @patch("openenv.auto._discovery._load_manifest_from_package")
    def test_create_env_info_without_manifest(self, mock_load_manifest):
        """Test creating env info when no manifest exists (uses conventions)."""
        mock_load_manifest.return_value = None

        env_info = _create_env_info_from_package(
            package_name="openenv-test-env", module_name="test_env", version="1.0.0"
        )

        assert env_info is not None
        assert env_info.env_key == "test"
        assert env_info.name == "test_env"
        assert env_info.client_class_name == "TestEnv"
        assert env_info.action_class_name == "TestAction"


class TestEnvironmentDiscovery:
    """Test EnvironmentDiscovery class."""

    @patch("importlib.metadata.distributions")
    @patch("openenv.auto._discovery._create_env_info_from_package")
    def test_discover_installed_packages(self, mock_create_info, mock_distributions):
        """Test discovering installed packages."""
        # Mock distribution objects
        mock_dist1 = Mock()
        mock_dist1.metadata = {"Name": "openenv-echo-env"}
        mock_dist1.version = "0.1.0"

        mock_dist2 = Mock()
        mock_dist2.metadata = {"Name": "openenv-coding_env"}
        mock_dist2.version = "0.2.0"

        mock_dist3 = Mock()
        mock_dist3.metadata = {"Name": "openenv-core"}  # Legacy core wheel
        mock_dist3.version = "1.0.0"

        mock_distributions.return_value = [mock_dist1, mock_dist2, mock_dist3]

        # Mock env info creation
        def create_info_side_effect(package_name, module_name, version):
            return EnvironmentInfo(
                env_key=module_name.replace("_env", ""),
                name=f"{module_name}",
                package_name=package_name,
                version=version,
                description=f"{module_name} environment",
                client_module_path=f"{module_name}.client",
                client_class_name=f"{module_name.replace('_env', '').capitalize()}Env",
                action_class_name=f"{module_name.replace('_env', '').capitalize()}Action",
                observation_class_name=f"{module_name.replace('_env', '').capitalize()}Observation",
                default_image=f"{module_name.replace('_', '-')}:latest",
            )

        mock_create_info.side_effect = create_info_side_effect

        discovery = EnvironmentDiscovery()
        envs = discovery._discover_installed_packages()

        # Should discover 2 environments, not the legacy core wheel
        assert len(envs) == 2
        assert "echo" in envs
        assert "coding" in envs

    def test_get_environment(self):
        """Test getting a specific environment."""
        discovery = EnvironmentDiscovery()

        # Mock the discover method
        with patch.object(discovery, "discover") as mock_discover:
            mock_discover.return_value = {
                "echo": EnvironmentInfo(
                    env_key="echo",
                    name="echo_env",
                    package_name="openenv-echo-env",
                    version="0.1.0",
                    description="Echo",
                    client_module_path="echo_env.client",
                    client_class_name="EchoEnv",
                    action_class_name="EchoAction",
                    observation_class_name="EchoObservation",
                    default_image="echo-env:latest",
                )
            }

            env = discovery.get_environment("echo")
            assert env is not None
            assert env.env_key == "echo"

    def test_get_environment_not_found(self):
        """Test getting a non-existent environment."""
        discovery = EnvironmentDiscovery()

        with patch.object(discovery, "discover") as mock_discover:
            mock_discover.return_value = {}

            env = discovery.get_environment("nonexistent")
            assert env is None

    def test_get_environment_by_name_flexible(self):
        """Test getting environment with flexible name matching."""
        discovery = EnvironmentDiscovery()

        mock_env = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            package_name="openenv-echo-env",
            version="0.1.0",
            description="Echo",
            client_module_path="echo_env.client",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest",
        )

        with patch.object(discovery, "discover") as mock_discover:
            mock_discover.return_value = {"echo": mock_env}

            # All these should work
            assert discovery.get_environment_by_name("echo") is not None
            assert discovery.get_environment_by_name("echo-env") is not None
            assert discovery.get_environment_by_name("echo_env") is not None

    def test_cache_management(self):
        """Test cache loading and saving."""
        discovery = EnvironmentDiscovery()

        # Create mock environment
        mock_env = EnvironmentInfo(
            env_key="test",
            name="test_env",
            package_name="openenv-test",
            version="1.0.0",
            description="Test",
            client_module_path="test_env.client",
            client_class_name="TestEnv",
            action_class_name="TestAction",
            observation_class_name="TestObservation",
            default_image="test-env:latest",
        )

        envs = {"test": mock_env}

        # Test saving cache
        discovery._save_cache(envs)
        assert discovery._cache_file.exists()

        # Test loading cache
        loaded = discovery._load_cache()
        assert loaded is not None
        assert "test" in loaded

        # Clean up
        discovery.clear_cache()
        assert not discovery._cache_file.exists()


class TestCacheSecurity:
    """The discovery cache must not be plantable/redirectable by another local user.

    A world-writable shared path let a local attacker pre-create the cache and
    redirect discovery's later ``import_module`` to attacker-chosen modules/classes
    (CWE-377 insecure temp file + CWE-427 uncontrolled load path).
    """

    def test_cache_file_is_per_user_not_shared_tmp(self):
        path = _default_cache_file()
        assert tempfile.gettempdir() not in str(path)
        assert path.parent.name == "openenv"

    def test_relative_xdg_cache_home_cannot_redirect_into_working_tree(
        self, tmp_path, monkeypatch
    ):
        """A relative XDG path must not trust a cache planted in the checkout."""
        checkout = tmp_path / "untrusted-checkout"
        planted = checkout / "cache" / "openenv" / "discovery_cache.json"
        planted.parent.mkdir(parents=True)
        planted.write_text("{}")

        monkeypatch.chdir(checkout)
        monkeypatch.setenv("XDG_CACHE_HOME", "cache")

        path = _default_cache_file()

        assert path == Path.home() / ".cache" / "openenv" / "discovery_cache.json"
        assert path.is_absolute()
        assert path.resolve() != planted.resolve()

    @pytest.mark.parametrize("xdg_cache_home", ["cache", ""])
    def test_relative_home_cannot_restore_a_relative_cache_path(
        self, tmp_path, monkeypatch, xdg_cache_home
    ):
        """Invalid XDG and home paths must fail closed, not trust the checkout."""
        checkout = tmp_path / "untrusted-checkout"
        checkout.mkdir()
        monkeypatch.chdir(checkout)
        monkeypatch.setenv("XDG_CACHE_HOME", xdg_cache_home)
        monkeypatch.setenv("HOME", "relative-home")

        with pytest.raises(RuntimeError, match="absolute home directory"):
            _default_cache_file()

    def test_world_writable_cache_is_not_trusted(self, tmp_path):
        f = tmp_path / "cache.json"
        f.write_text("{}")
        os.chmod(f, 0o666)
        # Rejected on POSIX; non-POSIX has no ownership model so it stays trusted.
        assert _is_trusted_cache_file(f) is (os.name != "posix")

    def test_owner_only_cache_is_trusted(self, tmp_path):
        f = tmp_path / "cache.json"
        f.write_text("{}")
        os.chmod(f, 0o600)
        assert _is_trusted_cache_file(f) is True

    def test_load_cache_ignores_world_writable_file(self, tmp_path):
        planted = tmp_path / "cache.json"
        planted.write_text(
            json.dumps(
                {
                    "evil": {
                        "env_key": "evil",
                        "name": "evil",
                        "package_name": "openenv-evil",
                        "version": "1.0.0",
                        "description": "",
                        "client_module_path": "os",
                        "client_class_name": "system",
                        "action_class_name": "A",
                        "observation_class_name": "O",
                        "default_image": "i",
                    }
                }
            )
        )
        os.chmod(planted, 0o666)
        discovery = EnvironmentDiscovery()
        discovery._cache_file = planted
        if os.name == "posix":
            assert discovery._load_cache() is None

    def test_saved_cache_is_owner_only(self, tmp_path):
        discovery = EnvironmentDiscovery()
        discovery._cache_file = tmp_path / "sub" / "cache.json"
        env = EnvironmentInfo(
            env_key="t",
            name="t",
            package_name="openenv-t",
            version="1.0.0",
            description="",
            client_module_path="t.client",
            client_class_name="T",
            action_class_name="A",
            observation_class_name="O",
            default_image="i",
        )
        discovery._save_cache({"t": env})
        assert discovery._cache_file.exists()
        if os.name == "posix":
            assert stat.S_IMODE(discovery._cache_file.stat().st_mode) == 0o600
        assert discovery._load_cache() is not None

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = tmp_path

    def test_symlinked_cache_is_refused(self, tmp_path):
        """The trust check must apply to the object actually read.

        Checking the path and then opening it separately leaves a window: an
        attacker who can write in the cache directory swaps the verified file
        for a symlink before the read. Opening with ``O_NOFOLLOW`` and
        inspecting the resulting descriptor closes it, so a symlink is refused
        outright rather than followed to a file that was never checked.
        """
        if os.name != "posix":
            return

        target = tmp_path / "attacker.json"
        target.write_text(
            json.dumps(
                {
                    "evil": {
                        "env_key": "evil",
                        "name": "evil",
                        "package_name": "openenv-evil",
                        "version": "1.0.0",
                        "description": "",
                        "client_module_path": "os",
                        "client_class_name": "system",
                        "action_class_name": "A",
                        "observation_class_name": "O",
                        "default_image": "i",
                    }
                }
            )
        )
        os.chmod(target, 0o600)

        link = tmp_path / "cache.json"
        link.symlink_to(target)

        discovery = EnvironmentDiscovery()
        discovery._cache_file = link
        assert discovery._load_cache() is None

    def test_save_does_not_rely_on_a_follow_up_chmod(self, tmp_path):
        """The cache must be created owner-only, not widened then narrowed.

        ``open()`` honours the umask, so creating the file and calling
        ``chmod`` afterwards leaves a window in which the cache is
        world-readable. Creating the descriptor with the mode already set
        removes it; `os.chmod` is made to fail here so the test only passes if
        nothing depends on it.
        """
        if os.name != "posix":
            return

        discovery = EnvironmentDiscovery()
        discovery._cache_file = tmp_path / "sub" / "cache.json"
        env = EnvironmentInfo(
            env_key="t",
            name="t",
            package_name="openenv-t",
            version="1.0.0",
            description="",
            client_module_path="t.client",
            client_class_name="T",
            action_class_name="A",
            observation_class_name="O",
            default_image="i",
        )

        real_umask = os.umask(0)
        real_chmod = os.chmod

        def _no_chmod(*args, **kwargs):
            raise AssertionError("cache permissions must not depend on chmod")

        os.chmod = _no_chmod
        try:
            discovery._save_cache({"t": env})
        finally:
            os.chmod = real_chmod
            os.umask(real_umask)

        assert discovery._cache_file.exists()
        assert stat.S_IMODE(discovery._cache_file.stat().st_mode) == 0o600

    def test_a_fifo_cache_does_not_block_discovery(self):
        """A planted FIFO must be refused, not waited on.

        Opening a FIFO read-only blocks until a writer appears, and that
        happens before the descriptor can be inspected, so a trust check that
        runs after the open never gets to reject it. The open must not block,
        and only a regular file is acceptable.
        """
        if os.name != "posix":
            return

        import threading

        fifo = self.tmp / "fifo.json"
        os.mkfifo(fifo, 0o600)

        outcome = {}

        def attempt():
            outcome["fd"] = _open_trusted_cache(fifo)

        worker = threading.Thread(target=attempt, daemon=True)
        worker.start()
        worker.join(timeout=5.0)

        assert not worker.is_alive(), "opening a FIFO cache blocked"
        assert outcome["fd"] is None

    def test_saving_over_a_world_writable_file_does_not_keep_its_mode(self):
        """`O_CREAT` only applies the mode when it creates the file.

        Writing through an existing inode therefore leaves whatever mode that
        file already had, so a cache that is already group/world-writable stays
        that way and the data just written is readable by everyone.
        """
        if os.name != "posix":
            return

        existing = self.tmp / "cache.json"
        existing.write_text("{}")
        os.chmod(existing, 0o666)

        discovery = EnvironmentDiscovery()
        discovery._cache_file = existing
        discovery._save_cache({})

        assert stat.S_IMODE(existing.stat().st_mode) == 0o600


class TestGlobalDiscovery:
    """Test global discovery instance management."""

    def test_get_discovery_singleton(self):
        """Test that get_discovery returns singleton."""
        reset_discovery()

        discovery1 = get_discovery()
        discovery2 = get_discovery()

        assert discovery1 is discovery2

    def test_reset_discovery(self):
        """Test resetting global discovery instance."""
        discovery1 = get_discovery()

        reset_discovery()

        discovery2 = get_discovery()

        # Should be different instances after reset
        assert discovery1 is not discovery2

    def test_reset_discovery_does_not_unlink_disk_cache(self, tmp_path, monkeypatch):
        """Singleton reset must not delete the persistent per-user cache.

        Regression for Bugbot on #1167: `reset_discovery()` used to call
        `clear_cache()`, and suites that only needed a fresh singleton
        (e.g. `test_auto_env.py`) would delete `~/.cache/openenv/...`.
        """
        cache = tmp_path / "discovery_cache.json"
        cache.write_text("{}")
        monkeypatch.setattr(_discovery_module, "_default_cache_file", lambda: cache)
        reset_discovery()

        discovery = get_discovery()
        assert discovery._cache_file == cache

        reset_discovery()

        assert cache.exists(), "reset_discovery() must not delete the on-disk cache"
        assert cache.read_text() == "{}"


class TestListEnvironments:
    """Test list_environments output."""

    def test_list_environments_with_envs(self, capsys):
        """Test listing when environments are found."""
        discovery = EnvironmentDiscovery()

        mock_envs = {
            "echo": EnvironmentInfo(
                env_key="echo",
                name="echo_env",
                package_name="openenv-echo-env",
                version="0.1.0",
                description="Echo environment",
                client_module_path="echo_env.client",
                client_class_name="EchoEnv",
                action_class_name="EchoAction",
                observation_class_name="EchoObservation",
                default_image="echo-env:latest",
            )
        }

        with patch.object(discovery, "discover", return_value=mock_envs):
            discovery.list_environments()

        captured = capsys.readouterr()
        assert "Available OpenEnv Environments" in captured.out
        assert "echo" in captured.out
        assert "Total: 1 environments" in captured.out

    def test_list_environments_empty(self, capsys):
        """Test listing when no environments are found."""
        discovery = EnvironmentDiscovery()

        with patch.object(discovery, "discover", return_value={}):
            discovery.list_environments()

        captured = capsys.readouterr()
        assert "No OpenEnv environments found" in captured.out
        assert "pip install openenv-" in captured.out
