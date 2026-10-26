# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for AutoEnv and AutoAction
======================================

Tests cover:
1. AutoEnv factory methods (from_hub, get_env_class, get_env_info, list_environments)
2. AutoAction factory methods (from_hub, from_env, get_action_info, list_actions)
3. Error handling for unknown environments
4. Name normalization and suggestions
5. Hub URL detection and handling
6. Integration with the discovery system
"""

from unittest.mock import Mock, patch

import pytest
from openenv.auto._discovery import (
    _is_hub_url,
    _normalize_env_name,
    EnvironmentDiscovery,
    EnvironmentInfo,
    reset_discovery,
)
from openenv.auto.auto_action import AutoAction
from openenv.auto.auto_env import AutoEnv


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_env_info():
    """Create a mock EnvironmentInfo for testing."""
    return EnvironmentInfo(
        env_key="echo",
        name="echo_env",
        package_name="openenv-echo-env",
        version="0.1.0",
        description="Echo environment for testing",
        client_module_path="echo_env.client",
        client_class_name="EchoEnv",
        action_class_name="EchoAction",
        observation_class_name="EchoObservation",
        default_image="echo-env:latest",
        spec_version=1,
    )


@pytest.fixture
def mock_coding_env_info():
    """Create a mock EnvironmentInfo for coding environment."""
    return EnvironmentInfo(
        env_key="coding",
        name="coding_env",
        package_name="openenv-coding_env",
        version="0.2.0",
        description="Coding environment with Python execution",
        client_module_path="coding_env.client",
        client_class_name="CodingEnv",
        action_class_name="CodeAction",  # Custom name
        observation_class_name="CodeObservation",  # Custom name
        default_image="coding-env:latest",
        spec_version=1,
    )


@pytest.fixture
def mock_discovery(mock_env_info, mock_coding_env_info):
    """Create a mock discovery instance with test environments."""
    discovery = Mock(spec=EnvironmentDiscovery)
    envs = {
        "echo": mock_env_info,
        "coding": mock_coding_env_info,
    }
    discovery.discover.return_value = envs
    discovery.get_environment.side_effect = lambda key: envs.get(key)
    discovery.get_environment_by_name.side_effect = lambda name: envs.get(
        _normalize_env_name(name).replace("_env", "")
    )
    return discovery


@pytest.fixture(autouse=True)
def reset_global_discovery():
    """Reset global discovery before and after each test."""
    reset_discovery()
    yield
    reset_discovery()


# ============================================================================
# AutoEnv Tests
# ============================================================================


class TestAutoEnvInstantiation:
    """Test that AutoEnv cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        """AutoEnv should raise TypeError when instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            AutoEnv()

        assert "factory class" in str(exc_info.value).lower()
        assert "AutoEnv.from_hub()" in str(exc_info.value)


class TestAutoEnvGetEnvClass:
    """Test AutoEnv.get_env_class() method."""

    def test_get_env_class_success(self, mock_discovery, mock_env_info):
        """Test getting environment class successfully."""
        # Mock the discovery
        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            # Mock the client class
            mock_client_class = Mock()
            mock_env_info.get_client_class = Mock(return_value=mock_client_class)

            result = AutoEnv.get_env_class("echo")

            assert result is mock_client_class
            mock_env_info.get_client_class.assert_called_once()

    def test_get_env_class_not_found(self, mock_discovery):
        """Test getting unknown environment raises ValueError."""
        mock_discovery.get_environment_by_name.return_value = None

        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoEnv.get_env_class("nonexistent")

            assert "Unknown environment" in str(exc_info.value)

    def test_get_env_class_with_different_name_formats(
        self, mock_discovery, mock_env_info
    ):
        """Test that different name formats resolve correctly."""
        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            mock_client_class = Mock()
            mock_env_info.get_client_class = Mock(return_value=mock_client_class)

            # All these should work
            for name in ["echo", "echo-env", "echo_env"]:
                mock_discovery.get_environment_by_name.return_value = mock_env_info
                result = AutoEnv.get_env_class(name)
                assert result is mock_client_class


class TestAutoEnvGetEnvInfo:
    """Test AutoEnv.get_env_info() method."""

    def test_get_env_info_success(self, mock_discovery, mock_env_info):
        """Test getting environment info successfully."""
        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            mock_discovery.get_environment_by_name.return_value = mock_env_info

            info = AutoEnv.get_env_info("echo")

            assert info["env_key"] == "echo"
            assert info["name"] == "echo_env"
            assert info["package"] == "openenv-echo-env"
            assert info["version"] == "0.1.0"
            assert info["description"] == "Echo environment for testing"
            assert info["env_class"] == "EchoEnv"
            assert info["action_class"] == "EchoAction"
            assert info["observation_class"] == "EchoObservation"
            assert info["module"] == "echo_env.client"
            assert info["default_image"] == "echo-env:latest"

    def test_get_env_info_not_found(self, mock_discovery):
        """Test getting info for unknown environment raises ValueError."""
        mock_discovery.get_environment_by_name.return_value = None

        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoEnv.get_env_info("nonexistent")

            assert "Unknown environment" in str(exc_info.value)


class TestAutoEnvListEnvironments:
    """Test AutoEnv.list_environments() method."""

    def test_list_environments(self, mock_discovery, capsys):
        """Test listing environments prints formatted output."""
        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            AutoEnv.list_environments()

        capsys.readouterr()  # Clear captured output
        # Should call discovery.list_environments()
        mock_discovery.list_environments.assert_called_once()


class TestAutoEnvFromName:
    """Test AutoEnv.from_hub() method."""

    def test_from_hub_unknown_env_with_suggestions(self, mock_discovery):
        """Test that unknown environment provides suggestions."""
        mock_discovery.get_environment_by_name.return_value = None
        mock_discovery.discover.return_value = {
            "echo": Mock(),
            "coding": Mock(),
        }

        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoEnv.from_hub("ech")  # Close to "echo"

            error_msg = str(exc_info.value)
            assert "Unknown environment" in error_msg or "ech" in error_msg
            # Should suggest similar names
            assert "echo" in error_msg.lower() or "available" in error_msg.lower()

    def test_from_hub_no_envs_available(self, mock_discovery):
        """Test error message when no environments are installed."""
        mock_discovery.get_environment_by_name.return_value = None
        mock_discovery.discover.return_value = {}

        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoEnv.from_hub("anyenv")

            error_msg = str(exc_info.value)
            assert "No OpenEnv environments found" in error_msg
            assert "pip install" in error_msg

    def test_from_hub_with_base_url(self, mock_discovery, mock_env_info):
        """Test from_hub with explicit base_url."""
        mock_discovery.get_environment_by_name.return_value = mock_env_info

        # Mock the client class
        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_env_info.get_client_class = Mock(return_value=mock_client_class)

        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            with patch(
                "openenv.auto.auto_env.AutoEnv._check_server_availability",
                return_value=True,
            ):
                result = AutoEnv.from_hub("echo", base_url="http://localhost:8000")

                assert result is mock_client_instance
                mock_client_class.assert_called_once_with(
                    base_url="http://localhost:8000", provider=None
                )


class TestAutoEnvHubDetection:
    """Test AutoEnv Hub URL detection and handling."""

    def test_documented_echo_space_path(self, mock_discovery, mock_env_info):
        """Test the canonical Echo Space ID with its explicit runtime URL."""
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_env_info.get_client_class = Mock(return_value=mock_client_class)
        base_url = "https://openenv-echo-env.hf.space"

        with (
            patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery),
            patch.object(AutoEnv, "_check_space_availability", return_value=False),
            patch.object(
                AutoEnv,
                "_ensure_package_from_hub",
                return_value="echo",
            ) as ensure_package,
            patch.object(AutoEnv, "_check_server_availability", return_value=True),
        ):
            result = AutoEnv.from_env(
                "openenv/echo_env",
                base_url=base_url,
            )

        assert result is mock_client_instance
        ensure_package.assert_called_once_with(
            "openenv/echo_env",
            trust_remote_code=False,
        )
        mock_client_class.assert_called_once_with(base_url=base_url, provider=None)

    def test_resolve_space_url(self):
        """Test resolving HuggingFace Space URL."""
        url = AutoEnv._resolve_space_url("wukaixingxp/coding-env-test")
        assert url == "https://wukaixingxp-coding-env-test.hf.space"

    @pytest.mark.parametrize(
        "repo_id, expected",
        [
            ("openenv/echo_env", "https://openenv-echo-env.hf.space"),
            ("openenv/repl_env-0.2.2", "https://openenv-repl-env-0-2-2.hf.space"),
            ("User/My_Env", "https://user-my-env.hf.space"),
        ],
    )
    def test_resolve_space_url_normalizes_subdomain(self, repo_id, expected):
        """Space subdomains are lowercase with non-alphanumerics replaced by '-'."""
        assert AutoEnv._resolve_space_url(repo_id) == expected

    def test_resolve_space_url_from_full_url(self):
        """Test resolving from full HuggingFace URL."""
        url = AutoEnv._resolve_space_url(
            "https://huggingface.co/wukaixingxp/coding-env-test"
        )
        assert url == "https://wukaixingxp-coding-env-test.hf.space"


# ============================================================================
# _is_local_url Tests
# ============================================================================


class TestIsLocalUrl:
    """Test AutoEnv._is_local_url() uses proper URL parsing, not substring matching."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8000",
            "http://localhost",
            "http://127.0.0.1:8000",
            "http://127.0.0.1",
            "http://127.5.5.5",  # rest of 127.0.0.0/8 loopback range
            "http://[::1]:8000",  # IPv6 loopback with port
            "http://[::1]",  # IPv6 loopback bare
            "http://0.0.0.0:8000",
            "http://0.0.0.0",
        ],
    )
    def test_local_urls_return_true(self, url):
        assert AutoEnv._is_local_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost.evil.com",  # headline false-positive bug
            "http://my-127.0.0.1.io",  # 127.0.0.1 as substring in domain
            "https://example.com",
            "https://example.com/path?q=localhost",  # only in query string
            "not-a-url",
            "",
        ],
    )
    def test_non_local_urls_return_false(self, url):
        assert AutoEnv._is_local_url(url) is False


# ============================================================================
# Git+ URL Installation Tests
# ============================================================================


class TestGitPlusUrlInstallation:
    """Test git+ URL installation functionality."""

    def test_get_hub_git_url(self):
        """Test generating git+ URL from repo ID."""
        url = AutoEnv._get_hub_git_url("burtenshaw/wordle")
        assert url == "git+https://huggingface.co/spaces/burtenshaw/wordle"

    def test_get_hub_git_url_from_full_url(self):
        """Test generating git+ URL from full HuggingFace URL."""
        url = AutoEnv._get_hub_git_url(
            "https://huggingface.co/spaces/burtenshaw/wordle"
        )
        assert url == "git+https://huggingface.co/spaces/burtenshaw/wordle"

    def test_install_from_hub_uses_git_url(self, mock_discovery):
        """Test that _install_from_hub uses git+ URL for installation."""
        with (
            patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery),
            patch("openenv.auto.auto_env._confirm_remote_install", return_value=True),
            patch("openenv.auto.auto_env.subprocess.run") as mock_run,
            patch("openenv.auto.auto_env._get_pip_command", return_value=["pip"]),
        ):
            mock_run.return_value = Mock(
                stdout="Successfully installed openenv-wordle_env-0.1.0",
                stderr="",
                returncode=0,
            )

            result = AutoEnv._install_from_hub("burtenshaw/wordle")

            # Verify git+ URL was used
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert (
                "git+https://huggingface.co/spaces/burtenshaw/wordle" in call_args[0][0]
            )
            # Verify package name is returned
            assert result == "openenv-wordle_env"

    def test_install_from_hub_respects_user_decline(self):
        """Test that installation is cancelled when user declines."""
        with patch("openenv.auto.auto_env._confirm_remote_install", return_value=False):
            with pytest.raises(ValueError) as exc_info:
                AutoEnv._install_from_hub("burtenshaw/wordle")

            assert "Installation cancelled" in str(exc_info.value)

    def test_install_from_hub_with_trust_remote_code(self):
        """Test that trust_remote_code=True skips confirmation."""
        with (
            patch("openenv.auto.auto_env._confirm_remote_install") as mock_confirm,
            patch("openenv.auto.auto_env.subprocess.run") as mock_run,
            patch("openenv.auto.auto_env._get_pip_command", return_value=["pip"]),
        ):
            mock_run.return_value = Mock(
                stdout="Successfully installed openenv-wordle_env-0.1.0",
                stderr="",
                returncode=0,
            )

            AutoEnv._install_from_hub("burtenshaw/wordle", trust_remote_code=True)

            # Confirmation should not be called when trust_remote_code=True
            mock_confirm.assert_not_called()


# ============================================================================
# uv pip Detection Tests
# ============================================================================


class TestUvPipDetection:
    """Test uv pip detection and command selection."""

    def test_has_uv_when_available(self):
        """Test _has_uv returns True when uv is installed."""
        from openenv.auto.auto_env import _has_uv

        with patch("shutil.which", return_value="/usr/local/bin/uv"):
            assert _has_uv() is True

    def test_has_uv_when_not_available(self):
        """Test _has_uv returns False when uv is not installed."""
        from openenv.auto.auto_env import _has_uv

        with patch("shutil.which", return_value=None):
            assert _has_uv() is False

    def test_get_pip_command_prefers_uv(self):
        """Test _get_pip_command returns uv pip when uv is available."""
        from openenv.auto.auto_env import _get_pip_command

        with patch("openenv.auto.auto_env._has_uv", return_value=True):
            cmd = _get_pip_command()
            assert cmd == ["uv", "pip"]

    def test_get_pip_command_falls_back_to_pip(self):
        """Test _get_pip_command returns pip when uv is not available."""
        import sys

        from openenv.auto.auto_env import _get_pip_command

        with patch("openenv.auto.auto_env._has_uv", return_value=False):
            cmd = _get_pip_command()
            assert cmd == [sys.executable, "-m", "pip"]


# ============================================================================
# User Confirmation Tests
# ============================================================================


class TestUserConfirmation:
    """Test user confirmation for remote code installation."""

    def test_confirm_skipped_with_env_var(self):
        """Test confirmation is skipped when OPENENV_TRUST_REMOTE_CODE is set."""
        import os

        from openenv.auto.auto_env import _confirm_remote_install

        with patch.dict(os.environ, {"OPENENV_TRUST_REMOTE_CODE": "1"}):
            result = _confirm_remote_install("test/repo")
            assert result is True

    def test_confirm_skipped_with_env_var_true(self):
        """Test confirmation is skipped when OPENENV_TRUST_REMOTE_CODE=true."""
        import os

        from openenv.auto.auto_env import _confirm_remote_install

        with patch.dict(os.environ, {"OPENENV_TRUST_REMOTE_CODE": "true"}):
            result = _confirm_remote_install("test/repo")
            assert result is True

    def test_confirm_returns_false_in_non_interactive(self):
        """Test confirmation returns False in non-interactive mode."""
        import os

        from openenv.auto.auto_env import _confirm_remote_install

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("sys.stdin.isatty", return_value=False),
        ):
            # Clear the env var if it exists
            os.environ.pop("OPENENV_TRUST_REMOTE_CODE", None)
            result = _confirm_remote_install("test/repo")
            assert result is False

    def test_confirm_prompts_user_when_interactive(self):
        """Test confirmation prompts user in interactive mode."""
        import os

        from openenv.auto.auto_env import _confirm_remote_install

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            os.environ.pop("OPENENV_TRUST_REMOTE_CODE", None)
            result = _confirm_remote_install("test/repo")
            assert result is True

    def test_confirm_user_declines(self):
        """Test confirmation returns False when user declines."""
        import os

        from openenv.auto.auto_env import _confirm_remote_install

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="n"),
        ):
            os.environ.pop("OPENENV_TRUST_REMOTE_CODE", None)
            result = _confirm_remote_install("test/repo")
            assert result is False


# ============================================================================
# AutoAction Tests
# ============================================================================


class TestAutoActionInstantiation:
    """Test that AutoAction cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        """AutoAction should raise TypeError when instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            AutoAction()

        assert "factory class" in str(exc_info.value).lower()
        assert "AutoAction.from_hub()" in str(exc_info.value)


class TestAutoActionFromName:
    """Test AutoAction.from_hub() method."""

    def test_from_hub_success(self, mock_discovery, mock_env_info):
        """Test getting action class successfully."""
        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            mock_discovery.get_environment_by_name.return_value = mock_env_info

            # Mock the action class
            mock_action_class = Mock()
            mock_env_info.get_action_class = Mock(return_value=mock_action_class)

            result = AutoAction.from_hub("echo")

            assert result is mock_action_class
            mock_env_info.get_action_class.assert_called_once()

    def test_from_hub_not_found(self, mock_discovery):
        """Test getting unknown action raises ValueError."""
        mock_discovery.get_environment_by_name.return_value = None
        mock_discovery.discover.return_value = {}

        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            with pytest.raises(ValueError) as exc_info:
                AutoAction.from_hub("nonexistent")

            error_msg = str(exc_info.value)
            assert "No OpenEnv environments found" in error_msg

    def test_from_hub_with_suggestions(self, mock_discovery):
        """Test that unknown action provides suggestions."""
        mock_discovery.get_environment_by_name.return_value = None
        mock_discovery.discover.return_value = {
            "echo": Mock(),
            "coding": Mock(),
        }

        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            with pytest.raises(ValueError) as exc_info:
                AutoAction.from_hub("ech")  # Close to "echo"

            error_msg = str(exc_info.value)
            assert "Unknown environment" in error_msg or "ech" in error_msg

    def test_from_hub_with_different_formats(self, mock_discovery, mock_env_info):
        """Test that different name formats work."""
        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            mock_action_class = Mock()
            mock_env_info.get_action_class = Mock(return_value=mock_action_class)

            # All these should work
            for name in ["echo", "echo-env", "echo_env"]:
                mock_discovery.get_environment_by_name.return_value = mock_env_info
                result = AutoAction.from_hub(name)
                assert result is mock_action_class


class TestAutoActionFromEnv:
    """Test AutoAction.from_env() method (alias for from_hub)."""

    def test_from_env_is_alias(self, mock_discovery, mock_env_info):
        """Test that from_env is an alias for from_hub."""
        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            mock_discovery.get_environment_by_name.return_value = mock_env_info

            mock_action_class = Mock()
            mock_env_info.get_action_class = Mock(return_value=mock_action_class)

            result = AutoAction.from_env("echo")

            assert result is mock_action_class


class TestAutoActionGetActionInfo:
    """Test AutoAction.get_action_info() method."""

    def test_get_action_info_success(self, mock_discovery, mock_env_info):
        """Test getting action info successfully."""
        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            mock_discovery.get_environment_by_name.return_value = mock_env_info

            info = AutoAction.get_action_info("echo")

            assert info["env_key"] == "echo"
            assert info["env_name"] == "echo_env"
            assert info["package"] == "openenv-echo-env"
            assert info["action_class"] == "EchoAction"
            assert info["observation_class"] == "EchoObservation"
            assert info["module"] == "echo_env.client"

    def test_get_action_info_with_custom_names(
        self, mock_discovery, mock_coding_env_info
    ):
        """Test getting action info with custom class names."""
        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            mock_discovery.get_environment_by_name.return_value = mock_coding_env_info

            info = AutoAction.get_action_info("coding")

            assert info["action_class"] == "CodeAction"
            assert info["observation_class"] == "CodeObservation"

    def test_get_action_info_not_found(self, mock_discovery):
        """Test getting info for unknown environment raises ValueError."""
        mock_discovery.get_environment_by_name.return_value = None

        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            with pytest.raises(ValueError) as exc_info:
                AutoAction.get_action_info("nonexistent")

            assert "Unknown environment" in str(exc_info.value)


class TestAutoActionListActions:
    """Test AutoAction.list_actions() method."""

    def test_list_actions_with_envs(
        self, mock_discovery, mock_env_info, mock_coding_env_info, capsys
    ):
        """Test listing actions prints formatted output."""
        mock_discovery.discover.return_value = {
            "echo": mock_env_info,
            "coding": mock_coding_env_info,
        }

        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            AutoAction.list_actions()

        captured = capsys.readouterr()
        assert "Available Action Classes" in captured.out
        assert "echo" in captured.out
        assert "EchoAction" in captured.out
        assert "coding" in captured.out
        assert "CodeAction" in captured.out
        assert "Total: 2 action classes" in captured.out

    def test_list_actions_empty(self, mock_discovery, capsys):
        """Test listing when no environments are found."""
        mock_discovery.discover.return_value = {}

        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            AutoAction.list_actions()

        captured = capsys.readouterr()
        assert "No OpenEnv environments found" in captured.out
        assert "pip install openenv-" in captured.out


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestNormalizeEnvName:
    """Test _normalize_env_name helper function."""

    def test_simple_name(self):
        """Test normalizing simple names."""
        assert _normalize_env_name("echo") == "echo_env"
        assert _normalize_env_name("coding") == "coding_env"

    def test_name_with_hyphen_suffix(self):
        """Test normalizing names with -env suffix."""
        assert _normalize_env_name("echo-env") == "echo_env"
        assert _normalize_env_name("coding-env") == "coding_env"

    def test_name_with_underscore_suffix(self):
        """Test normalizing names with _env suffix."""
        assert _normalize_env_name("echo_env") == "echo_env"
        assert _normalize_env_name("coding_env") == "coding_env"

    def test_name_with_hyphens(self):
        """Test normalizing names with hyphens."""
        assert _normalize_env_name("browser-gym") == "browser_gym_env"
        assert _normalize_env_name("sumo-rl") == "sumo_rl_env"


class TestIsHubUrl:
    """Test _is_hub_url helper function."""

    def test_org_repo_pattern(self):
        """Test Hub detection with org/repo pattern."""
        assert _is_hub_url("meta-pytorch/coding-env") is True
        assert _is_hub_url("myorg/myenv") is True
        assert _is_hub_url("wukaixingxp/echo-env-test") is True

    def test_full_url(self):
        """Test Hub detection with full URL."""
        assert _is_hub_url("https://huggingface.co/meta-pytorch/coding-env") is True
        assert _is_hub_url("huggingface.co/spaces/myenv") is True

    def test_local_names(self):
        """Test that local names are not detected as Hub URLs."""
        assert _is_hub_url("echo") is False
        assert _is_hub_url("coding-env") is False
        assert _is_hub_url("echo_env") is False
        assert _is_hub_url("browsergym") is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestAutoEnvAutoActionIntegration:
    """Test integration between AutoEnv and AutoAction."""

    def test_same_env_resolves_consistently(self, mock_discovery, mock_env_info):
        """Test that AutoEnv and AutoAction resolve the same environment."""
        with (
            patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery),
            patch(
                "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
            ),
        ):
            mock_discovery.get_environment_by_name.return_value = mock_env_info

            # Mock classes
            mock_client_class = Mock()
            mock_action_class = Mock()
            mock_env_info.get_client_class = Mock(return_value=mock_client_class)
            mock_env_info.get_action_class = Mock(return_value=mock_action_class)

            env_class = AutoEnv.get_env_class("echo")
            action_class = AutoAction.from_hub("echo")

            # Both should resolve from the same env_info
            assert env_class is mock_client_class
            assert action_class is mock_action_class

    def test_env_info_matches_action_info(self, mock_discovery, mock_env_info):
        """Test that env info and action info are consistent."""
        with (
            patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery),
            patch(
                "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
            ),
        ):
            mock_discovery.get_environment_by_name.return_value = mock_env_info

            env_info = AutoEnv.get_env_info("echo")
            action_info = AutoAction.get_action_info("echo")

            # Should have consistent information
            assert env_info["action_class"] == action_info["action_class"]
            assert env_info["observation_class"] == action_info["observation_class"]
            assert env_info["module"] == action_info["module"]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling in AutoEnv and AutoAction."""

    def test_import_error_handling(self, mock_discovery, mock_env_info):
        """Test handling of import errors when loading classes."""
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_env_info.get_client_class = Mock(
            side_effect=ImportError("Module not found")
        )

        with patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery):
            with pytest.raises(ImportError) as exc_info:
                AutoEnv.from_hub("echo", base_url="http://localhost:8000")

            error_msg = str(exc_info.value)
            assert "Failed to import" in error_msg
            assert "pip install" in error_msg or "reinstall" in error_msg

    def test_action_import_error_handling(self, mock_discovery, mock_env_info):
        """Test handling of import errors when loading action classes."""
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_env_info.get_action_class = Mock(
            side_effect=ImportError("Module not found")
        )

        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            with pytest.raises(ImportError) as exc_info:
                AutoAction.from_hub("echo")

            error_msg = str(exc_info.value)
            assert "Failed to import" in error_msg


class TestNameVariations:
    """Test various name format variations work correctly."""

    @pytest.mark.parametrize(
        "name,expected_key",
        [
            ("echo", "echo"),
            ("echo-env", "echo"),
            ("echo_env", "echo"),
            ("coding", "coding"),
            ("coding-env", "coding"),
            ("coding_env", "coding"),
            ("browser-gym", "browser_gym"),
            ("browser_gym", "browser_gym"),
            ("sumo-rl", "sumo_rl"),
            ("sumo_rl", "sumo_rl"),
        ],
    )
    def test_name_normalization_variations(self, name, expected_key):
        """Test that various name formats normalize correctly."""
        normalized = _normalize_env_name(name)
        key = normalized.replace("_env", "")
        assert key == expected_key


# ============================================================================
# Real Integration Tests - HuggingFace Space
# ============================================================================
# These tests require network access and connect to real HuggingFace Spaces.
# Run with: pytest -m integration tests/envs/test_auto_env.py
# Or: pytest -m "integration and network" tests/envs/test_auto_env.py


@pytest.mark.integration
@pytest.mark.network
class TestHuggingFaceSpaceIntegration:
    """
    Real integration tests that connect to HuggingFace Spaces.

    These tests require:
    - Network access to huggingface.co and *.hf.space
    - The HuggingFace Space to be running and accessible

    Run these tests with:
        pytest -m "integration and network" tests/envs/test_auto_env.py -v
    """

    # Test Space URL - this is a real HuggingFace Space
    HF_SPACE_REPO = "openenv/coding_env"

    @pytest.fixture
    def check_space_availability(self):
        """Check if the HuggingFace Space is accessible before running tests."""
        import requests

        space_url = AutoEnv._resolve_space_url(self.HF_SPACE_REPO)
        try:
            response = requests.get(f"{space_url}/health", timeout=10)
            if response.status_code != 200:
                pytest.skip(f"HuggingFace Space not accessible at {space_url}")
        except requests.RequestException as e:
            pytest.skip(f"Cannot reach HuggingFace Space: {e}")

    def test_connect_to_hf_space(self, check_space_availability):
        """
        Test connecting to a real HuggingFace Space using AutoEnv.

        This test:
        1. Connects to wukaixingxp/coding-env-test Space
        2. Resets the environment
        3. Verifies we get a valid observation
        """
        # Connect to HuggingFace Space
        env = AutoEnv.from_hub(self.HF_SPACE_REPO)

        try:
            # Reset the environment
            result = env.reset()

            # Verify we got a valid result
            assert result is not None
            assert hasattr(result, "observation")

            print(
                f"✅ Successfully connected to HuggingFace Space: {self.HF_SPACE_REPO}"
            )
            print(f"   Reset observation: {result.observation}")
        finally:
            # Clean up
            env.close()

    def test_execute_action_on_hf_space(self, check_space_availability):
        """
        Test executing an action on a real HuggingFace Space.

        This test:
        1. Connects to wukaixingxp/coding-env-test Space
        2. Gets the action class using AutoAction
        3. Executes Python code
        4. Verifies the output
        """
        # Connect to HuggingFace Space
        env = AutoEnv.from_hub(self.HF_SPACE_REPO)

        try:
            # Reset the environment
            env.reset()

            # Get action class using AutoAction
            CodeAction = AutoAction.from_hub(self.HF_SPACE_REPO)

            # Create and execute action
            action = CodeAction(code="print('Hello from pytest!')")
            result = env.step(action)

            # Verify the result
            assert result is not None
            assert hasattr(result, "observation")
            assert hasattr(result, "reward")
            assert hasattr(result, "done")

            # Check if stdout contains our message
            if hasattr(result.observation, "stdout"):
                assert "Hello from pytest!" in result.observation.stdout
                print("✅ Code execution successful!")
                print(f"   stdout: {result.observation.stdout}")

            print(f"   reward: {result.reward}")
            print(f"   done: {result.done}")
        finally:
            # Clean up
            env.close()

    def test_autoenv_and_autoaction_same_space(self, check_space_availability):
        """
        Test that AutoEnv and AutoAction work together seamlessly.

        Verifies that calling both with the same HF Space repo ID
        doesn't cause duplicate downloads or installations.
        """
        # First call - AutoEnv
        env = AutoEnv.from_hub(self.HF_SPACE_REPO)

        try:
            # Second call - AutoAction (should use cached package)
            ActionClass = AutoAction.from_hub(self.HF_SPACE_REPO)

            # Verify both work
            result = env.reset()
            assert result is not None

            # Create an action instance
            action = ActionClass(code="x = 1 + 1")
            step_result = env.step(action)

            assert step_result is not None
            print("✅ AutoEnv and AutoAction work together correctly")
        finally:
            env.close()

    def test_space_availability_check(self):
        """Test the Space availability check functionality."""

        # Test with real Space URL
        space_url = AutoEnv._resolve_space_url(self.HF_SPACE_REPO)

        # Check availability (this is a real network call)
        try:
            is_available = AutoEnv._check_space_availability(space_url, timeout=10.0)
            print(f"Space {space_url} availability: {is_available}")
            # We don't assert True because the space might be down
        except Exception as e:
            pytest.skip(f"Network error checking Space availability: {e}")


# ============================================================================
# Real Integration Tests - Local Docker
# ============================================================================
# These tests require Docker to be installed and running.
# Run with: pytest -m "integration and docker" tests/envs/test_auto_env.py


@pytest.mark.integration
@pytest.mark.docker
class TestDockerIntegration:
    """
    Real integration tests that start Docker containers.

    These tests require:
    - Docker to be installed and running
    - Docker images to be built (e.g., echo-env:latest)

    Build the Docker image first:
        cd src/envs/echo_env/server && docker build -t echo-env:latest .

    Run these tests with:
        pytest -m "integration and docker" tests/envs/test_auto_env.py -v
    """

    @pytest.fixture
    def check_docker_available(self):
        """Check if Docker is available and the required image exists."""
        import shutil
        import subprocess

        # Check if docker command exists
        if not shutil.which("docker"):
            pytest.skip("Docker is not installed")

        # Check if Docker daemon is running
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("Docker daemon is not running")
        except subprocess.TimeoutExpired:
            pytest.skip("Docker daemon not responding")
        except Exception as e:
            pytest.skip(f"Cannot access Docker: {e}")

    @pytest.fixture
    def check_echo_env_image(self, check_docker_available):
        """Check if the echo-env Docker image is available."""
        import subprocess

        result = subprocess.run(
            ["docker", "images", "-q", "echo-env:latest"],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            pytest.skip(
                "Docker image 'echo-env:latest' not found. "
                "Build it with: cd src/envs/echo_env/server && docker build -t echo-env:latest ."
            )

    def test_autoenv_with_docker_echo_env(self, check_echo_env_image):
        """
        Test AutoEnv with a real Docker container (echo-env).

        This test:
        1. Starts an echo-env Docker container using AutoEnv
        2. Sends a message
        3. Verifies the echo response
        4. Cleans up the container
        """
        from openenv.core.env_server.mcp_types import CallToolAction

        # Start Docker container using AutoEnv
        env = AutoEnv.from_hub("echo", docker_image="echo-env:latest")

        try:
            # Reset the environment
            result = env.reset()
            assert result is not None
            assert hasattr(result, "observation")

            print("✅ Docker container started successfully")
            print(f"   Reset observation: {result.observation}")

            # Send a message using MCP
            action = CallToolAction(
                tool_name="echo_message",
                arguments={"message": "Hello from Docker test!"},
            )
            step_result = env.step(action)

            # Verify the echo
            assert step_result is not None
            assert step_result.observation is not None

            print("✅ Message echoed successfully")
            print(f"   result: {step_result.observation}")
        finally:
            # Clean up - this should stop the container
            env.close()

    def test_autoaction_with_docker_echo_env(self, check_echo_env_image):
        """
        Test AutoAction with a real Docker container (echo-env).

        This test uses GenericEnvClient with skip_install=True for pure MCP environments.
        """
        from openenv.core.env_server.mcp_types import CallToolAction
        from openenv.core.generic_client import GenericEnvClient

        # Start Docker container using GenericEnvClient (MCP-first approach)
        env = GenericEnvClient.from_docker_image("echo-env:latest")

        try:
            # Reset
            env.reset()

            # Create MCP action
            action = CallToolAction(
                tool_name="echo_message", arguments={"message": "Dynamic action!"}
            )
            step_result = env.step(action)

            # Verify
            assert step_result is not None

            print("✅ MCP with Docker works correctly")
        finally:
            env.close()

    def test_env_info_for_docker_env(self, check_docker_available):
        """Test getting environment info for a Docker-based environment."""
        try:
            info = AutoEnv.get_env_info("echo")

            assert info is not None
            assert info["env_key"] == "echo"
            assert info["default_image"] == "echo-env:latest"

            print("✅ Environment info retrieved successfully")
            print(f"   env_key: {info['env_key']}")
            print(f"   default_image: {info['default_image']}")
            print(f"   env_class: {info['env_class']}")
        except ValueError as e:
            pytest.skip(f"Echo environment not installed: {e}")


# ============================================================================
# Real Integration Tests - Local Server
# ============================================================================
# These tests connect to a local server without Docker


@pytest.mark.integration
class TestLocalServerIntegration:
    """
    Integration tests that connect to a locally running server.

    These tests require a server to be running on localhost.

    Start a server first:
        cd src && python -m envs.echo_env.server.app

    Run these tests with:
        pytest -m integration tests/envs/test_auto_env.py::TestLocalServerIntegration -v
    """

    @pytest.fixture
    def local_echo_server(self):
        """Check if local echo server is running."""
        import requests

        base_url = "http://localhost:8000"
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Local echo server not healthy")
            return base_url
        except requests.RequestException:
            pytest.skip(
                "Local echo server not running. "
                "Start it with: cd src && python -m envs.echo_env.server.app"
            )

    def test_autoenv_with_local_server(self, local_echo_server):
        """
        Test AutoEnv connecting to a local server using base_url.

        This test:
        1. Connects to localhost:8000 using MCPToolClient
        2. Resets the environment
        3. Sends a message
        4. Verifies the response
        """
        from echo_env import EchoEnv

        # Connect to local server
        with EchoEnv(base_url=local_echo_server) as env:
            # Reset
            result = env.reset()
            assert result is not None

            print(f"✅ Connected to local server at {local_echo_server}")

            # Send message using call_tool
            result = env.call_tool("echo_message", message="Hello local server!")

            assert result is not None
            assert "Hello local server!" in result

            print("✅ Local server test passed")
            print(f"   echoed_message: {result}")

    def test_multiple_steps_local_server(self, local_echo_server):
        """Test multiple steps on local server."""
        from echo_env import EchoEnv

        with EchoEnv(base_url=local_echo_server) as env:
            env.reset()

            messages = ["First message", "Second message", "Third message"]

            for i, msg in enumerate(messages):
                result = env.call_tool("echo_message", message=msg)

                assert msg in result
                print(f"✅ Step {i + 1}: '{msg}' → '{result}'")

            print(f"✅ Multiple steps test passed ({len(messages)} steps)")


# ============================================================================
# Test Markers Configuration
# ============================================================================
# Add this to conftest.py or pyproject.toml:
#
# [tool.pytest.ini_options]
# markers = [
#     "integration: mark test as integration test (may require external resources)",
#     "network: mark test as requiring network access",
#     "docker: mark test as requiring Docker",
# ]
