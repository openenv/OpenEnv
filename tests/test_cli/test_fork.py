# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenv fork command."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openenv.cli.__main__ import app
from typer.testing import CliRunner

runner = CliRunner()


def test_fork_requires_source_space() -> None:
    """Test that fork requires SOURCE_SPACE argument."""
    result = runner.invoke(app, ["fork"])
    assert result.exit_code != 0
    assert "source" in result.output.lower() or "argument" in result.output.lower()


def test_fork_validates_source_space_format() -> None:
    """Test that fork validates source space format (owner/name)."""
    result = runner.invoke(app, ["fork", "invalid-no-slash"])
    assert result.exit_code != 0
    assert "format" in result.output.lower() or "invalid" in result.output.lower()


def test_fork_calls_duplicate_space_with_from_id() -> None:
    """Test that fork calls HfApi.duplicate_space with correct from_id."""
    with (
        patch("openenv.cli.commands.fork.whoami") as mock_whoami,
        patch("openenv.cli.commands.fork.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_api = MagicMock()
        mock_api.duplicate_space.return_value = (
            "https://huggingface.co/spaces/testuser/source-space"
        )
        mock_hf_api_class.return_value = mock_api

        result = runner.invoke(app, ["fork", "owner/source-space"])

        assert result.exit_code == 0
        mock_api.duplicate_space.assert_called_once()
        call_kwargs = mock_api.duplicate_space.call_args[1]
        assert call_kwargs["from_id"] == "owner/source-space"
        assert call_kwargs["private"] is False
        # HF API requires hardware; default to free cpu-basic when not specified
        assert call_kwargs["hardware"] == "cpu-basic"


def test_fork_authenticates_object_whoami_response() -> None:
    """Test that fork accepts object-shaped whoami responses."""
    with (
        patch("openenv.cli.commands.fork.whoami") as mock_whoami,
        patch("openenv.cli.commands.fork.login") as mock_login,
        patch("openenv.cli.commands.fork.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = SimpleNamespace(username="testuser")
        mock_api = MagicMock()
        mock_api.duplicate_space.return_value = (
            "https://huggingface.co/spaces/testuser/source-space"
        )
        mock_hf_api_class.return_value = mock_api

        result = runner.invoke(app, ["fork", "owner/source-space"])

        assert result.exit_code == 0
        mock_login.assert_not_called()
        mock_api.duplicate_space.assert_called_once()


def test_fork_passes_private_and_to_id() -> None:
    """Test that fork passes --private and --repo-id to duplicate_space."""
    with (
        patch("openenv.cli.commands.fork.whoami") as mock_whoami,
        patch("openenv.cli.commands.fork.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_api = MagicMock()
        mock_api.duplicate_space.return_value = (
            "https://huggingface.co/spaces/myuser/my-fork"
        )
        mock_hf_api_class.return_value = mock_api

        result = runner.invoke(
            app,
            ["fork", "owner/source-space", "--private", "--repo-id", "myuser/my-fork"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_api.duplicate_space.call_args[1]
        assert call_kwargs["private"] is True
        assert call_kwargs["to_id"] == "myuser/my-fork"


def test_fork_passes_variables_and_secrets() -> None:
    """Test that fork passes --set-env and --set-secret to duplicate_space."""
    with (
        patch("openenv.cli.commands.fork.whoami") as mock_whoami,
        patch("openenv.cli.commands.fork.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_api = MagicMock()
        mock_api.duplicate_space.return_value = (
            "https://huggingface.co/spaces/testuser/source-space"
        )
        mock_hf_api_class.return_value = mock_api

        result = runner.invoke(
            app,
            [
                "fork",
                "owner/source-space",
                "--set-env",
                "KEY1=val1",
                "--set-secret",
                "SECRET1=secretval",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_api.duplicate_space.call_args[1]
        assert call_kwargs["variables"] == [{"key": "KEY1", "value": "val1"}]
        assert call_kwargs["secrets"] == [{"key": "SECRET1", "value": "secretval"}]


def test_fork_validates_set_env_format() -> None:
    """Test that fork validates KEY=VALUE format for --set-env."""
    with patch("openenv.cli.commands.fork.whoami") as mock_whoami:
        mock_whoami.return_value = {"name": "testuser"}

        result = runner.invoke(
            app,
            ["fork", "owner/source-space", "--set-env", "no-equals-sign"],
        )

        assert result.exit_code != 0
        assert "KEY=VALUE" in result.output or "format" in result.output.lower()


def test_fork_handles_duplicate_space_error() -> None:
    """Test that fork handles duplicate_space API errors."""
    with (
        patch("openenv.cli.commands.fork.whoami") as mock_whoami,
        patch("openenv.cli.commands.fork.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_api = MagicMock()
        mock_api.duplicate_space.side_effect = Exception("Space not found")
        mock_hf_api_class.return_value = mock_api

        result = runner.invoke(app, ["fork", "owner/source-space"])

        assert result.exit_code != 0
        assert "fork" in result.output.lower() or "failed" in result.output.lower()
