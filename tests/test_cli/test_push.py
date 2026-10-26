# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenv push command."""

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from openenv.cli.__main__ import app
from openenv.cli.commands.push import _rewrite_dockerfile_for_space
from typer.testing import CliRunner


runner = CliRunner()
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from CLI output for stable assertions."""
    return ANSI_ESCAPE_RE.sub("", text)


def _create_test_openenv_env(env_dir: Path, env_name: str = "test_env") -> None:
    """Create a complete OpenEnv environment for testing."""
    # Create openenv.yaml
    manifest = {
        "spec_version": 1,
        "name": env_name,
        "type": "space",
        "runtime": "fastapi",
        "app": "server.app:app",
        "port": 8000,
    }
    with open(env_dir / "openenv.yaml", "w") as f:
        yaml.dump(manifest, f)

    # Create pyproject.toml (required by validate_env_structure)
    pyproject_content = f"""[project]
name = "{env_name}"
version = "0.1.0"
dependencies = ["openenv>=0.2.0"]
"""
    (env_dir / "pyproject.toml").write_text(pyproject_content)

    # Create __init__.py
    (env_dir / "__init__.py").write_text("# Test environment\n")

    # Create client.py (required by validate_env_structure)
    (env_dir / "client.py").write_text("# Test client\n")

    # Create models.py
    (env_dir / "models.py").write_text("# Test models\n")

    # Create server directory and files
    (env_dir / "server").mkdir(exist_ok=True)
    (env_dir / "server" / "__init__.py").write_text("# Server module\n")
    (env_dir / "server" / "app.py").write_text("# App module\n")
    (env_dir / "server" / "Dockerfile").write_text(
        'FROM openenv-base:latest\nCMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]\n'
    )

    # Create README.md with frontmatter
    readme_content = """---
title: Test Environment
sdk: docker
app_port: 8000
---

# Test Environment
"""
    (env_dir / "README.md").write_text(readme_content)


def test_push_validates_openenv_directory(tmp_path: Path) -> None:
    """Test that push validates openenv.yaml is present."""
    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)

    assert result.exit_code != 0
    assert (
        "openenv.yaml" in result.output.lower() or "manifest" in result.output.lower()
    )


def test_push_validates_openenv_yaml_format(tmp_path: Path) -> None:
    """Test that push validates openenv.yaml format."""
    # Create complete env structure then overwrite openenv.yaml with invalid content
    _create_test_openenv_env(tmp_path)
    (tmp_path / "openenv.yaml").write_text("invalid: yaml: content: [")

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)

    assert result.exit_code != 0
    assert "parse" in result.output.lower() or "yaml" in result.output.lower()


def test_push_validates_openenv_yaml_has_name(tmp_path: Path) -> None:
    """Test that push validates openenv.yaml has a name field."""
    import yaml

    # Create complete env structure then overwrite openenv.yaml without name
    _create_test_openenv_env(tmp_path)
    manifest = {"spec_version": 1, "type": "space"}
    with open(tmp_path / "openenv.yaml", "w") as f:
        yaml.dump(manifest, f)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)

    assert result.exit_code != 0
    assert "name" in result.output.lower()


def test_push_authenticates_with_hf(tmp_path: Path) -> None:
    """Test that push ensures Hugging Face authentication."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        # Mock whoami to return user info
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt

        # Mock HfApi
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        # Verify whoami was called
        assert mock_whoami.called


def test_push_authenticates_after_login(tmp_path: Path) -> None:
    """Test that push verifies the username after an interactive login."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.side_effect = [Exception("Not authenticated"), {"name": "testuser"}]
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        mock_login.assert_called_once()
        assert mock_api.upload_folder.called


def test_push_enables_web_interface_in_dockerfile(tmp_path: Path) -> None:
    """Test that push enables web interface in Dockerfile."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        # Verify API was called (upload_folder)
        assert mock_api.upload_folder.called


def test_push_updates_readme_frontmatter(tmp_path: Path) -> None:
    """Test that push updates README frontmatter with base_path."""
    _create_test_openenv_env(tmp_path)

    # Create README without base_path
    readme_content = """---
title: Test Environment
sdk: docker
app_port: 8000
---

# Test Environment
"""
    (tmp_path / "README.md").write_text(readme_content)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        # Verify API was called
        assert mock_api.upload_folder.called


def test_push_uses_repo_id_option(tmp_path: Path) -> None:
    """Test that push respects --repo-id option."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--repo-id", "custom-org/my-env"])
        finally:
            os.chdir(old_cwd)

        # Verify create_repo was called with correct repo_id
        mock_api.create_repo.assert_called_once()
        call_args = mock_api.create_repo.call_args
        assert call_args.kwargs["repo_id"] == "custom-org/my-env"


def test_push_uses_default_repo_id(tmp_path: Path) -> None:
    """Test that push uses default repo-id from username and env name."""
    _create_test_openenv_env(tmp_path, env_name="test_env")

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        # Verify create_repo was called with default repo_id
        mock_api.create_repo.assert_called_once()
        call_args = mock_api.create_repo.call_args
        assert call_args.kwargs["repo_id"] == "testuser/test_env"


def test_push_uses_private_option(tmp_path: Path) -> None:
    """Test that push respects --private option."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--private"])
        finally:
            os.chdir(old_cwd)

        # Verify create_repo was called with private=True
        mock_api.create_repo.assert_called_once()
        call_args = mock_api.create_repo.call_args
        assert call_args.kwargs["private"] is True


def test_push_uses_hardware_option(tmp_path: Path) -> None:
    """Test that push respects --hardware option."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--hardware", "t4-medium"])
        finally:
            os.chdir(old_cwd)

        # Verify create_repo was called with space_hardware="t4-medium"
        mock_api.create_repo.assert_called_once()
        call_kwargs = mock_api.create_repo.call_args.kwargs
        assert call_kwargs["space_hardware"] == "t4-medium"


def test_push_default_hardware_is_none(tmp_path: Path) -> None:
    """Test that push does not pass space_hardware when --hardware is not specified."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        # Verify create_repo was called without space_hardware
        mock_api.create_repo.assert_called_once()
        call_kwargs = mock_api.create_repo.call_args.kwargs
        assert "space_hardware" not in call_kwargs


def test_push_uses_base_image_option(tmp_path: Path) -> None:
    """Test that push respects --base-image option."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--base-image", "custom-base:latest"])
        finally:
            os.chdir(old_cwd)

        # Verify API was called (we can't easily test Dockerfile modification without reading staging dir)
        assert mock_api.upload_folder.called


def test_push_uses_directory_argument(tmp_path: Path) -> None:
    """Test that push respects directory argument."""
    env_dir = tmp_path / "my_env"
    env_dir.mkdir()
    _create_test_openenv_env(env_dir)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        # Directory is a positional argument, not an option
        result = runner.invoke(
            app,
            ["push", str(env_dir)],
        )

        # Verify API was called
        assert mock_api.upload_folder.called


def test_push_accepts_dockerfile_at_env_root(tmp_path: Path) -> None:
    """Test that push works when Dockerfile is at environment root instead of server/."""
    _create_test_openenv_env(tmp_path)
    # Move Dockerfile from server/ to env root
    root_dockerfile = tmp_path / "Dockerfile"
    (tmp_path / "server" / "Dockerfile").rename(root_dockerfile)

    staged_files: list[list[str]] = []

    def _capture_staging(*, folder_path: str, **_: object) -> None:
        staging = Path(folder_path)
        staged_files.append(
            sorted(
                str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()
            )
        )

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_api.upload_folder.side_effect = _capture_staging
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert mock_api.upload_folder.called

        # Verify the staging directory has Dockerfile at root, not inside server/
        files = staged_files[0]
        assert "Dockerfile" in files
        assert "server/Dockerfile" not in files


def test_push_accepts_env_without_models(tmp_path: Path) -> None:
    """Test that push works without models.py, as in MCP environments like echo_env."""
    _create_test_openenv_env(tmp_path)
    (tmp_path / "models.py").unlink()

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert mock_api.upload_folder.called


def test_push_handles_missing_dockerfile(tmp_path: Path) -> None:
    """Test that push fails when Dockerfile is missing (required for deployment)."""
    _create_test_openenv_env(tmp_path)
    # Remove Dockerfile (no root Dockerfile either)
    (tmp_path / "server" / "Dockerfile").unlink()

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)

    # Dockerfile is now required - should fail
    assert result.exit_code != 0
    assert "dockerfile" in result.output.lower() or "missing" in result.output.lower()


def test_push_handles_missing_readme(tmp_path: Path) -> None:
    """Test that push fails when README.md is missing (required for deployment)."""
    _create_test_openenv_env(tmp_path)
    # Remove README
    (tmp_path / "README.md").unlink()

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)

    # README.md is now required - should fail
    assert result.exit_code != 0
    assert "readme" in result.output.lower() or "missing" in result.output.lower()


def test_push_initializes_hf_api_without_token(tmp_path: Path) -> None:
    """Test that push initializes HfApi without token parameter."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        # Verify HfApi was initialized without token parameter
        mock_hf_api_class.assert_called_once()
        call_args = mock_hf_api_class.call_args
        # Should not have token in kwargs
        assert "token" not in (call_args.kwargs or {})


def test_push_validates_repo_id_format(tmp_path: Path) -> None:
    """Test that push rejects repo-ids with more than one slash."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_hf_api_class.return_value = MagicMock()

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--repo-id", "org/repo/extra"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code != 0
        assert "repo-id" in result.output.lower() or "format" in result.output.lower()


def test_push_bare_repo_id_expands_to_username(tmp_path: Path) -> None:
    """Bare repo-name (no slash) is expanded to username/repo-name before push."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
        patch("openenv.cli.commands.push._upload_to_hf_space") as mock_upload,
        patch("openenv.cli.commands.push._create_hf_space") as mock_create,
        patch("openenv.cli.commands.push._prepare_staging_directory") as _mock_stage,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_hf_api_class.return_value = MagicMock()
        mock_create.return_value = None
        mock_upload.return_value = None

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--repo-id", "my-env"])
        finally:
            os.chdir(old_cwd)

        assert "Invalid repo-id format" not in result.output
        mock_create.assert_called_once()
        assert mock_create.call_args.args[0] == "testuser/my-env"
        assert "testuser/my-env" in result.output


def test_push_validates_manifest_is_dict(tmp_path: Path) -> None:
    """Test that push validates manifest is a dictionary."""
    import yaml

    # Create complete env structure then overwrite openenv.yaml with non-dict
    _create_test_openenv_env(tmp_path)
    with open(tmp_path / "openenv.yaml", "w") as f:
        yaml.dump("not a dict", f)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)

    assert result.exit_code != 0
    assert "dictionary" in result.output.lower() or "yaml" in result.output.lower()


def test_push_handles_whoami_object_return(tmp_path: Path) -> None:
    """Test that push handles whoami returning an object instead of dict."""
    _create_test_openenv_env(tmp_path)

    # Create a mock object with name attribute
    class MockUser:
        def __init__(self):
            self.name = "testuser"

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = MockUser()
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        # Verify it worked with object return type
        assert mock_api.upload_folder.called


def test_push_handles_authentication_failure(tmp_path: Path) -> None:
    """Test that push handles authentication failure."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        # First whoami call fails (not authenticated)
        # Login also fails
        mock_whoami.side_effect = Exception("Not authenticated")
        mock_login.side_effect = Exception("Login failed")
        # Mock HfApi to prevent actual API calls
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code != 0
        assert (
            "authentication" in result.output.lower()
            or "login" in result.output.lower()
        )


def test_push_handles_whoami_missing_username(tmp_path: Path) -> None:
    """Test that push handles whoami response without username."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        # Return dict without name, fullname, or username
        mock_whoami.return_value = {}
        # Mock login to prevent actual login prompt
        mock_login.return_value = None
        # Mock HfApi to prevent actual API calls
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code != 0
        assert "username" in result.output.lower() or "extract" in result.output.lower()


def test_push_handles_readme_without_frontmatter(tmp_path: Path) -> None:
    """Test that push handles README without frontmatter."""
    _create_test_openenv_env(tmp_path)

    # Create README without frontmatter
    (tmp_path / "README.md").write_text("# Test Environment\nNo frontmatter here.\n")

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        # Verify it still works (should add frontmatter)
        assert mock_api.upload_folder.called


def test_push_handles_hf_api_create_repo_error(tmp_path: Path) -> None:
    """Test that push handles HF API create_repo error."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_api.create_repo.side_effect = Exception("API Error")
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            # Should continue despite error (warns but doesn't fail)
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        # Should still attempt upload
        assert mock_api.upload_folder.called


def test_push_handles_hf_api_upload_error(tmp_path: Path) -> None:
    """Test that push handles HF API upload_folder error."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_api.upload_folder.side_effect = Exception("Upload failed")
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code != 0
        assert "upload" in result.output.lower() or "failed" in result.output.lower()


def test_push_handles_base_image_not_found_in_dockerfile(tmp_path: Path) -> None:
    """Test that push handles Dockerfile without FROM line."""
    _create_test_openenv_env(tmp_path)

    # Create Dockerfile without FROM line
    (tmp_path / "server" / "Dockerfile").write_text(
        'RUN echo \'test\'\nCMD ["echo", "test"]\n'
    )

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--base-image", "custom-base:latest"])
        finally:
            os.chdir(old_cwd)

        # Should still work (adds FROM at beginning)
        assert mock_api.upload_folder.called


def test_rewrite_dockerfile_skips_healthcheck_cmd() -> None:
    """Test that web-interface ENV is not inserted before a HEALTHCHECK CMD."""
    dockerfile = "\n".join(
        [
            "FROM openenv-base:latest",
            "HEALTHCHECK --interval=30s \\",
            "  CMD curl -f http://localhost:8000/health || exit 1",
            'CMD ["uvicorn", "server.app:app"]',
        ]
    )

    rewritten, changes = _rewrite_dockerfile_for_space(
        dockerfile,
        base_image=None,
        enable_interface=True,
    )

    assert changes == ["enabled web interface"]
    assert (
        "HEALTHCHECK --interval=30s \\\n"
        "  CMD curl -f http://localhost:8000/health || exit 1\n"
        "ENV ENABLE_WEB_INTERFACE=true\n"
        'CMD ["uvicorn", "server.app:app"]'
    ) in rewritten


def test_push_excludes_files_from_ignore_file(tmp_path: Path) -> None:
    """Test that push excludes files using patterns loaded via --exclude."""
    _create_test_openenv_env(tmp_path)

    # Create files/folders to verify exclusion behavior.
    (tmp_path / "excluded_dir").mkdir()
    (tmp_path / "excluded_dir" / "secret.txt").write_text("do not upload")
    (tmp_path / "weights.bin").write_text("binary payload")
    (tmp_path / "keep.txt").write_text("keep me")

    ignore_file = tmp_path / ".openenvignore"
    ignore_file.write_text(
        """
# comments and empty lines are ignored
excluded_dir/
*.bin
"""
    )

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        def _assert_upload_payload(*_unused_args, **kwargs):
            ignore_patterns = kwargs["ignore_patterns"]
            assert "excluded_dir/" in ignore_patterns
            assert "*.bin" in ignore_patterns
            assert ".*" in ignore_patterns

            staged = Path(kwargs["folder_path"])
            assert not (staged / "excluded_dir").exists()
            assert not (staged / "weights.bin").exists()
            assert (staged / "keep.txt").exists()

        mock_api.upload_folder.side_effect = _assert_upload_payload

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app,
                ["push", "--exclude", ".openenvignore"],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        assert mock_api.upload_folder.called


def test_push_does_not_use_gitignore_as_default_excludes(tmp_path: Path) -> None:
    """Test that .gitignore patterns are not used by default."""
    _create_test_openenv_env(tmp_path)
    (tmp_path / ".gitignore").write_text("excluded_from_gitignore/\n")
    (tmp_path / "excluded_from_gitignore").mkdir()
    (tmp_path / "excluded_from_gitignore" / "secret.txt").write_text("upload me")
    (tmp_path / "keep.txt").write_text("keep me")

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        def _assert_upload_payload(*_unused_args, **kwargs):
            ignore_patterns = kwargs["ignore_patterns"]
            assert "excluded_from_gitignore/" not in ignore_patterns

            staged = Path(kwargs["folder_path"])
            assert (staged / "excluded_from_gitignore").exists()
            assert (staged / "keep.txt").exists()

        mock_api.upload_folder.side_effect = _assert_upload_payload

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        assert mock_api.upload_folder.called


def test_push_excludes_build_artifacts_by_default(tmp_path: Path) -> None:
    """Test that local build artifacts are never uploaded to the Space."""
    _create_test_openenv_env(tmp_path)
    (tmp_path / "build" / "lib" / "test_env").mkdir(parents=True)
    (tmp_path / "build" / "lib" / "test_env" / "client.py").write_text("# stale\n")
    (tmp_path / "test_env.egg-info").mkdir()
    (tmp_path / "test_env.egg-info" / "PKG-INFO").write_text("Name: test_env\n")

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        def _assert_upload_payload(*_unused_args, **kwargs):
            staged = Path(kwargs["folder_path"])
            assert not (staged / "build").exists()
            assert not (staged / "test_env.egg-info").exists()
            assert (staged / "client.py").exists()

        mock_api.upload_folder.side_effect = _assert_upload_payload

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        assert mock_api.upload_folder.called


def test_push_deletes_remote_files_only_with_delete(tmp_path: Path) -> None:
    """Test that push deletes nothing by default and forwards --delete patterns, as `hf upload` does."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            default = runner.invoke(app, ["push"])
            default_kwargs = mock_api.upload_folder.call_args[1]
            with_delete = runner.invoke(
                app, ["push", "--delete", "*.py", "--delete", "server/*"]
            )
            delete_kwargs = mock_api.upload_folder.call_args[1]
        finally:
            os.chdir(old_cwd)

        assert default.exit_code == 0
        assert default_kwargs["delete_patterns"] is None
        assert with_delete.exit_code == 0
        assert delete_kwargs["delete_patterns"] == ["*.py", "server/*"]


def test_push_fails_when_exclude_file_missing(tmp_path: Path) -> None:
    """Test that push fails if --exclude points to a missing file."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app,
                ["push", "--exclude", "missing.ignore"],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code != 0
        assert "exclude file" in result.output.lower()


def test_push_create_pr_sets_upload_flag_and_skips_create_repo(tmp_path: Path) -> None:
    """Test that --create-pr uploads with PR mode and skips repo creation."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app, ["push", "--repo-id", "my-org/my-env", "--create-pr"]
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        mock_api.upload_folder.assert_called_once()
        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs.get("create_pr") is True
        # When create_pr we do not create the repo (target repo must exist)
        mock_api.create_repo.assert_not_called()


def test_push_count_deploys_multiple_spaces(tmp_path: Path) -> None:
    """Test that --count 3 calls create_repo and upload_folder 3 times with suffixed repo IDs."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app,
                ["push", "--repo-id", "testuser/my-env", "--count", "3"],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0

        # Verify create_repo was called 3 times with suffixed repo IDs
        assert mock_api.create_repo.call_count == 3
        create_repo_ids = [
            call.kwargs["repo_id"] for call in mock_api.create_repo.call_args_list
        ]
        assert create_repo_ids == [
            "testuser/my-env-1",
            "testuser/my-env-2",
            "testuser/my-env-3",
        ]

        # Verify upload_folder was called 3 times with suffixed repo IDs
        assert mock_api.upload_folder.call_count == 3
        upload_repo_ids = [
            call.kwargs["repo_id"] for call in mock_api.upload_folder.call_args_list
        ]
        assert upload_repo_ids == [
            "testuser/my-env-1",
            "testuser/my-env-2",
            "testuser/my-env-3",
        ]

        # Verify progress messages in output
        clean_output = _strip_ansi(result.output)
        assert "[1/3]" in clean_output
        assert "[2/3]" in clean_output
        assert "[3/3]" in clean_output
        assert "All 3 instances deployed" in clean_output


def test_push_count_one_is_default_behavior(tmp_path: Path) -> None:
    """Test that --count 1 behaves exactly like no flag (no suffix)."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app,
                ["push", "--repo-id", "testuser/my-env", "--count", "1"],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0

        # Verify create_repo was called once without suffix
        mock_api.create_repo.assert_called_once()
        assert mock_api.create_repo.call_args.kwargs["repo_id"] == "testuser/my-env"

        # Verify upload_folder was called once without suffix
        mock_api.upload_folder.assert_called_once()
        assert mock_api.upload_folder.call_args.kwargs["repo_id"] == "testuser/my-env"


def test_push_count_with_registry_errors(tmp_path: Path) -> None:
    """Test that --count 2 --registry fails with error."""
    _create_test_openenv_env(tmp_path)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(
            app,
            ["push", "--count", "2", "--registry", "docker.io/user"],
        )
    finally:
        os.chdir(old_cwd)

    assert result.exit_code != 0
    assert "--count" in result.output.lower() or "count" in result.output.lower()
    assert "--registry" in result.output.lower() or "registry" in result.output.lower()


def test_push_count_with_create_pr_errors(tmp_path: Path) -> None:
    """Test that --count 2 --create-pr fails with error."""
    _create_test_openenv_env(tmp_path)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(
            app,
            [
                "push",
                "--repo-id",
                "testuser/my-env",
                "--count",
                "2",
                "--create-pr",
            ],
        )
    finally:
        os.chdir(old_cwd)

    assert result.exit_code != 0
    assert "--count" in result.output.lower() or "count" in result.output.lower()
    assert (
        "--create-pr" in result.output.lower() or "create-pr" in result.output.lower()
    )


def test_push_sets_space_variable_from_cli(tmp_path: Path) -> None:
    """-e KEY=VALUE triggers add_space_variable after upload."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app,
                ["push", "-e", "OPENSPIEL_GAME=tic_tac_toe"],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        mock_api.add_space_variable.assert_called_once()
        call_kwargs = mock_api.add_space_variable.call_args.kwargs
        assert call_kwargs["repo_id"] == "testuser/test_env"
        assert call_kwargs["key"] == "OPENSPIEL_GAME"
        assert call_kwargs["value"] == "tic_tac_toe"
        mock_api.add_space_secret.assert_not_called()


def test_push_sets_space_secret_from_cli(tmp_path: Path) -> None:
    """--secret KEY=VALUE triggers add_space_secret after upload."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app,
                ["push", "--secret", "OPENAI_API_KEY=sk-super-secret-123"],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        mock_api.add_space_secret.assert_called_once()
        call_kwargs = mock_api.add_space_secret.call_args.kwargs
        assert call_kwargs["repo_id"] == "testuser/test_env"
        assert call_kwargs["key"] == "OPENAI_API_KEY"
        assert call_kwargs["value"] == "sk-super-secret-123"
        mock_api.add_space_variable.assert_not_called()


def test_push_secret_value_is_not_logged(tmp_path: Path) -> None:
    """Secret values must never appear in CLI output (only the key)."""
    _create_test_openenv_env(tmp_path)

    secret_value = "sk-super-secret-do-not-leak"

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app,
                ["push", "--secret", f"OPENAI_API_KEY={secret_value}"],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert secret_value not in result.output
        # Key itself should be mentioned so the user sees confirmation
        assert "OPENAI_API_KEY" in result.output


def test_push_loads_variables_from_openenv_yaml(tmp_path: Path) -> None:
    """variables: in openenv.yaml are applied via add_space_variable."""
    _create_test_openenv_env(tmp_path)
    # Overwrite openenv.yaml adding a variables section
    manifest = {
        "spec_version": 1,
        "name": "test_env",
        "type": "space",
        "runtime": "fastapi",
        "app": "server.app:app",
        "port": 8000,
        "variables": {
            "OPENSPIEL_GAME": "tic_tac_toe",
            "MAX_STEPS": "100",
        },
    }
    with open(tmp_path / "openenv.yaml", "w") as f:
        yaml.dump(manifest, f)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert mock_api.add_space_variable.call_count == 2
        vars_seen = {
            call.kwargs["key"]: call.kwargs["value"]
            for call in mock_api.add_space_variable.call_args_list
        }
        assert vars_seen == {
            "OPENSPIEL_GAME": "tic_tac_toe",
            "MAX_STEPS": "100",
        }


def test_push_cli_env_var_overrides_yaml(tmp_path: Path) -> None:
    """-e on CLI overrides the same key from openenv.yaml variables."""
    _create_test_openenv_env(tmp_path)
    manifest = {
        "spec_version": 1,
        "name": "test_env",
        "type": "space",
        "runtime": "fastapi",
        "app": "server.app:app",
        "port": 8000,
        "variables": {"OPENSPIEL_GAME": "catch"},
    }
    with open(tmp_path / "openenv.yaml", "w") as f:
        yaml.dump(manifest, f)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app,
                ["push", "-e", "OPENSPIEL_GAME=tic_tac_toe"],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        # Exactly one call for OPENSPIEL_GAME with CLI value
        assert mock_api.add_space_variable.call_count == 1
        kwargs = mock_api.add_space_variable.call_args.kwargs
        assert kwargs["key"] == "OPENSPIEL_GAME"
        assert kwargs["value"] == "tic_tac_toe"


def test_push_rejects_env_var_without_equals(tmp_path: Path) -> None:
    """-e KEY (no =) must fail with a clear error and not call HfApi."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "-e", "NO_EQUALS_HERE"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code != 0
        assert (
            "key=value" in result.output.lower()
            or "invalid" in result.output.lower()
            or "format" in result.output.lower()
        )
        mock_api.add_space_variable.assert_not_called()


def test_push_rejects_secret_without_equals_without_leaking_value(
    tmp_path: Path,
) -> None:
    """Malformed --secret must fail without echoing the secret value."""
    _create_test_openenv_env(tmp_path)

    secret_value = "sk-super-secret-do-not-leak"

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--secret", secret_value])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code != 0
        assert secret_value not in result.output
        assert "key=value" in result.output.lower()
        mock_api.add_space_secret.assert_not_called()


def test_push_rejects_env_var_with_empty_key(tmp_path: Path) -> None:
    """-e =VALUE must fail with a clear error."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "-e", "=value_only"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code != 0
        mock_api.add_space_variable.assert_not_called()


def test_push_env_var_value_can_contain_equals(tmp_path: Path) -> None:
    """-e KEY=a=b=c splits only on first '=' so values with '=' are preserved."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "-e", "TOKEN=a=b=c"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        kwargs = mock_api.add_space_variable.call_args.kwargs
        assert kwargs["key"] == "TOKEN"
        assert kwargs["value"] == "a=b=c"


def test_push_no_variables_is_backward_compatible(tmp_path: Path) -> None:
    """Without -e/--secret and without variables: in yaml, no HF var APIs are called."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        mock_api.add_space_variable.assert_not_called()
        mock_api.add_space_secret.assert_not_called()


def test_push_count_applies_variables_to_all_instances(tmp_path: Path) -> None:
    """--count N pushes variables and secrets to each of the N instances."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(
                app,
                [
                    "push",
                    "--repo-id",
                    "testuser/my-env",
                    "--count",
                    "3",
                    "-e",
                    "OPENSPIEL_GAME=tic_tac_toe",
                    "--secret",
                    "TOKEN=abc",
                ],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        var_repo_ids = sorted(
            call.kwargs["repo_id"]
            for call in mock_api.add_space_variable.call_args_list
        )
        secret_repo_ids = sorted(
            call.kwargs["repo_id"] for call in mock_api.add_space_secret.call_args_list
        )
        assert var_repo_ids == [
            "testuser/my-env-1",
            "testuser/my-env-2",
            "testuser/my-env-3",
        ]
        assert secret_repo_ids == [
            "testuser/my-env-1",
            "testuser/my-env-2",
            "testuser/my-env-3",
        ]


def test_push_rejects_non_mapping_variables_block(tmp_path: Path) -> None:
    """Falsey non-dict variables blocks must still be rejected."""
    _create_test_openenv_env(tmp_path)
    manifest = {
        "spec_version": 1,
        "name": "test_env",
        "type": "space",
        "runtime": "fastapi",
        "app": "server.app:app",
        "port": 8000,
        "variables": [],
    }
    with open(tmp_path / "openenv.yaml", "w") as f:
        yaml.dump(manifest, f)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)

    assert result.exit_code != 0
    assert "variables" in result.output.lower()
    assert "mapping" in result.output.lower()


def test_push_registry_rejects_space_settings_flags(tmp_path: Path) -> None:
    """Custom registry pushes must reject Space-only flags."""
    _create_test_openenv_env(tmp_path)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(
            app,
            [
                "push",
                "--registry",
                "docker.io/testuser",
                "-e",
                "OPENSPIEL_GAME=tic_tac_toe",
            ],
        )
    finally:
        os.chdir(old_cwd)

    clean_output = _strip_ansi(result.output)
    assert result.exit_code != 0
    assert "--registry" in clean_output
    assert "--env-var" in clean_output


def test_push_create_pr_rejects_space_settings_flags(tmp_path: Path) -> None:
    """PR uploads cannot apply Space-only flags before merge."""
    _create_test_openenv_env(tmp_path)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(
            app,
            [
                "push",
                "--repo-id",
                "testuser/test_env",
                "--create-pr",
                "--secret",
                "OPENAI_API_KEY=sk-super-secret",
            ],
        )
    finally:
        os.chdir(old_cwd)

    clean_output = _strip_ansi(result.output)
    assert result.exit_code != 0
    assert "--create-pr" in clean_output
    assert "--secret" in clean_output


def test_push_exits_cleanly_if_setting_space_variable_fails(tmp_path: Path) -> None:
    """HF variable configuration errors should not surface a traceback."""
    _create_test_openenv_env(tmp_path)

    with (
        patch("openenv.cli.commands.push.whoami") as mock_whoami,
        patch("openenv.cli.commands.push.login") as mock_login,
        patch("openenv.cli.commands.push.HfApi") as mock_hf_api_class,
    ):
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None
        mock_api = MagicMock()
        mock_api.add_space_variable.side_effect = RuntimeError("permission denied")
        mock_hf_api_class.return_value = mock_api

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "-e", "OPENSPIEL_GAME=catch"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code != 0
        assert "failed to set variable openspiel_game" in result.output.lower()
        assert "permission denied" in result.output.lower()
        assert "traceback" not in result.output.lower()
