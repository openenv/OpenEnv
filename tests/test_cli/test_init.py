# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenv init command."""

import asyncio
import importlib
import importlib.util
import os
import sys
from pathlib import Path

from openenv.cli.__main__ import app
from openenv.cli.commands.init import _resolve_template_path
from typer.testing import CliRunner


runner = CliRunner()


def _snake_to_pascal(snake_str: str) -> str:
    """Helper function matching the one in init.py"""
    return "".join(word.capitalize() for word in snake_str.split("_"))


def test_init_creates_directory_structure(tmp_path: Path) -> None:
    """Test that init creates the correct directory structure."""
    env_name = "test_env"
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0
    assert env_dir.exists()
    assert env_dir.is_dir()

    # Check for required files
    assert (env_dir / "__init__.py").exists()
    assert (env_dir / "models.py").exists()
    assert (env_dir / "client.py").exists()
    assert (env_dir / "README.md").exists()
    assert (env_dir / "openenv.yaml").exists()
    assert (env_dir / "server").exists()
    assert (env_dir / "server" / "__init__.py").exists()
    assert (env_dir / "server" / "app.py").exists()
    assert (env_dir / "server" / f"{env_name}_environment.py").exists()
    assert (env_dir / "server" / "Dockerfile").exists()
    assert (env_dir / "server" / "requirements.txt").exists()


def test_init_replaces_template_placeholders(tmp_path: Path) -> None:
    """Test that template placeholders are replaced correctly."""
    env_name = "my_game_env"
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    # Check models.py has correct class names
    # For 'my_game_env', prefix is 'MyGame' (removes trailing '_env')
    models_content = (env_dir / "models.py").read_text()
    assert "MyGameAction" in models_content
    assert "MyGameObservation" in models_content
    assert "__ENV_NAME__" not in models_content
    assert "__ENV_CLASS_NAME__" not in models_content

    # Check client.py has correct class names
    client_content = (env_dir / "client.py").read_text()
    assert "MyGameEnv" in client_content
    assert "MyGameAction" in client_content
    assert "MyGameObservation" in client_content
    assert "__ENV_NAME__" not in client_content

    # Check __init__.py has correct exports
    init_content = (env_dir / "__init__.py").read_text()
    assert "MyGameAction" in init_content
    assert "MyGameObservation" in init_content
    assert "MyGameEnv" in init_content

    # Check environment file has correct class name
    env_file = env_dir / "server" / f"{env_name}_environment.py"
    assert env_file.exists()
    env_content = env_file.read_text()
    assert "MyGameEnvironment" in env_content
    assert "__ENV_CLASS_NAME__" not in env_content


def test_init_generates_openenv_yaml(tmp_path: Path) -> None:
    """Test that openenv.yaml is generated correctly."""
    env_name = "test_env"
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    yaml_file = env_dir / "openenv.yaml"
    assert yaml_file.exists()

    yaml_content = yaml_file.read_text()
    assert f"name: {env_name}" in yaml_content
    assert "type: space" in yaml_content
    assert "runtime: fastapi" in yaml_content
    assert "app: server.app:app" in yaml_content
    assert "port: 8000" in yaml_content
    assert "validation:" in yaml_content
    assert "__ENV_NAME__" not in yaml_content

    validate_result = runner.invoke(
        app,
        [
            "validate",
            str(env_dir),
            "--level",
            "static",
            "--skip-build",
        ],
    )
    assert validate_result.exit_code == 0, validate_result.output
    assert "Verdict: PASS" in validate_result.output

    environment_file = env_dir / "server" / f"{env_name}_environment.py"
    assert "min(length * 0.1, 1.0)" in environment_file.read_text()


def test_init_readme_has_hf_frontmatter(tmp_path: Path) -> None:
    """Test that README has Hugging Face Space compatible frontmatter."""
    env_name = "test_env"
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    readme_file = env_dir / "README.md"
    assert readme_file.exists()

    readme_content = readme_file.read_text()

    # Check for required HF Space frontmatter
    assert "---" in readme_content
    assert "title:" in readme_content
    assert "sdk: docker" in readme_content
    assert "app_port: 8000" in readme_content
    assert "tags:" in readme_content
    assert "- openenv" in readme_content

    # Check that placeholders are replaced
    assert "__ENV_NAME__" not in readme_content
    assert "__ENV_TITLE_NAME__" not in readme_content


def test_init_validates_env_name(tmp_path: Path) -> None:
    """Test that invalid environment names are rejected."""
    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        # Invalid: starts with number
        result = runner.invoke(app, ["init", "123_env"], input="\n")
        assert result.exit_code != 0
        assert (
            "not a valid python identifier" in result.output.lower()
            or "not a valid identifier" in result.output.lower()
        )

        # Invalid: contains spaces
        result = runner.invoke(app, ["init", "my env"], input="\n")
        assert result.exit_code != 0

        # Invalid: contains hyphens
        result = runner.invoke(app, ["init", "my-env"], input="\n")
        assert result.exit_code != 0
    finally:
        os.chdir(old_cwd)


def test_init_handles_existing_directory(tmp_path: Path) -> None:
    """Test that init fails gracefully when directory exists."""
    env_name = "existing_env"
    env_dir = tmp_path / env_name
    env_dir.mkdir()
    (env_dir / "some_file.txt").write_text("existing content")

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code != 0
    assert (
        "already exists" in result.output.lower()
        or "not empty" in result.output.lower()
    )


def test_init_handles_empty_directory(tmp_path: Path) -> None:
    """Test that init works when directory exists but is empty."""
    env_name = "empty_env"
    env_dir = tmp_path / env_name
    env_dir.mkdir()

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    # Should work - empty directory is okay
    assert result.exit_code == 0
    assert (env_dir / "models.py").exists()


def test_init_with_output_dir(tmp_path: Path) -> None:
    """Test that init works with custom output directory."""
    env_name = "output_env"
    output_dir = tmp_path / "custom_output"
    output_dir.mkdir()
    env_dir = output_dir / env_name

    result = runner.invoke(
        app,
        ["init", env_name, "--output-dir", str(output_dir)],
        input="\n",
    )

    assert result.exit_code == 0
    assert env_dir.exists()
    assert (env_dir / "models.py").exists()


def test_resolve_template_path_root() -> None:
    """Test resolving the packaged init template root."""
    template_path = _resolve_template_path("openenv.cli.templates.openenv_env", "")

    assert template_path.is_dir()
    assert (template_path / "models.py").exists()


def test_resolve_template_path_nested_directory() -> None:
    """Test resolving a nested init template directory."""
    template_path = _resolve_template_path(
        "openenv.cli.templates.openenv_env", "server"
    )

    assert template_path.is_dir()
    assert (template_path / "Dockerfile").exists()


def test_init_filename_templating(tmp_path: Path) -> None:
    """Test that filenames with placeholders are renamed correctly."""
    env_name = "test_env"
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    # Check that environment file is renamed correctly
    env_file = env_dir / "server" / f"{env_name}_environment.py"
    assert env_file.exists()

    # Check that __ENV_NAME___environment.py doesn't exist (should be renamed)
    template_name = env_dir / "server" / "__ENV_NAME___environment.py"
    assert not template_name.exists()


def test_init_all_naming_conventions(tmp_path: Path) -> None:
    """Test that all naming conventions are replaced correctly."""
    env_name = "complex_test_env"
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    # Check PascalCase
    # For 'complex_test_env', prefix is 'ComplexTest' (removes trailing '_env')
    models_content = (env_dir / "models.py").read_text()
    assert "ComplexTestAction" in models_content
    assert "ComplexTestObservation" in models_content

    # Check snake_case in imports
    assert env_name in models_content  # Should see snake_case module name

    # Check Title Case in README
    readme_content = (env_dir / "README.md").read_text()
    assert (
        "Complex Test Env" in readme_content
        or env_name.lower() in readme_content.lower()
    )


def test_init_server_app_imports(tmp_path: Path) -> None:
    """Test that server/app.py has correct imports after templating."""
    env_name = "test_env"
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    app_content = (env_dir / "server" / "app.py").read_text()

    # Check imports use correct class names
    # For 'test_env', prefix is 'Test' (removes trailing '_env')
    # Template uses direct imports (PYTHONPATH includes env dir in Docker)
    assert f"from .{env_name}_environment import" in app_content
    assert "from models import" in app_content  # Direct import for Docker compatibility
    assert "TestEnvironment" in app_content  # Prefix is 'Test', not 'TestEnv'
    assert "TestAction" in app_content  # Prefix is 'Test', not 'TestEnv'
    assert "TestObservation" in app_content  # Prefix is 'Test', not 'TestEnv'

    # Check that no template placeholders remain
    assert "__ENV_NAME__" not in app_content
    assert "__ENV_CLASS_NAME__" not in app_content


def test_init_server_main_parses_host_and_port(tmp_path: Path, monkeypatch) -> None:
    """Test that the generated `server` entry point honors --host and --port."""
    import uvicorn

    env_name = "demo_env"
    result = runner.invoke(app, ["init", env_name, "--output-dir", str(tmp_path)])
    assert result.exit_code == 0

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv", ["server", "--host", "127.0.0.1", "--port", "8001"]
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    importlib.import_module(f"{env_name}.server.app").main()

    assert calls == [{"host": "127.0.0.1", "port": 8001}]


def test_init_next_steps_use_generated_names(tmp_path: Path) -> None:
    """Test that the printed next steps and image tags match the generated env."""
    env_name = "demo_env"
    result = runner.invoke(app, ["init", env_name, "--output-dir", str(tmp_path)])
    assert result.exit_code == 0

    output = " ".join(result.output.split())
    assert "envs/demo_env_env" not in output
    assert "demo_env_env:latest" not in output
    assert "Copy this directory to <repo_root>/envs/demo_env" in output
    assert "openenv build envs/demo_env" in output
    assert "docker run -p 8000:8000 openenv-demo:latest" in output

    env_dir = tmp_path / env_name
    for path in [env_dir / "README.md", env_dir / "client.py"]:
        content = path.read_text()
        assert "openenv-demo:latest" in content
        assert "demo_env-env" not in content


def test_init_dockerfile_uses_correct_base(tmp_path: Path) -> None:
    """Test that Dockerfile uses correct base image and paths."""
    env_name = "test_env"
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    dockerfile = env_dir / "server" / "Dockerfile"
    assert dockerfile.exists()

    dockerfile_content = dockerfile.read_text()

    # Check base image
    assert "ghcr.io/huggingface/openenv-base:latest" in dockerfile_content

    # Check CMD uses correct module path (could be in list format or string format)
    assert "server.app:app" in dockerfile_content

    # Check that no template placeholders remain
    assert "__ENV_NAME__" not in dockerfile_content


def test_init_requirements_file(tmp_path: Path) -> None:
    """Test that requirements.txt is generated correctly."""
    env_name = "test_env"
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    requirements = env_dir / "server" / "requirements.txt"
    assert requirements.exists()

    req_content = requirements.read_text()
    assert "fastapi" in req_content
    assert "uvicorn" in req_content
    assert "openenv>=0.2.0" in req_content


def test_init_validates_empty_env_name(tmp_path: Path) -> None:
    """Test that init validates empty environment name."""
    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", ""], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code != 0
    assert "cannot be empty" in result.output.lower()


def test_init_env_name_without_env_suffix(tmp_path: Path) -> None:
    """Test that init works with env names that don't end with _env."""
    env_name = "mygame"  # No _env suffix
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0
    assert env_dir.exists()

    # Check that prefix is correctly derived (should be "Mygame" for "mygame")
    models_content = (env_dir / "models.py").read_text()
    assert "MygameAction" in models_content or "Mygame" in models_content


def test_init_single_part_env_name(tmp_path: Path) -> None:
    """Test that init works with single-part env names."""
    env_name = "game"  # Single part, no underscores
    env_dir = tmp_path / env_name

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0
    assert env_dir.exists()


def test_init_handles_file_path_collision(tmp_path: Path) -> None:
    """Test that init fails when path exists as a file."""
    env_name = "existing_file"
    file_path = tmp_path / env_name
    file_path.write_text("existing file content")

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["init", env_name], input="\n")
    finally:
        os.chdir(old_cwd)

    # The command should fail with exit code 2 (typer bad parameter)
    assert result.exit_code != 0, (
        f"Expected command to fail, but it succeeded. Output: {result.output}"
    )
    # Check that it's a BadParameter error (exit code 2) and not just a usage error
    # Typer formats BadParameter errors in the Error section
    error_output = result.output.lower()
    # The error message should mention the path or file, or at least indicate an error
    # Exit code 2 indicates BadParameter, and "error" in output indicates it's an error
    assert (
        result.exit_code == 2  # BadParameter exit code
        or "error" in error_output
        or "exists" in error_output
        or "file" in error_output
        or str(file_path).lower() in error_output
        or env_name.lower() in error_output
    ), (
        f"Expected BadParameter error about file collision. Exit code: {result.exit_code}, Output: {result.output}"
    )


def test_init_environment_reset_accepts_seed_and_episode_id(tmp_path: Path) -> None:
    """Generated reset() must accept the Environment.reset() arguments."""
    env_name = "seeded_env"
    result = runner.invoke(
        app, ["init", env_name, "--output-dir", str(tmp_path)], input="\n"
    )
    assert result.exit_code == 0

    env_dir = tmp_path / env_name
    env_file = env_dir / "server" / f"{env_name}_environment.py"
    sys.path.insert(0, str(env_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"{env_name}_environment", env_file
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = module.SeededEnvironment()

        env.reset(seed=1, episode_id="ep-123")
        assert env.state.episode_id == "ep-123"

        env.reset(episode_id="")
        assert env.state.episode_id == ""

        env.reset()
        assert env.state.episode_id not in (None, "", "ep-123")

        # Environment.reset_async() forwards seed and episode_id to reset().
        asyncio.run(env.reset_async(seed=2, episode_id="ep-456"))
        assert env.state.episode_id == "ep-456"
    finally:
        sys.path.remove(str(env_dir))
        sys.modules.pop("models", None)
