# SPDX-License-Identifier: BSD-3-Clause

"""Build Docker images for OpenEnv environments."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from .._cli_utils import console

app = typer.Typer(help="Build Docker images for OpenEnv environments")

_OPENENV_RUNTIME_DEP_RE = re.compile(r"^openenv(?:\s*(?:$|[<>=!~@;])|\[)")


def _is_openenv_runtime_dependency(dep: str) -> bool:
    """Return True for the OpenEnv runtime distribution, not openenv-* envs."""
    normalized = dep.strip().lower()
    return (
        _OPENENV_RUNTIME_DEP_RE.match(normalized) is not None
        or normalized.startswith("openenv-core")
        or normalized.startswith("openenv_core")
    )


def _is_in_repo_env_path(env_path: Path, repo_root: Path) -> bool:
    """Return whether env_path points to an environment below repo_root/envs."""
    try:
        rel_path = env_path.relative_to(repo_root)
    except ValueError:
        return False
    return len(rel_path.parts) > 1 and rel_path.parts[0] == "envs"


def _detect_build_context(env_path: Path) -> tuple[str, Path, Path | None]:
    """
    Detect whether we're building a standalone or in-repo environment.

    Returns:
        `tuple` of `(build_mode, build_context_path, repo_root)` where `build_mode` is
        `"standalone"` or `"in-repo"`, `build_context_path` is the path to use as Docker
        build context, and `repo_root` is the path to the repo root (`None` for standalone).
    """
    # Ensure env_path is absolute for proper comparison
    env_path = env_path.absolute()

    # Check if we're in a git repository
    current = env_path
    repo_root = None

    # Walk up to find .git directory
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            repo_root = parent
            break

    if repo_root is None:
        # Not in a git repo = standalone
        return "standalone", env_path, None

    if _is_in_repo_env_path(env_path, repo_root):
        return "in-repo", repo_root, repo_root

    # Otherwise, it's standalone (environment outside repo structure)
    return "standalone", env_path, None


def _prepare_standalone_build(env_path: Path, temp_dir: Path) -> Path:
    """
    Prepare a standalone environment for building.

    For standalone builds:
    1. Copy environment to temp directory
    2. Ensure pyproject.toml depends on openenv

    Returns:
        `Path` to the prepared build directory.
    """
    console.print("[cyan]Preparing standalone build...[/cyan]")

    # Copy environment to temp directory
    build_dir = temp_dir / env_path.name
    shutil.copytree(env_path, build_dir, symlinks=True)

    console.print(f"[cyan]Copied environment to:[/cyan] {build_dir}")

    # Check if pyproject.toml has openenv dependency
    pyproject_path = build_dir / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            try:
                import tomli

                pyproject = tomli.load(f)
                deps = pyproject.get("project", {}).get("dependencies", [])

                # Check if openenv dependency is declared
                has_openenv = any(dep.startswith("openenv") for dep in deps)

                if not has_openenv:
                    console.print(
                        "[yellow]Warning:[/yellow] pyproject.toml doesn't list the openenv dependency",
                    )
                    console.print(
                        "[yellow]You may need to add:[/yellow] openenv>=0.2.0",
                    )
            except ImportError:
                console.print(
                    "[yellow]Warning:[/yellow] tomli not available, skipping dependency check",
                )

    return build_dir


def _prepare_inrepo_build(env_path: Path, repo_root: Path, temp_dir: Path) -> Path:
    """
    Prepare an in-repo environment for building.

    For in-repo builds:
    1. Create temp directory with environment and core
    2. Set up structure that matches expected layout

    Returns:
        `Path` to the prepared build directory.
    """
    console.print("[cyan]Preparing in-repo build...[/cyan]")

    # Copy environment to temp directory
    build_dir = temp_dir / env_path.name
    shutil.copytree(env_path, build_dir, symlinks=True)

    # Copy OpenEnv package metadata + sources to temp directory.
    # Keep the src/ layout since pyproject.toml uses package-dir = {"" = "src"}.
    package_src = repo_root / "src" / "openenv"
    package_dest = build_dir / "openenv"
    if package_src.exists():
        package_dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package_src, package_dest / "src" / "openenv", symlinks=True)

        for filename in ("pyproject.toml", "README.md"):
            src_file = repo_root / filename
            if src_file.exists():
                shutil.copy2(src_file, package_dest / filename)

        console.print(f"[cyan]Copied OpenEnv package to:[/cyan] {package_dest}")

        # Update pyproject.toml to reference local OpenEnv copy
        pyproject_path = build_dir / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                try:
                    import tomli

                    pyproject = tomli.load(f)
                    deps = pyproject.get("project", {}).get("dependencies", [])

                    # Replace OpenEnv package references with local source.
                    new_deps = []
                    for dep in deps:
                        if _is_openenv_runtime_dependency(dep):
                            # Skip - we'll use local core
                            continue
                        new_deps.append(dep)

                    # Write back with local OpenEnv reference
                    pyproject["project"]["dependencies"] = new_deps + [
                        "openenv @ file:///app/env/openenv"
                    ]

                    # Write updated pyproject.toml
                    with open(pyproject_path, "wb") as out_f:
                        import tomli_w

                        tomli_w.dump(pyproject, out_f)

                    console.print(
                        "[cyan]Updated pyproject.toml to use local core[/cyan]"
                    )

                    # Remove old lockfile since dependencies changed
                    lockfile = build_dir / "uv.lock"
                    if lockfile.exists():
                        lockfile.unlink()
                        console.print("[cyan]Removed outdated uv.lock[/cyan]")

                except ImportError:
                    console.print(
                        "[yellow]Warning:[/yellow] tomli/tomli_w not available, using pyproject.toml as-is",
                    )
    else:
        console.print(
            "[yellow]Warning:[/yellow] OpenEnv package not found, building without it"
        )

    console.print(f"[cyan]Build directory prepared:[/cyan] {build_dir}")
    return build_dir


def _run_command(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a shell command and handle errors."""
    console.print(f"[bold cyan]Running:[/bold cyan] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=check, capture_output=True, text=True
        )
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}", file=sys.stderr)
        if e.stdout:
            console.print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        if check:
            raise typer.Exit(1) from e
        return e


def _build_docker_image(
    env_path: Path,
    tag: str | None = None,
    context_path: Path | None = None,
    dockerfile: Path | None = None,
    build_args: dict[str, str] | None = None,
    no_cache: bool = False,
) -> bool:
    """Build Docker image for the environment with smart context detection."""

    # Detect build context (standalone vs in-repo)
    build_mode, detected_context, repo_root = _detect_build_context(env_path)

    console.print(f"[bold cyan]Build mode detected:[/bold cyan] {build_mode}")

    # Use detected context unless explicitly overridden
    if context_path is None:
        context_path = detected_context

    # Create temporary build directory
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Prepare build directory based on mode
        if build_mode == "standalone":
            build_dir = _prepare_standalone_build(env_path, temp_dir)
        else:  # in-repo
            build_dir = _prepare_inrepo_build(env_path, repo_root, temp_dir)

        # Determine Dockerfile path
        if dockerfile is None:
            # Look for Dockerfile in server/ subdirectory
            dockerfile = build_dir / "server" / "Dockerfile"
            if not dockerfile.exists():
                # Fallback to root of build directory
                dockerfile = build_dir / "Dockerfile"

        if not dockerfile.exists():
            console.print(
                f"[bold red]Error:[/bold red] Dockerfile not found at {dockerfile}",
            )
            return False

        # Generate tag if not provided
        if tag is None:
            env_name = env_path.name
            if env_name.endswith("_env"):
                env_name = env_name[:-4]
            tag = f"openenv-{env_name}"

        console.print(f"[bold cyan]Building Docker image:[/bold cyan] {tag}")
        console.print(f"[bold cyan]Build context:[/bold cyan] {build_dir}")
        console.print(f"[bold cyan]Dockerfile:[/bold cyan] {dockerfile}")

        # Prepare build args
        if build_args is None:
            build_args = {}

        # Add build mode and env name to build args
        build_args["BUILD_MODE"] = build_mode
        build_args["ENV_NAME"] = env_path.name.replace("_env", "")

        cmd = _docker_build_command(
            tag=tag,
            dockerfile=dockerfile,
            build_dir=build_dir,
            build_args=build_args,
            no_cache=no_cache,
        )

        result = _run_command(cmd, check=False)
        return result.returncode == 0


def _docker_build_command(
    tag: str,
    dockerfile: Path,
    build_dir: Path,
    build_args: dict[str, str],
    no_cache: bool,
) -> list[str]:
    """Build the docker build command for an OpenEnv environment."""
    cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile)]

    if no_cache:
        cmd.append("--no-cache")

    for key, value in build_args.items():
        cmd.extend(["--build-arg", f"{key}={value}"])

    cmd.append(str(build_dir))
    return cmd


def _push_docker_image(tag: str, registry: str | None = None) -> bool:
    """Push Docker image to registry."""
    if registry:
        full_tag = f"{registry}/{tag}"
        console.print(f"[bold cyan]Tagging image as {full_tag}[/bold cyan]")
        _run_command(["docker", "tag", tag, full_tag])
        tag = full_tag

    console.print(f"[bold cyan]Pushing image:[/bold cyan] {tag}")
    result = _run_command(["docker", "push", tag], check=False)
    return result.returncode == 0


def _parse_build_args(raw_args: list[str] | None) -> dict[str, str]:
    """Parse Docker build args from repeated KEY=VALUE CLI options."""
    build_args: dict[str, str] = {}

    for arg in raw_args or []:
        if "=" in arg:
            key, value = arg.split("=", 1)
            build_args[key] = value
        else:
            print(
                f"Warning: Invalid build arg format: {arg}",
                file=sys.stderr,
            )

    return build_args


@app.command()
def build(
    env_path: Annotated[
        str | None,
        typer.Argument(
            help="Path to the environment directory (default: current directory)"
        ),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option(
            "--tag",
            "-t",
            help="Docker image tag (default: openenv-<env_name>)",
        ),
    ] = None,
    context: Annotated[
        str | None,
        typer.Option(
            "--context",
            "-c",
            help="Build context path (default: <env_path>/server)",
        ),
    ] = None,
    dockerfile: Annotated[
        str | None,
        typer.Option(
            "--dockerfile",
            "-f",
            help="Path to Dockerfile (default: <context>/Dockerfile)",
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Build without using cache",
        ),
    ] = False,
    build_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--build-arg",
            help="Build arguments (can be used multiple times, format: KEY=VALUE)",
        ),
    ] = None,
) -> None:
    """
    Build Docker images for OpenEnv environments.

    This command builds Docker images using the environment's pyproject.toml
    and uv for dependency management. Run from the environment root directory.

    Examples:

        ```bash
        # Build from environment root (recommended)
        $ cd my_env
        $ openenv build

        # Build with custom tag
        $ openenv build -t my-custom-tag

        # Build without cache
        $ openenv build --no-cache

        # Build with custom build arguments
        $ openenv build --build-arg VERSION=1.0 --build-arg ENV=prod

        # Build from different directory
        $ openenv build envs/echo_env
        ```
    """
    # Determine environment path (default to current directory)
    if env_path is None:
        env_path_obj = Path.cwd()
    else:
        env_path_obj = Path(env_path)

    # Validate environment path
    if not env_path_obj.exists():
        print(
            f"Error: Environment path does not exist: {env_path_obj}",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    if not env_path_obj.is_dir():
        print(
            f"Error: Environment path is not a directory: {env_path_obj}",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    # Check for openenv.yaml to confirm this is an environment directory
    openenv_yaml = env_path_obj / "openenv.yaml"
    if not openenv_yaml.exists():
        print(
            f"Error: Not an OpenEnv environment directory (missing openenv.yaml): {env_path_obj}",
            file=sys.stderr,
        )
        print(
            "Hint: Run this command from the environment root directory or specify the path",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    console.print(f"[bold]Building Docker image for:[/bold] {env_path_obj.name}")
    console.print("=" * 60)

    build_args = _parse_build_args(build_arg)

    # Convert string paths to Path objects
    context_path_obj = Path(context) if context else None
    dockerfile_path_obj = Path(dockerfile) if dockerfile else None

    # Build Docker image
    success = _build_docker_image(
        env_path=env_path_obj,
        tag=tag,
        context_path=context_path_obj,
        dockerfile=dockerfile_path_obj,
        build_args=build_args if build_args else None,
        no_cache=no_cache,
    )

    if not success:
        print("✗ Docker build failed", file=sys.stderr)
        raise typer.Exit(1)

    console.print("[bold green]✓ Docker build successful[/bold green]")
    console.print("\n[bold green]Done![/bold green]")
