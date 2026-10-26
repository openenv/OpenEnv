# SPDX-License-Identifier: BSD-3-Clause

"""Serve OpenEnv environments locally (TO BE IMPLEMENTED)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .._cli_utils import console

app = typer.Typer(help="Serve OpenEnv environments locally")


@app.command()
def serve(
    env_path: Annotated[
        str | None,
        typer.Argument(
            help="Path to the environment directory (default: current directory)"
        ),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to serve on"),
    ] = 8000,
    host: Annotated[
        str,
        typer.Option("--host", help="Host to bind to"),
    ] = "0.0.0.0",
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable auto-reload on code changes"),
    ] = False,
) -> None:
    """
    Serve an OpenEnv environment locally.

    TODO: This command is currently not implemented and has been deferred for later.

    Planned functionality:
    - Run environment server locally without Docker
    - Support multiple deployment modes (local, notebook, cluster)
    - Auto-reload for development
    - Integration with environment's [project.scripts] entry point

    For now, use Docker-based serving:
        1. Build the environment: openenv build
        2. Run the container: docker run -p 8000:8000 <image-name>

    Or use uv directly:
        uv run --project . server --port 8000
    """
    console.print("[bold yellow]⚠ This command is not yet implemented[/bold yellow]\n")

    console.print(
        "The [bold cyan]openenv serve[/bold cyan] command has been deferred for later."
    )

    console.print("[bold]Alternative approaches:[/bold]\n")

    console.print("[cyan]Option 1: Docker-based serving (recommended)[/cyan]")
    console.print("  1. Build the environment:")
    console.print("     [dim]$ openenv build[/dim]")
    console.print("  2. Run the Docker container:")
    console.print(
        f"     [dim]$ docker run -p {port}:{port} openenv-<env-name>:latest[/dim]\n"
    )

    console.print("[cyan]Option 2: Direct execution with uv[/cyan]")

    # Determine environment path
    if env_path is None:
        env_path_obj = Path.cwd()
    else:
        env_path_obj = Path(env_path)

    # Check for openenv.yaml
    openenv_yaml = env_path_obj / "openenv.yaml"
    if openenv_yaml.exists():
        console.print("  From your environment directory:")
        console.print(f"     [dim]$ cd {env_path_obj}[/dim]")
        console.print(f"     [dim]$ uv run --project . server --port {port}[/dim]\n")
    else:
        console.print("  From an environment directory with pyproject.toml:")
        console.print(f"     [dim]$ uv run --project . server --port {port}[/dim]\n")

    raise typer.Exit(1)
