# SPDX-License-Identifier: BSD-3-Clause

"""Serve an OpenEnv environment locally (uvicorn, from ``openenv.yaml``)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .._cli_utils import console, validate_env_structure


def serve(
    env_path: Annotated[
        str | None,
        typer.Argument(
            help="Path to the environment directory (default: current directory)"
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind (default: ``port`` in openenv.yaml, else 8000)",
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option("--host", help="Host interface to bind (local development only)"),
    ] = "127.0.0.1",
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable autoreload (development)"),
    ] = False,
) -> None:
    """
    Run the environment FastAPI app on the host for local development.

    Uses ``openenv.yaml`` fields ``app`` (e.g. ``server.app:app``), ``port``, and
    ``runtime`` (must be ``fastapi``). Matches ``uv run --project . server`` layout:
    the environment directory is the working directory. Package-qualified app
    paths also support the environment package and the repo's ``envs`` package.

    For production or training, use Docker (``openenv build``) — this command runs
    on the host for local development only.
    """
    import uvicorn

    env_path_obj = (
        Path.cwd().resolve() if env_path is None else Path(env_path).resolve()
    )

    if not env_path_obj.exists():
        typer.echo(f"Error: Path does not exist: {env_path_obj}", err=True)
        raise typer.Exit(1)

    if not env_path_obj.is_dir():
        typer.echo(f"Error: Path is not a directory: {env_path_obj}", err=True)
        raise typer.Exit(1)

    try:
        validate_env_structure(env_path_obj)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"Not a valid OpenEnv environment: {exc}") from exc

    manifest_path = env_path_obj / "openenv.yaml"
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
    except OSError as exc:
        raise typer.BadParameter(f"Failed to read openenv.yaml: {exc}") from exc
    except yaml.YAMLError as exc:
        raise typer.BadParameter(f"Invalid YAML in openenv.yaml: {exc}") from exc

    if not isinstance(manifest, dict):
        raise typer.BadParameter("openenv.yaml must be a YAML dictionary")

    app_spec = manifest.get("app")
    if not app_spec or not isinstance(app_spec, str):
        raise typer.BadParameter(
            "openenv.yaml must contain a string 'app' field (e.g. server.app:app)"
        )
    if ":" not in app_spec:
        raise typer.BadParameter(
            f"openenv.yaml 'app' must look like 'module.path:attribute', got {app_spec!r}"
        )

    runtime = str(manifest.get("runtime", "fastapi")).lower()
    if runtime != "fastapi":
        raise typer.BadParameter(
            f"openenv serve only supports runtime 'fastapi' (got {runtime!r})"
        )

    raw_port = port if port is not None else manifest.get("port", 8000)
    try:
        listen_port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(
            f"Invalid port {raw_port!r}; expected an integer"
        ) from exc
    if not (1 <= listen_port <= 65535):
        raise typer.BadParameter(
            f"Invalid port {listen_port}; expected a value between 1 and 65535"
        )

    original_path = list(sys.path)
    prev_cwd = os.getcwd()
    env_root = str(env_path_obj)
    app_dir = env_path_obj
    module_name = app_spec.split(":", 1)[0]
    # Prefer a module inside the environment, even when its package name matches
    # the environment directory (for example, /work/server/server/app.py).
    local_module = env_path_obj.joinpath(*module_name.split("."))
    if not (
        local_module.with_suffix(".py").is_file()
        or (local_module / "__init__.py").is_file()
    ):
        if env_path_obj.parent.name == "envs" and module_name.startswith(
            f"envs.{env_path_obj.name}."
        ):
            app_dir = env_path_obj.parent.parent
        elif module_name.startswith(f"{env_path_obj.name}."):
            app_dir = env_path_obj.parent

    try:
        os.chdir(env_root)

        console.print(
            f"[bold green]Serving[/bold green] [cyan]{app_spec}[/cyan] on "
            f"[bold]http://{host}:{listen_port}/[/bold]  (cwd: {env_root})"
        )

        try:
            uvicorn.run(
                app_spec,
                host=host,
                port=listen_port,
                reload=reload,
                app_dir=str(app_dir),
            )
        except OSError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
    finally:
        try:
            os.chdir(prev_cwd)
        except OSError:
            pass
        sys.path[:] = original_path
