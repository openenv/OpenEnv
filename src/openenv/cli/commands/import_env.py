"""Import third-party environments into OpenEnv wrappers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from openenv.cli.importers import DEFAULT_IMPORTERS, ImporterRegistry

from .._cli_utils import console
from .init import _generate_uv_lock, _validate_env_name


def _select_match(
    registry: ImporterRegistry,
    source: Path,
    source_type: str | None,
    env_class: str | None,
):
    try:
        matches = registry.detect(source, source_type=source_type)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    if not matches:
        supported = ", ".join(registry.supported_types)
        raise typer.BadParameter(
            f"No supported environment found in {source}. Supported source types: {supported}."
        )

    if env_class:
        matches = [
            match
            for match in matches
            if match[1].class_name == env_class or match[1].qualified_name == env_class
        ]
        if not matches:
            raise typer.BadParameter(f"No detected environment matched {env_class!r}.")

    detected_types = {importer.source_type for importer, _ in matches}
    if source_type is None and len(detected_types) > 1:
        raise typer.BadParameter(
            "Multiple source formats were detected. Re-run with --type "
            f"({'/'.join(sorted(detected_types))})."
        )

    if len(matches) > 1:
        choices = ", ".join(detected.qualified_name for _, detected in matches)
        raise typer.BadParameter(
            f"Multiple environment entrypoints were detected ({choices}). "
            "Re-run with --env-class."
        )

    return matches[0]


def import_env(
    source: Annotated[
        str,
        typer.Argument(help="Local source repository or directory to import"),
    ],
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Name for the generated OpenEnv package"),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory where the generated package will be created",
        ),
    ],
    env_class: Annotated[
        str | None,
        typer.Option(
            "--env-class",
            help="Environment class name or module:Class when detection is ambiguous",
        ),
    ] = None,
    source_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            help="Optional source type override, such as 'ors'",
        ),
    ] = None,
) -> None:
    """Deterministically import a third-party environment into OpenEnv."""
    env_name = _validate_env_name(name)
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists() or not source_path.is_dir():
        raise typer.BadParameter(f"Source must be an existing directory: {source_path}")

    base_dir = Path(output_dir).expanduser().resolve()
    env_dir = base_dir / env_name
    try:
        env_dir.relative_to(source_path)
    except ValueError:
        pass
    else:
        raise typer.BadParameter("Output directory must not be inside the source tree")

    if env_dir.exists():
        if env_dir.is_file():
            raise typer.BadParameter(f"Path '{env_dir}' exists and is a file")
        if any(env_dir.iterdir()):
            raise typer.BadParameter(
                f"Directory '{env_dir}' already exists and is not empty."
            )

    registry = ImporterRegistry(DEFAULT_IMPORTERS)
    importer, detected = _select_match(
        registry,
        source_path,
        source_type=source_type,
        env_class=env_class,
    )

    try:
        env_dir.mkdir(parents=True, exist_ok=True)
        console.print(
            "[bold cyan]Importing environment[/bold cyan] "
            f"{detected.qualified_name} as '{env_name}' ({importer.source_type})"
        )
        importer.generate(
            source=source_path,
            destination=env_dir,
            env_name=env_name,
            detected=detected,
        )

        console.print("[bold green]OK[/bold green] Generated OpenEnv wrapper")
        if _generate_uv_lock(env_dir):
            console.print("[green]OK[/green] Generated uv.lock")
        else:
            console.print("[yellow]Warning:[/yellow] Could not generate uv.lock")

        console.print(f"[bold green]Environment created at: {env_dir}[/bold green]")
    except Exception as e:
        if env_dir.exists() and env_dir.is_dir():
            shutil.rmtree(env_dir, ignore_errors=True)
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e
