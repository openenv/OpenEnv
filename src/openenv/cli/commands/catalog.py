# SPDX-License-Identifier: BSD-3-Clause

"""Metadata catalog production and inspection. No environment is instantiated."""

import json
from pathlib import Path

import typer
from openenv.discovery import (
    build_catalog,
    CatalogError,
    load_catalog,
    resolve_entry,
    search_catalog,
    write_catalog,
)

app = typer.Typer(
    help="Build and inspect versioned, metadata-only environment catalogs."
)


@app.command("build")
def build(
    repository: Path = typer.Option(..., help="Local Git repository to read."),
    repository_uri: str = typer.Option(..., help="Declared GitHub HTTPS clone URI."),
    publisher: str = typer.Option(..., help="Explicit catalog publication authority."),
    output: Path = typer.Option(..., help="Destination snapshot JSON."),
    revision: str = typer.Option("HEAD", help="Source commit or ref."),
    root: str = typer.Option("envs", help="Direct-child environment inventory root."),
    namespace: str = typer.Option("openenv", help="Publisher-owned namespace."),
) -> None:
    """Build from committed metadata; uncommitted files and candidate code are not used."""
    try:
        snapshot = build_catalog(
            repository,
            repository_uri=repository_uri,
            publisher=publisher,
            namespace=namespace,
            revision=revision,
            root=root,
        )
        write_catalog(snapshot, output)
    except CatalogError as error:
        typer.echo(f"Catalog error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"{len(snapshot.entries)} records, snapshot {snapshot.digest}, "
        f"complete={str(snapshot.complete).lower()}",
        err=True,
    )
    for issue in snapshot.issues:
        typer.echo(f"{issue.severity}: {issue.path}: {issue.message}", err=True)
    if not snapshot.complete:
        raise typer.Exit(1)


@app.command("inspect")
def inspect(
    identifier: str = typer.Argument(..., help="Exact revision-card identifier."),
    catalog: Path = typer.Option(..., help="Complete configured snapshot."),
) -> None:
    """Return the complete selected entry without guessing or fetching a URL."""
    try:
        entry = resolve_entry(load_catalog(catalog), identifier)
    except CatalogError as error:
        typer.echo(f"Catalog error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(entry.model_dump_json(by_alias=True, exclude_none=True, indent=2))


def discover(
    query: str = typer.Argument("", help="Task to find an environment for."),
    catalog: Path = typer.Option(
        ..., help="Complete, explicitly configured local snapshot."
    ),
    limit: int = typer.Option(20, min=1, max=1000),
    filter: list[str] = typer.Option([], "--filter", help="Exact field=value filter."),
    as_json: bool = typer.Option(False, "--json", help="Emit complete ARD entry data."),
) -> None:
    """Find unfamiliar environments by task before any installation or execution."""
    filters = {}
    for item in filter:
        key, separator, value = item.partition("=")
        if not separator or not key or key in filters:
            typer.echo(
                "Catalog error: filters must be distinct field=value pairs", err=True
            )
            raise typer.Exit(1)
        filters[key] = value
    try:
        snapshot = load_catalog(catalog)
        matches = search_catalog(snapshot, query, filters=filters, limit=limit)
    except CatalogError as error:
        typer.echo(f"Catalog error: {error}", err=True)
        raise typer.Exit(1) from error
    if as_json:
        results = [
            {
                **match.entry.model_dump(by_alias=True, exclude_none=True, mode="json"),
                "score": match.score,
            }
            for match in matches
        ]
        typer.echo(
            json.dumps(
                {"results": results, "snapshot": snapshot.digest, "metadataOnly": True},
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    for match in matches:
        card = match.entry.data
        typer.echo(f"{match.entry.display_name}: {card.description}")
        typer.echo(f"  source: {card.source.uri}")
        typer.echo(f"  path: {card.source.path}")
        typer.echo(f"  revision: {card.source.revision}")
        typer.echo(f"  license: {card.license}; artifact: {card.artifact_availability}")
        typer.echo(f"  identifier: {match.entry.identifier}")
    typer.echo("Metadata only. Execution approval remains a separate decision.")
