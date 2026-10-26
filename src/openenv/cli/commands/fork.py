# SPDX-License-Identifier: BSD-3-Clause

"""Fork (duplicate) a Hugging Face Space using the Hub API."""

from __future__ import annotations

from typing import Annotated

import typer
from huggingface_hub import HfApi, login, whoami

from .._cli_utils import _extract_hf_username, console

app = typer.Typer(
    help="Fork (duplicate) an OpenEnv environment on Hugging Face to your account"
)


def _parse_key_value(s: str) -> tuple[str, str]:
    """Parse KEY=VALUE string. Raises BadParameter if no '='."""
    if "=" not in s:
        raise typer.BadParameter(
            f"Expected KEY=VALUE format, got: {s!r}. "
            "Use --set-env KEY=VALUE or --set-secret KEY=VALUE"
        )
    key, _, value = s.partition("=")
    key = key.strip()
    if not key:
        raise typer.BadParameter(f"Empty key in: {s!r}")
    return key, value.strip()


def _ensure_hf_authenticated() -> str:
    """Ensure user is authenticated with Hugging Face. Returns username."""
    try:
        username = _extract_hf_username(whoami())
        if not username:
            raise ValueError("Could not extract username from whoami response")
        console.print(f"[bold green]✓[/bold green] Authenticated as: {username}")
        return username
    except Exception:
        console.print(
            "[bold yellow]Not authenticated with Hugging Face. Please login...[/bold yellow]"
        )
        try:
            login()
            username = _extract_hf_username(whoami())
            if not username:
                raise ValueError("Could not extract username from whoami response")
            console.print(f"[bold green]✓[/bold green] Authenticated as: {username}")
            return username
        except Exception as e:
            raise typer.BadParameter(
                f"Hugging Face authentication failed: {e}. Please run login manually."
            ) from e


@app.command()
def fork(
    source_space: Annotated[
        str,
        typer.Argument(
            help="Source Space ID in format 'owner/space-name' (e.g. org/my-openenv-space)"
        ),
    ],
    repo_id: Annotated[
        str | None,
        typer.Option(
            "--repo-id",
            "-r",
            help="Target repo ID for the fork (default: created under your account with same name)",
        ),
    ] = None,
    private: Annotated[
        bool,
        typer.Option("--private", help="Create the forked Space as private"),
    ] = False,
    set_env: Annotated[
        list[str],
        typer.Option(
            "--set-env",
            "-e",
            help="Set Space variable (public). Can be repeated. Format: KEY=VALUE",
        ),
    ] = [],
    set_secret: Annotated[
        list[str],
        typer.Option(
            "--set-secret",
            "--secret",
            "-s",
            help="Set Space secret. Can be repeated. Format: KEY=VALUE",
        ),
    ] = [],
    hardware: Annotated[
        str | None,
        typer.Option(
            "--hardware",
            "-H",
            help="Request hardware (e.g. t4-medium, cpu-basic). See Hub docs for options.",
        ),
    ] = None,
) -> None:
    """
    Fork (duplicate) a Hugging Face Space to your account using the Hub API.

    Uses the Hugging Face duplicate_space API. You can set environment variables
    and secrets, and request hardware/storage/sleep time at creation time.

    Examples:

        ```bash
        $ openenv fork owner/source-space
        $ openenv fork owner/source-space --private
        $ openenv fork owner/source-space --repo-id myuser/my-fork
        $ openenv fork owner/source-space --set-env MODEL_ID=user/model --set-secret HF_TOKEN=hf_xxx
        $ openenv fork owner/source-space --hardware t4-medium
        ```
    """
    if "/" not in source_space or source_space.count("/") != 1:
        raise typer.BadParameter(
            f"Invalid source Space ID: {source_space!r}. Expected format: 'owner/space-name'"
        )

    _ensure_hf_authenticated()
    api = HfApi()

    # Build kwargs for duplicate_space (only pass what we have)
    dup_kwargs: dict = {
        "from_id": source_space,
        "private": private,
    }
    if set_env:
        dup_kwargs["variables"] = [
            {"key": k, "value": v} for k, v in (_parse_key_value(x) for x in set_env)
        ]
    if set_secret:
        dup_kwargs["secrets"] = [
            {"key": k, "value": v} for k, v in (_parse_key_value(x) for x in set_secret)
        ]
    # HF API requires hardware when duplicating; default to free cpu-basic
    dup_kwargs["hardware"] = hardware if hardware is not None else "cpu-basic"
    if repo_id is not None:
        if "/" not in repo_id or repo_id.count("/") != 1:
            raise typer.BadParameter(
                f"Invalid --repo-id: {repo_id!r}. Expected format: 'username/repo-name'"
            )
        dup_kwargs["to_id"] = repo_id

    console.print(f"[bold cyan]Forking Space {source_space}...[/bold cyan]")
    try:
        result = api.duplicate_space(**dup_kwargs)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Fork failed: {e}")
        raise typer.Exit(1) from e

    # result is RepoUrl (str-like) or similar; get repo_id for display
    if hasattr(result, "repo_id"):
        new_repo_id = result.repo_id
    elif isinstance(result, str):
        # URL like https://huggingface.co/spaces/owner/name -> owner/name
        if "/spaces/" in result:
            new_repo_id = result.split("/spaces/")[-1].rstrip("/")
        else:
            new_repo_id = result
    else:
        new_repo_id = getattr(result, "repo_id", str(result))

    console.print("[bold green]✓[/bold green] Space forked successfully")
    console.print(
        f"[bold]Space URL:[/bold] https://huggingface.co/spaces/{new_repo_id}"
    )
