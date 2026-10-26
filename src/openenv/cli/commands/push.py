# SPDX-License-Identifier: BSD-3-Clause

"""Push an OpenEnv environment to Hugging Face Spaces."""

from __future__ import annotations

import shutil
import sys
import tempfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated

import typer
import yaml
from huggingface_hub import HfApi, login, whoami

from .._cli_utils import _extract_hf_username, console, validate_env_structure

app = typer.Typer(help="Push an OpenEnv environment to Hugging Face Spaces")


DEFAULT_PUSH_IGNORE_PATTERNS = [".*", "__pycache__", "*.pyc", "build/", "*.egg-info"]


def _format_kv_entry_for_error(entry: str, *, flag: str) -> str:
    """Redact secret values from CLI parse errors."""
    if flag == "--secret":
        return "<redacted>"
    return repr(entry)


def _parse_kv_pairs(raw_pairs: list[str], *, flag: str) -> dict[str, str]:
    """Parse a list of 'KEY=VALUE' strings into a dict.

    Splits only on the first '=', so values can contain '='. Later entries
    override earlier ones with the same key (repeated CLI flags still work).
    """
    parsed: dict[str, str] = {}
    for entry in raw_pairs:
        entry_display = _format_kv_entry_for_error(entry, flag=flag)
        if "=" not in entry:
            raise typer.BadParameter(
                f"Invalid {flag} format: {entry_display}. Expected KEY=VALUE."
            )
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(
                f"Invalid {flag} format: {entry_display}. Key cannot be empty."
            )
        parsed[key] = value
    return parsed


def _apply_space_variables_and_secrets(
    repo_id: str,
    variables: dict[str, str],
    secrets: dict[str, str],
    api: HfApi,
) -> None:
    """Configure Space-level variables and secrets on a deployed repo.

    Idempotent: add_space_variable/secret with the same key overwrites.
    Secret values are never logged — only the key is printed.
    """
    for key, value in variables.items():
        try:
            api.add_space_variable(repo_id=repo_id, key=key, value=value)
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to set variable {key}: {e}")
            raise typer.Exit(1) from e
        console.print(f"[bold green]✓[/bold green] Set variable {key} on {repo_id}")
    for key, value in secrets.items():
        try:
            api.add_space_secret(repo_id=repo_id, key=key, value=value)
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to set secret {key}: {e}")
            raise typer.Exit(1) from e
        console.print(f"[bold green]✓[/bold green] Set secret {key} on {repo_id}")


def _path_matches_pattern(relative_path: Path, pattern: str) -> bool:
    """Return True if a relative path matches an exclude pattern."""
    normalized_pattern = pattern.strip()
    if normalized_pattern.startswith("!"):
        return False

    while normalized_pattern.startswith("./"):
        normalized_pattern = normalized_pattern[2:]

    if normalized_pattern.startswith("/"):
        normalized_pattern = normalized_pattern[1:]

    if not normalized_pattern:
        return False

    posix_path = relative_path.as_posix()
    pattern_candidates = [normalized_pattern]
    if normalized_pattern.startswith("**/"):
        # Gitignore-style "**/" can also match directly at the root.
        pattern_candidates.append(normalized_pattern[3:])

    # Support directory patterns such as "artifacts/" and "**/outputs/".
    if normalized_pattern.endswith("/"):
        dir_pattern_candidates: list[str] = []
        for candidate in pattern_candidates:
            base = candidate.rstrip("/")
            if not base:
                continue
            dir_pattern_candidates.extend([base, f"{base}/*"])

        return any(
            fnmatch(posix_path, candidate) for candidate in dir_pattern_candidates
        )

    # Match both full relative path and basename for convenience.
    return any(
        fnmatch(posix_path, candidate) for candidate in pattern_candidates
    ) or any(fnmatch(relative_path.name, candidate) for candidate in pattern_candidates)


def _should_exclude_path(relative_path: Path, ignore_patterns: list[str]) -> bool:
    """Return True when the path should be excluded from staging/upload."""
    return any(
        _path_matches_pattern(relative_path, pattern) for pattern in ignore_patterns
    )


def _read_ignore_file(ignore_path: Path) -> tuple[list[str], int]:
    """Read ignore patterns from a file and return (patterns, ignored_negations)."""
    patterns: list[str] = []
    ignored_negations = 0

    for line in ignore_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("!"):
            ignored_negations += 1
            continue
        patterns.append(stripped)

    return patterns, ignored_negations


def _load_ignore_patterns(env_dir: Path, exclude_file: str | None) -> list[str]:
    """Load ignore patterns from defaults and an optional ignore file."""
    patterns = list(DEFAULT_PUSH_IGNORE_PATTERNS)
    ignored_negations = 0

    def _merge_ignore_file(ignore_path: Path, *, source_label: str) -> None:
        nonlocal ignored_negations
        file_patterns, skipped_negations = _read_ignore_file(ignore_path)
        patterns.extend(file_patterns)
        ignored_negations += skipped_negations
        console.print(
            f"[bold green]✓[/bold green] Loaded {len(file_patterns)} ignore patterns from {source_label}: {ignore_path}"
        )

    # Optional source: explicit exclude file from CLI.
    if exclude_file:
        ignore_path = Path(exclude_file)
        if not ignore_path.is_absolute():
            ignore_path = env_dir / ignore_path
        ignore_path = ignore_path.resolve()

        if not ignore_path.exists() or not ignore_path.is_file():
            raise typer.BadParameter(
                f"Exclude file not found or not a file: {ignore_path}"
            )

        _merge_ignore_file(ignore_path, source_label="--exclude")

    # Keep stable order while removing duplicates.
    patterns = list(dict.fromkeys(patterns))

    if ignored_negations > 0:
        console.print(
            f"[bold yellow]⚠[/bold yellow] Skipped {ignored_negations} negated ignore patterns ('!') because negation is not supported for push excludes"
        )

    return patterns


def _copytree_ignore_factory(env_dir: Path, ignore_patterns: list[str]):
    """Build a shutil.copytree ignore callback from path-based patterns."""

    def _ignore(path: str, names: list[str]) -> set[str]:
        current_dir = Path(path)
        ignored: set[str] = set()

        for name in names:
            candidate = current_dir / name
            try:
                relative_path = candidate.relative_to(env_dir)
            except ValueError:
                # candidate is not under env_dir (e.g. symlink or
                # copytree root differs from env_dir); skip filtering.
                continue
            if _should_exclude_path(relative_path, ignore_patterns):
                ignored.add(name)

        return ignored

    return _ignore


def _validate_openenv_directory(directory: Path) -> tuple[str, dict]:
    """
    Validate that the directory is an OpenEnv environment.

    Returns:
        `tuple` of `(env_name, manifest_data)`.
    """
    # Use the comprehensive validation function
    try:
        warnings = validate_env_structure(directory)
        for warning in warnings:
            console.print(f"[bold yellow]⚠[/bold yellow] {warning}")
    except FileNotFoundError as e:
        raise typer.BadParameter(f"Invalid OpenEnv environment structure: {e}") from e

    # Load and validate manifest
    manifest_path = directory / "openenv.yaml"
    try:
        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)
    except Exception as e:
        raise typer.BadParameter(f"Failed to parse openenv.yaml: {e}") from e

    if not isinstance(manifest, dict):
        raise typer.BadParameter("openenv.yaml must be a YAML dictionary")

    env_name = manifest.get("name")
    if not env_name:
        raise typer.BadParameter("openenv.yaml must contain a 'name' field")

    return env_name, manifest


def _rewrite_dockerfile_for_space(
    dockerfile_content: str,
    *,
    base_image: str | None,
    enable_interface: bool,
) -> tuple[str, list[str]]:
    """Rewrite deployment Dockerfile content and return printed change labels."""
    lines = dockerfile_content.split("\n")
    new_lines = []
    cmd_found = False
    base_image_updated = False
    web_interface_env_exists = "ENABLE_WEB_INTERFACE" in dockerfile_content
    last_instruction = None

    for line in lines:
        stripped = line.strip()
        token = stripped.split(maxsplit=1)[0] if stripped else ""
        current_instruction = token.upper()

        is_healthcheck_continuation = last_instruction == "HEALTHCHECK"

        # Update base image if specified
        if base_image and stripped.startswith("FROM") and not base_image_updated:
            new_lines.append(f"FROM {base_image}")
            base_image_updated = True
            last_instruction = "FROM"
            continue

        if (
            stripped.startswith("CMD")
            and not cmd_found
            and not web_interface_env_exists
            and enable_interface
            and not is_healthcheck_continuation
        ):
            new_lines.append("ENV ENABLE_WEB_INTERFACE=true")
            cmd_found = True

        new_lines.append(line)

        if current_instruction:
            last_instruction = current_instruction

    if not cmd_found and not web_interface_env_exists and enable_interface:
        new_lines.append("ENV ENABLE_WEB_INTERFACE=true")

    if base_image and not base_image_updated:
        new_lines.insert(0, f"FROM {base_image}")

    changes = []
    if base_image and base_image_updated:
        changes.append("updated base image")
    if enable_interface and not web_interface_env_exists:
        changes.append("enabled web interface")

    return "\n".join(new_lines), changes


def _get_hf_username() -> str:
    """Return the authenticated Hugging Face username from whoami()."""
    username = _extract_hf_username(whoami())
    if not username:
        raise ValueError("Could not extract username from whoami response")

    console.print(f"[bold green]✓[/bold green] Authenticated as: {username}")
    return username


def _ensure_hf_authenticated() -> str:
    """
    Ensure user is authenticated with Hugging Face.

    Returns:
        `str`: username of the authenticated user.
    """
    try:
        return _get_hf_username()
    except Exception:
        # Not authenticated, prompt for login
        console.print(
            "[bold yellow]Not authenticated with Hugging Face. Please login...[/bold yellow]"
        )

        try:
            login()
            return _get_hf_username()
        except Exception as e:
            raise typer.BadParameter(
                f"Hugging Face authentication failed: {e}. Please run login manually."
            ) from e


def _prepare_staging_directory(
    env_dir: Path,
    env_name: str,
    staging_dir: Path,
    ignore_patterns: list[str],
    base_image: str | None = None,
    enable_interface: bool = True,
) -> None:
    """
    Prepare files for deployment.

    This includes:
    - Copying necessary files
    - Modifying Dockerfile to optionally enable web interface and update base image
    - Ensuring README has proper HF frontmatter (if interface enabled)
    """
    # Create staging directory structure
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from env directory
    copy_ignore = _copytree_ignore_factory(env_dir, ignore_patterns)
    for item in env_dir.iterdir():
        relative_path = item.relative_to(env_dir)
        if _should_exclude_path(relative_path, ignore_patterns):
            continue

        dest = staging_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True, ignore=copy_ignore)
        else:
            shutil.copy2(item, dest)

    # Dockerfile must be at repo root for Hugging Face. Prefer root if present
    # (it was copied there); otherwise move server/Dockerfile to root.
    dockerfile_server_path = staging_dir / "server" / "Dockerfile"
    dockerfile_root_path = staging_dir / "Dockerfile"
    dockerfile_path: Path | None = None

    if dockerfile_root_path.exists():
        dockerfile_path = dockerfile_root_path
    elif dockerfile_server_path.exists():
        dockerfile_server_path.rename(dockerfile_root_path)
        console.print(
            "[bold cyan]Moved Dockerfile to repository root for deployment[/bold cyan]"
        )
        dockerfile_path = dockerfile_root_path

    # Modify Dockerfile to optionally enable web interface and update base image
    if dockerfile_path and dockerfile_path.exists():
        dockerfile_content = dockerfile_path.read_text()
        dockerfile_content, changes = _rewrite_dockerfile_for_space(
            dockerfile_content,
            base_image=base_image,
            enable_interface=enable_interface,
        )
        dockerfile_path.write_text(dockerfile_content)
        if changes:
            console.print(
                f"[bold green]✓[/bold green] Updated Dockerfile: {', '.join(changes)}"
            )
    else:
        console.print(
            "[bold yellow]⚠[/bold yellow] No Dockerfile at server/ or repo root"
        )

    # Ensure README has proper HF frontmatter (only if interface enabled)
    if enable_interface:
        readme_path = staging_dir / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text()
            if "base_path: /web" not in readme_content:
                # Check if frontmatter exists
                if readme_content.startswith("---"):
                    # Add base_path to existing frontmatter
                    lines = readme_content.split("\n")
                    new_lines = []
                    _in_frontmatter = True
                    for i, line in enumerate(lines):
                        new_lines.append(line)
                        if line.strip() == "---" and i > 0:
                            # End of frontmatter, add base_path before this line
                            if "base_path:" not in "\n".join(new_lines):
                                new_lines.insert(-1, "base_path: /web")
                            _in_frontmatter = False
                    readme_path.write_text("\n".join(new_lines))
                else:
                    # No frontmatter, add it
                    frontmatter = f"""---
title: {env_name.replace("_", " ").title()} Environment Server
emoji: 🔊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

"""
                    readme_path.write_text(frontmatter + readme_content)
                console.print(
                    "[bold green]✓[/bold green] Updated README with HF Space frontmatter"
                )
        else:
            console.print("[bold yellow]⚠[/bold yellow] No README.md found")


def _create_hf_space(
    repo_id: str,
    api: HfApi,
    private: bool = False,
    hardware: str | None = None,
) -> None:
    """Create a Hugging Face Space if it doesn't exist."""
    console.print(f"[bold cyan]Creating/verifying space: {repo_id}[/bold cyan]")

    try:
        create_kwargs: dict = {
            "repo_id": repo_id,
            "repo_type": "space",
            "space_sdk": "docker",
            "private": private,
            "exist_ok": True,
        }
        if hardware is not None:
            create_kwargs["space_hardware"] = hardware
        api.create_repo(**create_kwargs)
        console.print(f"[bold green]✓[/bold green] Space {repo_id} is ready")
    except Exception as e:
        # Space might already exist, which is okay with exist_ok=True
        # But if there's another error, log it
        console.print(f"[bold yellow]⚠[/bold yellow] Space creation: {e}")


def _upload_to_hf_space(
    repo_id: str,
    staging_dir: Path,
    api: HfApi,
    ignore_patterns: list[str],
    private: bool = False,
    create_pr: bool = False,
    commit_message: str | None = None,
    delete_patterns: list[str] | None = None,
) -> None:
    """Upload files to Hugging Face Space."""
    if create_pr:
        console.print(
            f"[bold cyan]Uploading files to {repo_id} (will open a Pull Request)...[/bold cyan]"
        )
    else:
        console.print(f"[bold cyan]Uploading files to {repo_id}...[/bold cyan]")

    upload_kwargs: dict = {
        "folder_path": str(staging_dir),
        "repo_id": repo_id,
        "repo_type": "space",
        "create_pr": create_pr,
        "ignore_patterns": ignore_patterns,
        "delete_patterns": delete_patterns,
    }
    if commit_message:
        upload_kwargs["commit_message"] = commit_message

    try:
        result = api.upload_folder(**upload_kwargs)
        console.print("[bold green]✓[/bold green] Upload completed successfully")
        if create_pr and result is not None and hasattr(result, "pr_url"):
            console.print(f"[bold]Pull request:[/bold] {result.pr_url}")
        console.print(
            f"[bold]Space URL:[/bold] https://huggingface.co/spaces/{repo_id}"
        )
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Upload failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def push(
    directory: Annotated[
        str | None,
        typer.Argument(
            help="Directory containing the OpenEnv environment (default: current directory)"
        ),
    ] = None,
    repo_id: Annotated[
        str | None,
        typer.Option(
            "--repo-id",
            "-r",
            help="Repository ID as 'repo_name' or 'namespace/repo_name'. Defaults to 'username/env-name' from openenv.yaml.",
        ),
    ] = None,
    base_image: Annotated[
        str | None,
        typer.Option(
            "--base-image",
            "-b",
            help="Base Docker image to use (overrides Dockerfile FROM)",
        ),
    ] = None,
    interface: Annotated[
        bool,
        typer.Option(
            "--interface",
            help="Enable web interface (default: True if no registry specified)",
        ),
    ] = None,
    no_interface: Annotated[
        bool,
        typer.Option(
            "--no-interface",
            help="Disable web interface",
        ),
    ] = False,
    registry: Annotated[
        str | None,
        typer.Option(
            "--registry",
            help="Custom registry URL (e.g., docker.io/username). Disables web interface by default.",
        ),
    ] = None,
    private: Annotated[
        bool,
        typer.Option(
            "--private",
            help="Deploy the space as private",
        ),
    ] = False,
    create_pr: Annotated[
        bool,
        typer.Option(
            "--create-pr",
            help="Create a Pull Request instead of pushing to the default branch",
        ),
    ] = False,
    exclude: Annotated[
        str | None,
        typer.Option(
            "--exclude",
            help="Optional additional ignore file with newline-separated glob patterns to exclude from Hugging Face uploads",
        ),
    ] = None,
    delete: Annotated[
        list[str] | None,
        typer.Option(
            "--delete",
            help="Glob pattern of remote Space files to delete in the same commit, as in `hf upload --delete` (repeatable). Use '*' to remove every file not uploaded again.",
        ),
    ] = None,
    hardware: Annotated[
        str | None,
        typer.Option(
            "--hardware",
            "-H",
            help="Request hardware for Hugging Face Space (e.g. t4-medium, cpu-basic). See HF docs for options.",
        ),
    ] = None,
    count: Annotated[
        int,
        typer.Option(
            "--count",
            "-n",
            help="Number of Space instances to deploy. Each gets a numeric suffix (e.g. env-1, env-2).",
            min=1,
        ),
    ] = 1,
    env_vars: Annotated[
        list[str] | None,
        typer.Option(
            "--env-var",
            "-e",
            help="Public Space variable as KEY=VALUE (repeatable). Overrides matching keys from openenv.yaml variables:.",
        ),
    ] = None,
    secrets: Annotated[
        list[str] | None,
        typer.Option(
            "--secret",
            help="Private Space secret as KEY=VALUE (repeatable). Value is never logged.",
        ),
    ] = None,
) -> None:
    """
    Push an OpenEnv environment to Hugging Face Spaces or a custom Docker registry.

    This command:
    1. Validates that the directory is an OpenEnv environment (openenv.yaml present)
    2. Builds and pushes to Hugging Face Spaces or custom Docker registry
    3. Optionally enables web interface for deployment

    The web interface is enabled by default when pushing to HuggingFace Spaces,
    but disabled by default when pushing to a custom Docker registry.

    Examples:

        ```bash
        # Push to HuggingFace Spaces from current directory (web interface enabled)
        $ cd my_env
        $ openenv push

        # Push to HuggingFace repo and open a Pull Request
        $ openenv push my-org/my-env --create-pr
        $ openenv push --repo-id my-org/my-env --create-pr

        # Push to HuggingFace without web interface
        $ openenv push --no-interface

        # Push to Docker Hub
        $ openenv push --registry docker.io/myuser

        # Push to GitHub Container Registry
        $ openenv push --registry ghcr.io/myorg

        # Push to custom registry with web interface
        $ openenv push --registry myregistry.io/path1/path2 --interface

        # Push to specific HuggingFace repo
        $ openenv push --repo-id my-org/my-env

        # Push privately with custom base image
        $ openenv push --private --base-image ghcr.io/huggingface/openenv-base:latest

        # Push with GPU hardware
        $ openenv push --hardware t4-medium

        # Set a public Space variable (overrides openenv.yaml variables:)
        $ openenv push -e OPENSPIEL_GAME=tic_tac_toe -e MAX_STEPS=100

        # Set a private Space secret (value never logged)
        $ openenv push --secret OPENAI_API_KEY=sk-...
        ```
    """
    # Validate --count flag combinations
    if count > 1 and registry:
        console.print(
            "[bold red]Error:[/bold red] --count cannot be used with --registry",
        )
        raise typer.Exit(1)

    if count > 1 and create_pr:
        console.print(
            "[bold red]Error:[/bold red] --count cannot be used with --create-pr",
        )
        raise typer.Exit(1)

    # Handle interface flag logic
    if no_interface and interface:
        console.print(
            "[bold red]Error:[/bold red] Cannot specify both --interface and --no-interface",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    # Determine if web interface should be enabled
    if no_interface:
        enable_interface = False
    elif interface is not None:
        enable_interface = interface
    elif registry is not None:
        # Custom registry: disable interface by default
        enable_interface = False
    else:
        # HuggingFace: enable interface by default
        enable_interface = True

    # Determine directory
    if directory:
        env_dir = Path(directory).resolve()
    else:
        env_dir = Path.cwd().resolve()

    if not env_dir.exists() or not env_dir.is_dir():
        raise typer.BadParameter(f"Directory does not exist: {env_dir}")

    # Check for openenv.yaml to confirm this is an environment directory
    openenv_yaml = env_dir / "openenv.yaml"
    if not openenv_yaml.exists():
        console.print(
            f"[bold red]Error:[/bold red] Not an OpenEnv environment directory (missing openenv.yaml): {env_dir}",
        )
        console.print(
            "[yellow]Hint:[/yellow] Run this command from the environment root directory",
        )
        raise typer.Exit(1)

    # Validate OpenEnv environment
    console.print(
        f"[bold cyan]Validating OpenEnv environment in {env_dir}...[/bold cyan]"
    )
    env_name, manifest = _validate_openenv_directory(env_dir)
    console.print(f"[bold green]✓[/bold green] Found OpenEnv environment: {env_name}")

    # Parse and merge Space variables (yaml < CLI) and secrets (CLI only).
    yaml_variables_raw = manifest.get("variables")
    if yaml_variables_raw is None:
        yaml_variables_raw = {}
    elif not isinstance(yaml_variables_raw, dict):
        raise typer.BadParameter(
            "openenv.yaml 'variables' must be a mapping of KEY: value"
        )
    if registry and (env_vars or secrets):
        raise typer.BadParameter(
            "--env-var/--secret cannot be used with --registry because custom registry pushes do not configure Hugging Face Space settings"
        )
    if create_pr and (env_vars or secrets):
        raise typer.BadParameter(
            "--env-var/--secret cannot be used with --create-pr because Space settings are only applied to the live Space after merge"
        )
    merged_variables: dict[str, str] = {
        str(k): str(v) for k, v in yaml_variables_raw.items()
    }
    cli_variables = _parse_kv_pairs(env_vars or [], flag="--env-var")
    merged_variables.update(cli_variables)
    cli_secrets = _parse_kv_pairs(secrets or [], flag="--secret")
    if registry and merged_variables:
        console.print(
            "[bold yellow]⚠[/bold yellow] openenv.yaml variables: are only applied to Hugging Face Spaces and will be ignored with --registry"
        )
    if create_pr and merged_variables:
        console.print(
            "[bold yellow]⚠[/bold yellow] openenv.yaml variables: are not applied when using --create-pr; configure them after the PR is merged"
        )

    # Handle custom registry push
    if registry:
        console.print("[bold cyan]Preparing to push to custom registry...[/bold cyan]")
        if enable_interface:
            console.print("[bold cyan]Web interface will be enabled[/bold cyan]")

        # Import build functions
        from .build import _build_docker_image, _push_docker_image

        # Prepare build args for custom registry deployment
        build_args = {}
        if enable_interface:
            build_args["ENABLE_WEB_INTERFACE"] = "true"

        # Build Docker image from the environment directory
        tag = f"{registry}/{env_name}"
        console.print(f"[bold cyan]Building Docker image: {tag}[/bold cyan]")

        success = _build_docker_image(
            env_path=env_dir,
            tag=tag,
            build_args=build_args if build_args else None,
        )

        if not success:
            console.print("[bold red]✗ Docker build failed[/bold red]")
            raise typer.Exit(1)

        console.print("[bold green]✓ Docker build successful[/bold green]")

        # Push to registry
        console.print(f"[bold cyan]Pushing to registry: {registry}[/bold cyan]")

        success = _push_docker_image(
            tag, registry=None
        )  # Tag already includes registry

        if not success:
            console.print("[bold red]✗ Docker push failed[/bold red]")
            raise typer.Exit(1)

        console.print("\n[bold green]✓ Deployment complete![/bold green]")
        console.print(f"[bold]Image:[/bold] {tag}")
        return

    ignore_patterns = _load_ignore_patterns(env_dir, exclude)

    # Ensure authentication for HuggingFace
    username = _ensure_hf_authenticated()

    # Determine repo_id
    if not repo_id:
        repo_id = f"{username}/{env_name}"

    if repo_id.count("/") > 1:
        raise typer.BadParameter(
            f"Invalid repo-id format: {repo_id!r}. Repo id must be in the form 'repo_name' or 'namespace/repo_name'."
        )
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    # Initialize Hugging Face API
    api = HfApi()

    # Prepare staging directory
    deployment_type = (
        "with web interface" if enable_interface else "without web interface"
    )
    console.print(
        f"[bold cyan]Preparing files for Hugging Face deployment ({deployment_type})...[/bold cyan]"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir) / "staging"
        _prepare_staging_directory(
            env_dir,
            env_name,
            staging_dir,
            ignore_patterns=ignore_patterns,
            base_image=base_image,
            enable_interface=enable_interface,
        )

        if count > 1:
            base_repo_id = repo_id
            for i in range(1, count + 1):
                instance_repo_id = f"{base_repo_id}-{i}"
                console.print(
                    f"\n[bold cyan][{i}/{count}] Deploying {instance_repo_id}...[/bold cyan]"
                )
                _create_hf_space(
                    instance_repo_id, api, private=private, hardware=hardware
                )
                _upload_to_hf_space(
                    instance_repo_id,
                    staging_dir,
                    api,
                    private=private,
                    create_pr=False,
                    ignore_patterns=ignore_patterns,
                    delete_patterns=delete,
                )
                _apply_space_variables_and_secrets(
                    instance_repo_id, merged_variables, cli_secrets, api
                )
            console.print(
                f"\n[bold green]✓ All {count} instances deployed![/bold green]"
            )
            for i in range(1, count + 1):
                console.print(
                    f"Visit instance {i}: https://huggingface.co/spaces/{base_repo_id}-{i}"
                )
        else:
            # Create/verify space (no-op if exists; needed when pushing to own new repo)
            if not create_pr:
                _create_hf_space(repo_id, api, private=private, hardware=hardware)
            # When create_pr we rely on upload_folder to create branch and PR

            # Upload files
            _upload_to_hf_space(
                repo_id,
                staging_dir,
                api,
                private=private,
                create_pr=create_pr,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete,
            )

            # Skip variable/secret configuration in PR mode since the target
            # repo's live environment should only change when the PR is merged.
            if not create_pr:
                _apply_space_variables_and_secrets(
                    repo_id, merged_variables, cli_secrets, api
                )

            console.print("\n[bold green]✓ Deployment complete![/bold green]")
            console.print(
                f"Visit your space at: https://huggingface.co/spaces/{repo_id}"
            )
