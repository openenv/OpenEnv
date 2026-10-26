"""Initialize a new OpenEnv environment."""

from __future__ import annotations

import random
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Annotated, Dict, List, Tuple

import typer

from .._cli_utils import console

app = typer.Typer(help="Initialize a new OpenEnv environment")


def _snake_to_pascal(snake_str: str) -> str:
    """Convert snake_case to PascalCase (e.g., 'my_env' -> 'MyEnv')."""
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _get_env_prefix(env_name: str) -> str:
    """Extract the prefix for class names (e.g., 'my_env' -> 'My', 'test_env' -> 'Test')."""
    # Remove trailing '_env' if present
    if env_name.endswith("_env"):
        base = env_name[:-4]  # Remove '_env'
    else:
        base = env_name

    # If empty or just one part, use the whole thing
    if not base or "_" not in base:
        return base.capitalize() if base else env_name.capitalize()

    # PascalCase all parts except the last
    parts = base.split("_")
    return "".join(word.capitalize() for word in parts)


def _snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase (e.g., 'my_env' -> 'myEnv')."""
    parts = snake_str.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def _snake_to_title(snake_str: str) -> str:
    """Convert snake_case to Title Case (e.g., 'my_env' -> 'My Env')."""
    return " ".join(word.capitalize() for word in snake_str.split("_"))


def _validate_env_name(name: str) -> str:
    """Validate environment name (must be valid Python identifier in snake_case)."""
    if not name:
        raise typer.BadParameter("Environment name cannot be empty")

    # Check if it's a valid Python identifier
    if not name.isidentifier():
        raise typer.BadParameter(
            f"Environment name '{name}' is not a valid Python identifier. Use snake_case (e.g., 'my_env', 'game_env')."
        )

    # Check if it starts with a number
    if name[0].isdigit():
        raise typer.BadParameter(
            f"Environment name '{name}' cannot start with a number."
        )

    return name


def _get_random_hf_space_config() -> Dict[str, str]:
    """
    Get random Hugging Face Space configuration values.

    Returns:
        `dict` with keys `emoji`, `colorFrom`, and `colorTo`.
    """
    # Valid emojis (emoji-only characters)
    emojis = [
        "🎮",
        "🎯",
        "🚀",
        "🌟",
        "🎨",
        "🎪",
        "🎭",
        "🎬",
        "🎤",
        "🎧",
        "🎵",
        "🎶",
        "🎸",
        "🎹",
        "🥁",
        "🎺",
        "🎻",
        "🎼",
        "🎯",
        "🎲",
        "🎳",
        "🎰",
        "🎴",
        "🃏",
        "🀄",
        "🎴",
        "🎨",
        "🖼️",
        "🎬",
        "🎭",
        "🎪",
        "🎤",
        "🎧",
        "🎵",
        "🎶",
        "🎸",
        "🎹",
        "🎺",
        "🎻",
        "🥁",
        "🎯",
        "🎲",
        "🎳",
        "🎰",
        "🏀",
        "⚽",
        "🏈",
        "⚾",
        "🎾",
        "🏐",
        "🏉",
        "🎱",
        "🏓",
        "🏸",
        "🥅",
        "🏒",
        "🏑",
        "🏏",
        "⛳",
        "🏹",
        "🎣",
        "🥊",
        "🥋",
        "🎽",
        "🏅",
        "🎖️",
        "🏆",
        "🥇",
        "🥈",
        "🥉",
        "🔊",
        "🔉",
        "🔈",
        "🔇",
        "📢",
        "📣",
        "📯",
        "🔔",
        "🔕",
        "📻",
        "📡",
        "💻",
        "🖥️",
        "🖨️",
        "⌨️",
        "🖱️",
        "🖲️",
        "🕹️",
        "🗜️",
        "💾",
        "💿",
        "📀",
        "📼",
        "📷",
        "📸",
        "📹",
        "🎥",
        "📽️",
        "🎞️",
        "📞",
        "☎️",
        "📟",
        "📠",
        "📺",
        "📻",
        "🎙️",
        "🎚️",
        "🎛️",
        "⏱️",
        "⏲️",
        "⏰",
        "🕰️",
        "⌚",
        "📱",
        "📲",
        "💻",
        "⌨️",
        "🖥️",
        "🖨️",
        "🖱️",
    ]

    # Valid colors from HF Spaces config reference
    colors = ["red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"]

    return {
        "emoji": random.choice(emojis),
        "colorFrom": random.choice(colors),
        "colorTo": random.choice(colors),
    }


def _create_template_replacements(env_name: str) -> Dict[str, str]:
    """
    Create comprehensive template replacement dictionary.

    Supports all naming conventions:
    - PascalCase for class names
    - camelCase for variable names
    - snake_case for module names, file paths
    """
    env_prefix = _get_env_prefix(env_name)
    env_camel = _snake_to_camel(env_name)
    env_title = _snake_to_title(env_name)

    # Get random HF Space config values
    hf_config = _get_random_hf_space_config()

    replacements = {
        # Template placeholders (MUST come first - full class names before partial)
        "__ENV_CLASS_NAME__Environment": f"{env_prefix}Environment",
        "__ENV_CLASS_NAME__Action": f"{env_prefix}Action",
        "__ENV_CLASS_NAME__Observation": f"{env_prefix}Observation",
        "__ENV_CLASS_NAME__Env": f"{env_prefix}Env",
        # Template placeholders (partial - must come after full replacements)
        "__ENV_NAME__": env_name,
        "__ENV_CLASS_NAME__": env_prefix,  # Use prefix, not full PascalCase
        "__ENV_TITLE_NAME__": env_title,
        "__ENV_CAMEL_NAME__": env_camel,
        # Default image tag of `openenv build`
        "__ENV_IMAGE_NAME__": f"openenv-{env_name.removesuffix('_env')}",
        # Hugging Face Space config placeholders
        "__HF_EMOJI__": hf_config["emoji"],
        "__HF_COLOR_FROM__": hf_config["colorFrom"],
        "__HF_COLOR_TO__": hf_config["colorTo"],
    }

    return replacements


def _replace_in_content(content: str, replacements: Dict[str, str]) -> str:
    """Replace all occurrences in content using case-sensitive replacements."""
    result = content
    # Sort by length (longest first) to avoid partial replacements
    for old, new in sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(old, new)
    return result


def _should_rename_file(filename: str, env_name: str) -> Tuple[bool, str]:
    """
    Check if a file should be renamed and return the new name.

    Handles template placeholders in filenames like:
    - `__ENV_NAME___environment.py` → `<env_name>_environment.py`
    """
    # Check for template placeholder
    if "__ENV_NAME__" in filename:
        new_name = filename.replace("__ENV_NAME__", env_name)
        return True, new_name

    return False, filename


def _copy_and_template_file(
    src_path: Path,
    dest_path: Path,
    replacements: Dict[str, str],
) -> None:
    """Copy a file and apply template replacements."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Read source file
        content = src_path.read_bytes()

        # Try to decode as text and apply replacements
        try:
            text = content.decode("utf-8")
            # Normalize line endings to LF before applying replacements
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = _replace_in_content(text, replacements)
            dest_path.write_text(text, encoding="utf-8", newline="\n")
        except UnicodeDecodeError:
            # Binary file, just copy
            dest_path.write_bytes(content)
    except Exception as e:
        raise RuntimeError(
            f"Failed to copy template file {src_path} to {dest_path}: {e}"
        ) from e


def _resolve_template_path(template_pkg: str, template_dir: str) -> Path:
    """Resolve the template package directory used by `openenv init`."""
    # Get the package path using importlib.resources but avoid importing the template package
    # We'll use the package's __file__ to get the directory path
    import importlib

    try:
        # Import the parent package (not the template package itself)
        if "." in template_pkg:
            parent_pkg = ".".join(template_pkg.split(".")[:-1])
            pkg = importlib.import_module(parent_pkg)
            template_path = Path(pkg.__file__).parent / template_pkg.split(".")[-1]
        else:
            pkg = importlib.import_module(template_pkg.split(".")[0])
            template_path = Path(pkg.__file__).parent / template_pkg.split(".")[-1]
    except Exception:
        # Fallback: try to use resources.files but handle import errors
        try:
            base = resources.files(template_pkg.split(".")[0])
            template_path = base.joinpath(*template_pkg.split(".")[1:])
            if not template_path.exists():
                raise FileNotFoundError(f"Template directory not found: {template_pkg}")
        except Exception as e:
            raise FileNotFoundError(
                f"Template directory not found: {template_pkg}"
            ) from e

    if template_dir:
        template_path = template_path / template_dir

    if not template_path.exists() or not template_path.is_dir():
        raise FileNotFoundError(
            f"Template directory not found: {template_pkg}.{template_dir}"
        )

    return template_path


def _copy_template_directory(
    template_pkg: str,
    template_dir: str,
    dest_dir: Path,
    replacements: Dict[str, str],
    env_name: str,
) -> List[Path]:
    """Recursively copy template directory and apply replacements."""
    created_files: List[Path] = []
    template_path = _resolve_template_path(template_pkg, template_dir)

    # Walk through all files in template directory using Path
    for item in template_path.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(template_path)
            dest_path = dest_dir / rel_path

            # Apply filename templating
            should_rename, new_name = _should_rename_file(dest_path.name, env_name)
            if should_rename:
                dest_path = dest_path.parent / new_name

            # Copy and apply replacements
            _copy_and_template_file(item, dest_path, replacements)
            created_files.append(dest_path)

    return created_files


def _generate_uv_lock(env_dir: Path) -> bool:
    """Generate uv.lock from pyproject.toml using uv."""
    pyproject_path = env_dir / "pyproject.toml"

    if not pyproject_path.exists():
        return False

    try:
        cmd = [
            "uv",
            "lock",
            "--directory",
            str(env_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.stdout:
            console.print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        console.print(
            f"[yellow]Warning: Could not generate uv.lock: {e.stderr}[/yellow]"
        )
        return False
    except FileNotFoundError:
        console.print(
            "[yellow]Warning: 'uv' not found. Install it to generate uv.lock[/yellow]"
        )
        return False


@app.command()
def init(
    env_name: Annotated[
        str,
        typer.Argument(
            help="Name of the environment to create (snake_case, e.g., 'my_env')"
        ),
    ],
    output_dir: Annotated[
        str | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory (defaults to current working directory)",
        ),
    ] = None,
) -> None:
    """
    Initialize a new OpenEnv environment.

    Creates a new directory with the environment name and generates all necessary
    files based on the OpenEnv template structure.

    Examples:

        ```bash
        $ openenv init my_game_env
        $ openenv init my_env --output-dir /path/to/projects
        ```
    """
    # Validate environment name
    env_name = _validate_env_name(env_name)

    # Determine output directory
    base_dir = Path(output_dir).resolve() if output_dir else Path.cwd().resolve()
    env_dir = base_dir / env_name

    # Check if directory already exists
    if env_dir.exists():
        if env_dir.is_file():
            raise typer.BadParameter(f"Path '{env_dir}' exists and is a file")
        if any(env_dir.iterdir()):
            raise typer.BadParameter(
                f"Directory '{env_dir}' already exists and is not empty. "
                "Please choose a different name or remove the existing directory."
            )

    try:
        # Create template replacements
        replacements = _create_template_replacements(env_name)

        # Create environment directory
        env_dir.mkdir(parents=True, exist_ok=True)

        console.print(
            f"[bold cyan]Creating OpenEnv environment '{env_name}'...[/bold cyan]"
        )

        # Copy template files from template structure
        template_pkg = "openenv.cli.templates.openenv_env"
        created_files = _copy_template_directory(
            template_pkg,
            "",
            env_dir,
            replacements,
            env_name,
        )

        console.print(f"[bold green]✓[/bold green] Created {len(created_files)} files")

        # Generate uv.lock
        console.print("\n[bold]Generating uv.lock...[/bold]")
        if _generate_uv_lock(env_dir):
            console.print("[green]✓[/green] Generated uv.lock")
        else:
            console.print("[yellow]⚠[/yellow] Could not generate uv.lock automatically")
            console.print("    You can generate it manually with:")
            console.print(f"    cd {env_dir} && uv lock")

        console.print(
            f"\n[bold green]Environment created successfully at: {env_dir}[/bold green]"
        )
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  cd {env_dir}")
        console.print(
            f"  # Edit your environment implementation in server/{env_name}_environment.py"
        )
        console.print("  # Edit your models in models.py")
        console.print("  # Install dependencies: uv sync")
        console.print("\n  # To integrate into OpenEnv repo:")
        console.print(f"  # 1. Copy this directory to <repo_root>/envs/{env_name}")
        console.print(f"  # 2. Build from repo root: openenv build envs/{env_name}")
        console.print(
            f"  # 3. Run your image: docker run -p 8000:8000 {replacements['__ENV_IMAGE_NAME__']}:latest"
        )

    except Exception as e:
        # Cleanup on error
        if env_dir.exists() and env_dir.is_dir():
            try:
                shutil.rmtree(env_dir)
            except Exception:
                pass

        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e
