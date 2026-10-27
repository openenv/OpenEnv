# SPDX-License-Identifier: BSD-3-Clause

"""
OpenEnv CLI entry point.

This module provides the main entry point for the OpenEnv command-line interface,
following the Hugging Face CLI pattern.
"""

import sys

import typer
from openenv.cli.commands import (
    build,
    catalog,
    collect,
    fork,
    harbor,
    import_env,
    init,
    push,
    serve,
    skills,
    validate,
)

# Create the main CLI app
app = typer.Typer(
    name="openenv",
    help="OpenEnv - An e2e framework for creating, deploying and using isolated execution environments for agentic RL training",
    no_args_is_help=True,
)

# Register commands
app.command(name="init", help="Initialize a new OpenEnv environment")(init.init)
app.command(name="import", help="Import a third-party environment into OpenEnv")(
    import_env.import_env
)
app.command(name="build", help="Build Docker images for OpenEnv environments")(
    build.build
)
app.add_typer(
    harbor.app,
    name="harbor",
    help="Run Harbor tasks with token-level capture (requires: pip install openenv\\[harbor])",
)
app.command(name="validate", help="Validate a package against the OpenEnv quality bar")(
    validate.validate
)
app.command(
    name="push",
    help="Push an OpenEnv environment to Hugging Face Spaces or custom registry",
)(push.push)
app.command(name="serve", help="Serve environments locally (TODO: Phase 4)")(
    serve.serve
)
app.command(
    name="fork",
    help="Fork (duplicate) a Hugging Face Space to your account",
)(fork.fork)
app.add_typer(
    skills.app,
    name="skills",
    help="Manage OpenEnv skills for AI assistants",
)
app.command(
    name="collect",
    help="Collect rollouts from a deployed OpenEnv environment",
)(collect.collect)
app.add_typer(catalog.app, name="catalog")
app.command(name="discover")(catalog.discover)


# Entry point for setuptools
def main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
