# SPDX-License-Identifier: BSD-3-Clause

"""`openenv check`: see whether openenvd isolates here, and prove that it does."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Check whether and how openenvd isolates an environment")


@app.command("preflight")
def preflight(
    manifest: Annotated[
        Path, typer.Option(help="The environment's openenv.yaml")
    ] = Path("openenv.yaml"),
    cgroup_root: Annotated[Path, typer.Option(help="The unit's cgroup root")] = Path(
        "/sys/fs/cgroup"
    ),
) -> None:
    """Say whether openenvd would isolate here, and if not, why."""
    from openenv.core.openenvd.daemon import main

    raise typer.Exit(
        main(
            [
                "--preflight",
                "--manifest",
                str(manifest),
                "--cgroup-root",
                str(cgroup_root),
            ]
        )
    )


@app.command("enforcement")
def enforcement(
    manifest: Annotated[
        Path, typer.Option(help="The environment's openenv.yaml")
    ] = Path("openenv.yaml"),
    assets: Annotated[
        Path | None, typer.Option(help="Asset root (default: the manifest's directory)")
    ] = None,
    state: Annotated[
        Path, typer.Option(help="Scratch directory for openenvd's state")
    ] = Path("/var/lib/openenvd/check"),
    cgroup_root: Annotated[Path, typer.Option(help="The unit's cgroup root")] = Path(
        "/sys/fs/cgroup"
    ),
) -> None:
    """
    Start a throwaway container in every zone and try what the zone must not do.

    Run it inside the unit, privileged. It exits non-zero if isolation is off or
    if any zone could do something it shouldn't.
    """
    from openenv.core.openenvd.daemon import main

    argv = [
        "--check",
        "--manifest",
        str(manifest),
        "--state",
        str(state),
        "--cgroup-root",
        str(cgroup_root),
    ]
    if assets is not None:
        argv += ["--assets", str(assets)]
    raise typer.Exit(main(argv))
