# SPDX-License-Identifier: BSD-3-Clause

"""Every `openenv.yaml` `app` field must name a module that actually imports.

The cloud providers do not read the Dockerfile `CMD` when a manifest is
present: `ModalProvider._discover_server_cmd` (and its Daytona twin) locate
`openenv.yaml` inside the sandbox, take the `app` field verbatim, and run
`cd /app/env && python -m uvicorn <app>`. So the `app` field alone decides
whether the env starts on Modal or Daytona, and a wrong value is invisible
locally, where `docker run` uses the Dockerfile `CMD` instead.

Two things have to hold, and file existence is only the first:

1. The module has to exist on disk under the env directory.
2. It has to be importable *under the name the manifest declares*. Declaring
   `server.app:app` makes `server` a top-level package, so a `from ..models`
   inside it climbs past the top level and raises `ImportError: attempted
   relative import beyond top-level package`. An env whose server package
   reaches outside itself must be declared through its own distribution
   package (`<env>.server.app:app`), which is what makes `..models` resolve.

Checking (2) by importing is not an option: it would pull in playwright,
carla, dm_control and the rest of the optional-dependency tail, so the module
would skip on exactly the CI machines that should be guarding this. The escape
is visible in the source, so this walks the import graph with `ast` instead --
no env dependency is imported, and the check runs anywhere.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENVS_DIR = REPO_ROOT / "envs"


def _manifests_with_app() -> list[pathlib.Path]:
    """Env manifests that declare an `app` field.

    A few envs still carry the pre-spec manifest format, which has no `app`
    field at all. Bringing those onto the current schema is a separate problem;
    skipping them keeps this test about the one thing it can prove.
    """
    found = []
    for manifest in sorted(ENVS_DIR.glob("*/openenv.yaml")):
        data = yaml.safe_load(manifest.read_text()) or {}
        if isinstance(data, dict) and isinstance(data.get("app"), str):
            found.append(manifest)
    return found


MANIFESTS = _manifests_with_app()


def _runtime_launches_under_envs_prefix(env_dir: pathlib.Path) -> bool:
    """Whether this env's *image* imports its server as `envs.<env>....`.

    Resolving against the repo root unconditionally would be wrong: the source
    tree always contains `envs/<env>/server/app.py`, so every env would accept
    an `envs.` prefix -- including envs whose image has no `envs` directory at
    all. The repo's layout says nothing about the container's.

    A repo-root build context is not the signal either. `coding_env` is built
    from the repo root but `pip install ./envs/coding_env/` turns it into a real
    distribution, so its `CMD` launches `coding_env.server.app:app` with no
    prefix. The only evidence that the prefix is importable at runtime is the
    image launching it that way -- as `grid_world_env` does, keeping the env at
    `/app/envs/grid_world_env` with `/app` on `PYTHONPATH`.
    """
    prefix = f"envs.{env_dir.name}."
    for script in [env_dir / "server" / "Dockerfile", *env_dir.rglob("*.sh")]:
        try:
            text = script.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if any(prefix in line for line in text.splitlines() if "uvicorn" in line):
            return True
    return False


def _resolve(env_dir: pathlib.Path, dotted: str) -> pathlib.Path | None:
    """Map a dotted module path to its file, relative to the env directory.

    The env directory is both the container's working directory and the root of
    the env's own distribution package (`pyproject.toml` maps the package name
    to `.`), so `<env>.server.app` and `server.app` reach the same file.
    """
    parts = dotted.split(".")
    candidates = [(env_dir, parts)]
    # `<env>.server.app`: the env directory is itself that package.
    if parts and parts[0] == env_dir.name:
        candidates.append((env_dir, parts[1:]))
    # `envs.<env>.server.app`: only for images that really do import it that
    # way, never as a blanket fallback. See the helper above.
    if _runtime_launches_under_envs_prefix(env_dir):
        candidates.append((REPO_ROOT, parts))

    for root, rest in candidates:
        if not rest:
            found = root / "__init__.py"
            if found.is_file():
                return found
            continue
        relative = pathlib.Path(*rest)
        module = root / relative.with_suffix(".py")
        if module.is_file():
            return module
        package = root / relative / "__init__.py"
        if package.is_file():
            return package
    return None


def _guarded(tree: ast.Module) -> set[ast.ImportFrom]:
    """Relative imports wrapped in a `try` that catches the resulting error.

    Most envs support both layouts by attempting the in-repo relative import and
    falling back to a flat one:

        try:
            from ..models import ChatAction
        except ImportError:
            from models import ChatAction

    Those cannot strand the server, so they are not evidence of a wrong `app`
    value. Only an unguarded escape is.
    """
    guarded: set[ast.ImportFrom] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches = any(
            handler.type is None
            or (isinstance(handler.type, ast.Name) and handler.type.id in _CAUGHT)
            or (
                isinstance(handler.type, ast.Tuple)
                and any(
                    isinstance(e, ast.Name) and e.id in _CAUGHT
                    for e in handler.type.elts
                )
            )
            for handler in node.handlers
        )
        if not catches:
            continue
        # Both halves of the pattern are guarded: the attempt in the `try`, and
        # the fallback in the handler, which only runs when the attempt failed.
        # Envs write it in either order -- relative first with a flat fallback,
        # or flat first with a relative fallback.
        for stmt in list(node.body) + [s for h in node.handlers for s in h.body]:
            for inner in ast.walk(stmt):
                if isinstance(inner, ast.ImportFrom):
                    guarded.add(inner)
    return guarded


_CAUGHT = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _escaping_imports(env_dir: pathlib.Path, declared: str) -> list[tuple[str, str]]:
    """Relative imports that climb above `declared`'s root package.

    Walks every module reachable from the declared entry point by relative
    import, tracking the depth each one sits at *in the declared namespace*.
    A `from ..x import y` at level 2 is fine inside `<env>.server.app` (whose
    containing package is two deep) and fatal inside `server.app` (one deep) --
    which is precisely the difference the manifest chooses.

    Returns (module, offending import) pairs.
    """
    # Importing `a.b.c` executes every ancestor package first, so they are part
    # of the reachable set.
    parts = declared.split(".")
    queue = [".".join(parts[: i + 1]) for i in range(len(parts))]

    seen: set[str] = set()
    offenders: list[tuple[str, str]] = []

    while queue:
        dotted = queue.pop()
        if dotted in seen:
            continue
        seen.add(dotted)

        path = _resolve(env_dir, dotted)
        if path is None:
            continue

        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover - not our problem to report
            continue

        # The package that contains this module: for a package's __init__ that
        # is the package itself, otherwise the parent.
        own = dotted.split(".")
        container = own if path.name == "__init__.py" else own[:-1]
        guarded = _guarded(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            if node in guarded:
                continue

            if node.level > len(container):
                target = "." * node.level + (node.module or "")
                offenders.append((dotted, f"from {target} import ..."))
                continue

            # Resolve the sibling/parent module it points at and keep walking.
            base = container[: len(container) - (node.level - 1)]
            child = base + (node.module.split(".") if node.module else [])
            if child:
                queue.append(".".join(child))

    return offenders


def test_manifests_with_app_field_were_found() -> None:
    """Guard against the glob silently matching nothing."""
    assert len(MANIFESTS) > 20


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.parent.name)
def test_manifest_app_target_exists(manifest: pathlib.Path) -> None:
    env_dir = manifest.parent
    app = yaml.safe_load(manifest.read_text())["app"]

    assert ":" in app, (
        f"{manifest.relative_to(REPO_ROOT)} declares app={app!r}, which is not "
        "in uvicorn's '<module>:<attribute>' form."
    )

    declared = app.split(":", 1)[0]
    assert _resolve(env_dir, declared) is not None, (
        f"{manifest.relative_to(REPO_ROOT)} declares app={app!r}, but module "
        f"{declared!r} does not exist under "
        f"{env_dir.relative_to(REPO_ROOT)}. The cloud providers run "
        f"'cd /app/env && python -m uvicorn {app}', so this env fails to start "
        "on Modal and Daytona."
    )


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.parent.name)
def test_manifest_app_target_is_importable_as_declared(
    manifest: pathlib.Path,
) -> None:
    env_dir = manifest.parent
    app = yaml.safe_load(manifest.read_text())["app"]
    declared = app.split(":", 1)[0]

    if _resolve(env_dir, declared) is None:
        pytest.skip("covered by test_manifest_app_target_exists")

    offenders = _escaping_imports(env_dir, declared)
    detail = "\n".join(f"    {mod}: {imp}" for mod, imp in offenders)
    packaged = (
        f"{env_dir.name}.{declared}"
        if not declared.startswith(f"{env_dir.name}.")
        else declared
    )

    assert not offenders, (
        f"{manifest.relative_to(REPO_ROOT)} declares app={app!r}, but modules "
        f"reachable from {declared!r} import above its root package:\n{detail}\n"
        f"Imported under that name the server raises 'ImportError: attempted "
        f"relative import beyond top-level package', so the env cannot start "
        f"on Modal or Daytona, which run "
        f"'cd /app/env && python -m uvicorn {app}'. Declare it through the "
        f"env's own package instead: {packaged}:app"
    )


# The checks above only prove that the values currently in the tree pass. That
# is not the same as proving a wrong value fails, and the difference is not
# academic: an earlier version of this file resolved every dotted path against
# the repo root, so the very bug this suite exists to catch sailed through it.
# These pin the two sides of that distinction.


def test_rejects_envs_prefix_for_env_root_layout() -> None:
    """AWM's original value must not resolve, and must not become resolvable.

    `agent_world_model_env` is built with its own directory as the context
    (`COPY . /app/env`), so the image has no `envs` package and
    `envs.agent_world_model_env.server.app` cannot be imported -- even though
    `envs/agent_world_model_env/server/app.py` does exist in the source tree.
    """
    env_dir = ENVS_DIR / "agent_world_model_env"

    assert not _runtime_launches_under_envs_prefix(env_dir)
    assert _resolve(env_dir, "envs.agent_world_model_env.server.app") is None
    # The value it was corrected to still resolves.
    assert _resolve(env_dir, "agent_world_model_env.server.app") is not None


def test_accepts_envs_prefix_for_repo_root_layout() -> None:
    """`grid_world_env` genuinely runs under the prefix, so it must be allowed.

    Its image keeps the env at `/app/envs/grid_world_env` with `/app` on
    `PYTHONPATH` and launches `envs.grid_world_env.server.app:app`.
    """
    env_dir = ENVS_DIR / "grid_world_env"

    assert _runtime_launches_under_envs_prefix(env_dir)
    assert _resolve(env_dir, "envs.grid_world_env.server.app") is not None


def test_repo_root_build_context_alone_does_not_allow_the_prefix() -> None:
    """`coding_env` is built from the repo root but installs itself as a dist.

    Its `CMD` is `coding_env.server.app:app`, so the prefix is not importable
    there. This pins the distinction between "built from the repo root" and
    "imports under the envs prefix", which are not the same predicate.
    """
    assert not _runtime_launches_under_envs_prefix(ENVS_DIR / "coding_env")
