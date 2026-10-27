# SPDX-License-Identifier: BSD-3-Clause

"""An env that declares a `State` subclass must serve it on `/schema` and `/state`.

`HTTPEnvServer` defaults `state_cls` to the base `State`, which publishes only
`episode_id` and `step_count` and strips every field a subclass declares from the
`/state` body. An env therefore has to pass its own class to the app factory; declaring
`class FooState(State)` is not enough on its own.

The WebSocket `state` frame calls `model_dump()` on the live environment and keeps those
fields either way, so an env that forgets this does not fail anywhere. It just serves two
different answers for the same object depending on the transport, which is what makes the
omission worth a test rather than a review note.

Checked with `ast` rather than by importing: importing every env's `server/app.py` would
pull in playwright, carla, dm_control and the rest of the optional-dependency tail, so this
would skip on exactly the machines that should be guarding it.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENVS_DIR = REPO_ROOT / "envs"

# Every helper that ends up constructing an HTTPEnvServer.
APP_FACTORIES = {"create_app", "create_fastapi_app", "HTTPEnvServer"}


def _parse(path: pathlib.Path) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            # Parsing every env module surfaces pre-existing SyntaxWarnings (an env with
            # an unescaped backslash in a docstring, say). They belong to that env, not to
            # what this test checks, and they drown the run's own output.
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def _state_subclasses(env_dir: pathlib.Path) -> set[str]:
    """Names in this env that subclass `State` directly."""
    found: set[str] = set()
    for module in env_dir.rglob("*.py"):
        tree = _parse(module)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                getattr(base, "id", getattr(base, "attr", None)) == "State"
                for base in node.bases
            ):
                found.add(node.name)
    return found


def _factory_calls(tree: ast.Module) -> list[ast.Call]:
    """Every app-factory call in the module.

    Checking only the last one is not enough: `repl_env` and `textarena_env` build the app
    in one of two branches chosen by `inspect.signature(create_app)`, and the branch that
    actually runs against current OpenEnv is the first of the two.
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in APP_FACTORIES
    ]


def _compatibility_fallback_calls(tree: ast.Module) -> set[int]:
    """Factory calls that only run against an openenv too old to accept the parameter.

    `repl_env` and `textarena_env` pin `openenv>=0.2.2` and probe
    `inspect.signature(create_app)` before passing anything newer. The `else` arm of that
    probe is the path for a release predating these parameters, so it must NOT pass
    `state_cls`: doing so would raise `TypeError` on exactly the installation it exists to
    support. Those calls are therefore exempt, while the arm that runs against current
    OpenEnv still has to be wired.
    """
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not node.orelse:
            continue
        if "parameters" not in ast.unparse(node.test):
            continue
        for fallback in node.orelse:
            for inner in ast.walk(fallback):
                if (
                    isinstance(inner, ast.Call)
                    and getattr(inner.func, "id", "") in APP_FACTORIES
                ):
                    exempt.add(id(inner))
    return exempt


def _state_cls_from_kwargs_dict(tree: ast.Module, dict_name: str) -> str | None:
    """Resolve `state_cls` for a call that splats a kwargs dict.

    Those two envs gate each newer parameter behind `inspect.signature`, so the value is
    assigned into a dict (`kwargs["state_cls"] = FooState`) rather than passed inline.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == dict_name
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "state_cls"
                ):
                    return ast.unparse(node.value)
    return None


def _state_cls_argument(tree: ast.Module, call: ast.Call) -> str | None:
    """The `state_cls` value this call passes, inline or through a splatted dict."""
    for keyword in call.keywords:
        if keyword.arg == "state_cls":
            return ast.unparse(keyword.value)
    for keyword in call.keywords:
        if keyword.arg is None and isinstance(keyword.value, ast.Name):
            resolved = _state_cls_from_kwargs_dict(tree, keyword.value.id)
            if resolved is not None:
                return resolved
    return None


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _envs_declaring_state() -> list[str]:
    envs = []
    for env_dir in sorted(p for p in ENVS_DIR.iterdir() if p.is_dir()):
        if not (env_dir / "server" / "app.py").exists():
            continue
        if _state_subclasses(env_dir):
            envs.append(env_dir.name)
    return envs


ENVS_WITH_STATE = _envs_declaring_state()


def test_fixture_finds_envs():
    """Guard against the discovery above silently matching nothing."""
    assert len(ENVS_WITH_STATE) > 20, ENVS_WITH_STATE


@pytest.mark.parametrize("env_name", ENVS_WITH_STATE)
def test_env_serves_its_own_state_class(env_name: str):
    env_dir = ENVS_DIR / env_name
    tree = _parse(env_dir / "server" / "app.py")
    assert tree is not None, f"{env_name}: server/app.py does not parse"

    exempt = _compatibility_fallback_calls(tree)
    calls = [call for call in _factory_calls(tree) if id(call) not in exempt]
    if not calls:
        pytest.skip(f"{env_name} builds its app outside a recognised factory call")

    declared = sorted(_state_subclasses(env_dir))
    imported = _imported_names(tree)
    for call in calls:
        where = f"{env_name} server/app.py line {call.lineno}"
        passed = _state_cls_argument(tree, call)
        assert passed is not None, (
            f"{where} declares {declared} but this app factory call does not pass "
            "state_cls, so /schema and /state fall back to the base State model and "
            "drop every field the subclass adds"
        )
        assert passed in declared, (
            f"{where} passes state_cls={passed}, which is not a State subclass declared "
            f"in this env ({declared})"
        )
        assert passed in imported, (
            f"{where} passes state_cls={passed} but never imports it"
        )
