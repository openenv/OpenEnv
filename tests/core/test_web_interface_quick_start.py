# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Quick Start markdown shown next to the web interface."""

import re
import sys
import textwrap
from pathlib import Path

import pytest
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Action, EnvironmentMetadata
from openenv.core.env_server.web_interface import get_quick_start_markdown


class DemoAction(Action):
    text: str
    tokens: list[int]
    count: int = 1


def _write_package(root, name, init):
    """An environment package with a client module, as the server sees it on disk."""
    package = root / name
    package.mkdir()
    (package / "client.py").write_text(
        textwrap.dedent(
            """
            from openenv.core.env_client import EnvClient

            class Helper:
                pass

            class DemoEnv(
                EnvClient[object, object, object]
            ):
                pass
            """
        )
    )
    (package / "__init__.py").write_text(textwrap.dedent(init))


@pytest.fixture
def demo_env(tmp_path, monkeypatch):
    _write_package(
        tmp_path,
        "demo_env",
        """
        from .client import DemoEnv
        from .models import DemoAction
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("SPACE_HOST", raising=False)
    yield "demo_env"
    for module in [m for m in sys.modules if m.startswith("demo_env")]:
        sys.modules.pop(module)


def _metadata(name):
    return EnvironmentMetadata(name=name, description="")


def test_uses_the_client_class_and_real_action_fields(demo_env):
    md = get_quick_start_markdown(_metadata(demo_env), DemoAction, CallToolObservation)

    assert "from demo_env import DemoAction, DemoEnv" in md
    assert 'DemoEnv(base_url="http://localhost:8000")' in md
    assert 'env.step(DemoAction(text="...", tokens=[]))' in md
    assert "pip install" not in md


def test_does_not_import_the_package_or_its_client(demo_env):
    get_quick_start_markdown(_metadata(demo_env), DemoAction, CallToolObservation)

    assert "demo_env" not in sys.modules
    assert "demo_env.client" not in sys.modules


def test_finds_lazy_exports(tmp_path, monkeypatch):
    _write_package(
        tmp_path,
        "lazy_env",
        """
        __all__ = ["DemoEnv", "DemoAction"]

        def __getattr__(name):
            if name == "DemoEnv":
                from .client import DemoEnv
                return DemoEnv
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    md = get_quick_start_markdown(
        _metadata("lazy_env"), DemoAction, CallToolObservation
    )

    assert "from lazy_env import DemoAction, DemoEnv" in md


def test_on_a_space_uses_its_url_and_install_line(demo_env, monkeypatch):
    monkeypatch.setenv("SPACE_ID", "openenv/demo_env")
    monkeypatch.setenv("SPACE_HOST", "openenv-demo-env.hf.space")

    md = get_quick_start_markdown(_metadata(demo_env), DemoAction, CallToolObservation)

    assert "pip install git+https://huggingface.co/spaces/openenv/demo_env" in md
    assert 'DemoEnv(base_url="https://openenv-demo-env.hf.space")' in md


def test_mcp_env_lists_tools(demo_env):
    md = get_quick_start_markdown(
        _metadata(demo_env), CallToolAction, CallToolObservation
    )

    assert "from demo_env import DemoEnv" in md
    assert "env.list_tools()" in md
    assert "CallToolAction(" not in md


def test_env_without_client_package_points_to_readme(monkeypatch):
    monkeypatch.delenv("SPACE_HOST", raising=False)

    md = get_quick_start_markdown(
        _metadata("no_such_env_package"), CallToolAction, CallToolObservation
    )

    assert "README" in md
    assert "```python" not in md


def test_env_names_match_their_packages():
    """The Quick Start finds an env's client by the `env_name` it passes to `create_app`."""
    envs = Path(__file__).parents[2] / "envs"
    for app in envs.glob("*/server/app.py"):
        for name in re.findall(r'env_name="([^"]+)"', app.read_text()):
            assert name == app.parents[1].name, app
