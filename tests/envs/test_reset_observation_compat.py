"""Env apps must start on openenv-core releases without `reset_observation_cls`."""

import importlib
import sys

import pytest
from fastapi import FastAPI
from openenv.core.env_server import http_server
from openenv.core.env_server.types import Observation

ENV_APPS = [
    "coding_tools_env.server.app",
    "echo_env.server.app",
    "finqa_env.server.app",
    "jupyter_env.server.app",
    "terminus_env.server.app",
]


def _released_create_app(
    env,
    action_cls,
    observation_cls,
    env_name=None,
    max_concurrent_envs=None,
    concurrency_config=None,
    gradio_builder=None,
    custom_tab_name="Custom",
    custom_tab_primary=False,
    show_default_tab=True,
    title_override=None,
):
    """Released `create_app` signature: no reset model, no `**kwargs`."""
    return FastAPI()


def _package_modules(package):
    return [m for m in sys.modules if m == package or m.startswith(package + ".")]


@pytest.fixture
def import_fresh():
    """Import an env app with a patched `create_app`, then restore the module cache."""
    saved, packages = {}, set()

    def load(module_name):
        package = module_name.split(".")[0]
        packages.add(package)
        for name in _package_modules(package):
            saved.setdefault(name, sys.modules.pop(name))
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            pytest.skip(f"{module_name} dependencies unavailable: {exc}")

    yield load
    for package in packages:
        for name in _package_modules(package):
            del sys.modules[name]
    sys.modules.update(saved)


@pytest.mark.parametrize("module_name", ENV_APPS)
def test_app_starts_on_core_without_reset_observation_cls(
    monkeypatch, import_fresh, module_name
):
    monkeypatch.setattr(http_server, "create_app", _released_create_app)
    module = import_fresh(module_name)
    assert isinstance(module.app, FastAPI)


@pytest.mark.parametrize("module_name", ENV_APPS)
def test_app_declares_reset_observation_when_core_supports_it(
    monkeypatch, import_fresh, module_name
):
    calls = []

    def current_create_app(*args, reset_observation_cls=None, state_cls=None, **kwargs):
        calls.append((reset_observation_cls, state_cls))
        return FastAPI()

    monkeypatch.setattr(http_server, "create_app", current_create_app)
    module = import_fresh(module_name)
    state_models = {
        "coding_tools_env.server.app": "CodingToolsState",
        "finqa_env.server.app": "FinQAState",
        "jupyter_env.server.app": "JupyterState",
        "terminus_env.server.app": "TerminusState",
    }
    state_name = state_models.get(module_name)
    state_cls = getattr(module, state_name) if state_name else None
    assert calls == [(Observation, state_cls)]
