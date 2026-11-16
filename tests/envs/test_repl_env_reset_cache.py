# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for REPL reset-time LLM cache invalidation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class _DummyLocalPythonExecutor:
    def __init__(self, *args, **kwargs) -> None:
        self.state = {}

    def send_tools(self, tools) -> None:
        self.tools = tools

    def send_variables(self, variables) -> None:
        self.state.update(variables)


def _install_smolagents_stub_if_needed() -> bool:
    if importlib.util.find_spec("smolagents") is not None:
        return False

    module = ModuleType("smolagents")
    module.LocalPythonExecutor = _DummyLocalPythonExecutor
    sys.modules["smolagents"] = module
    return True


_ENVS_ROOT = Path(__file__).resolve().parents[2] / "envs"
sys.path.insert(0, str(_ENVS_ROOT))
_installed_stub = _install_smolagents_stub_if_needed()
try:
    from repl_env.server.repl_environment import REPLEnvironment
finally:
    if _installed_stub:
        sys.modules.pop("smolagents", None)


def test_reset_rebuilds_llm_on_model_change(monkeypatch):
    """Changing `llm_model` between resets must rebuild the LLM functions."""
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    env = REPLEnvironment()
    calls: list[tuple[str | None, str | None]] = []

    def tracked(token, model):
        calls.append((token, model))
        env.llm_query_fn = lambda *args, **kwargs: ""
        env._runtime_controller = object()

    monkeypatch.setattr(env, "_create_llm_functions", tracked)
    env.reset(llm_model="model-A")
    env.reset(llm_model="model-B")

    assert [model for _, model in calls] == ["model-A", "model-B"]


def test_reset_no_rebuild_when_model_resolves_to_same_default(monkeypatch):
    """reset(llm_model=<default>) then reset() must not rebuild."""
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    env = REPLEnvironment()
    calls: list[tuple[str | None, str | None]] = []

    def tracked(token, model):
        calls.append((token, model))
        env.llm_query_fn = lambda *args, **kwargs: ""
        env._runtime_controller = object()

    monkeypatch.setattr(env, "_create_llm_functions", tracked)
    default = REPLEnvironment._resolve_model(None)
    env.reset(llm_model=default)
    env.reset()

    assert len(calls) == 1


def test_reset_preserves_constructor_llm_functions(monkeypatch):
    """Constructor-provided LLM functions are not HF runtime cache entries."""
    env = REPLEnvironment(llm_query_fn=lambda prompt: f"custom: {prompt}")
    calls: list[tuple[str | None, str | None]] = []

    def tracked(token, model):
        calls.append((token, model))

    monkeypatch.setattr(env, "_create_llm_functions", tracked)
    env.reset()

    assert calls == []
    assert env.llm_query_fn("prompt") == "custom: prompt"


@pytest.mark.parametrize("hf_token", [None, "", True, 0])
def test_reset_refuses_a_url_model_with_the_server_token(monkeypatch, hf_token):
    """`InferenceClient` sends its token to a model given as a URL and falls back to the server's
    `HF_TOKEN` for an empty or non-string one, so a client URL needs the client's own token."""
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    env = REPLEnvironment()
    with pytest.raises(ValueError, match="own hf_token"):
        env.reset(llm_model="https://attacker.example", hf_token=hf_token)


def test_reset_accepts_a_url_model_with_the_client_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    env = REPLEnvironment()
    env.reset(llm_model="https://my-endpoint.example", hf_token="client-tok")
    assert env.llm_query_fn is not None
