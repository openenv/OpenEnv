"""Seam behaviour: how each harness learns about the capture proxy, and how it is bounded.

A seam is the only place per-agent knowledge lives, so these tests assert on what a seam HANDS the
agent rather than on any rollout outcome.
"""

from __future__ import annotations

import sys
import types

import pytest
from openenv.harbor import seams

# --- step limits --------------------------------------------------------------------------------
# A cap is a training requirement, not a cost preference: AsyncGRPO packs every turn of a rollout
# into one row and each turn re-sends the whole conversation, so packed length grows with the SQUARE
# of the turn count. An unbounded rollout OOMs the loss step while every rollout log line reads fine.


def test_a_step_limit_reaches_the_agent_config():
    _, kwargs, _, _ = seams.get("mini-swe-agent").resolve(
        base_url="http://proxy", session="s1", model="Qwen/Qwen3.5-4B", step_limit=20
    )
    assert kwargs["config"]["agent"]["step_limit"] == 20


def test_no_step_limit_leaves_the_config_alone():
    _, kwargs, _, _ = seams.get("mini-swe-agent").resolve(
        base_url="http://proxy", session="s1", model="Qwen/Qwen3.5-4B"
    )
    assert "config" not in kwargs


def test_training_profile_disables_nested_opencode_calls_without_losing_provider():
    seam = seams.get("opencode")
    _, train, _, _ = seam.resolve(
        base_url="http://proxy", session="s1", model="m", training=True
    )
    config = train["opencode_config"]
    assert config["tools"] == {"task": False, "question": False, "webfetch": False}
    provider = config["provider"][seams.OPENCODE_PROVIDER]
    assert provider["options"]["baseURL"] == "http://proxy/v1"
    assert provider["options"]["apiKey"] == "s1"
    _, evaluation, _, _ = seam.resolve(base_url="http://proxy", session="s2", model="m")
    assert "tools" not in evaluation["opencode_config"]


def test_a_step_limit_merges_instead_of_replacing_a_seams_own_config():
    """opencode carries a provider block. A shallow update would drop it along with the base_url."""
    seam = seams.Seam(
        name="fake",
        dialect="openai_chat",
        kwargs=lambda base_url, session, model: {
            "config": {"provider": {"baseURL": base_url}}
        },
        step_limit=lambda n: {"config": {"agent": {"step_limit": n}}},
    )
    _, kwargs, _, _ = seam.resolve(
        base_url="http://proxy", session="s1", model="m", step_limit=7
    )
    assert kwargs["config"]["provider"]["baseURL"] == "http://proxy"
    assert kwargs["config"]["agent"]["step_limit"] == 7


def test_a_harness_that_cannot_cap_steps_warns_rather_than_dropping_it(
    caplog, monkeypatch
):
    """Silently ignoring the cap would let a caller believe its rollouts are bounded."""
    monkeypatch.setattr(seams, "_STEP_LIMIT_WARNED", set())
    with caplog.at_level("WARNING"):
        _, kwargs, _, _ = seams.get("opencode").resolve(
            base_url="http://proxy", session="s1", model="m", step_limit=20
        )
    assert "capture proxy enforces the model-call budget" in caplog.text
    assert "step_limit" not in str(kwargs)


def test_the_step_limit_warning_is_said_once_per_harness(caplog, monkeypatch):
    """A training run resolves a seam for every rollout; one line per harness is enough."""
    monkeypatch.setattr(seams, "_STEP_LIMIT_WARNED", set())
    with caplog.at_level("WARNING"):
        for session in ("s1", "s2", "s3"):
            seams.get("opencode").resolve(
                base_url="http://proxy", session=session, model="m", step_limit=20
            )
        seams.get("qwen-coder").resolve(
            base_url="http://proxy", session="s4", model="m", step_limit=20
        )
    warned = [
        r.getMessage()
        for r in caplog.records
        if "no native step-limit" in r.getMessage()
    ]
    assert len(warned) == 2
    assert warned[0].startswith("opencode") and warned[1].startswith("qwen-coder")


def test_copilot_byok_routes_without_a_github_model_service():
    from openenv.harbor.seams import SEAMS

    _, _, env, _ = SEAMS["copilot-cli"].resolve(
        base_url="https://capture.test", session="session-key", model="Qwen3.5-4B"
    )
    assert env["COPILOT_PROVIDER_BASE_URL"] == "https://capture.test/v1"
    assert env["COPILOT_PROVIDER_API_KEY"] == "session-key"
    assert env["COPILOT_OFFLINE"] == "true"


def test_grok_primary_and_auxiliary_models_share_capture_endpoint():
    from openenv.harbor.seams import SEAMS

    _, kwargs, env, _ = SEAMS["grok-build"].resolve(
        base_url="https://capture.test", session="session-key", model="Qwen3.5-4B"
    )
    config = kwargs["grok_config"]
    assert set(config["models"].values()) == {"Qwen3.5-4B"}
    assert config["model"]["Qwen3.5-4B"]["base_url"] == "https://capture.test/v1"
    assert config["model"]["Qwen3.5-4B"]["env_key"] == "OPENAI_API_KEY"
    assert env["OPENAI_API_KEY"] == "session-key"


@pytest.fixture
def custom_agents(monkeypatch):
    """A module of agents to name by import path: one host-side, one opencode subclass, one unknown
    installed agent."""
    # Not bare "harbor": without Harbor (Python 3.11), `examples/harbor` imports under that name
    # once another test puts `examples/` on sys.path.
    pytest.importorskip("harbor.agents.base")
    from harbor.agents.base import BaseAgent
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.agents.installed.opencode import OpenCode

    module = types.ModuleType("custom_agents_for_test")

    class HostSideAgent(BaseAgent):
        @staticmethod
        def name() -> str:
            return "my-host-side-agent"

    class MyOpenCode(OpenCode):
        pass

    class UnknownInstalledAgent(BaseInstalledAgent):
        @staticmethod
        def name() -> str:
            return "my-installed-agent"

    for cls in (HostSideAgent, MyOpenCode, UnknownInstalledAgent):
        setattr(module, cls.__name__, cls)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.__name__


def test_a_host_side_agent_of_your_own_gets_the_proxy_as_kwargs(custom_agents):
    """`--harness module:Class` is documented; it used to raise "no seam" for any import path."""
    name = f"{custom_agents}:HostSideAgent"
    seam = seams.get(name)
    model, kwargs, agent_env, proc_env = seam.resolve(
        base_url="https://capture.test", session="session-key", model="Qwen/Qwen3.5-9B"
    )
    assert seam.import_path == name
    assert seam.status == "untested"
    assert model == "Qwen3.5-9B"
    assert kwargs == {"api_base": "https://capture.test/v1", "api_key": "session-key"}
    # Nothing in the sandbox's or our process's environment, so nothing can reach a grader.
    assert agent_env == {} and proc_env == {}


def test_a_subclass_of_a_known_agent_is_wired_like_it(custom_agents):
    name = f"{custom_agents}:MyOpenCode"
    seam = seams.get(name)
    model, kwargs, _, _ = seam.resolve(
        base_url="https://capture.test", session="session-key", model="m"
    )
    assert seam.import_path == name
    assert model == seams.SEAMS["opencode"].model_fmt.format(model="m")
    provider = kwargs["opencode_config"]["provider"][seams.OPENCODE_PROVIDER]
    assert provider["options"]["apiKey"] == "session-key"


def test_an_unknown_installed_agent_is_refused_rather_than_guessed(custom_agents):
    with pytest.raises(ValueError, match="installed into the sandbox"):
        seams.get(f"{custom_agents}:UnknownInstalledAgent")


def test_a_bad_import_path_says_why():
    pytest.importorskip("harbor.agents.base")
    with pytest.raises(KeyError, match="cannot import harness"):
        seams.get("no_such_module_xyz:Agent")


def test_antigravity_uses_the_existing_native_google_endpoint_setting():
    from openenv.harbor.seams import SEAMS

    model, _, env, _ = SEAMS["antigravity-cli"].resolve(
        base_url="https://capture.test", session="session-key", model="Qwen3.5-4B"
    )
    assert model == "google/Qwen3.5-4B"
    assert env["GOOGLE_GEMINI_BASE_URL"] == "https://capture.test"
    assert env["GEMINI_API_KEY"] == "session-key"
    assert env["AGY_ADC_AUTH"] == "false"
