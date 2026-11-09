"""Harness construction failures should surface before a paid rollout."""

import json
import os
import sys
import types
from pathlib import Path

import pytest
from openenv.cli.commands.harbor import app
from openenv.harbor.capabilities import Capabilities
from typer.testing import CliRunner


@pytest.fixture
def info_only(monkeypatch):
    # No endpoint, dataset download, sandbox checks or server are needed for these tests.
    monkeypatch.setattr("openenv.harbor.startup.prepare", lambda **_: Capabilities())


@pytest.fixture
def agents(monkeypatch):
    pytest.importorskip("harbor.agents.base")
    from harbor.agents.base import BaseAgent
    from harbor.agents.installed.pi import Pi

    module = types.ModuleType("info_test_agents")
    module.constructed = []

    class SilentAgent(BaseAgent):
        @staticmethod
        def name():
            return "info-test-agent"

        def version(self):
            return None

        async def setup(self, environment):
            raise AssertionError("preflight must not set up an agent")

        async def run(self, instruction, environment, context):
            raise AssertionError("preflight must not run an agent")

    class DeclaredAgent(SilentAgent):
        def __init__(self, *, api_base, api_key, **kwargs):
            super().__init__(**kwargs)
            print("constructor output must not corrupt JSON")
            module.constructed.append((api_base, api_key, self.logs_dir))

    class BrokenAgent(DeclaredAgent):
        def __init__(self, *, required_setting, **kwargs):
            super().__init__(**kwargs)

    class WrongPi(Pi):
        pass

    for cls in (DeclaredAgent, SilentAgent, BrokenAgent, WrongPi):
        setattr(module, cls.__name__, cls)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_info_without_a_harness_keeps_the_existing_json(info_only):
    result = CliRunner().invoke(app, ["info", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == Capabilities().to_dict()


def test_info_constructs_with_dummy_capture_wiring(info_only, agents, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "operator-key")
    # litellm (which Harbor's agents import) runs load_dotenv() when first imported, and picks up any .env
    # above the checkout. Import it first, so the check below covers only what the preflight does to os.environ.
    try:
        import litellm  # noqa: F401
    except ImportError:
        pass

    before = dict(os.environ)
    result = CliRunner().invoke(
        app,
        [
            "info",
            "--harness",
            "info_test_agents:DeclaredAgent",
            "--model",
            "org/model",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    checked = json.loads(result.stdout)["harness"]
    assert checked["ok"] is True
    assert checked["host_side"] is True
    assert checked["status"] == "untested"
    assert checked["model"] == "model"
    assert checked["kwargs"] == {
        "api_base": "http://127.0.0.1:9/v1",
        "api_key": "openenv-harness-preflight",
    }
    assert checked["agent_env"] == checked["process_env"] == {}
    assert dict(os.environ) == before
    assert "operator-key" not in result.output
    url, key, logs = agents.constructed[0]
    assert url == checked["kwargs"]["api_base"]
    assert key == checked["kwargs"]["api_key"]
    assert not Path(logs).exists()


def test_info_accepts_declared_options(info_only, agents):
    options = pytest.importorskip("harbor.agents.options")

    class Options(options.AgentOptions):
        api_base: str
        api_key: str

    class OptionsAgent(agents.SilentAgent):
        options_model = Options

    agents.OptionsAgent = OptionsAgent
    result = CliRunner().invoke(
        app, ["info", "--harness", "info_test_agents:OptionsAgent", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["harness"]["ok"] is True


@pytest.mark.parametrize(
    ("agent", "detail"),
    [
        ("SilentAgent", "must declare api_base, api_key"),
        ("BrokenAgent", "required_setting"),
        ("WrongPi", "install_fixes:InterceptPi"),
        ("Missing", "cannot import harness"),
    ],
)
def test_info_reports_bad_harnesses_with_a_failing_exit_code(
    info_only, agents, agent, detail
):
    result = CliRunner().invoke(
        app, ["info", "--harness", f"info_test_agents:{agent}", "--json"]
    )
    assert result.exit_code == 1, result.output
    checked = json.loads(result.stdout)["harness"]
    assert checked["ok"] is False
    assert detail in checked["detail"]


def test_info_checks_the_installed_agent_using_the_rollout_configuration(
    info_only, monkeypatch
):
    pytest.importorskip("harbor.agents.base")
    monkeypatch.setenv("OPENENV_HARBOR_AGENT_VERSIONS", '{"opencode": "1.18.30"}')
    result = CliRunner().invoke(app, ["info", "--harness", "opencode", "--json"])
    assert result.exit_code == 0, result.output
    checked = json.loads(result.stdout)["harness"]
    assert checked["kwargs"]["version"] == "1.18.30"
    assert checked["host_side"] is False


def test_info_text_explains_the_limit_of_a_successful_check(info_only, agents):
    result = CliRunner().invoke(
        app, ["info", "--harness", "info_test_agents:DeclaredAgent"]
    )
    assert result.exit_code == 0
    assert "construction only; capture is untested" in result.stdout


@pytest.mark.parametrize("fails", [False, True])
def test_info_restores_process_credentials_after_construction(
    info_only, monkeypatch, fails
):
    pytest.importorskip("harbor.agents.base")
    from harbor.agents.factory import AgentFactory

    original = AgentFactory.create_agent_from_config

    def construct(config, **kwargs):
        assert os.environ["ANTHROPIC_API_KEY"] == "openenv-harness-preflight"
        if fails:
            raise ValueError("constructor failed")
        return original(config, **kwargs)

    monkeypatch.setattr(AgentFactory, "create_agent_from_config", construct)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "operator-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    before = dict(os.environ)
    result = CliRunner().invoke(app, ["info", "--harness", "claude-code", "--json"])
    assert result.exit_code == int(fails), result.output
    assert json.loads(result.stdout)["harness"]["ok"] is not fails
    assert "operator-key" not in result.output
    assert dict(os.environ) == before
