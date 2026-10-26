"""Seam behaviour: how each harness learns about the capture proxy, and how it is bounded.

A seam is the only place per-agent knowledge lives, so these tests assert on what a seam HANDS the
agent rather than on any rollout outcome.
"""

from __future__ import annotations

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


def test_a_harness_that_cannot_cap_steps_warns_rather_than_dropping_it(caplog):
    """Silently ignoring the cap would let a caller believe its rollouts are bounded."""
    with caplog.at_level("WARNING"):
        _, kwargs, _, _ = seams.get("opencode").resolve(
            base_url="http://proxy", session="s1", model="m", step_limit=20
        )
    assert "capture proxy enforces the model-call budget" in caplog.text
    assert "step_limit" not in str(kwargs)


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


def test_antigravity_uses_the_existing_native_google_endpoint_setting():
    from openenv.harbor.seams import SEAMS

    model, _, env, _ = SEAMS["antigravity-cli"].resolve(
        base_url="https://capture.test", session="session-key", model="Qwen3.5-4B"
    )
    assert model == "google/Qwen3.5-4B"
    assert env["GOOGLE_GEMINI_BASE_URL"] == "https://capture.test"
    assert env["GEMINI_API_KEY"] == "session-key"
    assert env["AGY_ADC_AUTH"] == "false"
