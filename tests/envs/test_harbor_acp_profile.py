"""The ACP profile uses Harbor's existing registry schema and isolates credentials."""

import json

import pytest

pytest.importorskip("harbor.agents.installed.acp")

from harbor.agents.installed.acp import AcpRegistryEntry
from openenv.harbor.seams import acp_opencode_config, get


def test_acp_opencode_profile_is_valid_and_routes_primary_and_auxiliary_calls():
    first = acp_opencode_config("https://capture.example", "session-one", "Qwen3.5-4B")
    entry = AcpRegistryEntry.model_validate(first["registry_entry"])
    assert entry.distribution.npx.package == "opencode-ai@1.18.30"
    assert entry.distribution.npx.args == ["acp"]
    config = json.loads(entry.distribution.npx.env["OPENCODE_CONFIG_CONTENT"])
    assert config["model"] == config["small_model"] == "intercepted/Qwen3.5-4B"
    assert (
        config["provider"]["intercepted"]["options"]["baseURL"]
        == "https://capture.example/v1"
    )
    assert config["provider"]["intercepted"]["options"]["apiKey"] == "session-one"
    second = acp_opencode_config("https://other.example", "session-two", "other-model")
    assert "session-one" not in json.dumps(second)
    assert "session-two" not in json.dumps(first)
    assert get("acp").kwargs is None  # Generic ACP must not silently select an agent.


def test_profile_selection_is_per_rollout_and_preserves_generic_adapter(tmp_path):
    from openenv.harbor.rollout import build_trial_config

    generic = get("acp")
    config = build_trial_config(
        task_dir=tmp_path,
        harness="acp",
        sandbox="e2b",
        intercept_url="https://capture.example",
        session_id="session-profile",
        model="Qwen3.5-4B",
        trial_name="trial",
        trials_dir=tmp_path,
        harness_profile="opencode-1.18.30",
    )
    assert config.agent.model_name == "intercepted/Qwen3.5-4B"
    entry = AcpRegistryEntry.model_validate(config.agent.kwargs["registry_entry"])
    assert entry.distribution.npx.package == "opencode-ai@1.18.30"
    assert get("acp") is generic
    assert generic.kwargs is None


def test_unknown_profile_is_not_silently_ignored():
    import pytest

    with pytest.raises(ValueError, match="unsupported harness profile"):
        get("acp", profile="unverified-agent")
    with pytest.raises(ValueError, match="unsupported harness profile"):
        get("codex", profile="opencode-1.18.30")
