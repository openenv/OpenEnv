# SPDX-License-Identifier: BSD-3-Clause

from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import Observation, ResetResponse, StepResponse
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient


class CustomObservation(Observation):
    ally_tree: str = ""
    task_instruction: str = ""


def test_serialize_observation_promotes_metadata_to_top_level() -> None:
    obs = Observation(
        done=False,
        reward=0.5,
        metadata={"total_nodes": 5, "task_id": 1},
    )

    result = serialize_observation(obs)

    assert result["metadata"] == {"total_nodes": 5, "task_id": 1}


def test_serialize_observation_keeps_metadata_in_nested_payload() -> None:
    obs = Observation(metadata={"step": 3})

    result = serialize_observation(obs)

    assert result["observation"]["metadata"] == {"step": 3}


def test_serialize_observation_omits_empty_top_level_metadata() -> None:
    obs = Observation(done=False, reward=0.0, metadata={})

    result = serialize_observation(obs)

    assert "metadata" not in result
    assert result["observation"]["metadata"] == {}


def test_serialize_observation_preserves_subclass_fields() -> None:
    obs = CustomObservation(
        ally_tree="[ref=btn_1 role=button]",
        task_instruction="Book a ticket",
        done=False,
        reward=0.2,
        metadata={"variant": "label_drift"},
    )

    result = serialize_observation(obs)

    assert result["observation"]["ally_tree"] == "[ref=btn_1 role=button]"
    assert result["observation"]["task_instruction"] == "Book a ticket"
    assert result["metadata"] == {"variant": "label_drift"}


def test_reset_response_accepts_top_level_metadata() -> None:
    serialized = serialize_observation(Observation(metadata={"reset_key": "val"}))

    reset_response = ResetResponse(**serialized)

    assert reset_response.metadata == {"reset_key": "val"}


def test_step_response_accepts_top_level_metadata() -> None:
    serialized = serialize_observation(Observation(metadata={"step_key": "val"}))

    step_response = StepResponse(**serialized)

    assert step_response.metadata == {"step_key": "val"}


def test_generic_client_receives_metadata() -> None:
    payload = serialize_observation(
        Observation(
            done=False,
            reward=0.42,
            metadata={"total_nodes": 6, "completed": ["origin", "dest"]},
        )
    )

    client = GenericEnvClient.__new__(GenericEnvClient)
    step_result = client._parse_result(payload)

    assert step_result.reward == 0.42
    assert step_result.done is False
    assert step_result.metadata == {
        "total_nodes": 6,
        "completed": ["origin", "dest"],
    }


def test_generic_client_handles_missing_metadata() -> None:
    payload = {"observation": {"text": "hello"}, "reward": 0.0, "done": False}

    client = GenericEnvClient.__new__(GenericEnvClient)
    step_result = client._parse_result(payload)

    assert step_result.metadata is None


def test_mcp_client_prefers_top_level_metadata() -> None:
    payload = {
        "observation": {"tools": [], "metadata": {"nested": "old"}},
        "reward": 1.0,
        "done": False,
        "metadata": {"top_level": "new"},
    }

    client = MCPToolClient.__new__(MCPToolClient)
    step_result = client._parse_result(payload)

    assert step_result.metadata == {"top_level": "new"}
    assert step_result.observation.metadata == {"top_level": "new"}
