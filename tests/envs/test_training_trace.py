# SPDX-License-Identifier: BSD-3-Clause


import pytest
from openenv.core.harness import TrainingTrace, TrainingTurn
from openenv.core.harness.capture.contract import to_training_trace as graph_trace
from openenv.core.harness.capture.graph import RolloutGraph, TurnNode
from openenv.harbor.contract import to_training_trace
from openenv.harbor.models import HarborRolloutResult, HarborTurn


def turn(**changes):
    return TrainingTurn(
        **{
            "node_id": "agent",
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3, 4, 5],
            "per_token_logps": [-0.1, -0.2, -0.3],
            "loss_mask": [0, 0, 1, 0, 1],
            **changes,
        }
    )


def test_partial_mask_and_zero_mask_survive_wire_round_trip():
    original = TrainingTrace(turns=[turn(), turn(node_id="context", loss_mask=[0] * 5)])
    restored = TrainingTrace.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.turns[0].loss_mask == [0, 0, 1, 0, 1]
    assert restored.turns[1].per_token_logps == [-0.1, -0.2, -0.3]


@pytest.mark.parametrize(
    "changes",
    [
        {"loss_mask": None},
        {"loss_mask": [0, 0, 1]},
        {"loss_mask": [1, 0, 1, 0, 1]},
        {"loss_mask": [0, 0, True, 0, 1]},
        {"completion_token_ids": [3, True, 5]},
        {"per_token_logps": []},
        {"per_token_logps": [-0.1, float("nan"), -0.3]},
        {"per_token_logps": [-0.1, 0.2, -0.3]},
        {"per_token_logps": [-0.1, True, -0.3]},
        {"prompt_token_ids": []},
    ],
)
def test_malformed_capture_fails_at_producer_boundary(changes):
    with pytest.raises(ValueError):
        turn(**changes)


def test_missing_mask_and_unknown_version_rejected():
    values = turn().model_dump(exclude={"loss_mask"})
    with pytest.raises(ValueError):
        TrainingTurn.model_validate(values)
    with pytest.raises(ValueError):
        TrainingTrace(schema_version=2, turns=[])


def test_duplicate_call_is_not_supervised_twice():
    with pytest.raises(ValueError, match="duplicate node_ids"):
        TrainingTrace(turns=[turn(), turn()])


def test_shared_graph_prefix_is_emitted_once_and_partial_masks_preserved():
    graph = RolloutGraph()
    for name, prompt, sampled in [
        ("root", [1], [2, 3]),
        ("left", [1, 2, 3, 8], [4]),
        ("right", [1, 2, 3, 9], [5]),
        ("aux", [90], [91]),
    ]:
        graph.add_turn(
            TurnNode(
                node_id=name,
                prompt_ids=prompt,
                sampled_ids=sampled,
                sampled_logprobs=[-0.1] * len(sampled),
            )
        )
    document = {
        "sequences": [
            {
                "role": "agent",
                "node_ids": ["root", "left"],
                "loss_mask": [0, 1, 0, 0, 1],
            },
            {
                "role": "agent",
                "node_ids": ["root", "right"],
                "loss_mask": [0, 1, 0, 0, 1],
            },
            {"role": "auxiliary", "node_ids": ["aux"], "loss_mask": [0, 1]},
        ]
    }
    trace = graph_trace(graph, document)
    assert [t.node_id for t in trace.turns] == ["root", "left", "right"]
    assert sum(sum(t.loss_mask) for t in trace.turns) == 3
    assert trace.turns[0].loss_mask == [0, 1, 0]


def test_harbor_masks_do_not_change_reward_or_drop_zero_masked_agent_calls():
    turns = [
        HarborTurn(
            turn=0, **turn().model_dump(exclude={"request", "response", "metadata"})
        ),
        HarborTurn(
            turn=1,
            **turn(node_id="context", loss_mask=[0] * 5).model_dump(
                exclude={"request", "response", "metadata"}
            ),
        ),
        HarborTurn(turn=2, role="auxiliary"),
        HarborTurn(turn=3, discarded=True),
    ]
    for reward in [None, 0.0, 1.0]:
        result = HarborRolloutResult(turns=turns, reward=reward)
        trace = to_training_trace(result)
        assert [t.node_id for t in trace.turns] == ["agent", "context"]
        assert result.reward == reward
        result.turns[0].loss_mask = None
        with pytest.raises(ValueError, match="explicit loss_mask"):
            to_training_trace(result)
        result.turns[0].loss_mask = [0, 0, 1, 0, 1]


def test_eval_and_fatal_capture_are_not_empty_successful_training():
    with pytest.raises(ValueError, match="eval-only"):
        to_training_trace(HarborRolloutResult(rollout_type="eval"))
    with pytest.raises(ValueError, match="fatal"):
        to_training_trace(HarborRolloutResult(findings=["[FATAL] bad capture"]))
