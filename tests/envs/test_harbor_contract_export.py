# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the trainer-facing contract includes, and what it must exclude.

Both directions are silent when wrong. Dropping agent turns trains on part of a rollout while
reporting the whole reward; including auxiliary turns credits a next-speaker classification with
solving the task.
"""

from __future__ import annotations

import pytest

contract = pytest.importorskip("openenv.core.harness.capture.contract")
graph_mod = pytest.importorskip("openenv.core.harness.capture.graph")

RolloutGraph = graph_mod.RolloutGraph
TurnNode = graph_mod.TurnNode


def node(node_id: str, prompt: list[int], sampled: list[int]) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        prompt_ids=prompt,
        sampled_ids=sampled,
        sampled_logprobs=[-0.1] * len(sampled),
        request_messages=[{"role": "user", "content": "hi"}],
        response_message={"content": "ok"},
    )


@pytest.fixture
def two_agent_roots():
    """A harness that rewrote its system prompt mid-run, so the rollout has two agent roots."""
    g = RolloutGraph()
    g.add_turn(node("a1", [1], [2, 3]))
    g.add_turn(node("a2", [1, 2, 3, 9], [4]))
    g.add_turn(node("b1", [500], [6]))
    document = {
        "sequences": [
            {"role": "agent", "root_id": "a1", "node_ids": ["a1", "a2"]},
            {"role": "agent", "root_id": "b1", "node_ids": ["b1"]},
        ]
    }
    return g, document


def test_every_agent_root_is_exported(two_agent_roots):
    """The regression: only the first agent sequence was kept, so later roots vanished."""
    g, document = two_agent_roots
    assert [n.node_id for n in contract._agent_nodes(g, document)] == ["a1", "a2", "b1"]


def test_turn_records_cover_every_agent_turn(two_agent_roots):
    g, document = two_agent_roots
    records = contract.to_turn_records(g, document)
    assert len(records) == 3
    for prompt_ids, output_ids, logps in records:
        assert output_ids, "a turn with no sampled tokens is not trainable"
        assert len(output_ids) == len(logps)


def test_trace_entries_cover_every_agent_turn(two_agent_roots):
    g, document = two_agent_roots
    assert len(contract.to_trace_entries(g, document)) == 3


def test_auxiliary_sequences_are_excluded():
    """An aux call must never be credited with the reward the agent earned."""
    g = RolloutGraph()
    g.add_turn(node("agent", [1], [2]))
    g.add_turn(node("aux", [900], [3]))
    document = {
        "sequences": [
            {"role": "agent", "root_id": "agent", "node_ids": ["agent"]},
            {"role": "auxiliary", "root_id": "aux", "node_ids": ["aux"]},
        ]
    }
    assert [n.node_id for n in contract._agent_nodes(g, document)] == ["agent"]


def test_discarded_sequences_are_excluded():
    g = RolloutGraph()
    g.add_turn(node("kept", [1], [2]))
    g.add_turn(node("dead", [1], [3]))
    document = {
        "sequences": [
            {"role": "agent", "root_id": "kept", "node_ids": ["kept"]},
            {"role": "discarded", "root_id": "kept", "node_ids": ["dead"]},
        ]
    }
    assert [n.node_id for n in contract._agent_nodes(g, document)] == ["kept"]


def test_a_node_shared_by_two_paths_appears_once():
    """Forked paths share their prefix; the shared turn must not be exported twice."""
    g = RolloutGraph()
    g.add_turn(node("shared", [1], [2]))
    g.add_turn(node("left", [1, 2], [3]))
    g.add_turn(node("right", [1, 2], [4]))
    document = {
        "sequences": [
            {"role": "agent", "root_id": "shared", "node_ids": ["shared", "left"]},
            {"role": "agent", "root_id": "shared", "node_ids": ["shared", "right"]},
        ]
    }
    ids = [n.node_id for n in contract._agent_nodes(g, document)]
    assert ids.count("shared") == 1
    assert set(ids) == {"shared", "left", "right"}


def test_no_agent_sequences_is_empty_not_an_error():
    g = RolloutGraph()
    g.add_turn(node("aux", [1], [2]))
    document = {
        "sequences": [{"role": "auxiliary", "root_id": "aux", "node_ids": ["aux"]}]
    }
    assert contract._agent_nodes(g, document) == []
    assert contract.to_turn_records(g, document) == []
