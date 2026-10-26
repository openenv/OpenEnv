# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Turning a capture document into the per-turn rows a trainer consumes.

The training contract is `(prompt_token_ids, completion_token_ids, per_token_logps)` per turn, and
every failure here is silent: the rollout completes, the reward is real, and the rows are wrong.
"""

from __future__ import annotations

import pytest

export_mod = pytest.importorskip("openenv.core.harness.capture.export")
graph_mod = pytest.importorskip("openenv.core.harness.capture.graph")
models = pytest.importorskip("openenv.harbor.models")

RolloutGraph = graph_mod.RolloutGraph
TurnNode = graph_mod.TurnNode
turns_from_document = models.turns_from_document


class FakeSession:
    """The surface `export_session` needs."""

    session_id = "sess"
    metadata: dict = {}
    findings: list = []

    def __init__(self) -> None:
        self.graph = RolloutGraph()


def build(turns, *, context_between=2):
    """A linear agent conversation. `turns` is [(n_sampled, has_logprobs), ...].

    Returns the session and the ground truth each turn should round-trip to.
    """
    session = FakeSession()
    prompt = list(range(100, 140))
    truth = []
    for i, (n_sampled, has_logprobs) in enumerate(turns):
        sampled = list(range(1000 + i * 100, 1000 + i * 100 + n_sampled))
        node = TurnNode(
            node_id=f"n{i}",
            prompt_ids=list(prompt),
            sampled_ids=sampled,
            sampled_logprobs=[-0.1 * (i + 1)] * n_sampled if has_logprobs else None,
            index=i,
            response_message={"content": f"turn {i}"},
        )
        session.graph.add_turn(node)
        truth.append((list(prompt), sampled))
        # The harness inserts a tool result before the next call.
        prompt = node.end_ids + [7777] * context_between
    return session, truth


def test_every_turn_carries_its_own_prompt():
    """`prompt_token_ids` used to be populated only for turn 0, leaving the rest empty.

    On-policy training needs the engine's tokenisation of everything the model saw before it
    generated, per turn. Empty arrays for turns 1..n make `contract.json` unusable for exactly the
    multi-turn rollouts this environment exists to produce.
    """
    session, truth = build([(5, True), (7, True), (3, True)])
    rows = turns_from_document(
        export_mod.export_session(session, include_messages=True)
    )

    assert len(rows) == 3
    for row, (want_prompt, want_sampled) in zip(rows, truth):
        assert row.prompt_token_ids == want_prompt, (
            f"turn {row.turn} prompt is not exact"
        )
        assert row.completion_token_ids == want_sampled
        assert len(row.per_token_logps) == len(want_sampled)


def test_prompts_grow_monotonically_down_a_conversation():
    """Turn k+1 saw everything turn k did, plus the tool result in between."""
    session, _ = build([(4, True), (4, True), (4, True)])
    rows = turns_from_document(
        export_mod.export_session(session, include_messages=True)
    )
    lengths = [len(r.prompt_token_ids) for r in rows]
    assert lengths == sorted(lengths) and len(set(lengths)) == 3, lengths
    # Each prompt contains the previous one as a prefix.
    for earlier, later in zip(rows, rows[1:]):
        assert (
            later.prompt_token_ids[: len(earlier.prompt_token_ids)]
            == earlier.prompt_token_ids
        )


def test_a_turn_without_usable_logprobs_does_not_drop_the_turns_after_it():
    """The regression, and it corrupted rather than merely truncated.

    Turn boundaries came from `turn_lengths`, which counts runs of mask-1. A turn whose logprobs
    were missing contributes mask-0 and therefore no run, so `node_ids` and `turn_lengths` had
    different lengths: `zip` stopped early AND the surviving turns were paired with the wrong
    spans, so a turn could be handed the next turn's tokens.
    """
    session, truth = build([(4, True), (5, False), (6, True)])
    document = export_mod.export_session(session, include_messages=True)

    sequence = next(r for r in document["sequences"] if r["role"] == "agent")
    assert len(sequence["turn_lengths"]) < len(sequence["node_ids"]), (
        "fixture no longer reproduces the mismatch this test exists for"
    )

    rows = turns_from_document(document)
    assert len(rows) == 3, "every captured turn must survive"
    for row, (_, want_sampled) in zip(rows, truth):
        assert row.completion_token_ids == want_sampled, (
            f"turn {row.turn} was given another turn's tokens"
        )


def test_adjacent_turns_with_no_context_between_them_stay_separate():
    """Two turns with nothing inserted between them formed one mask-run and merged into one."""
    session, truth = build([(3, True), (4, True)], context_between=0)
    rows = turns_from_document(
        export_mod.export_session(session, include_messages=True)
    )
    assert len(rows) == 2
    assert [len(r.completion_token_ids) for r in rows] == [3, 4]


def test_auxiliary_sequences_produce_no_turns():
    """An auxiliary call must never be credited with the reward the agent earned."""
    session, _ = build([(3, True)])
    document = export_mod.export_session(session, include_messages=True)
    for sequence in document["sequences"]:
        sequence["role"] = "auxiliary"
    assert turns_from_document(document) == []


def test_logprobs_line_up_with_the_tokens_they_score():
    """A logprob against the wrong token makes GRPO's ratio a ratio against an invented number."""
    session, _ = build([(4, True), (6, True)])
    rows = turns_from_document(
        export_mod.export_session(session, include_messages=True)
    )
    for row in rows:
        assert len(row.per_token_logps) == len(row.completion_token_ids)
        assert all(p < 0 for p in row.per_token_logps), (
            "placeholder zeros, not real logprobs"
        )


def test_a_node_shared_by_two_live_paths_is_exported_once():
    """Forked paths share their prefix, and each live path is its own exported sequence.

    Without dedup the shared turns appear once per path, so a trainer credits the same model call
    more than once and quietly doubles its weight in the gradient.
    """
    session = FakeSession()
    prompt = list(range(100, 140))

    shared = TurnNode(
        node_id="shared",
        prompt_ids=list(prompt),
        sampled_ids=[1000, 1001],
        sampled_logprobs=[-0.1, -0.1],
        index=0,
        response_message={"content": "shared"},
    )
    session.graph.add_turn(shared)

    # Two branches off the same parent, each of which continues, so neither is a discarded retry.
    for i, branch in enumerate(("a", "b")):
        first = TurnNode(
            node_id=f"{branch}1",
            prompt_ids=shared.end_ids + [7777, 7777],
            sampled_ids=[2000 + i * 100, 2001 + i * 100],
            sampled_logprobs=[-0.2, -0.2],
            index=1 + i * 2,
            response_message={"content": f"{branch}1"},
        )
        session.graph.add_turn(first)
        session.graph.add_turn(
            TurnNode(
                node_id=f"{branch}2",
                prompt_ids=first.end_ids + [7777, 7777],
                sampled_ids=[3000 + i * 100],
                sampled_logprobs=[-0.3],
                index=2 + i * 2,
                response_message={"content": f"{branch}2"},
            )
        )

    document = export_mod.export_session(session, include_messages=True)
    agent_paths = [s for s in document["sequences"] if s["role"] == "agent"]
    assert len(agent_paths) == 2, "the fixture must actually produce two live paths"

    rows = turns_from_document(document)
    ids = [r.completion_token_ids[0] for r in rows]
    assert len(ids) == len(set(ids)), f"a turn was exported twice: {ids}"
    # The shared prefix once, plus two turns on each branch.
    assert len(rows) == 5
