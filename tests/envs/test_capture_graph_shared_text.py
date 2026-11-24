# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Repeated history in a multi-turn rollout is held once, and exports are unchanged.

Every chat request re-sends the whole conversation, and each request body is decoded separately, so
turn k carries its own fresh copy of every image attached in turns 1..k. Retaining those copies makes
graph memory grow quadratically with the number of turns.
"""

from __future__ import annotations

import json

import pytest

graph_mod = pytest.importorskip("openenv.core.harness.capture.graph")
export_mod = pytest.importorskip("openenv.core.harness.capture.export")

RolloutGraph = graph_mod.RolloutGraph
TurnNode = graph_mod.TurnNode

N_TURNS = 12


class FakeSession:
    def __init__(self, graph):
        self.session_id = "s1"
        self.metadata: dict = {}
        self.findings: list[str] = []
        self.graph = graph


def image(i: int) -> str:
    return "data:image/png;base64," + chr(ord("A") + i) * (64 * 1024)


def request_body(turn: int) -> list[dict]:
    """The messages of request `turn`, decoded from the wire like the capture server does."""
    messages: list[dict] = [{"role": "system", "content": "You are a helpful agent."}]
    for i in range(turn + 1):
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"step {i}"},
                    {"type": "image_url", "image_url": {"url": image(i)}},
                ],
            }
        )
        if i < turn:
            messages.append({"role": "assistant", "content": f"reply {i}"})
    return json.loads(json.dumps(messages))


def build_graph() -> RolloutGraph:
    graph = RolloutGraph()
    for turn in range(N_TURNS):
        graph.add_turn(
            TurnNode(
                node_id=f"t{turn}",
                prompt_ids=[],
                sampled_ids=[],
                request_messages=request_body(turn),
                response_message={"role": "assistant", "content": f"reply {turn}"},
            )
        )
    return graph


def image_urls(node: TurnNode) -> list[str]:
    return [
        part["image_url"]["url"]
        for message in node.request_messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]


def test_each_image_is_retained_once_across_turns():
    graph = build_graph()
    retained = {id(url): url for node in graph.nodes() for url in image_urls(node)}
    # Without sharing this is N_TURNS * (N_TURNS + 1) / 2 copies.
    assert len(retained) == N_TURNS
    first_seen = image_urls(graph.get("t0"))[0]
    for node in graph.nodes():
        assert image_urls(node)[0] is first_seen


def test_sharing_strings_does_not_alias_message_containers():
    graph = build_graph()
    graph.get("t1").request_messages[1]["content"][1]["image_url"]["url"] = "edited"
    assert graph.get("t0").request_messages[1]["content"][1]["image_url"]["url"] == (
        image(0)
    )
    assert graph.get("t2").request_messages[1]["content"][1]["image_url"]["url"] == (
        image(0)
    )


def test_export_is_byte_identical_to_unshared_graph(monkeypatch):
    def export(graph: RolloutGraph) -> bytes:
        document = export_mod.export_session(
            FakeSession(graph), include_messages=True, capture_level="text"
        )
        return json.dumps(document, sort_keys=True).encode()

    shared = export(build_graph())
    monkeypatch.setattr(RolloutGraph, "_share_text", lambda self, value: value)
    unshared = export(build_graph())

    assert shared == unshared
    assert len(json.loads(shared)["turns"]) == N_TURNS
