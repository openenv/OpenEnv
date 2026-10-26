# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The loop-owning training endpoint, and the contract types it serves.

`GET /sessions/{id}/trace_entries` exists because a loop-owning consumer -- an external agent that
drives its own tool loop, opencode or codex or claude-code -- wants per-model-call records, not the
stitched sequence document `/rollout` returns. `to_trace_entries` already produced that shape but
could not be asked for it over HTTP, because it needs the session's graph and that is server-side
state.

These tests pin the two things a consumer depends on and cannot check for itself:

  * the endpoint answers with `{"session_id", "entries"}` and 404s an unknown id rather than 500ing,
    so a caller can distinguish "no such rollout" from "the server broke";
  * `TraceEntry` carries exactly the five keys the record is defined to have. A consumer builds
    training rows off those key names, so a rename is a silent breakage -- the trainer would read
    empty token fields and report a rollout that learned nothing.

No engine is contacted: `llm_url` is the discard port, so a stray request would fail loudly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
capture_server = pytest.importorskip("openenv.core.harness.capture.server")
from fastapi.testclient import TestClient  # noqa: E402
from openenv.core.harness import LoopOwningSession, TraceEntry  # noqa: E402


# Never contacted. Discard port, so a request that escaped would fail rather than reach a host.
UNUSED_ENGINE = "http://127.0.0.1:9/v1"


@pytest.fixture
def client() -> TestClient:
    return TestClient(capture_server.create_app(llm_url=UNUSED_ENGINE, model="unused"))


def test_unknown_session_is_404_not_500(client: TestClient) -> None:
    # A 500 here would be read as "the capture server is broken" and send someone to the wrong
    # place; the distinction between an unknown rollout and a broken server has to survive.
    response = client.get("/sessions/no-such-session/trace_entries")
    assert response.status_code == 404
    assert response.json() == {"error": "unknown session"}


def test_fresh_session_returns_an_empty_entry_list(client: TestClient) -> None:
    session_id = client.post("/sessions", json={}).json()["session_id"]

    response = client.get(f"/sessions/{session_id}/trace_entries")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    # Empty, not absent: a rollout that captured nothing yet is a valid answer, and a consumer
    # must be able to tell it apart from a malformed reply.
    assert body["entries"] == []


def test_trace_entry_carries_exactly_the_documented_keys() -> None:
    # Consumers index these by name to build training rows, so a rename breaks them silently --
    # the token fields simply read empty and the rollout looks like it learned nothing.
    #
    # `prompt_token_ids`, `loss_mask`, `reward` and `metadata` were added 2026-09. The first is the
    # load-bearing one: without it a consumer must re-render the prompt with apply_chat_template,
    # which matched the engine on 0 of 28 measured turns on Qwen3.5-4B and collapsed a run at its
    # first weight update.
    assert set(TraceEntry.__annotations__) == {
        "request",
        "response",
        "prompt_token_ids",
        "completion_token_ids",
        "completion_tokens",
        "per_token_logps",
        "loss_mask",
        "reward",
        "metadata",
    }


def test_trace_entry_is_total_false_so_partial_records_are_legal() -> None:
    # An eval-tier rollout has no token fields by design. If the record required them, every
    # eval capture would be a type error rather than a legitimately partial record.
    assert TraceEntry.__total__ is False


def test_loop_owning_session_protocol_is_structural() -> None:
    """A session satisfies the protocol by shape, never by inheritance.

    That freedom is the whole point: opencode reads a file out of its sandbox, the capture proxy
    answers over HTTP, and the consumer distinguishes neither.
    """

    class ReadsAFile:
        def wait_for_completion(self, timeout_s: float | None = None) -> int:
            return 0

        def fetch_proxy_trace(self) -> list[TraceEntry]:
            return []

    class CallsAServer:
        def wait_for_completion(self, timeout_s: float | None = None) -> int:
            return 0

        def fetch_proxy_trace(self) -> list[TraceEntry]:
            return [{"request": {}, "response": {}, "completion_token_ids": [1]}]

    for candidate in (ReadsAFile(), CallsAServer()):
        assert isinstance(candidate, LoopOwningSession)

    class MissingTheTrace:
        def wait_for_completion(self, timeout_s: float | None = None) -> int:
            return 0

    assert not isinstance(MissingTheTrace(), LoopOwningSession)
