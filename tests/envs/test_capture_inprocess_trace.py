# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reading a rollout back out of a live `CaptureServer`, in process.

This is the path `CaptureServer` exists for, and why it runs as a thread rather than a subprocess:
the caller mints a session on the registry the proxy is writing into, then reads the graph straight
back out of it. Going through HTTP for that would add a serialisation round trip and a failure mode
for no benefit -- and it is also how a caller ends up never deleting the session, because over HTTP
there is no obvious place to.

The property under test is the one the whole training contract rests on: turn k+1's prompt IS turn
k's prompt plus its completion, so turns link by exact token prefix. When that breaks, one
conversation silently fragments into several short ones and every fragment still trains.

A stub engine stands in for vLLM so this needs no GPU.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from openenv.core.harness.capture import to_trace_entries
from openenv.core.harness.capture.export import export_session
from openenv.core.harness.capture.runner import CaptureServer
from openenv.core.harness.capture.sessions import Upstream


LLM_URL = "http://engine.invalid/v1"
MODEL = "test-model"


class _TokenEngine:
    """A vLLM served with `--return-tokens-as-token-ids --logprobs-mode processed_logprobs`."""

    served_model = MODEL
    param_fixes: dict[str, Any] = {}
    capture_level = "tokens"

    def __init__(self) -> None:
        self.turn = 0
        # The prompt grows by the previous turn's completion. Faking that relationship is the only
        # way the graph's prefix linking can be exercised at all.
        self._prompt = [1, 2, 3]

    async def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        self.turn += 1
        prompt = list(self._prompt)
        completion = [100 + self.turn, 200 + self.turn]
        self._prompt = prompt + completion
        return {
            "id": f"c{self.turn}",
            "object": "chat.completion",
            "model": MODEL,
            # `token_ids` on the choice, `prompt_token_ids` on the response: the shape vLLM returns
            # under `--return-tokens-as-token-ids`.
            "prompt_token_ids": prompt,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"turn {self.turn}"},
                    "finish_reason": "stop",
                    "token_ids": completion,
                    "logprobs": {"content": [{"logprob": -0.5} for _ in completion]},
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": 2,
                "total_tokens": 0,
            },
        }


@pytest.fixture
def server():
    """A `CaptureServer` that is never `start()`ed -- the registry is what this exercises.

    Binding a port would make the test flaky on a busy machine and would test uvicorn rather than the
    contract.
    """
    srv = CaptureServer(llm_url=LLM_URL, model=MODEL)
    engine = _TokenEngine()
    # Seed the ENGINE POOL, not only the default: a session that names its own upstream resolves
    # through the pool, which is what lets one server drive a train-tier engine and an eval-tier one
    # at the same time. Without this the proxy would probe `engine.invalid` for real.
    srv.app.state.upstreams._by_engine[
        Upstream(llm_url=LLM_URL, model=MODEL).cache_key
    ] = (
        engine,
        "tokens",
    )
    srv.app.state.upstreams._default = (engine, "tokens")
    return srv


def _mint(server, **kwargs):
    return server.registry.create(
        upstream=Upstream(llm_url=LLM_URL, model=MODEL),
        capture_level="tokens",
        **kwargs,
    )


def _chat(client: TestClient, session_id: str) -> None:
    client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {session_id}"},
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )


def _entries(server, session):
    document = export_session(
        session, include_messages=True, capture_level=session.capture_level
    )
    return to_trace_entries(session.graph, document)


def test_entries_carry_the_engines_own_prompt_tokens(server):
    session = _mint(server)
    with TestClient(server.app) as client:
        for _ in range(3):
            _chat(client, session.session_id)

    entries = _entries(server, session)
    assert len(entries) == 3
    for entry in entries:
        assert entry["prompt_token_ids"], (
            "an entry came back with no engine tokenisation"
        )
        assert entry["completion_token_ids"]
        assert len(entry["per_token_logps"]) == len(entry["completion_token_ids"])
        # The mask spans prompt + completion, and only the completion is trained.
        assert len(entry["loss_mask"]) == len(entry["prompt_token_ids"]) + len(
            entry["completion_token_ids"]
        )
        assert set(entry["loss_mask"][: len(entry["prompt_token_ids"])]) == {0}

    # THE CONTRACT: turn k+1's prompt is everything before it, token for token.
    first, second = entries[0], entries[1]
    assert (
        second["prompt_token_ids"]
        == first["prompt_token_ids"] + first["completion_token_ids"]
    )


def test_deleting_a_session_releases_it(server):
    session = _mint(server)
    sid = session.session_id
    assert server.registry.get(sid) is not None
    assert server.registry.delete(sid)
    # Sessions held past their rollout collide with the next run's claim and surface as a burst of
    # CAPACITY_REACHED on a server that looks idle, so this is not bookkeeping.
    assert server.registry.get(sid) is None


def test_a_budget_bounds_what_is_captured(server):
    session = _mint(server, max_model_calls=2)
    with TestClient(server.app) as client:
        for _ in range(5):
            _chat(client, session.session_id)
    # Five requests, two captured: the proxy answered the rest itself, before ingest.
    assert len(_entries(server, session)) == 2
