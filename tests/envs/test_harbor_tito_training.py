"""Regression tests for exact captured supervision and concurrent session isolation."""

import asyncio
import copy
import sys
import threading
from itertools import permutations
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from harbor_env.harness import to_trace_entries as harbor_entries
from openenv.core.harness.capture import contract, server
from openenv.core.harness.capture.export import export_session
from openenv.core.harness.capture.graph import RolloutGraph, TurnNode
from openenv.core.harness.capture.sessions import SessionRegistry, Upstream
from openenv.core.harness.capture.upstream import training_sampling
from openenv.core.harness.capture.validate import validate_training_turn
from openenv.harbor import rollout
from openenv.harbor.client import HarborEnv
from openenv.harbor.models import HarborRolloutResult, turns_from_document


def node(name, prompt, completion):
    return TurnNode(
        name,
        prompt,
        completion,
        [-0.2] * len(completion),
        n_tools=1,
        request_messages=[{"role": "user", "content": "hi"}],
        response_message={"role": "assistant", "content": "ok"},
    )


def test_partial_mask_survives_both_public_exports_and_wire_roundtrip():
    session = SessionRegistry().create()
    session.graph.add_turn(node("a", [1, 2], [3, 4]))
    document = export_session(session, include_messages=True)
    document["sequences"][0]["loss_mask"] = [0, 0, 1, 0]
    document["sequences"][0]["logprobs"][-1] = 0.0  # reconciler masked this token
    core = contract.to_trace_entries(session.graph, document)
    result = HarborRolloutResult(turns=turns_from_document(document))
    result = HarborRolloutResult.model_validate_json(result.model_dump_json())
    entries = harbor_entries(result)
    for entry in (core[0], entries[0]):
        assert entry["loss_mask"] == [0, 0, 1, 0]
        assert entry["per_token_logps"] == [-0.2, -0.2]
        assert entry["prompt_token_ids"] == [1, 2]
        assert entry["completion_token_ids"] == [3, 4]
    with pytest.raises(ValueError, match="partial"):
        contract.to_turn_records(session.graph, document)


def test_shared_node_with_conflicting_masks_cannot_depend_on_path_order():
    session = SessionRegistry().create()
    session.graph.add_turn(node("a", [1, 2], [3, 4]))
    document = export_session(session, include_messages=True)
    other = copy.deepcopy(document["sequences"][0])
    other["loss_mask"][-1] = 0
    document["sequences"].append(other)
    with pytest.raises(ValueError, match="inconsistent"):
        contract.to_trace_entries(session.graph, document)
    with pytest.raises(ValueError, match="inconsistent"):
        turns_from_document(document)


def test_fatal_rollout_cannot_export_otherwise_valid_tokens_for_training():
    session = SessionRegistry().create()
    session.graph.add_turn(node("a", [1, 2], [3, 4]))
    document = export_session(session, include_messages=True)
    result = HarborRolloutResult(
        ok=False,
        reward=0.0,
        rollout_type="train",
        capture_level="tokens",
        turns=turns_from_document(document),
        findings=[
            "[FATAL] turn_mismatch: captured actions disagree with the agent trace"
        ],
    )
    result = HarborRolloutResult.model_validate_json(result.model_dump_json())
    with pytest.raises(ValueError, match="fatal validation"):
        harbor_entries(result)
    # A failed task with valid capture still provides training signal. Failure of the
    # task itself, unlike failed capture validation, must not discard its actions.
    result.findings = ["[WARN] multiple_roots: rewritten context"]
    assert harbor_entries(result)[0]["completion_token_ids"] == [3, 4]


@pytest.mark.parametrize("order", list(permutations("abc")))
def test_longest_exact_parent_is_independent_of_response_arrival(order):
    graph = RolloutGraph()
    nodes = {
        "a": node("a", [1], [2]),
        "b": node("b", [1, 2, 3], [4]),
        "c": node("c", [1, 2, 3, 4, 5], [6]),
    }
    for name in order:
        graph.add_turn(nodes[name])
    assert {n.node_id: n.parent_id for n in graph.nodes()} == {
        "a": None,
        "b": "a",
        "c": "b",
    }
    assert graph.sequence_for("c").input_ids == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize(
    "prompt,completion,lp,mask",
    [
        ([1], [2], [float("nan")], [0, 1]),
        ([1], [2], [float("-inf")], [0, 1]),
        ([1], [2], [], [0, 1]),
        ([True], [2], [-0.1], [0, 1]),
        ([1], [-2], [-0.1], [0, 1]),
        ([1], [2], [-0.1], [1, 1]),
        ([1], [2], [-0.1], [0, 2]),
        ([1], [2], [-0.1], [0]),
    ],
)
def test_malformed_training_arrays_fail_before_export(prompt, completion, lp, mask):
    with pytest.raises(ValueError):
        validate_training_turn(prompt, completion, lp, mask)


def test_concurrent_probe_is_shared_and_waiter_cancellation_is_isolated(monkeypatch):
    pool = server.UpstreamPool(default_client=None, default_level="tokens")
    started, release = threading.Event(), threading.Event()
    calls = []

    def probe(upstream):
        calls.append(upstream.cache_key)
        started.set()
        assert release.wait(5)
        return "m", "tokens"

    monkeypatch.setattr(pool, "_probe", probe)

    async def check():
        endpoint = Upstream("http://engine.invalid", "m", "key-a")
        cancelled = asyncio.create_task(pool.resolve(endpoint))
        assert await asyncio.to_thread(started.wait, 5)
        waiters = [asyncio.create_task(pool.resolve(endpoint)) for _ in range(100)]
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        clients = await asyncio.gather(*waiters)
        assert len(calls) == 1
        assert all(client[0] is clients[0][0] for client in clients)
        other, _ = await pool.resolve(Upstream("http://engine.invalid", "m", "key-b"))
        assert other is not clients[0][0]
        assert len(calls) == 2

    try:
        asyncio.run(check())
    finally:
        release.set()


@pytest.mark.parametrize(
    "sampling",
    [
        {},
        {"temperature": 0},
        {"temperature": True},
        {"temperature": float("inf")},
        {"temperature": 0.8, "top_p": 0.9},
        {"temperature": 0.8, "seed": 1},
    ],
)
def test_training_sampling_rejects_unrecomputable_policies(sampling):
    with pytest.raises(ValueError):
        training_sampling(sampling)


@pytest.mark.parametrize(
    "path,body",
    [
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "temperature": 0},
        ),
        ("/v1/responses", {"input": "hi", "temperature": 0}),
        (
            "/v1/messages",
            {"messages": [{"role": "user", "content": "hi"}], "temperature": 0},
        ),
        (
            "/v1beta/models/m:generateContent",
            {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "generationConfig": {"temperature": 0},
            },
        ),
    ],
)
def test_explicit_training_policy_reaches_engine_across_dialects(path, body):
    app = server.create_app(llm_url="http://engine.invalid", model="m")
    sent = []

    async def handle(request):
        import json

        data = json.loads(request.content)
        sent.append(data)
        return httpx.Response(
            200,
            json={
                "id": "reply",
                "model": "m",
                "prompt_token_ids": [1, 2],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "token_ids": [3],
                        "logprobs": {
                            "content": [{"token": "token_id:3", "logprob": -0.2}]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    app.state.inference._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://engine.invalid"
    )
    with TestClient(app) as client:
        sid = client.post(
            "/sessions", json={"sampling": {"temperature": 0.8}, "max_model_calls": 1}
        ).json()["session_id"]
        response = client.post(
            path,
            json={"model": "m", **body},
            headers={"Authorization": f"Bearer {sid}"},
        )
        assert response.status_code == 200, response.text
        session = app.state.registry.get(sid)
        captured = list(session.graph.nodes())[0]
        assert captured.sampling_params == training_sampling({"temperature": 0.8})
        assert captured.requested_sampling_params["temperature"] == 0
        assert len(sent) == 1
        assert sent[0]["temperature"] == 0.8 and sent[0]["logprobs"] is True
        assert "_openenv_sampling" not in response.json()


def test_rollout_cancellation_releases_session_and_applies_call_budget(
    monkeypatch, tmp_path
):
    registry = SessionRegistry()
    module = ModuleType("harbor.trial.trial")
    observed = []

    class Trial:
        @classmethod
        async def create(cls, config):
            observed.append(registry.get(registry.list_ids()[0]).max_model_calls)
            return cls()

        async def run(self):
            raise asyncio.CancelledError()

    module.Trial = Trial
    monkeypatch.setitem(sys.modules, "harbor.trial.trial", module)
    monkeypatch.setattr(rollout, "build_trial_config", lambda **_: None)
    monkeypatch.setattr(
        rollout.seams,
        "get",
        lambda _, *, profile=None: SimpleNamespace(
            resolve=lambda **_: ("m", {}, {}, {})
        ),
    )
    monkeypatch.setattr(
        "openenv.harbor.shared_template.enable_shared_templates", lambda: False
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            rollout.run_rollout(
                task_dir=tmp_path,
                harness="offline",
                sandbox="offline",
                registry=registry,
                intercept_url="http://offline.invalid",
                model="m",
                trials_dir=tmp_path,
                agent_step_limit=3,
            )
        )
    assert observed == [3]
    assert registry.list_ids() == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("logit_bias", {"1": 2}),
        ("response_format", {"type": "json_schema"}),
        ("tool_choice", "required"),
        ("min_tokens", 10),
        ("allowed_token_ids", [1, 2]),
    ],
)
def test_constrained_logits_are_rejected_without_spending_call_budget(field, value):
    app = server.create_app(llm_url="http://engine.invalid", model="m")
    with TestClient(app) as client:
        sid = client.post("/sessions", json={"sampling": {"temperature": 0.8}}).json()[
            "session_id"
        ]
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [],
                field: value,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {"type": "object"}},
                    }
                ],
            },
            headers={"Authorization": f"Bearer {sid}"},
        )
        assert response.status_code == 400
        assert app.state.registry.get(sid).model_calls == 0


def test_policy_changing_compatibility_fallback_cannot_produce_training_data():
    app = server.create_app(llm_url="http://engine.invalid", model="m")

    async def handle(request):
        import json

        if "temperature" in json.loads(request.content):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'temperature' is not supported with this model.",
                        "param": "temperature",
                        "code": "unsupported_parameter",
                    }
                },
            )
        return httpx.Response(200, json={"choices": []})

    app.state.inference._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://engine.invalid"
    )
    with TestClient(app) as client:
        sid = client.post("/sessions", json={"sampling": {"temperature": 0.8}}).json()[
            "session_id"
        ]
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": []},
            headers={"Authorization": f"Bearer {sid}"},
        )
        assert response.status_code == 502
        session = app.state.registry.get(sid)
        assert any("sampling_policy_changed" in finding for finding in session.findings)
        assert not list(session.graph.nodes())


def test_model_call_budget_holds_under_concurrent_requests():
    app = server.create_app(llm_url="http://engine.invalid", model="m")
    forwarded = []

    async def completion(request):
        forwarded.append(request)
        await asyncio.sleep(0)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }

    app.state.inference.completion = completion

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://capture"
        ) as client:
            sid = (await client.post("/sessions", json={"max_model_calls": 7})).json()[
                "session_id"
            ]
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json={"model": "m", "messages": []},
                        headers={"Authorization": f"Bearer {sid}"},
                    )
                    for _ in range(100)
                )
            )
            assert all(response.status_code == 200 for response in responses)
            assert len(forwarded) == 7
            assert app.state.registry.get(sid).model_calls == 7

    asyncio.run(check())


def test_every_output_limit_alias_is_capped_and_missing_limits_are_bounded():
    request = {
        "max_tokens": 16384,
        "max_completion_tokens": 64000,
        "max_output_tokens": 8192,
    }
    assert server.clamp_output_tokens(request, 4096) == 64000
    assert request == {
        "max_tokens": 4096,
        "max_completion_tokens": 4096,
        "max_output_tokens": 4096,
    }
    missing = {}
    server.clamp_output_tokens(missing, 4096)
    assert missing == {"max_tokens": 4096}


def test_client_serializes_sampling_in_the_mcp_rollout_arguments():
    client = object.__new__(HarborEnv)
    sent = {}

    def call(name, **kwargs):
        sent.update(tool=name, **kwargs)
        return HarborRolloutResult().model_dump()

    client._call = call
    client.run_rollout(sampling={"temperature": 0.8}, agent_step_limit=17)
    assert sent["tool"] == "run_rollout"
    assert sent["sampling"] == {"temperature": 0.8}
    assert sent["agent_step_limit"] == 17
    assert not {"provider", "purpose", "eval_sampling"} & sent.keys()


def test_client_never_drops_explicit_new_provider_semantics():
    client = object.__new__(HarborEnv)
    sent = {}

    def call(name, **kwargs):
        sent.update(kwargs)
        return HarborRolloutResult().model_dump()

    client._call = call
    client.run_rollout(provider="anthropic", purpose="eval", eval_sampling={})
    assert sent["provider"] == "anthropic"
    assert sent["purpose"] == "eval"
    assert sent["eval_sampling"] == {}
