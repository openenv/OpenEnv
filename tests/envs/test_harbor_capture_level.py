# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which capture level an endpoint gets, and what follows from it.

Two rollout types, and the endpoint decides which one you get: `train` when it returns token ids and
aligned logprobs, `eval` otherwise. The tests below cover the decision (a probe that negotiates its
way down through a provider's 400s) and the consequence that matters most — that an eval rollout
cannot be mistaken for, or converted into, a training one.
"""

from __future__ import annotations

import urllib.error

import pytest

validate_llm_mod = pytest.importorskip("openenv.core.harness.capture.validate_llm")
export_mod = pytest.importorskip("openenv.core.harness.capture.export")
contract_mod = pytest.importorskip("openenv.core.harness.capture.contract")
graph_mod = pytest.importorskip("openenv.core.harness.capture.graph")

validate_llm = validate_llm_mod.validate_llm


def http_400(payload: dict) -> urllib.error.HTTPError:
    import io
    import json

    return urllib.error.HTTPError(
        "http://engine/v1/chat/completions",
        400,
        "Bad Request",
        {},
        io.BytesIO(json.dumps(payload).encode()),
    )


def unsupported(param: str, code: str = "unsupported_parameter", message="") -> dict:
    return {
        "error": {
            "message": message
            or f"Unsupported parameter: '{param}' is not supported with this model.",
            "param": param,
            "code": code,
        }
    }


def reply(*, prompt_ids=None, token_ids=None, logprobs=None) -> dict:
    choice: dict = {"message": {"content": "ok"}, "finish_reason": "stop"}
    if token_ids is not None:
        choice["token_ids"] = token_ids
    if logprobs is not None:
        choice["logprobs"] = {"content": [{"logprob": lp} for lp in logprobs]}
    payload: dict = {"choices": [choice]}
    if prompt_ids is not None:
        payload["prompt_token_ids"] = prompt_ids
    return payload


def probe(monkeypatch, script):
    """Run `validate_llm` against a scripted endpoint. Returns the report and the bodies sent."""
    sent: list[dict] = []
    remaining = list(script)

    def fake_post(url, body, timeout, api_key=None, auth_header="Authorization"):
        sent.append(dict(body))
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(validate_llm_mod, "_post", fake_post)
    monkeypatch.setattr(validate_llm_mod, "list_models", lambda *a, **k: ["m"])
    # The logprobs-mode probe issues its own two calls and is exercised on its own below; leaving it
    # on here would make every scripted case account for requests it is not about.
    return (
        validate_llm(
            "http://engine", "m", check_logprobs_mode=False, check_tools=False
        ),
        sent,
    )


# --- the probe decides the level --------------------------------------------
def test_a_full_capture_engine_is_trainable(monkeypatch):
    report, sent = probe(
        monkeypatch,
        [reply(prompt_ids=[1, 2], token_ids=[7, 8], logprobs=[-0.1, -0.2])],
    )
    assert (report.capture_level, report.rollout_type) == ("tokens", "train")
    assert report.trainable and report.ok and report.reachable
    assert len(sent) == 1, "grading a working engine must cost exactly one completion"
    assert report.param_fixes == []


def test_token_ids_returned_as_null_land_on_the_logprobs_level(monkeypatch):
    """The HF router's shape: it accepts `return_token_ids` and answers `token_ids: null`. The level
    must come from what came back, not from the request having been accepted."""
    report, _ = probe(monkeypatch, [reply(token_ids=None, logprobs=[-0.1])])
    assert (report.capture_level, report.rollout_type) == ("logprobs", "eval")
    assert not report.trainable and report.reachable


def test_a_provider_rejecting_return_token_ids_is_retried_without_it(monkeypatch):
    report, sent = probe(
        monkeypatch,
        [
            http_400(
                {
                    "error": {
                        "message": "Unrecognized request argument supplied: "
                        "return_token_ids"
                    }
                }
            ),
            reply(logprobs=[-0.1]),
        ],
    )
    assert report.capture_level == "logprobs"
    assert "return_token_ids" in sent[0] and "return_token_ids" not in sent[1]
    assert report.param_fixes == ["dropped return_token_ids"]


def test_a_provider_rejecting_logprobs_too_lands_on_text(monkeypatch):
    """Every current OpenAI model: `return_token_ids` unknown, `logprobs` unsupported."""
    report, sent = probe(
        monkeypatch,
        [
            http_400(
                unsupported(
                    "return_token_ids",
                    "unknown_parameter",
                    "Unknown parameter: 'return_token_ids'.",
                )
            ),
            http_400(unsupported("logprobs")),
            reply(),
        ],
    )
    assert (report.capture_level, report.rollout_type) == ("text", "eval")
    assert "logprobs" not in sent[-1]
    assert "top_logprobs" not in sent[-1]
    assert report.reachable


def test_max_tokens_is_renamed_during_the_probe(monkeypatch):
    report, sent = probe(
        monkeypatch,
        [
            http_400(
                unsupported(
                    "max_tokens",
                    message="Unsupported parameter: 'max_tokens' is not supported with "
                    "this model. Use 'max_completion_tokens' instead.",
                )
            ),
            reply(prompt_ids=[1], token_ids=[2], logprobs=[-0.1]),
        ],
    )
    assert sent[1]["max_completion_tokens"] == sent[0]["max_tokens"]
    assert report.param_fixes == ["renamed max_tokens -> max_completion_tokens"]


def test_an_unreachable_endpoint_is_neither_trainable_nor_eval(monkeypatch):
    report, _ = probe(monkeypatch, [OSError("connection refused")])
    assert not report.reachable
    assert report.capture_level == ""
    assert "connection refused" in report.summary()


def test_a_model_the_endpoint_does_not_serve_fails_before_any_completion(monkeypatch):
    monkeypatch.setattr(
        validate_llm_mod, "list_models", lambda *a, **k: ["other-model"]
    )
    monkeypatch.setattr(
        validate_llm_mod,
        "_post",
        lambda *a, **k: pytest.fail("must not spend a completion on a bad model name"),
    )
    report = validate_llm("http://engine", "m")
    assert not report.ok and not report.reachable


def test_an_endpoint_that_publishes_no_model_list_is_still_probed(monkeypatch):
    """Some hosted gateways gate `/v1/models` differently from inference, or omit it. Refusing there
    would reject an endpoint that serves completions perfectly well."""
    monkeypatch.setattr(validate_llm_mod, "list_models", lambda *a, **k: [])
    monkeypatch.setattr(
        validate_llm_mod,
        "_post",
        lambda *a, **k: reply(prompt_ids=[1], token_ids=[2], logprobs=[-0.1]),
    )
    report = validate_llm("http://engine", "m")
    assert report.trainable


def test_require_llm_accepts_an_eval_endpoint_but_can_be_told_not_to(monkeypatch):
    monkeypatch.setattr(validate_llm_mod, "list_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(validate_llm_mod, "_post", lambda *a, **k: reply())

    report = validate_llm_mod.require_llm("http://engine", "m")
    assert report.capture_level == "text"

    with pytest.raises(RuntimeError, match="needs trainable rollouts"):
        validate_llm_mod.require_llm("http://engine", "m", require_tokens=True)


# --- what an eval level means downstream ------------------------------------
class FakeSession:
    def __init__(self, graph):
        self.session_id = "s1"
        self.metadata: dict = {}
        self.findings: list[str] = []
        self.graph = graph


def eval_graph(n_tools=1):
    graph = graph_mod.RolloutGraph()
    messages = [{"role": "user", "content": "go"}]
    graph.add_turn(
        graph_mod.TurnNode(
            node_id="a",
            prompt_ids=[],
            sampled_ids=[],
            request_messages=messages,
            response_message={"role": "assistant", "content": "step 1"},
            n_tools=n_tools,
        )
    )
    graph.add_turn(
        graph_mod.TurnNode(
            node_id="b",
            prompt_ids=[],
            sampled_ids=[],
            request_messages=[
                *messages,
                {"role": "assistant", "content": "step 1"},
                {"role": "user", "content": "result"},
            ],
            response_message={"role": "assistant", "content": "done"},
            n_tools=n_tools,
        )
    )
    return graph


def test_an_eval_export_keeps_the_whole_trace_and_claims_nothing_trainable():
    graph = eval_graph()
    document = export_mod.export_session(
        FakeSession(graph), include_messages=True, capture_level="logprobs"
    )
    assert document["rollout_type"] == "eval"
    assert document["capture_level"] == "logprobs"
    assert document["trainable"] is False

    # The trace is the payload of an eval rollout, and all of it has to survive.
    assert len(document["turns"]) == 2
    assert document["turns"][0]["request_messages"]

    # Structure survives too. `conversations_from_document` and `turns_from_document` both walk
    # `sequences`, so emptying it on an eval rollout would delete the very trace this exists for —
    # which is exactly the bug a live rollout caught: reward 1.0, six turns, zero conversations.
    assert len(document["sequences"]) == 1
    row = document["sequences"][0]
    assert row["node_ids"] == ["a", "b"]
    assert row["n_turns"] == 2
    # What is withheld is the training claim, not the structure.
    assert row["trainable"] is False
    assert row["input_ids"] == []
    assert document["stats"]["n_trainable_tokens"] == 0


def test_an_eval_rollout_still_rebuilds_its_conversations():
    """The regression a live run found: an eval result with a reward, six turns and no transcript."""
    models = pytest.importorskip("openenv.harbor.models")
    document = export_mod.export_session(
        FakeSession(eval_graph()), include_messages=True, capture_level="logprobs"
    )
    conversations = models.conversations_from_document(document)
    assert len(conversations) == 1
    # The deepest node replays the whole thread, plus its own reply.
    assert [m["role"] for m in conversations[0].messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    turns = models.turns_from_document(document)
    assert len(turns) == 2
    assert [t.text for t in turns] == ["step 1", "done"]
    assert all(t.prompt_token_ids == [] for t in turns)


@pytest.mark.parametrize("capture_level", ["text", "logprobs"])
@pytest.mark.parametrize("n_tools", [0, 1])
def test_eval_export_preserves_independent_conversations(capture_level, n_tools):
    models = pytest.importorskip("openenv.harbor.models")
    graph = eval_graph(n_tools=n_tools)
    graph.add_turn(
        graph_mod.TurnNode(
            node_id="independent",
            prompt_ids=[],
            sampled_ids=[],
            request_messages=[{"role": "user", "content": "Check another task"}],
            response_message={"role": "assistant", "content": "independent result"},
            n_tools=n_tools,
        )
    )
    document = export_mod.export_session(
        FakeSession(graph), include_messages=True, capture_level=capture_level
    )

    conversations = models.conversations_from_document(document)
    assert {conversation.root_id for conversation in conversations} == {
        "a",
        "independent",
    }
    turns = models.turns_from_document(document)
    assert {turn.node_id: turn.text for turn in turns} == {
        "a": "step 1",
        "b": "done",
        "independent": "independent result",
    }
    assert not any(turn["discarded"] for turn in document["turns"])
    assert document["trainable"] is False
    assert document["stats"]["n_trainable_tokens"] == 0
    assert all(not turn.trainable for turn in turns)


def test_a_training_contract_cannot_be_built_from_an_eval_rollout():
    graph = eval_graph()
    document = export_mod.export_session(FakeSession(graph), capture_level="text")
    for build in (contract_mod.to_turn_records, contract_mod.to_trace_entries):
        with pytest.raises(ValueError, match="EVAL rollout"):
            build(graph, document)


def test_a_train_export_is_unchanged():
    """The regression that matters: nothing on the trainable path may move."""
    graph = graph_mod.RolloutGraph()
    graph.add_turn(
        graph_mod.TurnNode(
            node_id="a",
            prompt_ids=[1, 2, 3],
            sampled_ids=[4, 5],
            sampled_logprobs=[-0.1, -0.2],
            n_tools=1,
        )
    )
    document = export_mod.export_session(FakeSession(graph))
    assert document["rollout_type"] == "train"
    assert document["capture_level"] == "tokens"
    assert len(document["sequences"]) == 1
    assert document["sequences"][0]["input_ids"] == [1, 2, 3, 4, 5]
    assert contract_mod.to_turn_records(graph, document) == [
        ([1, 2, 3], [4, 5], [-0.1, -0.2])
    ]


# --- the proxy reports its level, and never its credential ------------------
def app_client(**kwargs):
    from fastapi.testclient import TestClient

    server = pytest.importorskip("openenv.core.harness.capture.server")
    return TestClient(
        server.create_app(llm_url="http://127.0.0.1:9/v1", model="m", **kwargs)
    )


def test_health_states_the_rollout_type():
    with app_client(capture_level="logprobs") as client:
        body = client.get("/health").json()
    assert body["capture_level"] == "logprobs"
    assert body["rollout_type"] == "eval"


def test_health_confirms_auth_without_revealing_the_key():
    """On a Space this endpoint is public, and the key it holds buys paid inference."""
    with app_client(api_key="sk-secret-value") as client:
        body = client.get("/health").json()
    assert body["upstream_auth"] is True
    assert "sk-secret-value" not in str(body)


def test_health_says_train_by_default():
    with app_client() as client:
        body = client.get("/health").json()
    assert (body["capture_level"], body["rollout_type"]) == ("tokens", "train")
    assert body["upstream_auth"] is False


def test_an_eval_rollout_is_recorded_without_fatal_findings(monkeypatch):
    """`check_turn`'s FATALs (`no_prompt_ids`, `no_logprobs`) are the expected condition here.
    Letting them fire would mark every eval turn unusable and teach everyone to ignore findings.

    Two calls exercise the multi-turn capture contract without the single-call warning.
    """
    replies = [reply(), reply()]

    async def fake_completion(_self, _request):
        return replies.pop(0)

    server = pytest.importorskip("openenv.core.harness.capture.server")
    upstream = pytest.importorskip("openenv.core.harness.capture.upstream")
    monkeypatch.setattr(upstream.InferenceClient, "completion", fake_completion)

    app = server.create_app(
        llm_url="http://127.0.0.1:9/v1", model="m", capture_level="text"
    )
    from fastapi.testclient import TestClient

    first = [{"role": "user", "content": "go"}]
    second = [
        *first,
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "tool result"},
    ]
    with TestClient(app) as client:
        session = client.post("/sessions").json()["session_id"]
        for messages in (first, second):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": messages},
                headers={"Authorization": f"Bearer {session}"},
            )
            assert response.status_code == 200
        status = client.get(f"/sessions/{session}").json()
        rollout = client.get(f"/sessions/{session}/rollout").json()

    assert status["n_turns"] == 2
    # One root, via the message-prefix fallback: with no token ids the token rule would make each
    # call its own root and the rollout would read as two unrelated conversations. Note that
    # `reply()` omits `role` on the assistant message while the harness names it on the way back —
    # linking has to survive that asymmetry, since it is invisible until the graph collapses.
    assert status["n_roots"] == 1
    assert rollout["rollout_type"] == "eval"
    assert not [f for f in rollout["validation"] if f.startswith("[FATAL")], (
        "an eval rollout must not be reported as a capture failure"
    )


# --- the train path must not move -------------------------------------------
#
# The acceptance rows for a real vLLM and a real SGLang need a GPU. This is the gate that runs
# anywhere, including CI: a stub upstream returning exactly the shape vLLM returns with
# `--return-tokens-as-token-ids --logprobs-mode processed_logprobs`, driven through the whole proxy,
# with every field a trainer reads pinned. A change to the eval path that leaks into the token path
# fails here rather than in a loss curve days later.
def test_the_trainable_path_end_to_end_is_unchanged(monkeypatch):
    server = pytest.importorskip("openenv.core.harness.capture.server")
    upstream = pytest.importorskip("openenv.core.harness.capture.upstream")
    from fastapi.testclient import TestClient

    # Turn k's prompt is turn k-1's prompt + completion + one interstitial context token, which is
    # what a real engine returns once the harness has appended a tool result.
    scripted = [
        {"prompt": [1, 2, 3], "sampled": [10, 11], "logprobs": [-0.5, -0.25]},
        {
            "prompt": [1, 2, 3, 10, 11, 90],
            "sampled": [12, 13, 14],
            "logprobs": [-0.1, -0.2, -0.3],
        },
        {
            "prompt": [1, 2, 3, 10, 11, 90, 12, 13, 14, 91],
            "sampled": [15],
            "logprobs": [-0.05],
        },
    ]
    remaining = list(scripted)

    async def fake_completion(_self, _request):
        step = remaining.pop(0)
        return upstream.normalize_response(
            {
                "prompt_token_ids": step["prompt"],
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "step"},
                        "finish_reason": "stop",
                        "token_ids": step["sampled"],
                        "logprobs": {
                            "content": [{"logprob": lp} for lp in step["logprobs"]]
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(upstream.InferenceClient, "completion", fake_completion)
    app = server.create_app(llm_url="http://127.0.0.1:9/v1", model="m")

    with TestClient(app) as client:
        session = client.post("/sessions").json()["session_id"]
        for _ in scripted:
            assert (
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "go"}],
                        "tools": [{"type": "function", "function": {"name": "bash"}}],
                    },
                    headers={"Authorization": f"Bearer {session}"},
                ).status_code
                == 200
            )
        document = client.get(f"/sessions/{session}/rollout").json()
        graph = app.state.registry.get(session).graph

    assert document["rollout_type"] == "train"
    assert document["capture_level"] == "tokens"
    assert document["trainable"] is True
    assert document["stats"]["n_turns"] == 3
    assert document["stats"]["n_roots"] == 1

    (row,) = document["sequences"]
    assert row["role"] == "agent"
    assert row["trainable"] is True
    # The flattened sequence: every prompt token the model conditioned on, in order, with each
    # sampled span marked 1 and the interstitial context tokens marked 0.
    assert row["input_ids"] == [1, 2, 3, 10, 11, 90, 12, 13, 14, 91, 15]
    assert row["loss_mask"] == [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    assert row["logprobs"] == [
        0.0,
        0.0,
        0.0,
        -0.5,
        -0.25,
        0.0,
        -0.1,
        -0.2,
        -0.3,
        0.0,
        -0.05,
    ]
    assert row["prompt_len"] == 3
    assert row["turn_lengths"] == [2, 3, 1]
    assert row["n_trainable"] == 6
    assert document["stats"]["n_trainable_tokens"] == 6

    # And the contract a trainer actually consumes, per turn.
    assert contract_mod.to_turn_records(graph, document) == [
        ([1, 2, 3], [10, 11], [-0.5, -0.25]),
        ([1, 2, 3, 10, 11, 90], [12, 13, 14], [-0.1, -0.2, -0.3]),
        ([1, 2, 3, 10, 11, 90, 12, 13, 14, 91], [15], [-0.05]),
    ]


def test_the_probe_does_not_restate_the_tier_as_fatal_findings(monkeypatch):
    """An eval endpoint's findings must not read like a broken one.

    `no_prompt_token_ids` is reported FATAL by `check_upstream_response`, and it is the *definition*
    of an eval endpoint. Printing three fatal-looking lines under a heading that already says
    EVAL ONLY is how a findings list stops being read at all.
    """
    report, _ = probe(monkeypatch, [reply()])
    assert report.capture_level == "text"
    assert report.findings == []


def test_a_genuinely_broken_response_still_reports(monkeypatch):
    """`no_choices` is not "merely eval-only" — the endpoint answered with nothing at all."""
    report, _ = probe(monkeypatch, [{"id": "x"}])
    assert any("no_choices" in f for f in report.findings)


# --- raw vs processed logprobs ----------------------------------------------
#
# The one hole no other check here can see. `token_ids` arrives from the REQUEST parameter, not from a
# serving flag, so an engine started with neither flag returns aligned, negative, correctly-counted
# logprobs that are pre-temperature — and grades as fully trainable. vLLM's `logprobs_mode` defaults
# to `raw_logprobs`. The test follows from the definition: raw values cannot move with temperature.
def temperature_scripted(monkeypatch, by_temperature, *, fail=False):
    """Serve first-position `top_logprobs` per temperature. Returns the calls made."""
    calls: list[float] = []

    def fake_post(url, body, timeout, api_key=None, auth_header="Authorization"):
        if fail:
            raise OSError("endpoint refused")
        calls.append(body["temperature"])
        tops = by_temperature[body["temperature"]]
        return {
            "choices": [
                {
                    "message": {"content": "x"},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "a",
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": t, "logprob": lp}
                                    for t, lp in tops.items()
                                ],
                            }
                        ]
                    },
                }
            ]
        }

    monkeypatch.setattr(validate_llm_mod, "_post", fake_post)
    return calls


def test_an_unchanged_gap_is_raw(monkeypatch):
    """Live measurement: gap 6.7500 at both temperatures on a default-flags vLLM."""
    calls = temperature_scripted(
        monkeypatch,
        {1.0: {"a": -0.0037, "b": -6.7537}, 2.0: {"a": -0.0037, "b": -6.7537}},
    )
    assert validate_llm_mod.probe_logprobs_mode("http://engine", "m") == "raw"
    assert calls == [1.0, 2.0]


def test_a_halved_gap_is_processed(monkeypatch):
    """Live measurement: gap 6.7500 -> 3.3750, i.e. exactly T1/T2, on a processed_logprobs vLLM."""
    temperature_scripted(
        monkeypatch,
        {1.0: {"a": -0.0037, "b": -6.7537}, 2.0: {"a": -1.4380, "b": -4.8130}},
    )
    assert validate_llm_mod.probe_logprobs_mode("http://engine", "m") == "processed"


def test_a_constant_offset_between_calls_does_not_change_the_verdict(monkeypatch):
    """Why the gap, not the value: a data-parallel engine answers consecutive calls from different
    replicas, and comparing values directly misread one such engine (DP=4) as processed. A constant
    shift cancels in a difference, so the same gap survives it."""
    temperature_scripted(
        monkeypatch,
        {1.0: {"a": -0.0037, "b": -6.7537}, 2.0: {"a": -0.9037, "b": -7.6537}},
    )
    assert validate_llm_mod.probe_logprobs_mode("http://engine", "m") == "raw"


def test_a_distribution_too_flat_to_divide_is_unknown(monkeypatch):
    """Guessing from noise is how a check becomes something people override on principle."""
    temperature_scripted(
        monkeypatch, {1.0: {"a": -0.5, "b": -0.6}, 2.0: {"a": -0.5, "b": -0.9}}
    )
    assert validate_llm_mod.probe_logprobs_mode("http://engine", "m") == "unknown"


def test_a_single_top_logprob_is_unknown(monkeypatch):
    """A gap needs two values."""
    temperature_scripted(monkeypatch, {1.0: {"a": -0.5}, 2.0: {"a": -0.5}})
    assert validate_llm_mod.probe_logprobs_mode("http://engine", "m") == "unknown"


def test_an_unreachable_endpoint_is_unknown(monkeypatch):
    """Absence of evidence, not evidence of a problem."""
    temperature_scripted(monkeypatch, {}, fail=True)
    assert validate_llm_mod.probe_logprobs_mode("http://engine", "m") == "unknown"


def test_a_provider_that_refused_temperature_cannot_be_asked(monkeypatch):
    """gpt-5.6 drops `temperature`, so a question about temperature has no meaning — and must cost
    no calls at all."""
    compat = pytest.importorskip("openenv.core.harness.capture.compat")
    calls = temperature_scripted(monkeypatch, {})
    assert (
        validate_llm_mod.probe_logprobs_mode(
            "http://engine", "m", fixes=[compat.ParamFix(param="temperature")]
        )
        == "unknown"
    )
    assert calls == []


def test_raw_logprobs_demote_a_trainable_endpoint_to_eval(monkeypatch):
    """The endpoint answers fine and is a good eval backend; what it cannot do is train."""
    monkeypatch.delenv("OPENENV_ALLOW_RAW_LOGPROBS", raising=False)
    monkeypatch.setattr(validate_llm_mod, "list_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(
        validate_llm_mod,
        "probe_logprobs_mode",
        lambda *a, **k: "raw",
    )
    monkeypatch.setattr(
        validate_llm_mod,
        "_post",
        lambda *a, **k: reply(prompt_ids=[1], token_ids=[2], logprobs=[-0.1]),
    )
    report = validate_llm("http://engine", "m")
    assert report.logprobs_mode == "raw"
    assert report.capture_level == "logprobs", "must not stay trainable"
    assert report.trainable is False
    assert report.reachable is True, "still perfectly usable for eval"
    assert any("raw_logprobs" in f and "[FATAL]" in f for f in report.findings)
    # The measurement supersedes the inference that pointed at it; printing both says one thing twice.
    assert not any("token_strings" in f for f in report.findings)


def test_the_override_keeps_it_trainable_but_says_so(monkeypatch):
    """A refusal that cannot be overridden becomes a reason to stop trusting the tool."""
    monkeypatch.setenv("OPENENV_ALLOW_RAW_LOGPROBS", "1")
    monkeypatch.setattr(validate_llm_mod, "list_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(validate_llm_mod, "probe_logprobs_mode", lambda *a, **k: "raw")
    monkeypatch.setattr(
        validate_llm_mod,
        "_post",
        lambda *a, **k: reply(prompt_ids=[1], token_ids=[2], logprobs=[-0.1]),
    )
    report = validate_llm("http://engine", "m")
    assert report.capture_level == "tokens"
    assert report.trainable is True
    assert any("raw_logprobs_forced" in f for f in report.findings)


def test_the_mode_is_not_probed_below_the_tokens_tier(monkeypatch):
    """Nothing below `tokens` is trainable, so the answer would inform no decision."""
    monkeypatch.setattr(validate_llm_mod, "list_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(
        validate_llm_mod,
        "probe_logprobs_mode",
        lambda *a, **k: pytest.fail("must not spend two calls on an eval endpoint"),
    )
    monkeypatch.setattr(validate_llm_mod, "_post", lambda *a, **k: reply())
    assert validate_llm("http://engine", "m").logprobs_mode == ""


# --- can a coding agent work here at all? -----------------------------------
#
# The capture probe sends no tools; every validated harness sends one on every call. So this is the
# only signal about agent viability, and it caught a real failure that `harbor info` previously
# reported as a perfectly healthy endpoint.
def tool_reply(*, tool_call=True, finish="stop"):
    message = {"role": "assistant", "content": None if tool_call else "I'd run ls."}
    if tool_call:
        message["tool_calls"] = [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
        ]
    return {"choices": [{"message": message, "finish_reason": finish}]}


def test_a_tool_call_means_agents_can_work(monkeypatch):
    monkeypatch.setattr(validate_llm_mod, "_post", lambda *a, **k: tool_reply())
    assert validate_llm_mod.probe_tool_support("http://engine", "m")[0] == "ok"


def test_prose_instead_of_a_tool_call_is_reported(monkeypatch):
    monkeypatch.setattr(
        validate_llm_mod, "_post", lambda *a, **k: tool_reply(tool_call=False)
    )
    assert (
        validate_llm_mod.probe_tool_support("http://engine", "m")[0] == "no-tool-call"
    )


def test_truncation_is_inconclusive_not_a_failure(monkeypatch):
    """The false positive this avoids: Qwen3.6-35B-A3B spent 224 tokens reasoning and hit the cap, so
    a 64-token probe called it tool-incapable while it in fact worked with all 16 harnesses."""
    monkeypatch.setattr(
        validate_llm_mod,
        "_post",
        lambda *a, **k: tool_reply(tool_call=False, finish="length"),
    )
    assert validate_llm_mod.probe_tool_support("http://engine", "m")[0] == "unknown"


def test_an_endpoint_that_refuses_tools_outright_is_flagged(monkeypatch):
    """`tools` is protected from being dropped, so this cannot be papered over."""

    def refuse(*a, **k):
        raise http_400(
            {"error": {"message": "tools are not supported", "param": "tools"}}
        )

    monkeypatch.setattr(validate_llm_mod, "_post", refuse)
    assert validate_llm_mod.probe_tool_support("http://engine", "m")[0] == "rejected"


def test_reasoning_forced_off_warns_at_validate_time(monkeypatch):
    """The gpt-5.6 case: tools are accepted only with reasoning disabled, after which agentic loops
    make one model call and stop. Discovered only when the probe carries a tool manifest."""
    monkeypatch.setattr(validate_llm_mod, "list_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(
        validate_llm_mod, "probe_logprobs_mode", lambda *a, **k: "processed"
    )
    compat = pytest.importorskip("openenv.core.harness.capture.compat")
    monkeypatch.setattr(
        validate_llm_mod,
        "probe_tool_support",
        lambda *a, **k: (
            "ok",
            [compat.ParamFix(param="reasoning_effort", value="none")],
        ),
    )
    monkeypatch.setattr(
        validate_llm_mod,
        "_post",
        lambda *a, **k: reply(prompt_ids=[1], token_ids=[2], logprobs=[-0.1]),
    )
    report = validate_llm("http://engine", "m")
    assert any(
        "reasoning_effort" in f and "behaviour_changed" in f for f in report.findings
    )
    assert any("single model call" in f for f in report.findings)


# --- roles must not depend on token counts that eval endpoints never have ----
def test_a_toolless_harness_is_still_the_agent_on_an_eval_endpoint():
    """terminus-2 parses tool calls out of raw text, so it sends no manifest. Role assignment used
    `n_trainable` as the tiebreak when nothing had tools, and on an eval endpoint that is 0 for every
    sequence — so every path was labelled auxiliary, `result.turns` came back empty and the
    conversations were mistagged, on a rollout that had captured perfectly well."""
    models = pytest.importorskip("openenv.harbor.models")
    graph = graph_mod.RolloutGraph()
    first = [{"role": "user", "content": "go"}]
    graph.add_turn(
        graph_mod.TurnNode(
            node_id="a",
            prompt_ids=[],
            sampled_ids=[],
            n_tools=0,
            request_messages=first,
            response_message={"role": "assistant", "content": "step 1"},
        )
    )
    graph.add_turn(
        graph_mod.TurnNode(
            node_id="b",
            prompt_ids=[],
            sampled_ids=[],
            n_tools=0,
            request_messages=[
                *first,
                {"role": "assistant", "content": "step 1"},
                {"role": "user", "content": "result"},
            ],
            response_message={"role": "assistant", "content": "done"},
        )
    )
    document = export_mod.export_session(
        FakeSession(graph), include_messages=True, capture_level="logprobs"
    )
    assert [r["role"] for r in document["sequences"]] == ["agent"]
    assert len(models.turns_from_document(document)) == 2
    assert len(models.conversations_from_document(document)) == 1


def test_the_train_path_still_uses_trainable_tokens_as_the_tiebreak():
    """Where token counts DO mean something, a toolless sequence with nothing trainable is auxiliary."""
    graph = graph_mod.RolloutGraph()
    graph.add_turn(
        graph_mod.TurnNode(
            node_id="a",
            prompt_ids=[1, 2],
            sampled_ids=[3],
            sampled_logprobs=None,  # rejected on ingest -> masked out -> nothing trainable
            n_tools=0,
        )
    )
    document = export_mod.export_session(FakeSession(graph), capture_level="tokens")
    assert [r["role"] for r in document["sequences"]] == ["auxiliary"]
