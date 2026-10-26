# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`reconcile`: the only check on this path that is not self-referential.

It compares the capture against ATIF, the trace the harness writes independently, and its verdict is
what gates trainability. Until now only its helpers were tested — `load_trace` and
`atif_turn_lengths` — so every decision that actually decides whether a rollout may be trained on was
uncovered: the turn-count FATAL, the coverage floor, the auxiliary-subsequence inference, and the
subagent refusal.
"""

from __future__ import annotations

import pytest

atif = pytest.importorskip("openenv.harbor.atif")

reconcile = atif.reconcile


graph_mod = pytest.importorskip("openenv.core.harness.capture.graph")
export_mod = pytest.importorskip("openenv.core.harness.capture.export")


def document(lengths, *, role="agent", rollout_type="train"):
    """A real capture document with one sampled span per entry in `lengths`.

    Built by `export_session` over a real graph rather than hand-rolled, so the fixture cannot drift
    from the document contract `reconcile` reads — three separate KeyErrors while writing these tests
    were a hand-written dict missing fields the producer always sets.
    """
    graph = graph_mod.RolloutGraph()
    prompt = [1]
    for i, n_sampled in enumerate(lengths):
        sampled = list(range(1000 + i * 100, 1000 + i * 100 + n_sampled))
        graph.add_turn(
            graph_mod.TurnNode(
                node_id=f"n{i}",
                prompt_ids=list(prompt),
                sampled_ids=sampled,
                sampled_logprobs=[-0.1] * n_sampled,
                n_tools=1,
                finish_reason="stop",
            )
        )
        prompt = prompt + sampled + [9000 + i]

    class Session:
        session_id = "s"
        metadata: dict = {}
        findings: list = []

    session = Session()
    session.graph = graph
    doc = export_mod.export_session(
        session, capture_level="tokens" if rollout_type == "train" else "text"
    )
    if role != "agent":
        # Same mutation rollout.py's auxiliary demotion performs.
        for row in doc["sequences"]:
            row["role"] = role
    return doc


def trace(lengths, **extra):
    return {
        "schema_version": "1.0",
        "agent": {"name": "opencode"},
        "steps": [
            {"source": "agent", "metrics": {"completion_tokens": n}} for n in lengths
        ],
        **extra,
    }


def codes(report):
    return {f.code for f in report.findings}


def fatal_codes(report):
    return {f.code for f in report.fatal}


# --- the agreeing case ------------------------------------------------------
def test_identical_turn_lengths_reconcile():
    report = reconcile(document([10, 20, 30]), trace([10, 20, 30]))
    assert report.ok
    assert not fatal_codes(report)


def test_explicit_synthetic_api_error_is_not_a_model_response():
    native = trace([10, 20, 0])
    native["steps"][-1].update(
        model_name="<synthetic>", message="API Error: 400 unsupported image"
    )
    native["steps"][-1]["metrics"].update(prompt_tokens=0, cached_tokens=0)
    report = reconcile(document([10, 20]), native)
    assert report.ok
    assert "atif_synthetic_api_error" in codes(report)
    assert len(native["steps"]) == 3  # Preserve the native evidence.


@pytest.mark.parametrize(
    "change",
    [
        {"model_name": "Qwen3.5-2B"},
        {"message": "An ordinary empty reply"},
        {"tool_calls": [{"name": "read"}]},
        {"metrics": {"prompt_tokens": 2, "completion_tokens": 0}},
        {"metrics": {"prompt_tokens": 0, "completion_tokens": 1}},
    ],
)
def test_unproven_or_sampled_zero_token_steps_still_fail(change):
    native = trace([10, 20, 0])
    native["steps"][-1].update(
        model_name="<synthetic>",
        message="API Error: 400 unsupported image",
        metrics={"prompt_tokens": 0, "completion_tokens": 0},
    )
    native["steps"][-1].update(change)
    assert "turn_mismatch" in fatal_codes(reconcile(document([10, 20]), native))


def test_no_atif_is_not_a_failure():
    """Three of sixteen harnesses emit no trajectory; that is an absent cross-check, not a fault."""
    report = reconcile(document([10]), None)
    assert report.ok
    assert "no_atif" in codes(report)


# --- disagreement -----------------------------------------------------------
def test_atif_logging_more_calls_than_were_captured_is_fatal():
    """Calls the harness made that never reached the proxy mean the capture is incomplete."""
    report = reconcile(document([10, 20]), trace([10, 20, 30]))
    assert not report.ok


@pytest.mark.parametrize("recorded_stops", [0, 1])
def test_proxy_stop_requires_recorded_provenance_and_leaves_trace_unchanged(
    recorded_stops,
):
    doc = document([10, 20])
    doc["budget_stop_count"] = recorded_stops
    native = trace([10, 20, 0])
    native["steps"][-1]["message"] = atif.BUDGET_STOP_MESSAGE
    report = reconcile(doc, native)
    assert report.ok == bool(recorded_stops)
    assert len(native["steps"]) == 3


@pytest.mark.parametrize(
    "message,count",
    [("unexpected missing generation", 0), (atif.BUDGET_STOP_MESSAGE, 7)],
)
def test_stop_allowance_cannot_hide_unknown_or_sampled_model_turns(message, count):
    doc = document([10])
    doc["budget_stop_count"] = 1
    native = trace([10, count])
    native["steps"][-1]["message"] = message
    assert not reconcile(doc, native).ok


def test_stop_allowance_is_bounded_by_responses_actually_emitted():
    doc = document([10])
    doc["budget_stop_count"] = 1
    native = trace([10, 0, 0])
    for step in native["steps"][1:]:
        step["message"] = atif.BUDGET_STOP_MESSAGE
    assert not reconcile(doc, native).ok


def test_extra_captured_calls_embed_as_auxiliary_when_coverage_is_high():
    """Seeing MORE than the harness logged is benign and explainable: a next-speaker check, a title
    generator. The asymmetry is deliberate — missing calls mean we lost something, extra ones do not."""
    report = reconcile(document([58, 132, 266, 370, 33]), trace([58, 132, 266, 370]))
    assert report.ok
    assert "atif_aux_calls" in codes(report)
    assert report.aux_node_ids, "the aux call must be identified so it can be demoted"


def test_low_coverage_refuses_rather_than_discarding_most_of_a_rollout():
    """The mimo case: 49 captured calls, ATIF logged 5, and a subsequence match would have demoted 44
    to auxiliary under a warning. Below the floor that inference is as likely coincidence as signal."""
    captured = list(range(1, 50))
    report = reconcile(document(captured), trace(captured[:5]))
    assert not report.ok
    assert "atif_coverage_too_low" in fatal_codes(report)


# --- converters that give nothing to compare against ------------------------
def test_all_zero_token_counts_downgrade_to_no_cross_check():
    """vibe reports completion_tokens=0 on every step while capturing perfectly. Failing the rollout
    would punish it for its trace converter rather than for anything wrong."""
    report = reconcile(document([10, 20]), trace([0, 0]))
    assert report.ok
    assert "atif_no_token_counts" in codes(report)


def test_no_agent_steps_at_all_downgrades_the_same_way():
    report = reconcile(document([10, 20]), trace([]))
    assert report.ok
    assert "atif_no_token_counts" in codes(report)


# --- structural edge cases --------------------------------------------------
def test_nothing_captured_defers_to_the_rollouts_own_finding():
    """`check_rollout` already reports no_turns plainly; a second FATAL here buries the real cause."""
    empty: dict = {
        "rollout_type": "train",
        "sequences": [],
        "turns": [],
        "stats": {"n_turns": 0, "n_roots": 0, "n_discarded": 0},
    }
    report = reconcile(empty, trace([10]))
    assert "no_turns_upstream" in codes(report)
    assert not fatal_codes(report)


def test_calls_captured_but_none_labelled_agent_is_fatal():
    report = reconcile(document([10, 20], role="auxiliary"), trace([10, 20]))
    assert not report.ok
    assert "no_agent_sequence" in fatal_codes(report)


def test_subagent_trajectories_are_refused_not_merely_noted():
    """The warning used to say subagent turns must not carry the parent's reward and then did nothing
    to stop it: no node ids collected, no role changed. ATIF does not say which captured calls belong
    to the subagent, so the rollout cannot be attributed and is refused."""
    report = reconcile(
        document([10, 20]),
        trace([10, 20], subagent_trajectories=[{"agent": {"name": "sub"}}]),
    )
    assert not report.ok
    assert "atif_subagents" in fatal_codes(report)


# --- the eval path ----------------------------------------------------------
def test_eval_rollouts_compare_call_counts_since_token_counts_do_not_exist():
    report = reconcile(document([0, 0, 0], rollout_type="eval"), trace([10, 20, 30]))
    assert report.ok
    assert "eval_reconcile_counts_only" in codes(report)


def test_eval_rollouts_still_notice_a_truncated_harness_trace():
    """The one real bug reconciliation has ever caught was a truncated trajectory. Counts alone are
    enough to see it, which is why the eval path bothers comparing at all."""
    report = reconcile(document([0] * 2, rollout_type="eval"), trace([1] * 6))
    assert "atif_calls_missing" in codes(report)


def _partial_usage_pair():
    doc = document([64, 199, 10])
    native = trace([None, None, 10])
    for index in range(2):
        call_id = f"model-call-{index}"
        doc["turns"][index]["response_message"] = {"tool_calls": [{"id": call_id}]}
        native["steps"][index]["tool_calls"] = [{"tool_call_id": call_id}]
    return doc, native


def test_partial_usage_requires_exact_ordered_call_identity():
    doc, native = _partial_usage_pair()
    report = reconcile(doc, native)
    assert report.ok
    assert "atif_partial_usage" in codes(report)
    assert "turns_match" not in codes(report)
    assert not report.aux_node_ids


@pytest.mark.parametrize(
    "failure",
    [
        "different_id",
        "missing_id",
        "duplicate_id",
        "known_count",
        "explicit_zero",
        "extra_step",
    ],
)
def test_partial_usage_cannot_hide_disagreement(failure):
    doc, native = _partial_usage_pair()
    if failure == "different_id":
        native["steps"][0]["tool_calls"][0]["tool_call_id"] = "different"
    elif failure == "missing_id":
        doc["turns"][0]["response_message"] = {}
    elif failure == "duplicate_id":
        for index in range(2):
            doc["turns"][index]["response_message"]["tool_calls"][0]["id"] = "duplicate"
            native["steps"][index]["tool_calls"][0]["tool_call_id"] = "duplicate"
    elif failure == "known_count":
        native["steps"][-1]["metrics"]["completion_tokens"] = 11
    elif failure == "explicit_zero":
        native["steps"][0]["metrics"]["completion_tokens"] = 0
    else:
        native["steps"].append({"source": "agent", "metrics": {"completion_tokens": 5}})
    assert not reconcile(doc, native).ok
