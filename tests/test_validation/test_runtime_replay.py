import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openenv.validation.runtime import replay
from openenv.validation.runtime.collector import RuntimeCollectionInterrupted
from openenv.validation.runtime.contracts import (
    LaunchSpec,
    RuntimeEvidence,
    RuntimePlan,
    WireExchange,
)
from openenv.validation.types import ProviderCapability


def inputs(monkeypatch, *, judged=False, fresh_container=True):
    image = "sha256:" + "a" * 64
    original = SimpleNamespace(
        base_url="http://original",
        inspect=Mock(return_value={"container_id": "original", "image_id": image}),
        stop=Mock(),
    )
    fresh = SimpleNamespace(
        base_url="http://fresh",
        inspect=Mock(return_value={"container_id": "fresh", "image_id": image}),
        stop=Mock(),
    )
    provider = SimpleNamespace(
        capabilities={ProviderCapability.FRESH_CONTAINER} if fresh_container else set(),
        start=Mock(return_value=fresh),
    )
    spec = LaunchSpec(
        image_ref=image,
        resources={"cpu": 1, "memory_mb": 256, "disk_mb": 64, "episode_timeout_s": 30},
        network={"mode": "public"},
        run_id="validation-original",
        env_vars={"OPENENV_VALIDATION_TOKEN": "original-run-authorization"},
    )
    plan = RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {"episode_id": "same-episode", "seed": 42},
            "actions": [{"increment": 1}],
        }
    )
    collect = Mock(return_value=RuntimeEvidence())
    monkeypatch.setattr(replay, "collect_runtime_evidence", collect)
    return (
        provider,
        original,
        fresh,
        spec,
        plan,
        RuntimeEvidence(),
        SimpleNamespace(llm_judged=judged),
        collect,
    )


def run(values, **kwargs):
    provider, original, _, spec, plan, evidence, capabilities, _ = values
    return replay.collect_replays(
        provider, original, spec, plan, evidence, capabilities=capabilities, **kwargs
    )


def test_schedule_preserves_inputs_verifies_identity_and_cleans_fresh_container(
    monkeypatch,
):
    values = inputs(monkeypatch)
    provider, original, fresh, spec, plan, evidence, _, collect = values
    result = run(values)
    assert [row.scope for row in result.replays] == ["session", "seed", "container"]
    assert result.exchanges == evidence.exchanges
    assert result.replay_failure_reason is None
    assert [call.args[0] for call in collect.call_args_list] == [
        "http://original",
        "http://original",
        "http://fresh",
    ]
    assert [call.args[1].reset.seed for call in collect.call_args_list] == [42, 43, 42]
    assert all(
        call.args[1].reset.episode_id == plan.reset.episode_id
        for call in collect.call_args_list
    )
    assert all(call.args[1].actions == plan.actions for call in collect.call_args_list)
    new_spec = provider.start.call_args.args[0]
    assert new_spec.image_ref == spec.image_ref
    assert new_spec.run_id != spec.run_id
    assert (
        new_spec.env_vars["OPENENV_VALIDATION_TOKEN"]
        != spec.env_vars["OPENENV_VALIDATION_TOKEN"]
    )
    assert (
        collect.call_args_list[-1].kwargs["validation_token"]
        == new_spec.env_vars["OPENENV_VALIDATION_TOKEN"]
    )
    assert json.loads(result.replays[-1].provider_json)["container_id"] == "fresh"
    assert result.replays[-1].cleanup_complete is True
    fresh.stop.assert_called_once()
    original.stop.assert_not_called()


def test_judged_schedule_has_exactly_twenty_identical_inputs_plus_changed_seed(
    monkeypatch,
):
    values = inputs(monkeypatch, judged=True)
    result = run(values)
    assert 1 + sum(row.scope != "seed" for row in result.replays) == 20
    assert sum(row.scope == "seed" for row in result.replays) == 1
    assert values[-1].call_count == 20
    assert (
        len({call.args[1].model_dump_json() for call in values[-1].call_args_list}) == 2
    )


def test_provider_without_fresh_containers_collects_only_session_experiments(
    monkeypatch,
):
    values = inputs(monkeypatch, fresh_container=False, judged=True)
    result = run(values)
    assert [row.scope for row in result.replays] == ["session", "seed"]
    assert (
        result.replay_failure_reason == "missing provider capability: fresh_container"
    )
    values[0].start.assert_not_called()


@pytest.mark.parametrize(
    "inspection",
    [
        {"container_id": "original"},
        {"container_id": "fresh", "image_id": "wrong-image"},
        {"image_id": "missing-container-id"},
    ],
)
def test_identity_mismatch_cannot_supply_a_completed_fresh_container_sample(
    monkeypatch, inspection
):
    values = inputs(monkeypatch)
    fresh = values[2]
    fresh.inspect.return_value = {"image_id": values[3].image_ref, **inspection}
    result = run(values)
    assert result.replays[-1].evidence.failure_reason
    assert result.replays[-1].cleanup_complete is True
    assert "inspection failed" in result.replay_failure_reason
    assert values[-1].call_count == 2
    fresh.stop.assert_called_once()


def test_container_cleanup_failure_is_preserved_even_after_good_replay(monkeypatch):
    values = inputs(monkeypatch)
    values[2].stop.side_effect = RuntimeError("private-provider-message")
    result = run(values)
    assert result.replays[-1].cleanup_complete is False
    assert result.replay_failure_reason == "fresh container cleanup failed"
    assert "private-provider-message" not in result.replay_failure_reason


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_start_retains_unconfirmed_cleanup_without_a_returned_handle(
    monkeypatch, failure
):
    values = inputs(monkeypatch)
    values[0].start.side_effect = failure("private-provider-message")
    if failure is KeyboardInterrupt:
        with pytest.raises(RuntimeCollectionInterrupted) as error:
            run(values)
        result = error.value.evidence
    else:
        result = run(values)
    sample = result.replays[-1]
    assert sample.scope == "container"
    assert sample.cleanup_complete is False
    assert sample.provider_json is None
    assert not sample.evidence.exchanges
    assert "teardown could not be confirmed" in sample.evidence.failure_reason
    assert "private-provider-message" not in result.replay_failure_reason
    values[2].stop.assert_not_called()


def test_total_deadline_caps_episode_and_startup_budgets(monkeypatch):
    values = inputs(monkeypatch)
    monkeypatch.setattr(replay.time, "monotonic", lambda: 100.0)
    result = run(values, deadline=102.5)
    assert result.replay_failure_reason is None
    assert all(
        call.kwargs["episode_timeout_s"] == 2.5 for call in values[-1].call_args_list
    )
    assert values[0].start.call_args.args[0].startup_timeout_s == 2.5


def test_expired_total_budget_retains_baseline_without_starting_experiments(
    monkeypatch,
):
    values = inputs(monkeypatch)
    monkeypatch.setattr(replay.time, "monotonic", lambda: 100.0)
    result = run(values, deadline=99.0)
    assert result.exchanges == values[5].exchanges
    assert not result.replays
    assert "TimeoutError" in result.replay_failure_reason
    values[-1].assert_not_called()
    values[0].start.assert_not_called()


def test_collection_failure_retains_prefix_and_stops_replaying(monkeypatch):
    values = inputs(monkeypatch)
    values[-1].side_effect = [
        RuntimeEvidence(),
        RuntimeEvidence(failure_reason="step failed (TimeoutError)"),
    ]
    result = run(values)
    assert len(result.replays) == 2
    assert result.replays[-1].evidence.failure_reason == "step failed (TimeoutError)"
    values[0].start.assert_not_called()


@pytest.mark.parametrize("interrupted_call", [1, 3])
def test_interruption_keeps_partial_replay_and_cleans_fresh_container(
    monkeypatch, interrupted_call
):
    values = inputs(monkeypatch)
    partial = RuntimeEvidence(
        failure_phase="step", failure_reason="step failed (KeyboardInterrupt)"
    )
    values[-1].side_effect = [RuntimeEvidence()] * (interrupted_call - 1) + [
        RuntimeCollectionInterrupted(partial)
    ]
    with pytest.raises(RuntimeCollectionInterrupted) as error:
        run(values)
    result = error.value.evidence
    assert len(result.replays) == interrupted_call
    assert result.replays[-1].evidence is partial
    assert "interrupted" in result.replay_failure_reason
    if interrupted_call == 3:
        values[2].stop.assert_called_once()
        assert result.replays[-1].cleanup_complete is True


def test_failed_baseline_is_returned_without_further_execution(monkeypatch):
    values = list(inputs(monkeypatch))
    values[5] = replace(values[5], failure_reason="reset failed (ValueError)")
    assert run(values) is values[5]
    values[-1].assert_not_called()


@pytest.mark.parametrize("budget,retained_samples", [(14, 1), (26, 3), (38, 3)])
def test_retained_byte_budget_counts_primary_and_all_raw_fields(
    monkeypatch, budget, retained_samples
):
    values = list(inputs(monkeypatch, judged=True))
    values[5] = RuntimeEvidence(observation_schema_json="é")  # Two UTF-8 bytes.
    sample = RuntimeEvidence(
        exchanges=(WireExchange("step", "aa", "bbb"),),
        observation_schema_json="é",
        reset_observation_schema_json="é",
        telemetry_json="xyz",
    )
    assert replay._evidence_bytes(sample) == 12
    values[-1].return_value = sample
    monkeypatch.setattr(replay, "MAX_REPLAY_BYTES", budget)
    result = run(values)
    assert len(result.replays) == retained_samples
    assert "total retained replay evidence exceeds" in result.replay_failure_reason
    retained = replay._evidence_bytes(result) + sum(
        replay._evidence_bytes(row.evidence) for row in result.replays
    )
    assert retained <= budget
    assert result.observation_schema_json == "é"
    if budget >= 26:
        container = next(row for row in result.replays if row.scope == "container")
        assert container.cleanup_complete is True
        assert json.loads(container.provider_json)["container_id"] == "fresh"
        values[2].stop.assert_called_once()
        if budget == 26:
            assert not container.evidence.exchanges


@pytest.mark.parametrize("field", ["tools_json", "tasks_json"])
@pytest.mark.parametrize("budget,collections", [(3, 0), (5, 1)])
def test_discovery_payload_counts_toward_retained_replay_budget(
    monkeypatch, field, budget, collections
):
    values = list(inputs(monkeypatch))
    values[5] = RuntimeEvidence(**{field: '"é"'})  # Four UTF-8 bytes.
    values[-1].return_value = RuntimeEvidence(observation_schema_json="{}")
    monkeypatch.setattr(replay, "MAX_REPLAY_BYTES", budget)
    result = run(values)
    assert "total retained replay evidence exceeds" in result.replay_failure_reason
    assert not result.replays
    assert replay._evidence_bytes(result) <= budget
    assert values[-1].call_count == collections
    values[0].start.assert_not_called()


def test_oversized_primary_evidence_is_explicitly_incomplete_and_not_retained(
    monkeypatch,
):
    values = list(inputs(monkeypatch))
    values[5] = RuntimeEvidence(telemetry_json="x" * 20)
    monkeypatch.setattr(replay, "MAX_REPLAY_BYTES", 10)
    result = run(values)
    assert result.failure_phase == "replay evidence budget"
    assert result.failure_reason == result.replay_failure_reason
    assert replay._evidence_bytes(result) == 0
    values[-1].assert_not_called()


def test_byte_budget_does_not_swallow_interruption_or_discard_completed_prefix(
    monkeypatch,
):
    values = inputs(monkeypatch)
    sample = RuntimeEvidence(telemetry_json="x" * 10)
    values[-1].side_effect = [sample, RuntimeCollectionInterrupted(sample)]
    monkeypatch.setattr(replay, "MAX_REPLAY_BYTES", 15)
    with pytest.raises(RuntimeCollectionInterrupted) as error:
        run(values)
    assert len(error.value.evidence.replays) == 1
    assert error.value.evidence.replays[0].evidence is sample
    assert (
        "total retained replay evidence exceeds"
        in error.value.evidence.replay_failure_reason
    )
