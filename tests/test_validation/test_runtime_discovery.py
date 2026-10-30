"""Pure declaration/attribution checks; process tests cover their wire inputs."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from conftest import load_fixture_manifest
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.discovery import (
    RewardAttributionGrader,
    RubricIntrospectableGrader,
    TaskDeclarationAccuracyGrader,
    ToolDeclarationAccuracyGrader,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.types import CheckStatus
from support.runtime import exchange


def node(
    name="root",
    score=0.75,
    *,
    children=None,
    aggregation="leaf",
    config=None,
    evaluated=True,
):
    return {
        "name": name,
        "class_name": "fixture.Rubric",
        "children": children or [],
        "aggregation": aggregation,
        "config": config or {},
        "config_available": True,
        "score": score if evaluated else None,
        "evaluated": evaluated,
    }


def telemetry(nodes=None):
    nodes = nodes or [
        node(
            children=["root.a", "root.b"],
            aggregation="weighted_sum",
            config={"weights": [0.25, 0.75]},
        ),
        node("root.a", 0),
        node("root.b", 1),
    ]
    return {
        "schema_version": 1,
        "rubric": copy.deepcopy(nodes),
        "attribution": [{"step_index": 0, "rubric": copy.deepcopy(nodes)}],
    }


def subject(tmp_path, *, snapshot=None, reward=0.75, **fields):
    manifest = load_fixture_manifest("served_min_pass")
    manifest["capabilities"].update(
        rubric_tree=True,
        task_api=True,
        declared_tools=["echo"],
        declared_task_count={"train": 100},
    )
    evidence = SimpleNamespace(
        tools_json=json.dumps({"tools": [{"name": "echo"}]}),
        tools_error=None,
        tools_unavailable=None,
        tasks_json=json.dumps(
            {
                "splits": [{"name": "train"}],
                "counts": {"train": 100},
                "previews": {"train": [{"id": 0}, {"id": 1}]},
            }
        ),
        tasks_error=None,
        telemetry_json=json.dumps(telemetry() if snapshot is None else snapshot),
        telemetry_error=None,
        telemetry_unavailable=None,
        failure_reason=None,
        exchanges=(
            exchange(
                "step",
                {"type": "step", "data": {}},
                {
                    "type": "observation",
                    "data": {"reward": reward, "done": False, "observation": {}},
                },
            ),
        ),
    )
    for name, value in fields.items():
        setattr(evidence, name, value)
    return Subject(
        tmp_path,
        NormalizedManifest.model_validate(manifest),
        None,
        None,
        tmp_path,
        evidence,
    )


@pytest.mark.parametrize(
    "grader",
    [
        ToolDeclarationAccuracyGrader,
        TaskDeclarationAccuracyGrader,
        RubricIntrospectableGrader,
        RewardAttributionGrader,
    ],
)
def test_matching_discovery_and_fresh_attribution_pass(tmp_path, grader):
    assert grader().run(subject(tmp_path)).status is CheckStatus.PASS


@pytest.mark.parametrize("scored", [True, False])
@pytest.mark.parametrize("evaluated", [True, False])
def test_unscored_steps_do_not_require_an_emitted_root_score(
    tmp_path, scored, evaluated
):
    payload = telemetry()
    if not evaluated:
        for item in payload["attribution"][0]["rubric"]:
            item.update(evaluated=False, score=None)
    measured = subject(tmp_path, snapshot=payload, reward=None)
    if scored:
        payload["attribution"].append(
            {"step_index": 1, "rubric": copy.deepcopy(telemetry()["rubric"])}
        )
        measured.runtime_evidence.telemetry_json = json.dumps(payload)
        measured.runtime_evidence.exchanges += subject(
            tmp_path
        ).runtime_evidence.exchanges
    result = RewardAttributionGrader().run(measured)
    assert result.status is (CheckStatus.PASS if scored else CheckStatus.SKIP)


@pytest.mark.parametrize("mutation", ["identity", "configuration", "aggregation"])
def test_unscored_steps_still_validate_supplied_attribution(tmp_path, mutation):
    payload = telemetry()
    record = payload["attribution"][0]
    if mutation == "identity":
        record["step_index"] = 1
    elif mutation == "configuration":
        record["rubric"][0]["config"]["weights"] = [0.5, 0.5]
    else:
        record["rubric"][0]["score"] = 0.2
    result = RewardAttributionGrader().run(
        subject(tmp_path, snapshot=payload, reward=None)
    )
    assert result.status is CheckStatus.FAIL


def test_null_terminal_reward_is_not_an_unscored_step(tmp_path):
    measured = subject(tmp_path, reward=None)
    step = measured.runtime_evidence.exchanges[0]
    response = json.loads(step.response_json)
    response["data"]["done"] = True
    measured.runtime_evidence.exchanges = (
        replace(step, response_json=json.dumps(response)),
    )
    assert RewardAttributionGrader().run(measured).status is CheckStatus.FAIL


def test_genuinely_empty_tools_pass_but_unsupported_discovery_does_not(tmp_path):
    measured = subject(tmp_path, tools_json='{"tools": []}')
    measured.manifest.capabilities.declared_tools = []
    assert ToolDeclarationAccuracyGrader().run(measured).status is CheckStatus.PASS
    measured.runtime_evidence.tools_error = "unsupported"
    assert ToolDeclarationAccuracyGrader().run(measured).status is CheckStatus.FAIL


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("tools", [[], [{"name": "echo"}]])
def test_omitted_tools_are_unknown_but_explicit_empty_tools_are_checked(
    tmp_path, declared, tools
):
    measured = subject(tmp_path, tools_json=json.dumps({"tools": tools}))
    capabilities = measured.manifest.capabilities
    data = capabilities.model_dump(exclude={"declared_tools"})
    if declared:
        data["declared_tools"] = []
    measured.manifest.capabilities = type(capabilities).model_validate(data)
    grader = ToolDeclarationAccuracyGrader()
    assert grader.applies_to(measured.manifest) is declared
    assert grader.run(measured).status is (
        CheckStatus.SKIP
        if not declared
        else CheckStatus.FAIL
        if tools
        else CheckStatus.PASS
    )


@pytest.mark.parametrize("task_api", [False, True])
@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_omitted_task_counts_are_unknown_but_explicit_empty_counts_are_checked(
    tmp_path, task_api, declared, empty
):
    measured = subject(tmp_path)
    capabilities = measured.manifest.capabilities
    data = capabilities.model_dump(exclude={"declared_task_count"})
    data["task_api"] = task_api
    if declared:
        data["declared_task_count"] = {}
    measured.manifest.capabilities = type(capabilities).model_validate(data)
    if empty:
        measured.runtime_evidence.tasks_json = json.dumps(
            {"splits": [], "counts": {}, "previews": {}}
        )
    grader = TaskDeclarationAccuracyGrader()
    assert grader.applies_to(measured.manifest) is (task_api or declared)
    result = grader.run(measured)
    assert result.status is (
        CheckStatus.SKIP
        if not declared
        else CheckStatus.PASS
        if empty
        else CheckStatus.FAIL
    )
    if task_api and not declared:
        assert result.evidence == ["task counts were not declared"]


def test_omitted_counts_do_not_hide_an_invalid_declared_task_api(tmp_path):
    measured = subject(tmp_path)
    measured.manifest.capabilities.model_fields_set.discard("declared_task_count")
    measured.manifest.capabilities.declared_task_count.clear()
    payload = json.loads(measured.runtime_evidence.tasks_json)
    payload["counts"]["train"] = True
    measured.runtime_evidence.tasks_json = json.dumps(payload)
    assert TaskDeclarationAccuracyGrader().run(measured).status is CheckStatus.FAIL


@pytest.mark.parametrize("tools", [[], [{"name": "echo"}]])
def test_matching_or_partial_first_tool_page_is_explicitly_incomplete(tmp_path, tools):
    result = ToolDeclarationAccuracyGrader().run(
        subject(tmp_path, tools_json=json.dumps({"tools": tools, "nextCursor": "next"}))
    )
    assert result.status is CheckStatus.SKIP
    assert "pagination" in " ".join(result.evidence)


@pytest.mark.parametrize("cursor", [False, 1, [], {}])
def test_malformed_tool_cursor_fails(tmp_path, cursor):
    result = ToolDeclarationAccuracyGrader().run(
        subject(tmp_path, tools_json=json.dumps({"tools": [], "nextCursor": cursor}))
    )
    assert result.status is CheckStatus.FAIL


def test_undeclared_tool_on_first_page_still_fails(tmp_path):
    result = ToolDeclarationAccuracyGrader().run(
        subject(
            tmp_path,
            tools_json=json.dumps({"tools": [{"name": "extra"}], "nextCursor": "next"}),
        )
    )
    assert result.status is CheckStatus.FAIL
    assert "undeclared" in " ".join(result.evidence)


def test_task_preview_preserves_any_json_task_spec(tmp_path):
    measured = subject(tmp_path)
    tasks = json.loads(measured.runtime_evidence.tasks_json)
    tasks["previews"]["train"] = ["task-0", ["task-1", None]]
    measured.runtime_evidence.tasks_json = json.dumps(tasks)
    assert TaskDeclarationAccuracyGrader().run(measured).status is CheckStatus.PASS


def test_rubric_checks_do_not_apply_plan_size_limits_to_trajectory(tmp_path):
    payload = telemetry()
    payload["trajectory"] = {
        "records": [{"observation": [0] * 200} for _ in range(100)]
    }
    measured = subject(tmp_path, snapshot=payload)
    for grader in (RubricIntrospectableGrader, RewardAttributionGrader):
        assert grader().run(measured).status is CheckStatus.PASS


@pytest.mark.parametrize(
    "tools,reason",
    [
        ([], "missing"),
        ([{"name": "echo"}, {"name": "extra"}], "undeclared"),
        ([{"name": "echo"}, {"name": "echo"}], "duplicate"),
        ([{"name": True}], "malformed"),
    ],
)
def test_tool_mismatches_have_distinct_findings(tmp_path, tools, reason):
    result = ToolDeclarationAccuracyGrader().run(
        subject(tmp_path, tools_json=json.dumps({"tools": tools}))
    )
    assert result.status is CheckStatus.FAIL
    assert reason in " ".join(result.evidence)


@pytest.mark.parametrize("count", [True, "100", 100.0, -1, 99])
def test_invalid_or_wrong_task_count_cannot_be_coerced(tmp_path, count):
    measured = subject(tmp_path)
    tasks = json.loads(measured.runtime_evidence.tasks_json)
    tasks["counts"]["train"] = count
    measured.runtime_evidence.tasks_json = json.dumps(tasks)
    assert TaskDeclarationAccuracyGrader().run(measured).status is CheckStatus.FAIL


@pytest.mark.parametrize(
    "change", ["extra_split", "missing_count", "oversized_preview", "empty_preview"]
)
def test_inconsistent_task_discovery_fails(tmp_path, change):
    measured = subject(tmp_path)
    tasks = json.loads(measured.runtime_evidence.tasks_json)
    if change == "extra_split":
        tasks["splits"].append({"name": "test"})
        tasks["counts"]["test"] = 0
        tasks["previews"]["test"] = []
    elif change == "missing_count":
        tasks["counts"] = {}
    elif change == "oversized_preview":
        tasks["counts"]["train"] = 1
        measured.manifest.capabilities.declared_task_count["train"] = 1
    else:
        tasks["previews"]["train"] = []
    measured.runtime_evidence.tasks_json = json.dumps(tasks)
    assert TaskDeclarationAccuracyGrader().run(measured).status is CheckStatus.FAIL


@pytest.mark.parametrize(
    "grader,field,error",
    [
        (ToolDeclarationAccuracyGrader, "tools_json", "tools_error"),
        (TaskDeclarationAccuracyGrader, "tasks_json", "tasks_error"),
        (RubricIntrospectableGrader, "telemetry_json", "telemetry_error"),
        (RewardAttributionGrader, "telemetry_json", "telemetry_error"),
    ],
)
def test_missing_prerequisite_and_collection_failure_are_distinct(
    tmp_path, grader, field, error
):
    assert grader().run(subject(tmp_path, **{field: None})).status is CheckStatus.SKIP
    result = grader().run(
        subject(tmp_path, **{field: None, error: "private-exception-text"})
    )
    assert result.status is CheckStatus.FAIL
    assert "private-exception-text" not in result.model_dump_json()
    assert (
        grader().run(replace(subject(tmp_path), runtime_evidence=None)).status
        is CheckStatus.SKIP
    )


@pytest.mark.parametrize(
    "change",
    [
        "no_config",
        "no_root",
        "broken_edge",
        "duplicate_node",
        "nonfinite_score",
        "stale_score",
        "bad_version",
        "rubric_error",
        "empty_child_segment",
    ],
)
def test_missing_or_malformed_rubric_introspection_fails(tmp_path, change):
    payload = telemetry()
    if change == "no_config":
        payload["rubric"][0]["config_available"] = False
    elif change == "no_root":
        payload["rubric"] = []
    elif change == "broken_edge":
        payload["rubric"][0]["children"] = ["root.missing", "root.b"]
    elif change == "duplicate_node":
        payload["rubric"].append(payload["rubric"][0])
    elif change == "nonfinite_score":
        payload["rubric"][0]["score"] = float("nan")
    elif change == "stale_score":
        payload["rubric"][0]["evaluated"] = False
    elif change == "bad_version":
        payload["schema_version"] = True
    elif change == "empty_child_segment":
        payload["rubric"][0]["children"][0] = "root."
        payload["rubric"][1]["name"] = "root."
    else:
        payload["rubric_error"] = "private-error"
    result = RubricIntrospectableGrader().run(subject(tmp_path, snapshot=payload))
    assert result.status is CheckStatus.FAIL
    assert "private-error" not in result.model_dump_json()


@pytest.mark.parametrize(
    "change",
    [
        "root_reward",
        "child_total",
        "missing_child",
        "stale_root",
        "missing_step",
        "wrong_step",
        "config_changed",
    ],
)
def test_incorrect_or_stale_attribution_fails(tmp_path, change):
    payload = telemetry()
    record = payload["attribution"][0]
    if change == "root_reward":
        record["rubric"][0]["score"] = 0.5
    elif change == "child_total":
        record["rubric"][1]["score"] = 1
    elif change == "missing_child":
        record["rubric"][1].update(evaluated=False, score=None)
    elif change == "stale_root":
        record["rubric"][0].update(evaluated=False, score=None)
    elif change == "missing_step":
        payload["attribution"] = []
    elif change == "wrong_step":
        record["step_index"] = True
    else:
        record["rubric"][0]["config"]["weights"] = [0.5, 0.5]
    assert (
        RewardAttributionGrader().run(subject(tmp_path, snapshot=payload)).status
        is CheckStatus.FAIL
    )


@pytest.mark.parametrize("before,after", [(True, 1), (False, 0), (1, 1.0)])
def test_attribution_detects_rubric_configuration_type_changes(tmp_path, before, after):
    payload = telemetry()
    payload["rubric"][1]["config"] = {"nested": [{"value": after}]}
    payload["attribution"][0]["rubric"][1]["config"] = {"nested": [{"value": before}]}
    result = RewardAttributionGrader().run(subject(tmp_path, snapshot=payload))
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["step 0: rubric configuration changed"]


def test_attribution_ignores_rubric_configuration_key_order(tmp_path):
    payload = telemetry()
    payload["rubric"][1]["config"] = {"enabled": True, "threshold": 1}
    payload["attribution"][0]["rubric"][1]["config"] = {
        "threshold": 1,
        "enabled": True,
    }
    assert (
        RewardAttributionGrader().run(subject(tmp_path, snapshot=payload)).status
        is CheckStatus.PASS
    )


@pytest.mark.parametrize("aggregation", ["leaf", "gate", "sequential", "weighted_sum"])
def test_stock_aggregation_rules_and_sequential_short_circuit(tmp_path, aggregation):
    if aggregation == "leaf":
        nodes, reward = [node()], 0.75
    elif aggregation == "gate":
        nodes, reward = (
            [
                node(
                    score=0,
                    children=["root.a"],
                    aggregation="gate",
                    config={"threshold": 1},
                ),
                node("root.a", 0.75),
            ],
            0,
        )
    elif aggregation == "sequential":
        nodes, reward = (
            [
                node(score=0, children=["root.a", "root.b"], aggregation="sequential"),
                node("root.a", 0),
                node("root.b", evaluated=False),
            ],
            0,
        )
    else:
        nodes, reward = telemetry()["rubric"], 0.75
    assert (
        RewardAttributionGrader()
        .run(subject(tmp_path, snapshot=telemetry(nodes), reward=reward))
        .status
        is CheckStatus.PASS
    )


def test_unknown_aggregation_is_explicitly_incomplete_but_wrong_root_still_fails(
    tmp_path,
):
    nodes = [node(children=["root.a"], aggregation="unknown"), node("root.a")]
    result = RewardAttributionGrader().run(subject(tmp_path, snapshot=telemetry(nodes)))
    assert result.status is CheckStatus.SKIP
    assert "unsupported custom" in " ".join(result.evidence)
    assert (
        RewardAttributionGrader()
        .run(subject(tmp_path, snapshot=telemetry(nodes), reward=0))
        .status
        is CheckStatus.FAIL
    )


def test_attribution_cannot_pass_from_a_truncated_wire_prefix(tmp_path):
    measured = subject(tmp_path, failure_reason="step failed (TimeoutError)")
    assert RewardAttributionGrader().run(measured).status is CheckStatus.FAIL


def test_undeclared_task_and_rubric_capabilities_do_not_run(tmp_path):
    measured = subject(tmp_path)
    measured.manifest.capabilities.task_api = False
    measured.manifest.capabilities.declared_task_count = {}
    measured.manifest.capabilities.model_fields_set.discard("declared_task_count")
    measured.manifest.capabilities.rubric_tree = False
    for grader in (
        TaskDeclarationAccuracyGrader,
        RubricIntrospectableGrader,
        RewardAttributionGrader,
    ):
        assert not grader().applies_to(measured.manifest)
        assert grader().run(measured).status is CheckStatus.SKIP


@pytest.mark.parametrize(
    "grader", [RubricIntrospectableGrader, RewardAttributionGrader]
)
def test_unsupported_session_telemetry_is_a_named_skip(tmp_path, grader):
    result = grader().run(
        subject(
            tmp_path,
            telemetry_json=None,
            telemetry_unavailable="unsupported capability: session_telemetry",
        )
    )
    assert result.status is CheckStatus.SKIP
    assert result.evidence == ["unsupported capability: session_telemetry"]
