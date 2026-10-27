"""Declaration and rubric checks over bounded, independently collected evidence."""

import json
import math
import time

from ...report import CheckResult
from ...types import CheckStatus
from .basic import _RuntimeGrader
from .repeatability import _difference


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("runtime JSON numbers must be finite")
    return number


class _EvidenceGrader(_RuntimeGrader):
    """Discovery can be measured without an episode step; missing evidence cannot pass."""

    unavailable_field = None

    def run(self, subject):
        started = time.monotonic()
        problems, incomplete = [], None
        evidence = subject.runtime_evidence
        if not self.applies_to(subject.manifest):
            incomplete = "declared capability does not apply"
        elif evidence is None:
            incomplete = "runtime evidence is unavailable"
        elif getattr(evidence, self.error_field):
            problems = [f"{self.label} collection failed"]
        elif self.unavailable_field and getattr(evidence, self.unavailable_field):
            incomplete = getattr(evidence, self.unavailable_field)
        elif getattr(evidence, self.field) is None:
            incomplete = f"missing prerequisite: {self.label} evidence"
        else:
            try:
                payload = json.loads(
                    getattr(evidence, self.field),
                    parse_float=_json_float,
                    parse_constant=_json_float,
                )
                if not isinstance(payload, dict):
                    raise ValueError("evidence must be an object")
                problems, incomplete = self.check(subject, payload)
            except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
                problems = [f"malformed {self.label} evidence"]
        status = (
            CheckStatus.FAIL
            if problems
            else CheckStatus.SKIP
            if incomplete
            else CheckStatus.PASS
        )
        return CheckResult(
            check_id=self.check_id,
            status=status,
            evidence=problems[:20]
            or [incomplete or f"observed {self.label} contract holds"],
            duration_s=time.monotonic() - started,
        )


class ToolDeclarationAccuracyGrader(_EvidenceGrader):
    check_id = "runtime.tool_declaration_accuracy"
    field, error_field, label = "tools_json", "tools_error", "tool discovery"
    unavailable_field = "tools_unavailable"

    def applies_to(self, manifest):
        return "declared_tools" in manifest.capabilities.model_fields_set

    def check(self, subject, payload):
        tools = payload["tools"]
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")
        names = [tool["name"] for tool in tools]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("invalid tool name")
        cursor = payload.get("nextCursor")
        if cursor is not None and not isinstance(cursor, str):
            raise ValueError("invalid tool pagination cursor")
        declared = subject.manifest.capabilities.declared_tools
        problems = []
        if len(set(names)) != len(names):
            problems.append("discovery contains duplicate tool names")
        if len(set(declared)) != len(declared):
            problems.append("declaration contains duplicate tool names")
        missing = set(declared) - set(names)
        extra = set(names) - set(declared)
        if missing and cursor is None:
            problems.append(f"{len(missing)} declared tools are missing from discovery")
        if extra:
            problems.append(f"{len(extra)} discovered tools are undeclared")
        return problems, (
            "tool inventory is incomplete: pagination was not collected"
            if cursor is not None
            else None
        )


class TaskDeclarationAccuracyGrader(_EvidenceGrader):
    check_id = "runtime.task_declaration_accuracy"
    field, error_field, label = "tasks_json", "tasks_error", "task discovery"

    def applies_to(self, manifest):
        return manifest.capabilities.task_api or (
            "declared_task_count" in manifest.capabilities.model_fields_set
        )

    def check(self, subject, payload):
        splits, counts, previews = (
            payload["splits"],
            payload["counts"],
            payload["previews"],
        )
        if (
            not isinstance(splits, list)
            or not isinstance(counts, dict)
            or not isinstance(previews, dict)
        ):
            raise ValueError("invalid task discovery")
        names = [split["name"] for split in splits]
        if any(not isinstance(name, str) or not name for name in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("invalid split names")
        if set(names) != counts.keys() or set(names) != previews.keys():
            raise ValueError("incomplete split discovery")
        problems = []
        declared = subject.manifest.capabilities.declared_task_count
        has_counts = (
            "declared_task_count" in subject.manifest.capabilities.model_fields_set
        )
        if has_counts and set(names) != declared.keys():
            problems.append("discovered splits differ from declared task counts")
        for name in names:
            count, preview = counts[name], previews[name]
            if type(count) is not int or count < 0:
                raise ValueError("invalid task count")
            if not isinstance(preview, list):
                raise ValueError("invalid task preview")
            if len(preview) > count or (count > 0 and not preview):
                problems.append(
                    "task preview is inconsistent with the advertised count"
                )
            if name in declared and count != declared[name]:
                problems.append("advertised task count differs from the declaration")
        return problems, None if has_counts else "task counts were not declared"


def _rubric_nodes(payload):
    if not isinstance(payload, list) or not payload:
        raise ValueError("missing rubric tree")
    nodes = {}
    for node in payload:
        name = node["name"]
        if not isinstance(name, str) or not all(name.split(".")) or name in nodes:
            raise ValueError("invalid rubric name")
        if not isinstance(node["class_name"], str) or not node["class_name"]:
            raise ValueError("missing rubric class")
        children = node["children"]
        if not isinstance(children, list) or any(
            not isinstance(child, str) for child in children
        ):
            raise ValueError("invalid rubric children")
        if len(set(children)) != len(children):
            raise ValueError("duplicate rubric children")
        if type(node["config_available"]) is not bool or not isinstance(
            node["config"], dict
        ):
            raise ValueError("invalid rubric configuration")
        if type(node["evaluated"]) is not bool:
            raise ValueError("invalid evaluation marker")
        if node["evaluated"] and not _finite(node["score"]):
            raise ValueError("evaluated score is not finite")
        if not node["evaluated"] and node["score"] is not None:
            raise ValueError("unevaluated score must be absent")
        aggregation = node["aggregation"]
        if aggregation not in {"weighted_sum", "sequential", "gate", "leaf", "unknown"}:
            raise ValueError("invalid aggregation")
        if aggregation == "leaf" and children:
            raise ValueError("leaf has children")
        if aggregation == "weighted_sum":
            weights = node["config"]["weights"]
            if (
                not isinstance(weights, list)
                or len(weights) != len(children)
                or not all(_finite(weight) for weight in weights)
            ):
                raise ValueError("invalid rubric weights")
            if not math.isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-6):
                raise ValueError("rubric weights do not sum to one")
        if aggregation == "gate" and (
            len(children) != 1 or not _finite(node["config"]["threshold"])
        ):
            raise ValueError("invalid rubric gate")
        nodes[name] = node
    if "root" not in nodes:
        raise ValueError("missing rubric root")
    linked = set()
    for name, node in nodes.items():
        for child in node["children"]:
            if (
                child not in nodes
                or child in linked
                or child.rsplit(".", 1)[0] != name
                or child == name
            ):
                raise ValueError("invalid rubric edge")
            linked.add(child)
    if linked != nodes.keys() - {"root"}:
        raise ValueError("disconnected rubric tree")
    return nodes


class RubricIntrospectableGrader(_EvidenceGrader):
    check_id = "runtime.rubric_introspectable"
    requires_capabilities = frozenset({"rubric_tree"})
    field, error_field, label = "telemetry_json", "telemetry_error", "rubric telemetry"
    unavailable_field = "telemetry_unavailable"

    def applies_to(self, manifest):
        return manifest.capabilities.rubric_tree

    def check(self, subject, payload):
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported telemetry version")
        if payload.get("rubric_error") is not None:
            return ["subject reported unavailable rubric introspection"], None
        nodes = _rubric_nodes(payload["rubric"])
        return (
            ["rubric configuration is unavailable"]
            if any(not node["config_available"] for node in nodes.values())
            else []
        ), None


class RewardAttributionGrader(RubricIntrospectableGrader):
    check_id = "runtime.reward_attribution"
    depends_on = ("runtime.startup", "runtime.rubric_introspectable")

    def check(self, subject, payload):
        problems, _ = super().check(subject, payload)
        if subject.runtime_evidence.failure_reason:
            problems.append("wire replay is incomplete")
        if problems:
            return problems, None
        baseline = _rubric_nodes(payload["rubric"])
        steps = [
            row for row in subject.runtime_evidence.exchanges if row.operation == "step"
        ]
        records = payload["attribution"]
        if not isinstance(records, list):
            raise ValueError("invalid attribution")
        if not steps:
            return problems, "missing prerequisite: an observed step reward"
        if len(records) != len(steps):
            return problems + ["attribution does not cover every observed step"], None

        def definition(tree):
            return {
                name: {
                    key: value
                    for key, value in node.items()
                    if key not in {"score", "evaluated"}
                }
                for name, node in tree.items()
            }

        unsupported = False
        scored_steps = 0
        for index, (record, step) in enumerate(zip(records, steps, strict=True)):
            if type(record["step_index"]) is not int or record["step_index"] != index:
                problems.append(
                    f"step {index}: attribution identity differs from the wire"
                )
            nodes = _rubric_nodes(record["rubric"])
            if _difference(definition(nodes), definition(baseline)):
                problems.append(f"step {index}: rubric configuration changed")
            data = json.loads(step.response_json)["data"]
            reward = data["reward"]
            root = nodes["root"]
            unscored = reward is None and data.get("done") is False
            scored_steps += not unscored
            if not unscored and (
                not root["evaluated"]
                or not _finite(reward)
                or not math.isclose(root["score"], reward, rel_tol=1e-9, abs_tol=1e-9)
            ):
                problems.append(
                    f"step {index}: fresh root score differs from emitted reward"
                )
            for node in nodes.values():
                if not node["evaluated"]:
                    continue
                children = [nodes[name] for name in node["children"]]
                aggregation = node["aggregation"]
                if aggregation == "unknown":
                    unsupported = True
                    continue
                if aggregation == "leaf":
                    continue
                expected = 1.0 if aggregation == "sequential" else 0.0
                visited = 0
                for child in children:
                    if not child["evaluated"]:
                        problems.append(
                            f"step {index}: required child score was not evaluated"
                        )
                        break
                    visited += 1
                    score = child["score"]
                    if aggregation == "weighted_sum":
                        expected += score * node["config"]["weights"][visited - 1]
                    elif aggregation == "gate":
                        expected = 0.0 if score < node["config"]["threshold"] else score
                    else:
                        expected = score
                        if score == 0:
                            break
                if aggregation == "sequential" and any(
                    child["evaluated"] for child in children[visited:]
                ):
                    problems.append(
                        f"step {index}: sequential evaluated children after its gate"
                    )
                if not _finite(expected) or not math.isclose(
                    node["score"], expected, rel_tol=1e-9, abs_tol=1e-9
                ):
                    problems.append(
                        f"step {index}: child attribution does not match the parent score"
                    )
        incomplete = (
            "unsupported custom rubric aggregation semantics" if unsupported else None
        )
        if not scored_steps:
            incomplete = "missing prerequisite: an observed numeric step reward"
        return problems, incomplete
