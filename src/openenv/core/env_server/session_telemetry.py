# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, opt-in evidence emitted by the subject's replay session."""

import json
import secrets
from typing import Any, Literal

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import Gate, Sequential, WeightedSum
from pydantic import BaseModel, ConfigDict, Field, SecretStr

MAX_TELEMETRY_BYTES = 8 * 1024 * 1024
MAX_TRAJECTORY_ACTIONS = 100
MAX_TRAJECTORY_RECORDS = 202
MAX_RUBRIC_NODES = 128


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ValidationOpenData(_WireModel):
    schema_version: Literal[1]
    token: SecretStr = Field(min_length=32, max_length=256)


class ValidationOpenMessage(_WireModel):
    type: Literal["validation_open"]
    data: ValidationOpenData


class ValidationReadData(_WireModel):
    schema_version: Literal[1]
    capability: SecretStr = Field(min_length=32, max_length=256)


class ValidationReadMessage(_WireModel):
    type: Literal["validation_read"]
    data: ValidationReadData


class ValidationOpenedData(_WireModel):
    schema_version: Literal[1] = 1
    capability: str


class ValidationOpenedResponse(_WireModel):
    type: Literal["validation_open"] = "validation_open"
    data: ValidationOpenedData


class SeedAcceptance(_WireModel):
    requested: bool
    value: Any = None
    accepted: bool


class RubricNode(_WireModel):
    name: str
    class_name: str
    children: list[str]
    aggregation: Literal["weighted_sum", "sequential", "gate", "leaf", "unknown"]
    config: dict[str, Any]
    config_available: bool
    score: Any = None
    evaluated: bool = False


class StepAttribution(_WireModel):
    step_index: int
    rubric: list[RubricNode]


class TrajectoryRecord(_WireModel):
    operation: Literal["reset", "step", "state"]
    request: dict[str, Any]
    response: dict[str, Any]


class SubjectTrajectory(_WireModel):
    schema_version: Literal[1] = 1
    source: Literal["openenv-server"] = "openenv-server"
    records: list[TrajectoryRecord] = Field(default_factory=list)
    complete: bool = True
    reason: str | None = None


class ValidationSnapshot(_WireModel):
    schema_version: Literal[1] = 1
    seed: SeedAcceptance | None = None
    rubric: list[RubricNode] = Field(default_factory=list)
    rubric_error: str | None = None
    attribution: list[StepAttribution] = Field(default_factory=list)
    trajectory: SubjectTrajectory = Field(default_factory=SubjectTrajectory)


class ValidationResponse(_WireModel):
    type: Literal["validation"] = "validation"
    data: ValidationSnapshot


def _rubrics(root: Rubric | None) -> list[tuple[str, Rubric]]:
    """Visit the named tree with finite size, rejecting cycles and aliases."""
    if root is None:
        return []
    result, pending, seen = [], [("root", root)], set()
    while pending:
        name, rubric = pending.pop()
        if id(rubric) in seen or len(result) >= MAX_RUBRIC_NODES:
            raise ValueError("Rubric tree is cyclic, shared, or exceeds 128 nodes")
        seen.add(id(rubric))
        result.append((name, rubric))
        children = list(rubric.named_children())
        if any(not key or "." in key for key, _ in children):
            raise ValueError("Rubric child names must be nonempty path segments")
        pending.extend((f"{name}.{key}", child) for key, child in reversed(children))
    return result


def rubric_counts(root: Rubric | None) -> dict[int, int]:
    return {id(r): r._evaluation_count for _, r in _rubrics(root)}


def rubric_snapshot(
    root: Rubric | None, before: dict[int, int] | None = None
) -> list[RubricNode]:
    nodes = []
    for name, rubric in _rubrics(root):
        children = [f"{name}.{key}" for key, _ in rubric.named_children()]
        aggregation = "unknown" if children else "leaf"
        config = None
        # Exact types: a subclass can override scoring, so its semantics are unknown.
        if type(rubric) is WeightedSum:
            aggregation, config = "weighted_sum", {"weights": list(rubric._weights)}
        elif type(rubric) is Sequential:
            aggregation, config = "sequential", {}
        elif type(rubric) is Gate:
            aggregation, config = "gate", {"threshold": rubric.threshold}
        else:
            config = rubric.validation_config()
        evaluated = before is not None and rubric._evaluation_count > before.get(
            id(rubric), 0
        )
        nodes.append(
            RubricNode(
                name=name,
                class_name=f"{type(rubric).__module__}.{type(rubric).__qualname__}",
                children=children,
                aggregation=aggregation,
                config={} if config is None else config,
                config_available=config is not None,
                score=rubric.last_score if evaluated else None,
                evaluated=evaluated,
            )
        )
    json.dumps([node.model_dump() for node in nodes], allow_nan=False)
    return nodes


class SessionTelemetry:
    """Own one socket's capability and a detached record of completed operations."""

    def __init__(self):
        self.capability = secrets.token_urlsafe(32)
        self.snapshot = ValidationSnapshot()
        self._bytes = 0
        self._steps = 0

    def authorized(self, capability: str) -> bool:
        return secrets.compare_digest(self.capability.encode(), capability.encode())

    def append(
        self,
        operation: Literal["reset", "step", "state"],
        request: dict[str, Any],
        response: dict[str, Any],
        *,
        seed: SeedAcceptance | None = None,
        rubric: list[RubricNode] | None = None,
    ) -> None:
        if not self.snapshot.trajectory.complete:
            return
        try:
            # JSON round-trip detaches mutable observations/actions from the record.
            record = TrajectoryRecord.model_validate(
                json.loads(
                    json.dumps(
                        {
                            "operation": operation,
                            "request": request,
                            "response": response,
                        },
                        allow_nan=False,
                    )
                )
            )
            attribution = (
                StepAttribution(step_index=self._steps, rubric=rubric or [])
                if operation == "step"
                else None
            )
            byte_count = len(record.model_dump_json().encode())
            if attribution is not None:
                byte_count += len(attribution.model_dump_json().encode())
            byte_count += (
                len(seed.model_dump_json().encode()) if seed is not None else 0
            )
            # Reserve room for the current rubric, envelope, and seed metadata.
            current_bytes = len(
                json.dumps(
                    [
                        node.model_dump()
                        for node in (self.snapshot.rubric if rubric is None else rubric)
                    ],
                    allow_nan=False,
                ).encode()
            )
            if (
                len(self.snapshot.trajectory.records) >= MAX_TRAJECTORY_RECORDS
                or operation == "step"
                and self._steps >= MAX_TRAJECTORY_ACTIONS
                or self._bytes + byte_count + current_bytes + 4096 > MAX_TELEMETRY_BYTES
            ):
                self.incomplete("Session telemetry budget exceeded")
                return
            self._bytes += byte_count
            self.snapshot.trajectory.records.append(record)
            if seed is not None:
                self.snapshot.seed = seed.model_copy(deep=True)
            if rubric is not None:
                self.snapshot.rubric = [node.model_copy(deep=True) for node in rubric]
            if attribution is not None:
                self.snapshot.attribution.append(attribution.model_copy(deep=True))
                self._steps += 1
        except (TypeError, ValueError):
            self.incomplete("Session telemetry contains non-JSON data")

    def incomplete(self, reason: str) -> None:
        self.snapshot.trajectory.complete = False
        self.snapshot.trajectory.reason = reason
