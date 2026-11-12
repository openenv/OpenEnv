"""Evidence remains detached, bounded, and fresh across rubric evaluation paths."""

import asyncio

import pytest
from openenv.core.env_server import session_telemetry
from openenv.core.env_server.session_telemetry import (
    rubric_counts,
    rubric_snapshot,
    SessionTelemetry,
)
from openenv.core.rubrics import Gate, Rubric, Sequential, WeightedSum


class PublicScore(Rubric):
    def forward(self, action, observation):
        return action

    def validation_config(self):
        return {}


class AsyncScore(PublicScore):
    async def forward(self, action, observation):
        return action


@pytest.mark.parametrize("score_cls", [PublicScore, AsyncScore])
def test_gating_excludes_stale_scores_in_sync_and_async_paths(score_cls):
    rubric = Sequential(Gate(score_cls(), threshold=0.5), score_cls())

    def score(value):
        result = rubric(value, None)
        return asyncio.run(result) if asyncio.iscoroutine(result) else result

    score(1.0)
    before = rubric_counts(rubric)
    assert score(0.2) == 0.0
    nodes = {node.name: node for node in rubric_snapshot(rubric, before)}
    assert nodes["root"].evaluated and nodes["root"].score == 0.0
    assert nodes["root.rubric_0"].aggregation == "gate"
    assert nodes["root.rubric_0"].config == {"threshold": 0.5}
    assert nodes["root.rubric_0.rubric"].score == 0.2
    assert nodes["root.rubric_1"].evaluated is False
    assert nodes["root.rubric_1"].score is None
    assert rubric.rubric_1.last_score == 1.0


def test_weighted_semantics_and_private_configuration_are_explicit():
    class PrivateScore(Rubric):
        def __init__(self):
            super().__init__()
            self.api_key = "must-not-appear"

        def forward(self, action, observation):
            return 0.5

        def state_dict(self):
            return {"secret": self.api_key}

    rubric = WeightedSum([PublicScore(), PrivateScore()], [0.2, 0.8])
    before = rubric_counts(rubric)
    assert rubric(1.0, None) == pytest.approx(0.6)
    nodes = rubric_snapshot(rubric, before)
    assert nodes[0].aggregation == "weighted_sum"
    assert nodes[0].config == {"weights": [0.2, 0.8]}
    assert nodes[0].children == ["root.rubric_0", "root.rubric_1"]
    assert nodes[2].config_available is False
    assert "must-not-appear" not in str([node.model_dump() for node in nodes])


def test_records_are_detached_and_action_and_byte_limits_are_explicit(monkeypatch):
    subject = SessionTelemetry()
    response = {"type": "observation", "data": {"observation": {"counter": 1}}}
    subject.append("step", {"type": "step", "data": {}}, response)
    response["data"]["observation"]["counter"] = 999
    assert subject.snapshot.trajectory.records[0].response["data"]["observation"] == {
        "counter": 1
    }
    for _ in range(100):
        subject.append("step", {"type": "step", "data": {}}, response)
    assert len(subject.snapshot.trajectory.records) == 100
    assert subject.snapshot.trajectory.complete is False
    assert "budget" in subject.snapshot.trajectory.reason

    monkeypatch.setattr(session_telemetry, "MAX_TELEMETRY_BYTES", 4200)
    subject = SessionTelemetry()
    subject.append(
        "state", {"type": "state"}, {"type": "state", "data": {"x": "x" * 200}}
    )
    assert subject.snapshot.trajectory.complete is False
    assert subject.snapshot.trajectory.records == []


def test_non_json_or_cyclic_rubrics_cannot_claim_complete_evidence():
    subject = SessionTelemetry()
    subject.append("step", {"type": "step", "data": {}}, {"reward": float("nan")})
    assert subject.snapshot.trajectory.complete is False
    rubric = PublicScore()
    rubric.child = rubric
    with pytest.raises(ValueError, match="cyclic"):
        rubric_snapshot(rubric)
