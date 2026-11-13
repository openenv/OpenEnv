"""
Tests for the Cloud SRE Environment.

Validates all three tasks (easy, medium, hard) with both fixed
and seeded states, plus grading correctness.
"""

import pytest
from cloud_sre_env.models import SREAction
from cloud_sre_env.server.cloud_sre_environment import (
    CHAOS_EVENTS,
    CloudSREEnvironment,
    TASK_REGISTRY,
)


class TestEnvironmentBasics:
    """Test reset, step, state lifecycle."""

    def test_reset_returns_observation(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="phantom_volume_cleanup")
        assert len(obs.resources) > 0
        assert obs.step_number == 0
        assert obs.task_description != ""

    def test_step_increments(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        obs = env.step(SREAction(command="wait"))
        assert obs.step_number == 1

    def test_state_returns_sre_state(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        state = env.state
        assert state.task_id == "phantom_volume_cleanup"
        assert state.episode_id != ""

    def test_max_steps_ends_episode(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        for _ in range(15):
            env.step(SREAction(command="wait"))
        assert env.state.done is True

    def test_invalid_task_raises(self):
        env = CloudSREEnvironment()
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset(task_id="nonexistent_task")


class TestTask1PhantomVolumeCleanup:
    """Test the easy task — orphan volume termination."""

    def test_fixed_state_has_orphans(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="phantom_volume_cleanup")
        orphans = [r for r in obs.resources if r.get("status") == "available"]
        assert len(orphans) == 3

    def test_perfect_score(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        for oid in ["ebs-orphan-001", "ebs-orphan-002", "ebs-orphan-003"]:
            env.step(SREAction(command="terminate", resource_id=oid))
        score, breakdown = env.grade()
        assert score == 1.0
        assert len(breakdown["orphans_terminated"]) == 3

    def test_penalty_for_wrong_termination(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        env.step(SREAction(command="terminate", resource_id="ec2-web-001"))
        score, breakdown = env.grade()
        assert score == 0.0
        assert len(breakdown["active_resources_terminated"]) == 1

    def test_seeded_state_is_different(self):
        env = CloudSREEnvironment()
        obs1 = env.reset(task_id="phantom_volume_cleanup", seed=42)
        obs2 = env.reset(task_id="phantom_volume_cleanup", seed=99)
        ids1 = {r["id"] for r in obs1.resources}
        ids2 = {r["id"] for r in obs2.resources}
        assert ids1 != ids2


class TestTask2LatencySpikeRemediation:
    """Test the medium task — RDS scaling."""

    def test_fixed_state_has_rds(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="latency_spike_remediation")
        rds = [r for r in obs.resources if r.get("type") == "rds_database"]
        assert len(rds) >= 1

    def test_scaling_rds_scores(self):
        env = CloudSREEnvironment()
        env.reset(task_id="latency_spike_remediation")
        env.step(
            SREAction(
                command="scale",
                resource_id="rds-primary-001",
                params={"target_size": "db.t3.medium"},
            )
        )
        score, breakdown = env.grade()
        assert score >= 0.7
        assert breakdown["rds_scaled"] is True


class TestTask3NoisyNeighborIncident:
    """Test the hard task — rogue instance investigation."""

    def test_fixed_state_has_rogue_and_stopped_backend(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="noisy_neighbor_incident")
        ids = {r["id"] for r in obs.resources}
        assert "ec2-rogue-test-001" in ids
        assert "ec2-backend-prod-001" in ids

    def test_perfect_incident_response(self):
        env = CloudSREEnvironment()
        env.reset(task_id="noisy_neighbor_incident")
        env.step(SREAction(command="inspect", resource_id="ec2-rogue-test-001"))
        env.step(SREAction(command="terminate", resource_id="ec2-rogue-test-001"))
        env.step(SREAction(command="reboot", resource_id="ec2-backend-prod-001"))
        score, breakdown = env.grade()
        assert score == 1.0
        assert breakdown["inspected_rogue"] is True
        assert breakdown["terminated_rogue"] is True
        assert breakdown["rebooted_backend"] is True
        assert breakdown["alerts_resolved"] is True


class TestAllTasksRegistered:
    """Ensure all tasks are in the registry."""

    def test_registry_has_all_tasks(self):
        assert "phantom_volume_cleanup" in TASK_REGISTRY
        assert "latency_spike_remediation" in TASK_REGISTRY
        assert "noisy_neighbor_incident" in TASK_REGISTRY


def test_step_rewards_and_terminal_flag_reach_client():
    from cloud_sre_env.client import CloudSREEnv
    from openenv.core.env_server.serialization import serialize_observation

    env = CloudSREEnvironment()
    env.reset(task_id="phantom_volume_cleanup")
    client = CloudSREEnv.__new__(CloudSREEnv)
    for step in range(15):
        obs = env.step(SREAction(command="wait"))
        result = client._parse_result(serialize_observation(obs))
        assert result.reward == pytest.approx(-0.01)
        assert result.observation.reward == result.reward
        assert result.done is (step == 14)
        assert result.observation.done == result.done
        assert result.observation.step_number == step + 1
    assert env.step(SREAction(command="wait")).reward == 0.0


def test_cost_drift_chaos_survives_state_recalculation():
    import random

    def event_for_seed(candidate):
        rng = random.Random(candidate * 1000 + 4)
        if rng.random() > 0.25:
            return None
        return rng.choice(CHAOS_EVENTS)["type"]

    seed = next(
        candidate
        for candidate in range(1000)
        if event_for_seed(candidate) == "cost_drift"
    )
    env = CloudSREEnvironment()
    env.reset(task_id="phantom_volume_cleanup", seed=seed)
    for _ in range(4):
        observation = env.step(SREAction(command="wait"))

    resource_cost = sum(r["cost_per_hour"] for r in observation.resources)
    assert observation.total_hourly_cost > resource_cost
    drifted_cost = observation.total_hourly_cost
    env._recalculate_state()
    assert env._build_observation().total_hourly_cost == drifted_cost


def test_scale_reward_uses_pre_mutation_cpu():
    env = CloudSREEnvironment()
    env.reset(task_id="latency_spike_remediation")
    observation = env.step(
        SREAction(
            command="scale",
            resource_id="rds-primary-001",
            params={"target_size": "db.t3.medium"},
        )
    )
    assert observation.reward == pytest.approx(0.13)
    rds = next(r for r in observation.resources if r["id"] == "rds-primary-001")
    assert rds["cpu_utilization"] == 45.0


def test_server_app_uses_environment_factory():
    from cloud_sre_env.server.app import app

    assert app.title == "OpenEnv Environment HTTP API"


def test_reset_preserves_requested_episode_id_and_seed():
    env = CloudSREEnvironment()
    first = env.reset(seed=42, episode_id="requested-episode")
    assert env.state.episode_id == "requested-episode"
    second = env.reset(seed=42)
    assert env.state.episode_id
    assert env.state.episode_id != "requested-episode"
    assert second.resources == first.resources


@pytest.mark.parametrize("transport", ["http", "websocket"])
def test_app_exposes_complete_episode_state(transport):
    from cloud_sre_env.models import SREState
    from cloud_sre_env.server.app import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        schema = client.get("/schema").json()
        assert schema["state"] == SREState.model_json_schema()
        reset = {"task_id": "phantom_volume_cleanup", "episode_id": "state-test"}
        if transport == "http":
            response = client.get("/state")
            assert response.status_code == 200
            assert response.json() == SREState().model_dump()
            return
        else:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "reset", "data": reset})
                assert ws.receive_json()["type"] == "observation"
                ws.send_json({"type": "step", "data": {"command": "wait"}})
                assert ws.receive_json()["type"] == "observation"
                ws.send_json({"type": "state"})
                response = ws.receive_json()
                assert response["type"] == "state"
                state = response["data"]
        assert state == {
            "episode_id": "state-test",
            "step_count": 1,
            "task_id": "phantom_volume_cleanup",
            "current_step": 1,
            "done": False,
            "cumulative_reward": -0.01,
            "action_count": 1,
        }


@pytest.mark.parametrize(
    "commands, expected_score, inspected",
    [
        (["inspect"], 0.2, True),
        (["inspect", "wait"], 0.2, True),
        (["inspect", "inspect"], 0.2, True),
        (["terminate"], 0.3, False),
        (["inspect", "terminate"], 0.5, True),
        (["terminate", "inspect"], 0.3, False),
    ],
)
def test_rogue_inspection_is_independent(commands, expected_score, inspected):
    env = CloudSREEnvironment()
    env.reset(task_id="noisy_neighbor_incident")
    for command in commands:
        env.step(SREAction(command=command, resource_id="ec2-rogue-test-001"))
    score, breakdown = env.grade()
    assert score == expected_score
    assert breakdown["inspected_rogue"] is inspected


@pytest.mark.parametrize(
    "resource_id", ["ec2-app-001", "rds-primary-001", "alb-main-001"]
)
def test_production_termination_penalizes_reward_and_grade(resource_id):
    env = CloudSREEnvironment()
    env.reset(task_id="latency_spike_remediation")
    observation = env.step(SREAction(command="terminate", resource_id=resource_id))
    assert observation.reward == pytest.approx(-0.15)
    score, breakdown = env.grade()
    assert breakdown["under_budget"] is True
    assert breakdown["penalties"] == pytest.approx(0.3)
    assert score == 0.0


@pytest.mark.parametrize(
    "resource_id, expected", [("ebs-orphan-001", 0.078), ("ebs-data-001", -0.1)]
)
def test_volume_termination_rewards_unchanged(resource_id, expected):
    env = CloudSREEnvironment()
    env.reset(task_id="phantom_volume_cleanup")
    observation = env.step(SREAction(command="terminate", resource_id=resource_id))
    assert observation.reward == pytest.approx(expected)


@pytest.mark.parametrize(
    "resource_type", ["ec2_instance", "rds_database", "alb_load_balancer"]
)
def test_task2_grade_penalizes_production_by_identity(resource_type):
    env = CloudSREEnvironment()
    env.reset(task_id="latency_spike_remediation", seed=42)
    initial = env._initial_state
    resource = next(r for r in initial["resources"] if r["type"] == resource_type)
    # Grading must use resource metadata, not an ID prefix.
    resource["id"] = "production-resource"
    score, breakdown = env._task.grade(
        [{"command": "terminate", "resource_id": resource["id"]}],
        {"total_hourly_cost": 0},
        initial,
    )
    assert score == 0.0
    assert breakdown["prod_terminated"] == [resource["id"]]
    assert breakdown["penalties"] == pytest.approx(0.3)
