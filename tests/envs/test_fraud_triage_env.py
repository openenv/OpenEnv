# SPDX-License-Identifier: BSD-3-Clause

"""Exercise Fraud Triage's episode, rewards, and real WebSocket contract."""

import pytest
from fastapi.testclient import TestClient
from fraud_triage_env import FraudTriageAction, FraudTriageEnv, TriageDecision
from fraud_triage_env.models import FraudTriageObservation
from fraud_triage_env.server.fraud_triage_environment import FraudTriageEnvironment
from openenv.core.env_server import create_app
from pydantic import ValidationError


def test_seed_replays_transactions_and_rewards():
    env = FraudTriageEnvironment(episode_length=3)

    def episode():
        observations = [env.reset(seed=42, episode_id="replay").model_dump()]
        for _ in range(3):
            observations.append(env.step(FraudTriageAction(decision=0)).model_dump())
        return observations, env.state.model_dump()

    first = episode()
    assert first == episode()
    assert first[1]["episode_id"] == "replay"
    assert "is_fraud" not in first[0][0]
    assert first[0][-1]["done"]


@pytest.mark.parametrize("fraud_rate", [0.0, 1.0])
@pytest.mark.parametrize("decision", list(TriageDecision))
def test_each_decision_scores_hidden_label_and_updates_one_counter(
    fraud_rate, decision
):
    env = FraudTriageEnvironment(episode_length=1, fraud_rate=fraud_rate)
    env.reset(seed=7)
    observation = env.step(FraudTriageAction(decision=decision))
    flags = decision != TriageDecision.APPROVE
    expected = (1.0 if flags else -3.0) if fraud_rate else (-0.5 if flags else 0.1)
    if decision == TriageDecision.ESCALATE:
        expected -= 0.1
    assert observation.reward == pytest.approx(expected)
    assert observation.done
    assert env.state.cumulative_reward == pytest.approx(expected)
    assert env.state.step_count == 1
    assert env.state.true_positives == int(bool(fraud_rate) and flags)
    assert env.state.false_negatives == int(bool(fraud_rate) and not flags)
    assert env.state.false_positives == int(not fraud_rate and flags)
    assert env.state.true_negatives == int(not fraud_rate and not flags)


def test_step_after_done_cannot_repeat_reward_or_change_state():
    env = FraudTriageEnvironment(episode_length=1)
    action = FraudTriageAction(decision=0)
    env.step(action)
    completed = env.state.model_dump()
    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step(action)
    assert env.state.model_dump() == completed
    assert not env.reset(seed=1).done
    assert env.state.step_count == 0
    assert env.step(action).done


@pytest.mark.parametrize("decision", [-1, 4, 100])
def test_invalid_decision_is_rejected(decision):
    with pytest.raises(ValidationError):
        FraudTriageAction(decision=decision)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episode_length": 0},
        {"episode_length": -1},
        {"episode_length": True},
        {"fraud_rate": -0.1},
        {"fraud_rate": 1.1},
        {"fraud_rate": float("nan")},
    ],
)
def test_invalid_episode_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FraudTriageEnvironment(**kwargs)


def test_websocket_reset_step_and_state_match_client_parsers():
    app = create_app(
        lambda: FraudTriageEnvironment(episode_length=2),
        FraudTriageAction,
        FraudTriageObservation,
    )
    parser = FraudTriageEnv(base_url="http://testserver")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {"seed": 42, "episode_id": "wire"}})
            response = ws.receive_json()
            assert response["type"] == "observation"
            result = parser._parse_result(response["data"])
            assert result.observation.transaction_id
            assert result.observation.amount > 0
            assert result.observation.legal_actions == [0, 1, 2, 3]
            assert result.reward == 0.0
            assert not result.done
            for done in [False, True]:
                action = FraudTriageAction(decision=TriageDecision.FLAG)
                ws.send_json({"type": "step", "data": parser._step_payload(action)})
                response = ws.receive_json()
                assert response["type"] == "observation"
                result = parser._parse_result(response["data"])
                assert result.done is done
                assert result.observation.done is done
                assert result.observation.reward == result.reward
            ws.send_json({"type": "state"})
            response = ws.receive_json()
            assert response["type"] == "state", response
            state = parser._parse_state(response["data"])
            assert state.episode_id == "wire"
            assert state.step_count == 2
            assert state.true_positives + state.false_positives == 2
