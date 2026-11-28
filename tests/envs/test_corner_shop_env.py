# SPDX-License-Identifier: BSD-3-Clause

"""Exercise Corner Shop's episodes, going-concern reward, and WebSocket contract."""

import json

import pytest
from corner_shop_env import CornerShopAction, CornerShopEnv
from corner_shop_env.models import CornerShopObservation, CornerShopState
from corner_shop_env.server import going_concern as gc
from corner_shop_env.server.app import app as corner_shop_app
from corner_shop_env.server.corner_shop_environment import CornerShopEnvironment
from corner_shop_env.server.going_concern import rewards
from fastapi.testclient import TestClient
from openenv.core.env_server import create_app

FIRMS = [1, 2, 3]


def play(env, operator, firm, as_message=False):
    operator.reset()
    observation = env.reset(firm=firm)
    assert observation.system_prompt and observation.prompt
    assert observation.week == 0 and observation.reward == 0.0
    while not observation.done:
        decision = operator.act(observation.shop)
        if as_message:
            message = "Plan.\n```json\n" + json.dumps(decision) + "\n```"
            action = CornerShopAction(message=message)
        else:
            action = CornerShopAction(decision=decision)
        observation = env.step(action)
        if not observation.done:
            assert observation.reward == 0.0
    return observation


@pytest.mark.parametrize("firm", FIRMS)
def test_honest_scores_one_and_cheater_trades_the_business_for_cash(firm):
    env = CornerShopEnvironment()
    honest = play(env, gc.ReferenceOperator(), firm)
    cheater = play(env, gc.EndgameExploiter(), firm)

    assert honest.reward == pytest.approx(1.0)
    assert cheater.reward < honest.reward
    assert cheater.metrics["cash_usd"] > honest.metrics["cash_usd"]
    assert cheater.metrics["going_concern_usd"] < honest.metrics["going_concern_usd"]
    assert cheater.metrics["liquidated_units"] > 0

    for name, observation in [("honest", honest), ("cheater", cheater)]:
        operator = gc.ReferenceOperator if name == "honest" else gc.EndgameExploiter
        sim = gc.run_episode(operator(), firm).sim
        assert observation.metrics["cash_usd"] == pytest.approx(
            rewards.cash_reward(sim)
        )
        assert observation.metrics["going_concern_usd"] == pytest.approx(
            rewards.going_concern(sim)
        )


def test_message_path_matches_structured_path():
    env = CornerShopEnvironment()
    structured = play(env, gc.ReferenceOperator(), 1)
    text = play(env, gc.ReferenceOperator(), 1, as_message=True)
    assert text.reward == pytest.approx(structured.reward)
    assert text.metrics == structured.metrics


def test_unparseable_reply_is_reported_and_the_week_passes():
    env = CornerShopEnvironment()
    env.reset(firm=1)
    observation = env.step(CornerShopAction(message="no json here"))
    assert observation.week == 1
    assert observation.errors
    assert not observation.done


def test_malformed_list_fields_in_the_last_weeks_do_not_abort_the_episode():
    env = CornerShopEnvironment()
    observation = env.reset(firm=1)
    while not observation.done:
        observation = env.step(
            CornerShopAction(message='{"orders": 5, "pay": "INV-1"}')
        )
    assert observation.week == observation.weeks_total
    assert observation.metrics["orders_last_3_weeks"] == 0
    assert observation.metrics["payments_last_3_weeks"] == 0


def test_end_episode_runs_remaining_weeks_unattended_and_scores():
    env = CornerShopEnvironment()
    env.reset(firm=1)
    observation = env.step(CornerShopAction(end_episode=True))
    assert observation.done
    assert observation.metrics["weeks_unattended"] == observation.weeks_total
    assert env.state.done
    with pytest.raises(RuntimeError, match="episode is over"):
        env.step(CornerShopAction())


def test_seed_picks_firm_and_cash_reward_is_selectable():
    env = CornerShopEnvironment()
    assert env.reset(seed=6).firm == 7
    env.reset(firm=1, reward="cash", normalize=False)
    observation = env.step(CornerShopAction(end_episode=True))
    assert observation.reward == pytest.approx(observation.metrics["cash_usd"] * 1e-3)


@pytest.mark.parametrize("kwargs", [{"firm": 0}, {"firm": 101}, {"reward": "profit"}])
def test_invalid_reset_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CornerShopEnvironment().reset(**{"firm": 1, **kwargs})


def test_http_state_serves_the_state_subclass():
    with TestClient(corner_shop_app) as client:
        schema = client.get("/schema").json()
        assert "firm" in schema["state"]["properties"]


def test_websocket_episode_matches_client_parsers():
    app = create_app(
        CornerShopEnvironment,
        CornerShopAction,
        CornerShopObservation,
        state_cls=CornerShopState,
    )
    parser = CornerShopEnv(base_url="http://testserver")
    operator = gc.ReferenceOperator()
    operator.reset()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {"firm": 2, "episode_id": "wire"}})
            response = ws.receive_json()
            assert response["type"] == "observation"
            result = parser._parse_result(response["data"])
            assert result.observation.firm == 2
            assert result.observation.system_prompt
            assert not result.done
            steps = 0
            while not result.done:
                action = CornerShopAction(
                    decision=operator.act(result.observation.shop)
                )
                ws.send_json({"type": "step", "data": parser._step_payload(action)})
                response = ws.receive_json()
                assert response["type"] == "observation"
                result = parser._parse_result(response["data"])
                assert result.observation.done is result.done
                steps += 1
            assert result.reward == pytest.approx(1.0)
            assert result.observation.metrics["going_concern_vs_honest"] == (
                pytest.approx(1.0)
            )
            ws.send_json({"type": "state"})
            response = ws.receive_json()
            assert response["type"] == "state", response
            state = parser._parse_state(response["data"])
            assert state.episode_id == "wire"
            assert state.firm == 2
            assert state.step_count == steps
            assert state.done
