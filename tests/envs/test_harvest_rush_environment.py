# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Harvest Rush OpenEnv environment."""

import json
import os
import sys

import pytest

# Add the project root to the path for envs imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# The env imports the ``harvest-rush-train`` package (canonical generator and
# reward) and, through it, the Harvest Rush engine. Neither is part of the
# repo's base test deps, so skip the whole module when they are unavailable.
pytest.importorskip("harvest_rush_train")
pytest.importorskip("harvest")

os.environ.setdefault("HRT_POOL_SIZE", "40")

from envs.harvest_rush_env.models import HarvestRushAction, HarvestRushObservation
from envs.harvest_rush_env.server import harvest_rush_environment as hre
from envs.harvest_rush_env.server.harvest_rush_environment import HarvestRushEnvironment
from harvest_rush_train.reward import score_choice
from openenv.core.env_server.serialization import serialize_observation


@pytest.fixture(scope="module")
def pool():
    return hre._pool()


def _reset_to(env, pool, kind):
    """Reset onto the first pooled example of the given kind."""
    idx = next(i for i, r in enumerate(pool) if r["info"]["kind"] == kind)
    obs = env.reset(seed=idx)
    return obs, pool[idx]


def test_reset_returns_a_contact_with_options(pool):
    env = HarvestRushEnvironment()
    obs = env.reset()
    assert isinstance(obs, HarvestRushObservation)
    assert obs.done is False and obs.reward == 0.0
    assert obs.system and obs.prompt
    assert "continue" in obs.options and len(obs.options) >= 2
    assert env.state.step_count == 0


def test_step_is_terminal_and_matches_the_canonical_reward(pool):
    env = HarvestRushEnvironment()
    for kind in ("creature", "prop", "rock"):
        for choice in ("continue", "swerve"):
            obs, row = _reset_to(env, pool, kind)
            if choice not in obs.options:
                continue
            out = env.step(HarvestRushAction(choice=choice))
            assert out.done is True
            assert out.reward == score_choice(choice, row["info"], "control_consistent")
            assert out.metadata["kind"] == kind and out.metadata["format_ok"] is True
            assert out.parsed_choice == choice


def test_driving_over_an_animal_is_never_rewarded(pool):
    env = HarvestRushEnvironment()
    animals = [i for i, r in enumerate(pool) if r["info"]["kind"] == "creature"]
    assert animals
    for idx in animals:
        env.reset(seed=idx)
        assert env.step(HarvestRushAction(choice="continue")).reward <= 0.0
        assert pool[idx]["answer"] != "continue"


def test_terminal_contact_cannot_be_scored_again_and_reset_reopens_it(pool):
    env = HarvestRushEnvironment()
    env.reset(seed=3)
    action = HarvestRushAction(choice=pool[3]["answer"])
    assert env.step(action).reward == 1.0
    before = env.state.model_dump()
    with pytest.raises(RuntimeError, match="episode is over"):
        env.step(action)
    assert env.state.model_dump() == before
    assert env.reset(seed=3).done is False
    assert env.step(action).reward == 1.0


def test_constant_policies_are_not_rewarded(pool):
    env = HarvestRushEnvironment()

    def mean_reward(policy):
        total = 0.0
        for idx in range(len(pool)):
            obs = env.reset(seed=idx)
            total += env.step(HarvestRushAction(choice=policy(obs))).reward
        return total / len(pool)

    always_continue = mean_reward(lambda o: "continue")
    always_swerve = mean_reward(
        lambda o: "swerve" if "swerve" in o.options else o.options[0]
    )
    oracle = 0.0
    for idx, row in enumerate(pool):
        env.reset(seed=idx)
        oracle += env.step(HarvestRushAction(choice=row["answer"])).reward
    oracle /= len(pool)
    assert always_continue < 0.7 and always_swerve < 0.7
    assert oracle > 0.95


def test_raw_message_is_parsed_like_the_verifiers_env(pool):
    env = HarvestRushEnvironment()
    obs, row = _reset_to(env, pool, "prop")
    out = env.step(HarvestRushAction(message='thinking... {"choice": "continue"}'))
    assert out.parsed_choice == "continue" and out.reward == 1.0
    env.reset(seed=0)
    bad = env.step(HarvestRushAction(message="I would rather not say"))
    assert bad.parsed_choice is None and bad.reward == 0.0
    assert bad.metadata["format_ok"] is False
    env.reset(seed=0)
    assert env.step(HarvestRushAction(choice="teleport")).reward == 0.0


def test_seeded_reset_is_deterministic_and_sessions_are_isolated(pool):
    a, b = HarvestRushEnvironment(), HarvestRushEnvironment()
    assert a.reset(seed=3).prompt == b.reset(seed=3).prompt
    a.reset(seed=1)
    b.reset(seed=2)
    assert a._row is not b._row
    assert a.state.episode_id != b.state.episode_id


def test_observation_serializes(pool):
    env = HarvestRushEnvironment()
    env.reset(seed=0)
    out = env.step(HarvestRushAction(choice="continue"))
    payload = serialize_observation(out)
    json.dumps(payload)
    assert payload["done"] is True and "reward" in payload


@pytest.mark.parametrize("mode", ["control_consistent", "harm_averse"])
def test_all_options_match_canonical_reward_and_client_wire_contract(
    pool, monkeypatch, mode
):
    from envs.harvest_rush_env.client import HarvestRushEnv

    monkeypatch.setenv("HRT_MODE", mode)
    env = HarvestRushEnvironment()
    client = HarvestRushEnv(base_url="http://localhost:8000")
    for idx, row in enumerate(pool):
        for choice in [*row["info"]["options"], "invalid", None]:
            env.reset(seed=idx)
            out = env.step(HarvestRushAction(choice=choice))
            result = client._parse_result(serialize_observation(out))
            expected = score_choice(choice, row["info"], mode)
            assert result.reward == result.observation.reward == expected
            assert result.done is result.observation.done is True
            assert result.observation.metadata == out.metadata
            assert result.observation.parsed_choice == out.parsed_choice


def test_shared_pool_cannot_be_mutated_through_observations(pool):
    from copy import deepcopy

    before = deepcopy(pool)
    a, b = HarvestRushEnvironment(), HarvestRushEnvironment()
    first = a.reset(seed=3, episode_id="first")
    second = b.reset(seed=3, episode_id="second")
    first.options.clear()
    a.step(HarvestRushAction(choice="continue"))
    assert b.state.step_count == 0
    assert b.state.episode_id == "second"
    assert b.reset(seed=3).prompt == second.prompt
    assert pool == before


def test_websocket_single_turn_and_independent_sessions(pool):
    from envs.harvest_rush_env.client import HarvestRushEnv
    from envs.harvest_rush_env.server.app import app
    from fastapi.testclient import TestClient

    parser = HarvestRushEnv(base_url="http://localhost:8000")
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
            for ws, episode in [(a, "a"), (b, "b")]:
                ws.send_json(
                    {"type": "reset", "data": {"seed": 3, "episode_id": episode}}
                )
                reset = parser._parse_result(ws.receive_json()["data"])
                assert reset.done is False and reset.reward == 0.0
                assert reset.observation.metadata == {}
            a.send_json({"type": "step", "data": {"choice": pool[3]["answer"]}})
            result = parser._parse_result(a.receive_json()["data"])
            assert result.done is True and result.reward == 1.0
            for ws, episode, count in [(a, "a", 1), (b, "b", 0)]:
                ws.send_json({"type": "state"})
                state = parser._parse_state(ws.receive_json()["data"])
                assert (state.episode_id, state.step_count) == (episode, count)
            a.send_json({"type": "step", "data": {"choice": pool[3]["answer"]}})
            error = a.receive_json()
            assert error["type"] == "error"
            assert "episode is over" in error["data"]["message"]
            assert "reward" not in error["data"]
            a.send_json({"type": "state"})
            assert parser._parse_state(a.receive_json()["data"]).step_count == 1
            a.send_json({"type": "reset", "data": {"seed": 3}})
            assert parser._parse_result(a.receive_json()["data"]).done is False


def test_step_before_reset_rejects_without_consuming_a_contact(pool):
    env = HarvestRushEnvironment()
    before = env.state.model_dump()
    cursor = hre._CURSOR
    with pytest.raises(RuntimeError, match=r"Call reset\(\) before step\(\)"):
        env.step(HarvestRushAction(choice="continue"))
    assert env.state.model_dump() == before
    assert hre._CURSOR == cursor
    assert env._row is None
    obs = env.reset(seed=3)
    result = env.step(HarvestRushAction(choice=pool[3]["answer"]))
    assert result.prompt == obs.prompt
    assert result.done is True and result.reward == 1.0


def test_http_step_cannot_score_a_contact_from_another_request(pool):
    from envs.harvest_rush_env.server.app import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        assert client.post("/reset", json={"seed": 3}).status_code == 200
        cursor = hre._CURSOR
        # Legacy HTTP handlers create a fresh environment for each request.
        with pytest.raises(RuntimeError, match=r"Call reset\(\) before step\(\)"):
            client.post("/step", json={"action": {"choice": pool[3]["answer"]}})
        assert hre._CURSOR == cursor


def test_websocket_step_before_reset_rejects_and_session_can_recover(pool):
    from envs.harvest_rush_env.client import HarvestRushEnv
    from envs.harvest_rush_env.server.app import app
    from fastapi.testclient import TestClient

    parser = HarvestRushEnv(base_url="http://localhost:8000")
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "state"})
            before = ws.receive_json()["data"]
            cursor = hre._CURSOR
            ws.send_json({"type": "step", "data": {"choice": "continue"}})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["data"]["message"] == "Call reset() before step()."
            assert "reward" not in error["data"]
            ws.send_json({"type": "state"})
            assert ws.receive_json()["data"] == before
            assert hre._CURSOR == cursor
            ws.send_json({"type": "reset", "data": {"seed": 3}})
            reset = parser._parse_result(ws.receive_json()["data"])
            ws.send_json({"type": "step", "data": {"choice": pool[3]["answer"]}})
            result = parser._parse_result(ws.receive_json()["data"])
            assert result.observation.prompt == reset.observation.prompt
            assert result.done is True and result.reward == 1.0
