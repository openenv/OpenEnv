# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Unity environment's web playground drawing (no Unity binary needed)."""

from unity_env.models import UnityObservation
from unity_env.server.unity_environment import UnityMLAgentsEnvironment


def test_unity_web_playground(tmp_path):
    env = UnityMLAgentsEnvironment(cache_dir=str(tmp_path))
    obs = UnityObservation(
        visual_observations=["iVBORw0KGgo="],
        behavior_name="PushBlock?team=0",
        action_spec_info={"is_discrete": True, "discrete_branches": [7]},
    ).model_dump()
    assert '<img alt="Unity camera" src="data:image/png;base64,iVBORw0KGgo="' in (
        env.render_web(obs)
    )
    actions = env.web_actions(obs)
    assert len(actions) == 7
    assert actions[:2] == [
        ("0 · noop", {"discrete_actions": [0]}),
        ("1 · forward", {"discrete_actions": [1]}),
    ]

    obs = UnityObservation(
        vector_observations=[0.1, 0.2],
        behavior_name="3DBall?team=0",
        action_spec_info={"is_continuous": True, "discrete_branches": []},
    ).model_dump()
    assert env.render_web(obs) is None
    assert env.render_web({}) is None
    assert env.web_actions(obs) == []

    obs = UnityObservation(
        behavior_name="GridWorld?team=0",
        action_spec_info={"discrete_branches": [5]},
    ).model_dump()
    assert env.web_actions(obs)[4] == ("action 4", {"discrete_actions": [4]})


def test_unity_include_visual_default(tmp_path, monkeypatch):
    monkeypatch.setenv("UNITY_INCLUDE_VISUAL", "1")
    assert UnityMLAgentsEnvironment(cache_dir=str(tmp_path))._include_visual
    monkeypatch.delenv("UNITY_INCLUDE_VISUAL")
    assert not UnityMLAgentsEnvironment(cache_dir=str(tmp_path))._include_visual
