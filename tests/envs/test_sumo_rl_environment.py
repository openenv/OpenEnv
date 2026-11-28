# SPDX-License-Identifier: BSD-3-Clause

"""reset(seed=..., episode_id=...) in the SUMO-RL environment."""

import sys
import types

import pytest


class _FakeSumoRl:
    """Stands in for sumo_rl's SumoEnvironment, which needs the SUMO binary."""

    def __init__(self, sumo_seed=42, **kwargs):
        self.sumo_seed = sumo_seed
        self.action_space = types.SimpleNamespace(n=2)

    def reset(self, seed=None, **kwargs):
        # sumo_rl replaces its seed when one is given
        if seed is not None:
            self.sumo_seed = seed
        return [0.0, 0.0], {"step": 0.0}


@pytest.fixture
def env(monkeypatch):
    fake = types.ModuleType("sumo_rl")
    fake.SumoEnvironment = _FakeSumoRl
    monkeypatch.setitem(sys.modules, "sumo_rl", fake)
    monkeypatch.delitem(
        sys.modules, "sumo_rl_env.server.sumo_environment", raising=False
    )
    from sumo_rl_env.server.sumo_environment import SumoEnvironment

    return SumoEnvironment(net_file="net.xml", route_file="routes.xml")


def test_reset_passes_seed_to_sumo(env):
    assert env.env.sumo_seed == 42
    env.reset(seed=7)
    assert env.env.sumo_seed == 7
    env.reset()
    assert env.env.sumo_seed == 7


def test_reset_keeps_episode_id_or_generates_one(env):
    env.reset(episode_id="traffic-1")
    assert env.state.episode_id == "traffic-1"
    env.reset(episode_id="")
    assert env.state.episode_id == ""
    env.reset()
    first = env.state.episode_id
    env.reset()
    assert first and env.state.episode_id != first
