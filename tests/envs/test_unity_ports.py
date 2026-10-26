"""Each Unity env instance must talk to its Unity process on its own port."""

import sys
from types import SimpleNamespace

import pytest
from envs.unity_env.server.unity_environment import UnityMLAgentsEnvironment


def test_instances_use_different_ports(monkeypatch, tmp_path):
    ports = []

    class FakeRegistryEntry:
        def __init__(self, **kwargs):
            pass

        def make(self, base_port=5005, worker_id=0, **kwargs):
            ports.append(base_port + worker_id)
            raise RuntimeError("stop before starting Unity")

    entry = SimpleNamespace(identifier="PushBlock", expected_reward=0, description="")
    modules = {
        "mlagents_envs": SimpleNamespace(),
        "mlagents_envs.base_env": SimpleNamespace(ActionTuple=None),
        "mlagents_envs.registry": SimpleNamespace(
            default_registry={"PushBlock": entry}
        ),
        "mlagents_envs.registry.remote_registry_entry": SimpleNamespace(
            RemoteRegistryEntry=FakeRegistryEntry
        ),
        "mlagents_envs.side_channel": SimpleNamespace(),
        "mlagents_envs.side_channel.engine_configuration_channel": SimpleNamespace(
            EngineConfigurationChannel=lambda: None
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    for _ in range(2):
        env = UnityMLAgentsEnvironment(cache_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="stop before starting Unity"):
            env._load_environment("PushBlock")

    assert ports[0] != ports[1]
