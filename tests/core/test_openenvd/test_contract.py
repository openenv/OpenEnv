# SPDX-License-Identifier: BSD-3-Clause

import pytest
from openenv.core.openenvd.contract import (
    ManifestError,
    parse_manifest,
    Phase,
    ZoneKind,
)


def test_defaults_are_strict_for_the_agent_zone():
    m = parse_manifest({})
    assert m.zones.agent.network.egress == "relays-only"
    assert "env" in m.zones.agent.containers


def test_partial_agent_zone_keeps_agent_defaults():
    m = parse_manifest({"zones": {"agent": {"resources": {"pids": 10}}}})
    assert m.zones.agent.network.egress == "relays-only"
    assert m.zones.agent.resources.pids == 10


def test_partial_agent_network_keeps_relay_default():
    m = parse_manifest({"zones": {"agent": {"network": {"model": "openai"}}}})
    assert m.zones.agent.network.egress == "relays-only"
    assert m.zones.agent.network.model == "openai"


def test_app_and_port_default_to_the_top_level(tmp_path):
    from openenv.core.openenvd.contract import load_manifest

    path = tmp_path / "openenv.yaml"
    path.write_text("app: server.app:app\nport: 8123\n")
    m = load_manifest(path)
    assert (m.app, m.port) == ("server.app:app", 8123)
    path.write_text("app: a:b\nopenenvd:\n  app: c:d\n")
    assert load_manifest(path).app == "c:d"


@pytest.mark.parametrize(
    "block, message",
    [
        ({"zones": {"agent": {"network": {"egress": "allowlist"}}}}, "relays-only"),
        ({"zones": {"agent": {"network": {"egress": "none"}}}}, "relays-only"),
        (
            {"zones": {"agent": {"containers": {"env": {"reads": ["workspace"]}}}}},
            "only observers may declare reads",
        ),
        (
            {"zones": {"observers": {"containers": {"g": {"reads": ["cgroups"]}}}}},
            "at least one phase",
        ),
        (
            {
                "privileged_assets": {"oracle": "o.sh"},
                "zones": {
                    "observers": {
                        "containers": {
                            "g": {"reads": ["assets.oracle"], "phases": ["running"]}
                        }
                    }
                },
            },
            r"within \[frozen, grading\]",
        ),
        (
            {
                "privileged_assets": {"oracle": "o.sh"},
                "zones": {
                    "observers": {
                        "containers": {
                            "g": {
                                "reads": ["assets.oracle", "workspace"],
                                "phases": ["grading"],
                            }
                        }
                    }
                },
            },
            "split it into a runner",
        ),
        (
            {
                "zones": {
                    "agent": {
                        "resources": {"memory_mb": 100},
                        "containers": {"env": {"resources": {"memory_mb": 200}}},
                    }
                }
            },
            "exceeds the zone ceiling",
        ),
        (
            {
                "zones": {
                    "services": {
                        "containers": {"s": {"expose": {"host": "a", "port": 80}}}
                    }
                }
            },
            "listening port",
        ),
    ],
)
def test_zone_rules(block, message):
    with pytest.raises(ManifestError, match=message):
        parse_manifest(block)


def test_container_narrowing_and_phases():
    m = parse_manifest(
        {
            "zones": {
                "agent": {"resources": {"memory_mb": 1000, "pids": 50}},
                "observers": {
                    "containers": {
                        "metrics": {"reads": ["cgroups"], "phases": ["ready"]}
                    }
                },
            }
        }
    )
    assert m.zones.agent.effective_resources("env").memory_mb == 1000
    assert m.container_phases(ZoneKind.AGENT, "env") == (Phase.READY, Phase.RUNNING)
    assert m.container_phases(ZoneKind.OBSERVERS, "metrics") == (Phase.READY,)
