"""`runtime.network_policy` judges measured reachability, never provider claims."""

import json
from pathlib import Path

import pytest
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.network import NetworkPolicyGrader
from openenv.validation.manifest import NetworkPolicy
from openenv.validation.types import CheckStatus, ProviderCapability

REACHABLE = {"result": "reachable"}
DENIED = {"result": "denied", "errno": 101}
TIMEOUT = {"result": "inconclusive", "errno": None}
EACCES = {"result": "inconclusive", "errno": 13}


def evidence(
    mode, *, network_mode=None, interfaces=None, route=None, v6=0, probes=None
):
    isolated = mode == "no-network"
    return {
        "schema_version": 1,
        "requested_mode": mode,
        "subject_network_mode": network_mode or ("none" if isolated else "bridge"),
        "namespace": {
            "interfaces": interfaces or (["lo"] if isolated else ["eth0", "lo"]),
            "default_route": (not isolated) if route is None else route,
            "ipv6_non_loopback": v6,
            "listening_ports": [8000],
        },
        "probes": probes
        if probes is not None
        else [
            {
                "kind": kind,
                "control": REACHABLE,
                "subject": DENIED if isolated else REACHABLE,
            }
            for kind in ("tcp", "udp")
        ],
    }


class _Manifest:
    def __init__(self, mode):
        self.network = NetworkPolicy(
            mode=mode, allowed_hosts=["example.com"] if mode == "allowlist" else []
        )


def grade(mode, measured):
    subject = Subject(
        root=Path("."),
        manifest=_Manifest(mode),
        image_ref="sha256:" + "a" * 64,
        running=None,
        outputs_dir=Path("."),
        network_evidence_json=None if measured is None else json.dumps(measured),
    )
    return NetworkPolicyGrader().run(subject)


def test_grader_metadata():
    grader = NetworkPolicyGrader()
    assert grader.check_id == "runtime.network_policy"
    assert grader.requires_provider == frozenset({ProviderCapability.NETWORK_POLICY})
    assert "runtime.startup" in grader.depends_on


def test_allowlist_is_applicable_but_not_implemented():
    assert NetworkPolicyGrader().applies_to(_Manifest("allowlist"))
    result = grade("allowlist", evidence("public"))
    assert result.status is CheckStatus.SKIP
    assert "allowlist" in " ".join(result.evidence)


@pytest.mark.parametrize("mode", ["public", "no-network"])
def test_measured_policy_passes(mode):
    assert grade(mode, evidence(mode)).status is CheckStatus.PASS


def test_no_network_leak_fails_on_measurement_alone():
    leaked = evidence(
        "no-network",
        probes=[{"kind": "tcp", "control": REACHABLE, "subject": REACHABLE}],
    )
    result = grade("no-network", leaked)
    assert result.status is CheckStatus.FAIL
    assert any("tcp" in line and "reachable" in line for line in result.evidence)


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"network_mode": "bridge"}, "network mode"),
        ({"interfaces": ["eth0", "lo"]}, "interface"),
        ({"route": True}, "default route"),
        ({"v6": 1}, "IPv6"),
    ],
)
def test_no_network_namespace_must_be_loopback_only(changes, expected):
    result = grade("no-network", evidence("no-network", **changes))
    assert result.status is CheckStatus.FAIL
    assert any(expected in line for line in result.evidence)


def test_unproven_denial_is_skipped_not_passed():
    measured = evidence(
        "no-network",
        probes=[
            {"kind": "tcp", "control": REACHABLE, "subject": DENIED},
            {"kind": "udp", "control": REACHABLE, "subject": TIMEOUT},
        ],
    )
    result = grade("no-network", measured)
    assert result.status is CheckStatus.SKIP
    assert any("udp" in line and "not proven" in line for line in result.evidence)


def test_probe_kinds_without_a_working_control_are_not_evidence():
    # Podman refuses unprivileged ICMP sockets in every mode; that is not a denial.
    measured = evidence(
        "no-network",
        probes=[
            {"kind": "tcp", "control": REACHABLE, "subject": DENIED},
            {"kind": "icmp", "control": EACCES, "subject": EACCES},
        ],
    )
    result = grade("no-network", measured)
    assert result.status is CheckStatus.PASS
    assert result.measured["counted_probes"] == ["tcp"]


def test_unreachable_controlled_sink_skips():
    measured = evidence(
        "no-network",
        probes=[{"kind": "tcp", "control": TIMEOUT, "subject": DENIED}],
    )
    result = grade("no-network", measured)
    assert result.status is CheckStatus.SKIP
    assert any("sink" in line for line in result.evidence)


@pytest.mark.parametrize(
    "mode, changes",
    [
        ("public", {"network_mode": "none"}),
        ("no-network", {"network_mode": "bridge"}),
        ("no-network", {"interfaces": ["eth0", "lo"]}),
        ("no-network", {"route": True}),
        ("no-network", {"v6": 1}),
    ],
)
def test_failed_tcp_control_does_not_hide_namespace_violations(mode, changes):
    measured = evidence(
        mode,
        probes=[{"kind": "tcp", "control": TIMEOUT, "subject": TIMEOUT}],
        **changes,
    )
    assert grade(mode, measured).status is CheckStatus.FAIL


def test_failed_tcp_control_does_not_hide_a_counted_udp_leak():
    measured = evidence(
        "no-network",
        probes=[
            {"kind": "tcp", "control": TIMEOUT, "subject": DENIED},
            {"kind": "udp", "control": REACHABLE, "subject": REACHABLE},
        ],
    )
    result = grade("no-network", measured)
    assert result.status is CheckStatus.FAIL
    assert any("udp" in line and "reachable" in line for line in result.evidence)


@pytest.mark.parametrize("subject_outcome", [DENIED, TIMEOUT])
def test_public_requires_egress_from_the_subject_namespace(subject_outcome):
    measured = evidence(
        "public",
        probes=[{"kind": "tcp", "control": REACHABLE, "subject": subject_outcome}],
    )
    result = grade("public", measured)
    expected = CheckStatus.FAIL if subject_outcome is DENIED else CheckStatus.SKIP
    assert result.status is expected


def test_public_subject_on_no_network_fails():
    result = grade("public", evidence("public", network_mode="none"))
    assert result.status is CheckStatus.FAIL


def test_evidence_for_a_different_mode_fails():
    result = grade("no-network", evidence("public"))
    assert result.status is CheckStatus.FAIL


def test_missing_evidence_skips_and_malformed_evidence_fails():
    assert grade("no-network", None).status is CheckStatus.SKIP
    broken = grade("no-network", {"schema_version": 1, "probes": "nope"})
    assert broken.status is CheckStatus.FAIL
