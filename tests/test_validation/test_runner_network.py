"""The runner measures the network only through providers that enforce it."""

import shutil
from dataclasses import dataclass, field

from conftest import FIXTURES
from openenv.validation import runner
from openenv.validation.providers import ProviderError
from openenv.validation.runner import run_validation
from openenv.validation.types import CheckStatus, Level, ProviderCapability
from support.runtime import evidence, FakeRunningSubject, FakeRuntimeProvider

ENFORCING = frozenset(
    {
        ProviderCapability.IMAGE_BUILD,
        ProviderCapability.EXEC,
        ProviderCapability.NETWORK_POLICY,
    }
)


def isolated_evidence():
    return {
        "schema_version": 1,
        "requested_mode": "no-network",
        "subject_network_mode": "none",
        "namespace": {
            "interfaces": ["lo"],
            "default_route": False,
            "ipv6_non_loopback": 0,
            "listening_ports": [8000],
        },
        "probes": [
            {
                "kind": "tcp",
                "control": {"result": "reachable"},
                "subject": {"result": "denied", "errno": 101},
            }
        ],
    }


@dataclass
class MeasuredSubject(FakeRunningSubject):
    measurement: object = None
    measured: list = field(default_factory=list)

    def measure_network(self, timeout_s: float = 120) -> dict:
        self.measured.append(timeout_s)
        if isinstance(self.measurement, Exception):
            raise self.measurement
        return self.measurement


def package(tmp_path, mode):
    root = tmp_path / "subject"
    shutil.copytree(FIXTURES / "runtime" / "served_probe", root)
    manifest = root / "openenv.yaml"
    text = manifest.read_text()
    assert "  execution:" in text
    network = f"  network:\n    mode: {mode}\n"
    if mode == "allowlist":
        network += "    allowed_hosts: [example.com]\n"
    manifest.write_text(text.replace("  execution:", network + "  execution:", 1))
    return root


def validate(tmp_path, monkeypatch, mode, subject, capabilities=ENFORCING, modes=None):
    monkeypatch.setattr(runner, "collect_runtime_evidence", lambda *a, **k: evidence())
    provider = FakeRuntimeProvider(
        capabilities=capabilities,
        supported_network_modes=modes or frozenset({"public", "no-network"}),
        subject=subject,
    )
    report = run_validation(
        package(tmp_path, mode), max_level=Level.RUNTIME, provider=provider
    )
    return provider, {r.check_id: r for r in report.results}


def test_measured_no_network_evidence_is_graded(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=isolated_evidence())
    provider, results = validate(tmp_path, monkeypatch, "no-network", subject)
    assert provider.launches[0].network.mode == "no-network"
    assert subject.measured
    assert results["runtime.network_policy"].status is CheckStatus.PASS


def test_failed_measurement_leaves_the_check_unproven(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=ProviderError("Control network probe failed"))
    _, results = validate(tmp_path, monkeypatch, "no-network", subject)
    result = results["runtime.network_policy"]
    assert result.status is CheckStatus.SKIP
    assert "unavailable" in " ".join(result.evidence)


def test_provider_without_enforcement_is_not_asked_to_measure(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=isolated_evidence())
    _, results = validate(
        tmp_path,
        monkeypatch,
        "public",
        subject,
        capabilities=frozenset(
            {ProviderCapability.IMAGE_BUILD, ProviderCapability.EXEC}
        ),
        modes=frozenset({"public"}),
    )
    assert subject.measured == []
    result = results["runtime.network_policy"]
    assert result.status is CheckStatus.SKIP
    assert "network_policy" in " ".join(result.evidence)


def test_unsupported_mode_is_refused_before_launch(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=isolated_evidence())
    provider, results = validate(
        tmp_path, monkeypatch, "no-network", subject, modes=frozenset({"public"})
    )
    assert provider.launches == []
    assert results["runtime.startup"].status is CheckStatus.SKIP


def test_allowlist_keeps_a_named_network_policy_skip(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=None)
    provider, results = validate(tmp_path, monkeypatch, "allowlist", subject)
    assert provider.launches == []
    assert subject.measured == []
    assert results["static.manifest"].status is CheckStatus.PASS
    assert results["runtime.startup"].status is CheckStatus.SKIP
    assert results["runtime.network_policy"].status is CheckStatus.SKIP
