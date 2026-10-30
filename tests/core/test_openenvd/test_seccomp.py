# SPDX-License-Identifier: BSD-3-Clause

import json

import pytest
from openenv.core.openenvd.seccomp import (
    CLONE_NAMESPACE_FLAGS,
    DENIED_SYSCALLS,
    seccomp_architectures,
    seccomp_profile,
)


def _rules(profile, name):
    return [r for r in profile["syscalls"] if name in r["names"]]


def test_profile_allows_by_default_and_denies_listed_syscalls():
    profile = seccomp_profile(arch="x86_64")
    assert profile["defaultAction"] == "SCMP_ACT_ALLOW"
    assert profile["architectures"] == ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86"]
    for name in ("ptrace", "bpf", "mount", "unshare", "setns", "userfaultfd", "fsopen"):
        assert name in DENIED_SYSCALLS
        (rule,) = _rules(profile, name)
        assert rule["action"] == "SCMP_ACT_ERRNO" and rule["errnoRet"] == 1
        assert "args" not in rule
    json.loads(json.dumps(profile))


def test_clone3_returns_enosys():
    profile = seccomp_profile(arch="aarch64")
    assert profile["architectures"] == ["SCMP_ARCH_AARCH64"]
    (rule,) = _rules(profile, "clone3")
    assert rule["errnoRet"] == 38


def test_clone_denied_per_namespace_flag():
    profile = seccomp_profile(arch="x86_64")
    rules = _rules(profile, "clone")
    masks = sorted(r["args"][0]["value"] for r in rules)
    assert masks == sorted(CLONE_NAMESPACE_FLAGS.values())
    for rule in rules:
        (arg,) = rule["args"]
        assert arg["index"] == 0
        assert arg["op"] == "SCMP_CMP_MASKED_EQ"
        assert arg["value"] == arg["valueTwo"]
    assert CLONE_NAMESPACE_FLAGS["CLONE_NEWUSER"] == 0x10000000


def test_container_profile_only_denies_af_packet():
    profile = seccomp_profile(arch="x86_64")
    families = [r["args"][0]["value"] for r in _rules(profile, "socket")]
    assert families == [17]


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError):
        seccomp_architectures("riscv64")
