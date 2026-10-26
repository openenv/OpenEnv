# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation on ingest, and the capability checks that run before a rollout starts.

Both exist to convert a silent wrong answer into a loud one. A turn whose logprobs are misaligned
has to be caught while we still know which turn it was; a sandbox that cannot be constructed has to
be caught before it is offered rather than 90 seconds into a run.
"""

from __future__ import annotations

import sys
import types

import pytest

validate = pytest.importorskip("openenv.core.harness.capture.validate")
capabilities = pytest.importorskip("openenv.harbor.capabilities")

check_turn = validate.check_turn


def codes(report) -> set[str]:
    return {f.code for f in report.findings}


# --- per-turn validation ----------------------------------------------------
def test_a_well_formed_turn_passes():
    report = check_turn([1, 2, 3], [4, 5], [-0.1, -0.2], finish_reason="stop")
    assert report.ok


def test_missing_prompt_ids_is_fatal():
    """The endpoint was started without token-id capture: every rebuilt row would be empty."""
    report = check_turn([], [4], [-0.1])
    assert not report.ok
    assert "no_prompt_ids" in codes(report)


def test_logprob_count_must_match_sampled_count():
    """Off-by-one here silently trains on the wrong token's probability."""
    report = check_turn([1], [4, 5, 6], [-0.1, -0.2])
    assert not report.ok


def test_a_turn_that_sampled_nothing_is_reported_but_not_fatal():
    """Reported, not rejected: a model can legitimately stop without emitting a token.

    `ok` means usable, so an empty completion warns rather than invalidating the rollout. It still
    has to be visible, because a run of these means the agent is looping without producing anything.
    """
    report = check_turn([1, 2], [], [])
    assert report.ok
    assert "no_sampled_ids" in codes(report)


def test_findings_name_the_turn():
    """A misalignment is only actionable if you know which call produced it."""
    report = check_turn([], [1], [-0.1], index=7)
    assert any("turn 7" in str(f) for f in report.findings)


# --- sandbox capability -----------------------------------------------------
def _module_with(**flags):
    module = types.ModuleType("fake_backend_module")
    for key, value in flags.items():
        setattr(module, key, value)
    sys.modules[module.__name__] = module
    return module


def test_missing_sdk_is_detected_from_the_backends_own_flag():
    """Harbor guards each SDK with a module-level `_HAS_X` and raises from `__init__`.

    So the module imports, the class loads, and the check passes, with the failure arriving only
    once a rollout tries to build a sandbox, where it reads as a broken rollout rather than a
    missing dependency. This is the case that shipped a Space offering `e2b` it could never run.
    """
    module = _module_with(_HAS_E2B=False)
    cls = type("E2BEnvironment", (), {"__module__": module.__name__})
    detail = capabilities._missing_sdk(cls)
    assert detail and "e2b" in detail
    assert "harbor[cloud]" in detail or "openenv[harbor]" in detail


def test_present_sdk_reports_nothing():
    module = _module_with(_HAS_E2B=True)
    cls = type("E2BEnvironment", (), {"__module__": module.__name__})
    assert capabilities._missing_sdk(cls) == ""


def test_a_backend_without_flags_is_not_assumed_broken():
    module = _module_with(SOMETHING_ELSE=1)
    cls = type("Plain", (), {"__module__": module.__name__})
    assert capabilities._missing_sdk(cls) == ""


def test_several_missing_extras_are_all_named():
    module = _module_with(_HAS_MODAL=False, _HAS_DOCKERFILE_PARSE=False)
    cls = type("ModalEnvironment", (), {"__module__": module.__name__})
    detail = capabilities._missing_sdk(cls)
    assert "modal" in detail and "dockerfile_parse" in detail


def test_an_unimportable_module_is_not_a_crash():
    cls = type("Ghost", (), {"__module__": "module.that.does.not.exist"})
    assert capabilities._missing_sdk(cls) == ""


def test_unknown_sandbox_names_are_rejected_by_name():
    status = capabilities.check_sandbox("not-a-real-backend")
    assert status.available is False
    assert "unknown" in status.detail.lower() or "harbor" in status.detail.lower()


# --- capability reporting ---------------------------------------------------
def test_render_says_why_a_sandbox_is_unavailable():
    """The commonest cause of a rollout dying 90s in is a missing key, so it belongs at startup."""
    caps = capabilities.Capabilities(
        sandboxes=[
            capabilities.SandboxStatus("e2b", True),
            capabilities.SandboxStatus("daytona", False, "DAYTONA_API_KEY is not set"),
        ]
    )
    out = caps.render()
    assert "DAYTONA_API_KEY is not set" in out
    assert caps.available_sandboxes == ["e2b"]


def test_render_warns_when_nothing_is_usable():
    caps = capabilities.Capabilities(
        sandboxes=[capabilities.SandboxStatus("e2b", False, "no key")]
    )
    assert "WARNING" in caps.render()


def test_capabilities_serialise_for_the_wire():
    caps = capabilities.Capabilities(
        sandboxes=[capabilities.SandboxStatus("e2b", True)],
        llm={"model": "m", "ok": True},
    )
    payload = caps.to_dict()
    assert set(payload) == {"harnesses", "sandboxes", "datasets", "llm"}
    assert payload["llm"]["ok"] is True


def test_the_install_hint_does_not_point_at_the_unsatisfiable_extra():
    """`harbor[cloud]` cannot be installed: langsmith and tensorlake demand incompatible websockets.

    Both pyprojects avoid it for that reason, so an error message telling someone to install it
    would send them straight to a resolver failure.
    """
    module = _module_with(_HAS_DAYTONA=False)
    cls = type("DaytonaEnvironment", (), {"__module__": module.__name__})
    detail = capabilities._missing_sdk(cls)
    assert "harbor[cloud]" not in detail
    assert "openenv[harbor]" in detail
