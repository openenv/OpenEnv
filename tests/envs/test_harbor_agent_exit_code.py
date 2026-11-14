# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A non-zero agent exit after a graded run is a warning, not a failed rollout.

`opencode run` has been seen to exit 1 on runs that finished normally, and Harbor records that as
`NonZeroAgentExitCodeError` while still running the verifier. Reporting those rollouts as failed
made `openenv harbor rollout` exit 1 on good batches. A run the verifier never graded, or one the
agent's own output says was cut short, still fails.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

rollout = pytest.importorskip("openenv.harbor.rollout")
trial_module = pytest.importorskip("harbor.trial.trial")

from openenv.core.harness.capture.sessions import SessionRegistry  # noqa: E402

_OPENCODE_EXIT = (
    "Command failed (exit 1): [ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; opencode "
    "--model=intercepted/m run --format=json --thinking --dangerously-skip-permissions -- 'task'\n"
    'stdout: {"type":"step_finish","part":{"reason":"stop"}}\n'
    "stderr: None"
)


def _run(
    monkeypatch, tmp_path, *, rewards, exception_type, message, clean_capture=True
):
    class Trial:
        @classmethod
        async def create(cls, config):
            return cls()

        async def run(self):
            return SimpleNamespace(
                verifier_result=(
                    SimpleNamespace(rewards=rewards) if rewards is not None else None
                ),
                exception_info=SimpleNamespace(
                    exception_type=exception_type, exception_message=message
                ),
            )

    monkeypatch.setattr(trial_module, "Trial", Trial)
    monkeypatch.setattr(rollout, "build_trial_config", lambda **kwargs: None)
    if clean_capture:
        # A complete, validated capture, so only the exit-code handling decides `ok`.
        monkeypatch.setattr(
            rollout,
            "export_session",
            lambda session, **kwargs: {
                "stats": {"n_turns": 2, "n_roots": 1},
                "validation": [],
                "sequences": [],
                "turns": [],
            },
        )
        monkeypatch.setattr(rollout, "load_trace", lambda trial_dir: (None, ""))
        monkeypatch.setattr(
            rollout,
            "reconcile",
            lambda document, atif: SimpleNamespace(
                ok=True, findings=[], aux_node_ids=[]
            ),
        )
    return asyncio.run(
        rollout.run_rollout(
            task_dir=tmp_path,
            harness="opencode",
            sandbox="e2b",
            registry=SessionRegistry(),
            intercept_url="http://127.0.0.1:9",
            model="m",
            trials_dir=tmp_path,
            purpose="eval",
            capture_level="text",
        )
    )


@pytest.mark.parametrize("reward", [1.0, 0.0])
def test_graded_nonzero_exit_is_ok_with_a_warning(monkeypatch, tmp_path, reward):
    result = _run(
        monkeypatch,
        tmp_path,
        rewards={"reward": reward},
        exception_type="NonZeroAgentExitCodeError",
        message=_OPENCODE_EXIT,
    )
    assert result.ok is True
    assert result.reward == reward
    assert result.error is None and result.exception_type is None
    warnings = [f for f in result.findings if "agent exited non-zero" in f]
    assert len(warnings) == 1 and warnings[0].startswith("[WARN]")
    assert "NonZeroAgentExitCodeError: Command failed (exit 1)" in warnings[0]


def test_ungraded_nonzero_exit_still_fails(monkeypatch, tmp_path):
    result = _run(
        monkeypatch,
        tmp_path,
        rewards=None,
        exception_type="NonZeroAgentExitCodeError",
        message=_OPENCODE_EXIT,
    )
    assert result.ok is False
    assert result.exception_type == "NonZeroAgentExitCodeError"
    assert result.error.startswith("Command failed (exit 1)")


@pytest.mark.parametrize(
    "exception_type,message",
    [
        # Harbor's classification of a provider failure: a subclass, so a different name.
        ("ApiRateLimitError", "Command failed (exit 1): opencode run ...\nstdout: 429"),
        # opencode exited 0 but reported an error event, which Harbor raises itself.
        (
            "NonZeroAgentExitCodeError",
            "OpenCode emitted error event(s): provider returned 500",
        ),
        # opencode exited 1 BECAUSE of an error event in its own stream.
        (
            "NonZeroAgentExitCodeError",
            'Command failed (exit 1): opencode run ...\nstdout: {"type":"error",'
            '"error":{"name":"APIError"}}\nstderr: None',
        ),
        ("AgentTimeoutError", "Agent execution timed out after 600 seconds"),
    ],
)
def test_other_agent_failures_still_fail_even_when_graded(
    monkeypatch, tmp_path, exception_type, message
):
    result = _run(
        monkeypatch,
        tmp_path,
        rewards={"reward": 1.0},
        exception_type=exception_type,
        message=message,
    )
    assert result.ok is False
    assert result.exception_type == exception_type
    assert not any("agent exited non-zero" in f for f in result.findings)


def test_graded_nonzero_exit_with_an_empty_capture_still_fails(monkeypatch, tmp_path):
    """The capture gates still apply: a graded run with no model calls is not trainable data."""
    result = _run(
        monkeypatch,
        tmp_path,
        rewards={"reward": 1.0},
        exception_type="NonZeroAgentExitCodeError",
        message=_OPENCODE_EXIT,
        clean_capture=False,
    )
    assert result.ok is False
    assert result.exception_type is None
    assert result.error and "NonZeroAgentExitCodeError" not in result.error
