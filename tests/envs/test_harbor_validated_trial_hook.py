# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trusted trial hooks run before agent execution, without model or sandbox calls."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

rollout = pytest.importorskip("openenv.harbor.rollout")
runner = pytest.importorskip("openenv.harbor.runner")
trial_module = pytest.importorskip("harbor.trial.trial")

from openenv.core.harness.capture.sessions import SessionRegistry
from openenv.harbor import proc_env_context, shared_template
from openenv.harbor.models import HarborRolloutResult

ENVIRONMENT_KWARGS = {"auto_mounting": False, "keep_containers": True}


@pytest.mark.parametrize("environment_kwargs", [None, {}, ENVIRONMENT_KWARGS])
@pytest.mark.parametrize("keep_sandbox", [False, True])
def test_trial_config_environment_kwargs(
    monkeypatch, tmp_path, environment_kwargs, keep_sandbox
):
    monkeypatch.delenv("OPENENV_HARBOR_AGENT_VERSIONS", raising=False)
    options = (
        {} if environment_kwargs is None else {"environment_kwargs": environment_kwargs}
    )
    config = rollout.build_trial_config(
        task_dir=tmp_path,
        harness="opencode",
        sandbox="docker",
        intercept_url="http://capture.invalid",
        session_id="test",
        model="m",
        trial_name="test",
        trials_dir=tmp_path,
        keep_sandbox=keep_sandbox,
        **options,
    )

    assert config.environment.kwargs == (environment_kwargs or {})
    assert ("kwargs" in config.environment.model_fields_set) is (
        environment_kwargs is not None
    )
    assert config.environment.delete is (not keep_sandbox)


@pytest.mark.parametrize("credential_path", ["agent-env", "context-env", "locked-env"])
@pytest.mark.parametrize("hook_mode", ["attach", "reject", "omitted"])
def test_trial_hook_precedes_execution(
    monkeypatch, tmp_path, credential_path, hook_mode
):
    # Restore the global proxy as well as its install state after exercising the real contexts.
    monkeypatch.setattr(os, "environ", os.environ)
    monkeypatch.setattr(proc_env_context, "_installed", False)
    monkeypatch.setattr(proc_env_context, "_real_environ", None)
    monkeypatch.setenv(
        "OPENENV_CONCURRENT_PROC_ENV", "0" if credential_path == "locked-env" else "1"
    )
    env_key = "OPENENV_TRIAL_HOOK_TEST_KEY"
    monkeypatch.setenv(env_key, "baseline")
    proc_env = {} if credential_path == "agent-env" else {env_key: "trial-value"}
    expected_env = proc_env.get(env_key, "baseline")
    seam = SimpleNamespace(
        name="fake",
        resolve=lambda **kwargs: ("m", {}, {}, proc_env),
    )
    monkeypatch.setattr(rollout.seams, "get", lambda *args, **kwargs: seam)
    monkeypatch.setattr(shared_template, "enable_shared_templates", lambda: False)
    config = object()
    build = Mock(return_value=config)
    monkeypatch.setattr(rollout, "build_trial_config", build)
    monkeypatch.setattr(
        rollout,
        "export_session",
        lambda *args, **kwargs: {"stats": {}, "sequences": [], "turns": []},
    )
    monkeypatch.setattr(rollout, "load_trace", lambda path: (None, ""))
    monkeypatch.setattr(
        rollout,
        "reconcile",
        lambda *args: SimpleNamespace(ok=True, findings=[], aux_node_ids=[]),
    )
    events = []
    owned_trials = []
    agent_run = AsyncMock()

    class Trial:
        guarded = False
        _close_logger_handler = trial_module.Trial._close_logger_handler

        def __init__(self):
            self.config = SimpleNamespace(trial_name=tmp_path.name)
            self.paths = SimpleNamespace(log_path=tmp_path / "trial.log")
            trial_module.Trial._init_logger(self)

        @classmethod
        async def create(cls, trial_config):
            assert trial_config is config
            assert os.environ[env_key] == expected_env
            events.append("create")
            return trial

        async def run(self):
            assert os.environ[env_key] == expected_env
            assert hook_mode == "omitted" or self.guarded
            events.append("run")
            await agent_run()
            self._close_logger_handler()
            return SimpleNamespace(
                verifier_result=SimpleNamespace(rewards={"reward": 1.0})
            )

    trial = Trial()
    handler = trial._log_handler
    stream = handler.stream
    monkeypatch.setattr(trial_module, "Trial", Trial)

    def on_trial_created(created):
        assert created is trial
        assert os.environ[env_key] == expected_env
        events.append("hook")
        owned_trials.append(created)
        if hook_mode == "reject":
            raise ValueError("image guard rejected trial")
        created.guarded = True

    registry = SessionRegistry()
    options = (
        {}
        if hook_mode == "omitted"
        else {
            "on_trial_created": on_trial_created,
            "environment_kwargs": ENVIRONMENT_KWARGS,
        }
    )
    result = asyncio.run(
        rollout.run_rollout(
            task_dir=tmp_path,
            harness="fake",
            sandbox="docker",
            registry=registry,
            intercept_url="http://capture.invalid",
            model="m",
            trials_dir=tmp_path,
            purpose="eval",
            capture_level="text",
            agent_timeout_sec=12.5,
            agent_step_limit=3,
            **options,
        )
    )

    try:
        assert stream.closed
        assert handler not in trial.logger.handlers
        assert trial._log_handler is None
    finally:
        trial._close_logger_handler()

    assert build.call_args.kwargs["agent_timeout_sec"] == 12.5
    assert build.call_args.kwargs["agent_step_limit"] == 3
    assert build.call_args.kwargs["environment_kwargs"] is (
        None if hook_mode == "omitted" else ENVIRONMENT_KWARGS
    )
    assert registry.list_ids() == []
    assert os.environ[env_key] == "baseline"
    assert not rollout._PROC_ENV_LOCK.locked()
    assert owned_trials == ([] if hook_mode == "omitted" else [trial])
    if hook_mode == "reject":
        assert events == ["create", "hook"]
        agent_run.assert_not_awaited()
        assert result.ok is False
        assert result.reward is None
        assert result.error == "image guard rejected trial"
        assert result.exception_type == "ValueError"
    else:
        assert events == (
            ["create", "run"] if hook_mode == "omitted" else ["create", "hook", "run"]
        )
        agent_run.assert_awaited_once()
        assert result.ok is True
        assert result.reward == 1.0


@pytest.mark.parametrize("bounded", [False, True])
def test_batch_forwards_trial_hook_and_agent_budgets(monkeypatch, tmp_path, bounded):
    monkeypatch.setattr(
        "openenv.harbor.startup.prepare",
        lambda **kwargs: SimpleNamespace(
            llm={"model": "m", "capture_level": "text"},
            available_sandboxes=("docker",),
        ),
    )
    monkeypatch.setattr(
        runner, "resolve_task_dirs", lambda dataset: [tmp_path, tmp_path]
    )
    capture = SimpleNamespace(
        registry=object(), inference=None, start=Mock(), stop=Mock()
    )
    monkeypatch.setattr(runner, "CaptureServer", lambda **kwargs: capture)
    forwarder = SimpleNamespace(
        name="direct", start=Mock(return_value="http://capture.invalid"), stop=Mock()
    )
    monkeypatch.setattr(
        "openenv.core.harness.capture.forwarding.make_forwarder",
        lambda expose: forwarder,
    )
    run = AsyncMock(return_value=HarborRolloutResult(rollout_type="eval"))
    monkeypatch.setattr(runner, "run_rollout", run)
    hook = Mock()
    options = (
        {
            "on_trial_created": hook,
            "agent_timeout_sec": 12.5,
            "agent_step_limit": 3,
            "environment_kwargs": ENVIRONMENT_KWARGS,
        }
        if bounded
        else {}
    )

    results = asyncio.run(
        runner.run_batch(
            llm_url="http://llm.invalid/v1",
            dataset="local",
            task_indices=[0, 1],
            sandbox="docker",
            expose="direct",
            trials_dir=tmp_path,
            **options,
        )
    )

    assert len(results) == 2
    assert run.await_count == 2
    for call in run.await_args_list:
        assert call.kwargs["on_trial_created"] is (hook if bounded else None)
        assert call.kwargs["agent_timeout_sec"] == (12.5 if bounded else None)
        assert call.kwargs["agent_step_limit"] == (3 if bounded else None)
        assert call.kwargs["environment_kwargs"] is (
            ENVIRONMENT_KWARGS if bounded else None
        )
    capture.stop.assert_called_once()
    forwarder.stop.assert_called_once()
