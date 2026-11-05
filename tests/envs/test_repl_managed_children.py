"""Unit contracts and real-process cleanup coverage for managed child queries."""

import functools
import os
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("smolagents")

import repl_env.runner as runner_module
from repl_env import LocalREPLEnv, LocalRLMRunner
from repl_env.managed_children import ChildCleanupError, ManagedChildConfig
from repl_env.recursive_controller import create_server_recursive_controller


pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="POSIX worker supervision"
)


def fake_chat_factory(directory):
    def chat(messages, model=None):
        Path(directory, f"{os.getpid()}.started").touch()
        prompt = messages[-1]["content"]
        if "blocked" in prompt:
            if "ignore-term" in prompt:
                signal.signal(
                    signal.SIGTERM,
                    lambda *_: Path(directory, f"{os.getpid()}.term").touch(),
                )
            Path(directory, f"{os.getpid()}.ready").touch()
            while not Path(directory, "release").exists():
                time.sleep(0.01)
        if "raise-error" in prompt:
            raise ValueError("scripted model failure")
        if "launch-nested" in prompt:
            return "```repl\nchild = rlm_query('blocked ignore-term', blocking=False); llm_query('blocked ignore-term')\n```"
        if "nested" in prompt:
            return "```repl\nprint(FINAL(rlm_query('answer')))\n```"
        return "```repl\nprint(FINAL(42))\n```"

    return chat


def make_controller(tmp_path, **kwargs):
    config = ManagedChildConfig(
        chat_factory=functools.partial(fake_chat_factory, str(tmp_path)),
        shutdown_grace_s=0.1,
        terminate_grace_s=0.15,
        kill_timeout_s=2,
    )
    return create_server_recursive_controller(
        fake_chat_factory(str(tmp_path)),
        managed_children=config,
        max_depth=kwargs.pop("max_depth", 3),
        max_iterations=2,
        **kwargs,
    )


def wait_ready(tmp_path, count=1, *, manager=None, phase="ready"):
    deadline = time.monotonic() + 30 * count
    details = []
    while time.monotonic() < deadline:
        files = list(tmp_path.glob(f"*.{phase}"))
        if len(files) >= count:
            return [int(path.stem) for path in files]
        if manager is not None:
            with manager._lock:
                entries = list(manager._jobs.items())
            jobs = [job for _, job in entries]
            details = [
                (job.pid, manager.status(handle), job.exitcode, job.error)
                for handle, job in entries
            ]
            if any(
                job.finished.is_set() and job.status in ("failed", "cleanup_failed")
                for job in jobs
            ):
                pytest.fail(f"Worker failed before {phase}: {details}")
        time.sleep(0.01)
    pytest.fail(f"Expected {count} workers at {phase}; found {len(files)}: {details}")


@pytest.fixture
def fault_controller(tmp_path):
    controller = make_controller(tmp_path)
    yield controller
    try:
        controller.close()
    finally:
        for job in controller.children._jobs.values():
            if job.exitcode is None:
                job.process.kill()
                job.process.join(5)
                job.exitcode = job.process.exitcode
                assert job.exitcode is not None
                if job.thread is not None and job.thread.ident is not None:
                    job.thread.join(5)
                job.channel.close()
                job.shutdown.close()
                job.process.close()


def assert_reaped(pids):
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


def start_call(function, name):
    done = threading.Event()
    errors = []

    def invoke():
        try:
            function()
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=invoke, name=name)
    thread.start()
    return thread, done, errors


@pytest.mark.parametrize(
    "model_argument, expected_model",
    [
        ("", None),
        (", 'fake-model'", "fake-model"),
        (", model='fake-model'", "fake-model"),
    ],
)
def test_nonblocking_query_routes_model_arguments(
    tmp_path, monkeypatch, model_argument, expected_model
):
    controller = make_controller(tmp_path)
    spawn = Mock(return_value="child-handle")
    wait = Mock(return_value="42")
    monkeypatch.setattr(controller.children, "spawn", spawn)
    monkeypatch.setattr(controller.children, "wait", wait)
    monkeypatch.setattr(controller.children, "status", Mock(return_value="completed"))
    monkeypatch.setattr(controller.children, "result", Mock(return_value="42"))
    with LocalREPLEnv(recursive_controller=controller) as env:
        reset = env.reset()
        assert "blocking=False" in reset.observation.result.stdout
        assert "rlm_spawn" not in env.list_variables()
        result = env.execute(
            f"child = rlm_query('answer'{model_argument}, blocking=False); parent_work = 6 * 7"
        )
        assert result.observation.result.success
        spawn.assert_called_once_with("answer", expected_model)
        assert (
            env.execute("print(parent_work)").observation.result.stdout.strip() == "42"
        )
        result = env.execute("print(rlm_wait(child, timeout_s=15))")
        assert result.observation.result.stdout.strip() == "42"
        wait.assert_called_once_with("child-handle", timeout_s=15)
        assert (
            env.execute("print(rlm_status(child))").observation.result.stdout.strip()
            == "completed"
        )
        assert (
            env.execute("print(rlm_result(child))").observation.result.stdout.strip()
            == "42"
        )
        assert not controller.children._jobs


def test_nonblocking_query_runs_supervised_worker(tmp_path):
    controller = make_controller(tmp_path)
    with LocalREPLEnv(recursive_controller=controller) as env:
        env.reset()
        result = env.execute(
            "child = rlm_query('answer', model='fake-model', blocking=False); parent_work = 6 * 7"
        )
        assert result.observation.result.success
        result = env.execute("print(parent_work, rlm_wait(child, timeout_s=15))")
        assert result.observation.result.success
        assert result.observation.result.stdout.strip() == "42 42"
        result = env.execute("print(rlm_status(child), rlm_result(child))")
        assert result.observation.result.stdout.strip() == "completed 42"
        job = next(iter(controller.children._jobs.values()))
        assert job.model == "fake-model"
        assert job.pid != os.getpid()
        assert job.finished.is_set()
        assert_reaped([job.pid])


@pytest.mark.parametrize("blocking_argument", ["", ", blocking=True"])
def test_blocking_query_returns_answer_by_default(
    tmp_path, monkeypatch, blocking_argument
):
    controller = make_controller(tmp_path)
    query = Mock(return_value="42")
    monkeypatch.setattr(controller, "rlm_query_fn", query)
    with LocalREPLEnv(recursive_controller=controller) as env:
        env.reset()
        result = env.execute(
            f"print(rlm_query('answer', 'fake-model'{blocking_argument}))"
        )
        assert result.observation.result.success
        assert result.observation.result.stdout.strip() == "42"
        query.assert_called_once_with("answer", "fake-model")
        assert not controller.children._jobs


@pytest.mark.parametrize("blocking_argument", ["", ", blocking=True"])
@pytest.mark.parametrize("recursive", [False, True])
def test_unmanaged_blocking_query_keeps_existing_call_signature(
    blocking_argument, recursive
):
    calls = []

    def query(prompt, model=None):
        calls.append((prompt, model))
        return "answer"

    with LocalREPLEnv(
        llm_query_fn=query, subcall_fn=query if recursive else None
    ) as env:
        env.reset()
        result = env.execute(
            f"print(rlm_query('question', 'existing-model'{blocking_argument}))"
        )
        assert result.observation.result.success
        assert result.observation.result.stdout.strip() == "answer"
        assert calls == [("question", "existing-model")]


@pytest.mark.parametrize("recursive", [False, True])
def test_nonblocking_query_requires_managed_mode(recursive):
    calls = []

    def query(prompt, model=None):
        calls.append((prompt, model))
        return "answer"

    with LocalREPLEnv(
        llm_query_fn=query, subcall_fn=query if recursive else None
    ) as env:
        env.reset()
        result = env.execute("rlm_query('question', blocking=False)")
        assert not result.observation.result.success
        assert "requires managed children" in result.observation.result.stderr
        assert calls == []


def test_blocking_option_is_keyword_only():
    calls = []
    with LocalREPLEnv(llm_query_fn=lambda *args: calls.append(args)) as env:
        env.reset()
        result = env.execute("rlm_query('question', None, False)")
        assert not result.observation.result.success
        assert "positional" in result.observation.result.stderr
        assert calls == []


@pytest.mark.parametrize("limit", ["max_children_total", "max_batch_workers"])
def test_nonblocking_query_respects_supervisor_limits(tmp_path, monkeypatch, limit):
    controller = make_controller(
        tmp_path, **{limit: 0 if limit == "max_children_total" else 1}
    )
    launcher = Mock(side_effect=AssertionError("Unexpected worker launch"))
    monkeypatch.setattr(
        "repl_env.managed_children.multiprocessing.get_context", launcher
    )
    with LocalREPLEnv(recursive_controller=controller) as env:
        env.reset()
        if limit == "max_batch_workers":
            controller.children._jobs["running-child"] = SimpleNamespace(
                finished=threading.Event()
            )
        admitted = len(controller.children._jobs)
        try:
            result = env.execute("rlm_query('answer', blocking=False)")
            assert not result.observation.result.success
            expected_error = (
                "max_children_total" if limit == "max_children_total" else "concurrency"
            )
            assert expected_error in result.observation.result.stderr
            assert len(controller.children._jobs) == admitted
            launcher.assert_not_called()
        finally:
            controller.children._jobs.clear()


def test_nonblocking_query_at_depth_limit_still_uses_worker(tmp_path):
    controller = make_controller(tmp_path, max_depth=1)
    with LocalREPLEnv(recursive_controller=controller) as env:
        env.reset()
        result = env.execute("child = rlm_query('blocked ignore-term', blocking=False)")
        assert result.observation.result.success
        pids = wait_ready(tmp_path, manager=controller.children)
        handle = next(iter(controller.children._jobs))
        with pytest.raises(TimeoutError):
            controller.children.wait(handle, timeout_s=0.01)
        assert controller.children.status(handle) == "running"
        with pytest.raises(RuntimeError, match="not finished"):
            controller.children.result(handle)
        controller.children.cancel(handle)
        with pytest.raises(RuntimeError, match="cancelled"):
            controller.children.result(handle)
        env.close()
        assert_reaped(pids)


@pytest.mark.parametrize(
    "ignore_term, exitcode", [(False, -signal.SIGTERM), (True, -signal.SIGKILL)]
)
def test_reset_stops_blocked_call_and_invalidates_handles(
    tmp_path, ignore_term, exitcode
):
    controller = make_controller(tmp_path)
    with LocalREPLEnv(recursive_controller=controller) as env:
        env.reset(context="old")
        prompt = "blocked ignore-term" if ignore_term else "blocked"
        handle = controller.children.spawn(prompt)
        pids = wait_ready(tmp_path)
        job = controller.children._jobs[handle]
        env.reset(context="new")
        assert job.exitcode == exitcode
        with pytest.raises(ValueError, match="closed"):
            _ = job.process.sentinel
        assert_reaped(pids)
        with pytest.raises(KeyError, match="handle"):
            controller.children.status(handle)
        assert env.state().context == "new"


def test_close_stops_all_workers_and_is_idempotent(tmp_path):
    controller = make_controller(tmp_path)
    env = LocalREPLEnv(recursive_controller=controller)
    env.reset()
    for _ in range(2):
        controller.children.spawn("blocked ignore-term")
    pids = wait_ready(tmp_path, 2)
    env.close()
    assert_reaped(pids)
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_cooperative_shutdown_exits_before_term(tmp_path):
    controller = make_controller(tmp_path)
    controller.children.config = replace(controller.children.config, shutdown_grace_s=2)
    handle = controller.children.spawn("blocked")
    pids = wait_ready(tmp_path)
    job = controller.children._jobs[handle]
    timer = threading.Timer(0.03, lambda: Path(tmp_path, "release").touch())
    timer.start()
    try:
        controller.children.cancel(handle)
        assert job.exitcode == 0
        assert controller.children.status(handle) == "cancelled"
        assert_reaped(pids)
    finally:
        timer.join()
        controller.close()


@pytest.mark.parametrize(
    "final_code",
    ["print(FINAL(42))", "answer.update({'content': '42', 'ready': True})"],
)
def test_parent_finalization_stops_children(tmp_path, final_code):
    controller = make_controller(tmp_path)
    with LocalREPLEnv(recursive_controller=controller) as env:
        env.reset()
        controller.children.spawn("blocked ignore-term")
        pids = wait_ready(tmp_path)
        assert env.execute(final_code).done
        assert_reaped(pids)
        assert not env.execute(
            "rlm_query('answer', blocking=False)"
        ).observation.result.success


@pytest.mark.parametrize(
    "final_code",
    ["print(FINAL(42))", "answer.update({'content': '42', 'ready': True})"],
)
def test_final_answer_survives_cleanup_failure(tmp_path, monkeypatch, final_code):
    controller = make_controller(tmp_path)
    manager = controller.children
    env = LocalREPLEnv(recursive_controller=controller)
    env.reset(context="original episode")
    finished = Mock(spec=threading.Event)
    finished.wait.return_value = False
    job = SimpleNamespace(
        stop=threading.Event(), finished=finished, status="running", exitcode=None
    )
    manager._jobs["unconfirmed-child"] = job
    launcher = Mock(side_effect=AssertionError("Unexpected worker launch"))
    monkeypatch.setattr(
        "repl_env.managed_children.multiprocessing.get_context", launcher
    )
    try:
        with pytest.raises(ChildCleanupError, match="exit could not be confirmed"):
            env.execute(final_code)
        assert env.state().final_answer == "42"
        assert manager._cleanup_failed
        assert job.stop.is_set()
        finished.wait.assert_called_once()
        with pytest.raises(ChildCleanupError, match="cannot be reused"):
            env.execute("print('must not execute')")
        with pytest.raises(ChildCleanupError, match="cannot be reused"):
            manager.spawn("answer")
        with pytest.raises(ChildCleanupError, match="exit could not be confirmed"):
            env.reset(context="replacement episode")
        assert env.state().context == "original episode"
        assert env.state().final_answer == "42"
        launcher.assert_not_called()
    finally:
        manager._jobs.clear()
        env.close()


def test_iteration_limit_stops_children(tmp_path):
    controller = make_controller(tmp_path)
    with LocalREPLEnv(recursive_controller=controller) as env:
        env.reset(max_iterations=1)
        controller.children.spawn("blocked ignore-term")
        pids = wait_ready(tmp_path)
        assert env.execute("pass").done
        assert_reaped(pids)


def test_worker_failure_is_observable(tmp_path):
    controller = make_controller(tmp_path)
    try:
        failed = controller.children.spawn("raise-error")
        with pytest.raises(RuntimeError, match="scripted model failure"):
            controller.children.wait(failed, timeout_s=15)
        assert controller.children.status(failed) == "failed"
        assert_reaped([controller.children._jobs[failed].pid])
    finally:
        controller.close()


def test_cleanup_failure_prevents_reuse_and_can_be_retried(tmp_path, monkeypatch):
    controller = make_controller(tmp_path)
    completions = []
    callback_done = threading.Event()

    def on_complete(*args):
        completions.append(args)
        callback_done.set()

    controller.children.on_subcall_complete = on_complete
    env = LocalREPLEnv(recursive_controller=controller)
    env.reset(context="old")
    controller.children.spawn("blocked ignore-term")
    pids = wait_ready(tmp_path)
    send_signal = controller.children._signal
    monkeypatch.setattr(controller.children, "_signal", lambda *args: None)
    try:
        with pytest.raises(ChildCleanupError):
            env.reset(context="new")
        assert env.state().context == "old"
        with pytest.raises(ChildCleanupError):
            env.execute("print(42)")
        with pytest.raises(ChildCleanupError):
            controller.children.spawn("answer")
    finally:
        monkeypatch.setattr(controller.children, "_signal", send_signal)
        env.close()
    assert_reaped(pids)
    assert callback_done.wait(5)
    assert len(completions) == 1


def test_nested_recursion_shares_child_budget(tmp_path):
    controller = make_controller(tmp_path, max_children_total=1)
    try:
        handle = controller.children.spawn("nested")
        assert "max_children_total" in controller.children.wait(handle, timeout_s=15)
        with pytest.raises(RuntimeError, match="max_children_total"):
            controller.children.spawn("answer")
    finally:
        controller.close()


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
def test_child_budget_counts_all_admitted_jobs_until_reset(
    tmp_path, monkeypatch, outcome
):
    controller = make_controller(tmp_path, max_children_total=1)
    manager = controller.children
    finished = threading.Event()
    finished.set()
    manager._jobs["prior-child"] = SimpleNamespace(
        finished=finished,
        stop=threading.Event(),
        status=outcome,
        exitcode=0,
    )
    launcher = Mock(side_effect=RuntimeError("Worker launch reached"))
    monkeypatch.setattr(
        "repl_env.managed_children.multiprocessing.get_context", launcher
    )
    try:
        assert manager.status("prior-child") == outcome
        with pytest.raises(RuntimeError, match="max_children_total"):
            manager.spawn("answer")
        launcher.assert_not_called()
        manager.reset()
        with pytest.raises(KeyError, match="handle"):
            manager.status("prior-child")
        with pytest.raises(RuntimeError, match="Worker launch reached"):
            manager.spawn("answer")
        launcher.assert_called_once_with("spawn")
    finally:
        controller.close()


def test_zero_child_budget_never_admits_a_job(tmp_path):
    controller = make_controller(tmp_path, max_children_total=0)
    try:
        with pytest.raises(RuntimeError, match="max_children_total"):
            controller.children.spawn("answer")
        controller.children.reset()
        with pytest.raises(RuntimeError, match="max_children_total"):
            controller.children.spawn("answer")
        assert not controller.children._jobs
    finally:
        controller.close()


def test_runner_reuses_its_managed_controller(tmp_path, monkeypatch):
    created = []
    received = []
    create_controller = runner_module.create_server_recursive_controller
    local_env = runner_module.LocalREPLEnv

    def track_controller(*args, **kwargs):
        controller = create_controller(*args, **kwargs)
        created.append(controller)
        return controller

    def track_environment(*args, **kwargs):
        received.append(kwargs["recursive_controller"])
        return local_env(*args, **kwargs)

    monkeypatch.setattr(
        runner_module, "create_server_recursive_controller", track_controller
    )
    monkeypatch.setattr(runner_module, "LocalREPLEnv", track_environment)
    runner = LocalRLMRunner(
        fake_chat_factory(str(tmp_path)),
        max_iterations=2,
        max_depth=3,
        max_children_total=1,
        managed_children=ManagedChildConfig(
            chat_factory=functools.partial(fake_chat_factory, str(tmp_path)),
        ),
    )
    result = runner.run("", "nested")
    assert result.final_answer == "42"
    assert len(result.child_traces) == 1
    assert len(created) == len(received) == 1
    assert received[0] is created[0]
    assert created[0].backend.limits.max_children_total == 1
    assert created[0].children._closed
    assert all(job.exitcode is not None for job in created[0].children._jobs.values())


def test_nested_recursion_returns_result_from_supervised_worker(tmp_path):
    controller = make_controller(tmp_path)
    try:
        handle = controller.children.spawn("nested")
        wait_ready(tmp_path, 2, manager=controller.children, phase="started")
        assert controller.children.wait(handle, timeout_s=15) == "42"
        assert len(controller.children._jobs) == 2
        assert all(job.exitcode == 0 for job in controller.children._jobs.values())
    finally:
        controller.close()


def test_reset_tracks_and_reaps_nested_managed_workers(tmp_path):
    controller = make_controller(tmp_path, max_depth=4)
    with LocalREPLEnv(recursive_controller=controller) as env:
        env.reset()
        controller.children.spawn("launch-nested")
        pids = wait_ready(tmp_path, 2, manager=controller.children)
        assert len(controller.children._jobs) == 2
        env.reset()
        assert_reaped(pids)


def test_parent_exit_is_not_proof_nested_worker_stopped(tmp_path, monkeypatch):
    controller = make_controller(tmp_path, max_depth=4)
    send_signal = controller.children._signal
    try:
        handle = controller.children.spawn("launch-nested")
        pids = wait_ready(tmp_path, 2, manager=controller.children)
        monkeypatch.setattr(
            controller.children,
            "_signal",
            lambda job, signum: send_signal(job, signum) if job.depth == 1 else None,
        )
        with pytest.raises(ChildCleanupError):
            controller.children.cancel(handle)
        assert controller.children._jobs[handle].exitcode is not None
        with pytest.raises(ChildCleanupError):
            controller.children.spawn("answer")
    finally:
        monkeypatch.setattr(controller.children, "_signal", send_signal)
        controller.close()
    assert_reaped(pids)


def test_watcher_start_failure_reaps_started_worker(
    tmp_path, monkeypatch, fault_controller
):
    manager = fault_controller.children
    original_start = threading.Thread.start

    def start(thread):
        if thread._target == manager._watch:
            wait_ready(tmp_path, manager=manager)
            raise RuntimeError("watcher unavailable")
        original_start(thread)

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", start)
        with pytest.raises(RuntimeError, match="watcher unavailable"):
            manager.spawn("blocked ignore-term")
    job = next(iter(manager._jobs.values()))
    assert job.exitcode == -signal.SIGKILL
    assert job.finished.is_set()
    assert_reaped([job.pid])
    manager.close()


@pytest.mark.parametrize("operation", ["reset", "close", "wait"])
def test_broken_channel_still_stops_and_reaps_worker(
    tmp_path, fault_controller, operation
):
    manager = fault_controller.children
    env = LocalREPLEnv(recursive_controller=fault_controller)
    env.reset()
    handle = manager.spawn("blocked ignore-term")
    pids = wait_ready(tmp_path, manager=manager)
    job = manager._jobs[handle]
    job.channel.close()
    operations = {"reset": env.reset, "close": env.close}
    if operation == "wait":
        assert job.finished.wait(5)
        assert manager.status(handle) == "failed"
    else:
        operations[operation]()
    assert job.finished.is_set()
    assert job.exitcode == -signal.SIGKILL
    assert_reaped(pids)
    if operation == "reset":
        result = env.execute(
            "print(rlm_wait(rlm_query('answer', blocking=False), timeout_s=15))"
        )
        assert result.observation.result.success
        assert result.observation.result.stdout.strip() == "42"


def test_unmanaged_controller_forwards_subcall_callbacks(tmp_path):
    events = []
    controller = create_server_recursive_controller(
        fake_chat_factory(str(tmp_path)),
        max_depth=3,
        max_iterations=2,
        on_subcall_start=lambda *args: events.append("start"),
        on_subcall_complete=lambda *args: events.append("complete"),
    )
    try:
        assert controller.rlm_query_fn("answer") == "42"
        assert events == ["start", "complete"]
    finally:
        controller.close()


def test_callback_dispatcher_start_failure_does_not_leak_worker(
    tmp_path, monkeypatch, fault_controller
):
    manager = fault_controller.children
    manager.on_subcall_start = lambda *args: None
    manager.on_subcall_complete = lambda *args: None
    original_start = threading.Thread.start

    def start(thread):
        if thread._target == manager._run_callbacks:
            raise RuntimeError("observer unavailable")
        original_start(thread)

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", start)
        handle = manager.spawn("blocked ignore-term")
        pids = wait_ready(tmp_path, manager=manager)
        manager.close()
    assert manager._jobs[handle].exitcode == -signal.SIGKILL
    assert manager._callback_thread is None
    assert_reaped(pids)


def test_concurrency_limit_rejects_without_starting_extra_worker(tmp_path):
    controller = make_controller(tmp_path, max_batch_workers=1)
    try:
        controller.children.spawn("blocked ignore-term")
        pids = wait_ready(tmp_path)
        with pytest.raises(RuntimeError, match="concurrency"):
            controller.children.spawn("blocked ignore-term")
        assert len(list(tmp_path.glob("*.ready"))) == 1
    finally:
        controller.close()
    assert_reaped(pids)


def test_child_timeout_interrupts_inflight_call(tmp_path):
    controller = make_controller(tmp_path, per_child_timeout_s=0.3)
    try:
        handle = controller.children.spawn("blocked ignore-term")
        pids = wait_ready(tmp_path)
        with pytest.raises(RuntimeError, match="timeout"):
            controller.children.wait(handle, timeout_s=15)
        assert_reaped(pids)
    finally:
        controller.close()


def test_runner_exception_cleans_up_managed_children(tmp_path):
    def root_chat(messages, model=None):
        if len(messages) > 2:
            wait_ready(tmp_path)
            raise ValueError("root failed")
        return "```repl\nchild = rlm_query('blocked ignore-term', blocking=False)\n```"

    runner = LocalRLMRunner(
        root_chat,
        max_depth=3,
        managed_children=make_controller(tmp_path).children.config,
    )
    with pytest.raises(ValueError, match="root failed"):
        runner.run("", "question")
    assert_reaped([int(path.stem) for path in tmp_path.glob("*.ready")])


def test_non_importable_factory_rejected():
    with pytest.raises(ValueError, match="pickl"):
        ManagedChildConfig(chat_factory=lambda: lambda *_: "answer")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": 0},
        {"max_batch_workers": 0},
        {"max_children_total": -1},
        {"per_child_timeout_s": float("nan")},
    ],
)
def test_invalid_limits_rejected_before_spawning(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_controller(tmp_path, **kwargs)


def test_callbacks_and_trace_run_in_supervisor(tmp_path, monkeypatch):
    controller = make_controller(tmp_path)
    events = []
    callback_done = threading.Event()

    def on_complete(*args):
        events.append((os.getpid(), "complete", args, threading.get_ident()))
        callback_done.set()

    controller.children.on_subcall_start = lambda *args: events.append(
        (os.getpid(), "start", args, threading.get_ident())
    )
    controller.children.on_subcall_complete = on_complete
    try:
        handle = controller.children.spawn("answer", model="fake-model")
        assert controller.children.wait(handle, timeout_s=15) == "42"
        assert callback_done.wait(5)
        assert [event[:2] for event in events] == [
            (os.getpid(), "start"),
            (os.getpid(), "complete"),
        ]
        assert all(event[2][1] == "fake-model" for event in events)
        assert events[0][3] == events[1][3]
        assert events[0][3] != threading.get_ident()
        assert events[0][3] != controller.children._jobs[handle].thread.ident
        assert controller.backend.child_traces[0].result_preview == "42"
        assert_reaped([controller.children._jobs[handle].pid])

        def unexpected_signal(*args):
            pytest.fail("A reaped worker's PID may already belong to another process")

        monkeypatch.setattr(os, "kill", unexpected_signal)
        controller.children._signal(controller.children._jobs[handle], signal.SIGKILL)
    finally:
        controller.close()


def test_invalid_prompt_does_not_start_worker(tmp_path):
    controller = make_controller(tmp_path)
    try:
        with pytest.raises(TypeError):
            controller.children.spawn(None)
        assert not controller.children._jobs
    finally:
        controller.close()


@pytest.mark.parametrize("first_operation", ["reset", "close"])
def test_reset_and_close_are_serialized(tmp_path, monkeypatch, first_operation):
    controller = make_controller(tmp_path)
    manager = controller.children
    drained = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    original_stop = manager.stop

    def pause_after_drain():
        original_stop()
        if threading.current_thread().name == "first-lifecycle":
            drained.set()
            if not release_first.wait(10):
                raise TimeoutError("lifecycle test barrier was not released")

    operations = {"reset": manager.reset, "close": manager.close}
    second_operation = "close" if first_operation == "reset" else "reset"

    def second_call():
        second_entered.set()
        operations[second_operation]()

    monkeypatch.setattr(manager, "stop", pause_after_drain)
    second_thread = None
    first_thread, first_done, first_errors = start_call(
        operations[first_operation], "first-lifecycle"
    )
    try:
        assert drained.wait(10)
        second_thread, second_done, second_errors = start_call(
            second_call, "second-lifecycle"
        )
        assert second_entered.wait(5)
        assert not second_done.wait(0.25)
    finally:
        release_first.set()
        first_thread.join(10)
        if second_thread is not None:
            second_thread.join(10)
        monkeypatch.setattr(manager, "stop", original_stop)
        manager.close()

    assert first_done.is_set() and not first_thread.is_alive()
    assert second_done.is_set() and not second_thread.is_alive()
    assert not first_errors
    if second_operation == "reset":
        assert len(second_errors) == 1
        assert isinstance(second_errors[0], RuntimeError)
        assert "closed" in str(second_errors[0])
    else:
        assert not second_errors
    with pytest.raises(RuntimeError, match="closed"):
        manager.spawn("answer")


@pytest.mark.parametrize("callback_phase", ["start", "complete"])
@pytest.mark.parametrize("operation", ["reset", "close"])
def test_blocked_callbacks_do_not_block_lifecycle(tmp_path, callback_phase, operation):
    controller = make_controller(tmp_path)
    manager = controller.children
    env = LocalREPLEnv(recursive_controller=controller)
    env.reset()
    entered = threading.Event()
    release_callback = threading.Event()
    callback_done = threading.Event()
    events = []
    handles = []

    def observe(phase):
        events.append(phase)
        if phase == callback_phase:
            entered.set()
            if not release_callback.wait(20):
                raise TimeoutError("callback test barrier was not released")

    def on_complete(*args):
        observe("complete")
        callback_done.set()

    manager.on_subcall_start = lambda *args: observe("start")
    manager.on_subcall_complete = on_complete
    prompt = "blocked ignore-term" if callback_phase == "start" else "answer"
    spawn_thread, spawn_done, spawn_errors = start_call(
        lambda: handles.append(manager.spawn(prompt)), "callback-spawn"
    )
    operation_thread = None
    try:
        assert entered.wait(15)
        assert spawn_done.wait(0.5)
        assert not spawn_errors
        job = manager._jobs[handles[0]]
        pids = wait_ready(tmp_path) if callback_phase == "start" else [job.pid]
        operations = {"reset": env.reset, "close": env.close}
        operation_thread, operation_done, operation_errors = start_call(
            operations[operation], "callback-lifecycle"
        )
        assert operation_done.wait(4)
        assert not operation_errors
        assert job.finished.is_set()
        assert_reaped(pids)
        assert not release_callback.is_set()
        assert not callback_done.is_set()
    finally:
        release_callback.set()
        spawn_thread.join(10)
        if operation_thread is not None:
            operation_thread.join(10)
        env.close()

    assert not spawn_thread.is_alive()
    assert not operation_thread.is_alive()
    assert callback_done.wait(5)
    assert events == ["start", "complete"]
    manager._callback_thread.join(5)
    assert not manager._callback_thread.is_alive()


@pytest.mark.parametrize("callback_phase", ["start", "complete"])
def test_callback_exceptions_do_not_stop_dispatcher(tmp_path, callback_phase):
    controller = make_controller(tmp_path)
    events = []
    callback_done = threading.Event()

    def observe(phase):
        events.append(phase)
        if phase == callback_phase:
            raise ValueError("observer failed")

    def on_complete(*args):
        try:
            observe("complete")
        finally:
            callback_done.set()

    controller.children.on_subcall_start = lambda *args: observe("start")
    controller.children.on_subcall_complete = on_complete
    after_failure = threading.Event()
    try:
        controller.children._notify(
            controller.children.on_subcall_start, 1, "default", "answer"
        )
        controller.children._notify(
            controller.children.on_subcall_complete, 1, "default", 0.0, None
        )
        controller.children._notify(after_failure.set)
        assert callback_done.wait(5)
        assert after_failure.wait(5)
        assert events == ["start", "complete"]
        controller.children.check_ready()
        assert not controller.children._jobs
    finally:
        controller.close()
    controller.children._callback_thread.join(5)
    assert not controller.children._callback_thread.is_alive()
