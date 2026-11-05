"""The sandbox supervisor: ownership, idempotent cleanup, the creation limit and signal chaining.

Fake environments stand in for Harbor's; the real docker backend is exercised in
`test_harbor_supervisor_docker.py`.
"""

from __future__ import annotations

import asyncio
import signal
import threading
import time
from types import SimpleNamespace

import pytest

supervisor_mod = pytest.importorskip("openenv.harbor.supervisor")
SandboxSupervisor = supervisor_mod.SandboxSupervisor


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """Starting a sandbox installs handlers in this (main) thread; keep them out of other tests."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


class FakeEnv:
    def __init__(self, log: list, name: str = "env", start_s: float = 0.0):
        self.log = log
        self.name = name
        self.start_s = start_s

    async def start(self, force_build: bool) -> None:
        self.log.append((self.name, "start"))
        await asyncio.sleep(self.start_s)
        self.log.append((self.name, "started"))

    async def stop(self, delete: bool) -> None:
        self.log.append((self.name, "stop", delete))


def make_trial(env, delete: bool = True):
    return SimpleNamespace(
        agent_environment=env,
        config=SimpleNamespace(environment=SimpleNamespace(delete=delete)),
    )


def stops(log):
    return [entry for entry in log if entry[1] == "stop"]


async def test_cleanup_only_touches_sandboxes_this_supervisor_started():
    log: list = []
    mine, theirs = SandboxSupervisor(), SandboxSupervisor()
    a, b, never_started = FakeEnv(log, "a"), FakeEnv(log, "b"), FakeEnv(log, "c")
    mine.adopt(make_trial(a))
    mine.adopt(make_trial(never_started))
    theirs.adopt(make_trial(b))
    await a.start(force_build=False)
    await b.start(force_build=False)

    await mine.aclose()

    assert stops(log) == [("a", "stop", True)]
    assert mine.owned == 0
    assert theirs.owned == 1


async def test_cleanup_is_safe_to_call_more_than_once():
    log: list = []
    sup = SandboxSupervisor()
    env = FakeEnv(log)
    sup.adopt(make_trial(env, delete=False))
    await env.start(force_build=False)

    # Signal cleanup and Harbor's own `_finalize` reaching stop at the same time.
    await asyncio.gather(sup.aclose(), sup.aclose(), env.stop(delete=False))
    await sup.aclose()
    await env.stop(delete=False)

    assert stops(log) == [("env", "stop", False)]


async def test_no_new_sandbox_starts_once_cleanup_begins():
    log: list = []
    sup = SandboxSupervisor(max_starts=1)
    first, queued = FakeEnv(log, "first", start_s=0.05), FakeEnv(log, "queued")
    sup.adopt(make_trial(first))
    sup.adopt(make_trial(queued))

    first_start = asyncio.ensure_future(first.start(force_build=False))
    await asyncio.sleep(0)  # `first` holds the only permit
    queued_start = asyncio.ensure_future(queued.start(force_build=False))
    await asyncio.sleep(0)  # `queued` is waiting for it

    await sup.aclose()
    await first_start
    with pytest.raises(RuntimeError, match="shutting down"):
        await queued_start

    assert ("queued", "start") not in log
    assert stops(log) == [("first", "stop", True)]
    assert sup.owned == 0


async def test_adopting_twice_wraps_once():
    log: list = []
    sup = SandboxSupervisor(max_starts=1)
    trial = make_trial(FakeEnv(log))
    sup.adopt(trial)
    sup.adopt(trial)
    await trial.agent_environment.start(force_build=False)
    await trial.agent_environment.stop(delete=True)
    assert log == [("env", "start"), ("env", "started"), ("env", "stop", True)]


async def test_creation_limit_does_not_limit_rollouts():
    sup = SandboxSupervisor(max_starts=2)
    starting = peak_starting = live = peak_live = 0

    class CountingEnv:
        async def start(self, force_build: bool) -> None:
            nonlocal starting, peak_starting
            starting += 1
            peak_starting = max(peak_starting, starting)
            await asyncio.sleep(0.02)
            starting -= 1

        async def stop(self, delete: bool) -> None:
            pass

    async def rollout():
        nonlocal live, peak_live
        env = CountingEnv()
        sup.adopt(make_trial(env))
        await env.start(force_build=False)
        live += 1
        peak_live = max(peak_live, live)
        await asyncio.sleep(
            0.2
        )  # the agent runs; its sandbox no longer holds a creation slot
        live -= 1
        await env.stop(delete=True)

    await asyncio.gather(*(rollout() for _ in range(6)))

    assert peak_starting == 2
    assert peak_live == 6


async def test_cancel_during_start_stops_only_after_the_start_finishes():
    log: list = []
    sup = SandboxSupervisor(max_starts=1)
    env = FakeEnv(log, start_s=0.2)
    sup.adopt(make_trial(env))

    starting = asyncio.ensure_future(env.start(force_build=False))
    await asyncio.sleep(0.05)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    # What Harbor's `Trial.run` does next: `_finalize` -> shielded `stop`.
    await env.stop(delete=True)

    assert log == [("env", "start"), ("env", "started"), ("env", "stop", True)]
    assert sup.owned == 0
    # The permit came back once the abandoned start actually finished.
    nxt = FakeEnv(log, "next")
    sup.adopt(make_trial(nxt))
    await asyncio.wait_for(nxt.start(force_build=False), 1)


async def test_cancel_while_waiting_for_a_permit_owns_nothing():
    log: list = []
    sup = SandboxSupervisor(max_starts=1)
    first, queued = FakeEnv(log, "first", start_s=0.2), FakeEnv(log, "queued")
    sup.adopt(make_trial(first))
    sup.adopt(make_trial(queued))

    running = asyncio.ensure_future(first.start(force_build=False))
    await asyncio.sleep(0.01)
    waiting = asyncio.ensure_future(queued.start(force_build=False))
    await asyncio.sleep(0.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await running

    assert ("queued", "start") not in log
    assert sup.owned == 1
    third = FakeEnv(log, "third")
    sup.adopt(make_trial(third))
    await asyncio.wait_for(third.start(force_build=False), 1)


def test_creation_limit_holds_across_event_loops():
    sup = SandboxSupervisor(max_starts=1)
    windows: list[tuple[float, float]] = []

    class TimedEnv:
        async def start(self, force_build: bool) -> None:
            began = time.monotonic()
            await asyncio.sleep(0.1)
            windows.append((began, time.monotonic()))

        async def stop(self, delete: bool) -> None:
            pass

    def worker():
        env = TimedEnv()
        sup.adopt(make_trial(env))
        asyncio.run(env.start(force_build=False))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    windows.sort()
    assert len(windows) == 3
    assert all(prev[1] <= nxt[0] for prev, nxt in zip(windows, windows[1:]))


async def _until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        await asyncio.sleep(0.01)


async def test_signal_stops_sandboxes_then_reaches_the_previous_handler():
    log: list = []
    seen: list[int] = []

    def previous(sig, frame):
        log.append(("previous", "called"))
        seen.append(sig)

    signal.signal(signal.SIGTERM, previous)
    sup = SandboxSupervisor()
    env = FakeEnv(log)
    sup.adopt(make_trial(env))
    await env.start(force_build=False)  # installs the handler on top of `previous`
    assert signal.getsignal(signal.SIGTERM) == sup._on_signal

    signal.raise_signal(signal.SIGTERM)
    await _until(lambda: seen)

    assert log[-2:] == [("env", "stop", True), ("previous", "called")]
    assert seen == [signal.SIGTERM]
    assert signal.getsignal(signal.SIGTERM) is previous


async def test_second_signal_skips_the_wait():
    seen: list[int] = []
    release = asyncio.Event()

    class HangingEnv:
        async def start(self, force_build: bool) -> None:
            pass

        async def stop(self, delete: bool) -> None:
            await release.wait()

    signal.signal(signal.SIGTERM, lambda sig, frame: seen.append(sig))
    sup = SandboxSupervisor()
    env = HangingEnv()
    sup.adopt(make_trial(env))
    await env.start(force_build=False)

    signal.raise_signal(signal.SIGTERM)
    await asyncio.sleep(0.05)
    assert seen == []  # still cleaning up
    signal.raise_signal(signal.SIGTERM)
    await _until(lambda: seen)
    release.set()
    await asyncio.sleep(0.05)

    assert seen == [signal.SIGTERM]  # chained once, not again when cleanup finished


async def test_signal_reaches_a_loop_level_handler_only_after_cleanup():
    # uvicorn before 0.29 registers its handler with `loop.add_signal_handler`. The loop delivers
    # that one itself, so a `signal.signal` wrapper would let it run at once, and a second time
    # when the signal is chained.
    log: list = []
    loop = asyncio.get_running_loop()

    def previous(sig):
        log.append(("previous", sig))

    loop.add_signal_handler(signal.SIGTERM, previous, signal.SIGTERM)
    try:
        sup = SandboxSupervisor()
        env = FakeEnv(log)
        sup.adopt(make_trial(env))
        await env.start(force_build=False)

        signal.raise_signal(signal.SIGTERM)
        await _until(lambda: ("previous", signal.SIGTERM) in log)
        await asyncio.sleep(0.05)

        assert log[-2:] == [("env", "stop", True), ("previous", signal.SIGTERM)]
        assert log.count(("previous", signal.SIGTERM)) == 1
        assert loop._signal_handlers[signal.SIGTERM]._callback is previous
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


async def test_signal_a_loop_delivers_by_itself_is_not_delivered_again():
    # uvloop does not expose its signal callbacks, so the supervisor cannot take their place. Its
    # Python-level handler is a method of the loop; the loop has already run its callback by the
    # time cleanup ends, and re-raising would hand the server a second signal.
    class OpaqueLoop(asyncio.SelectorEventLoop):
        def sighandler(self, sig, frame):
            seen.append(sig)

    seen: list[int] = []
    log: list = []
    opaque = OpaqueLoop()
    try:
        signal.signal(signal.SIGTERM, opaque.sighandler)
        sup = SandboxSupervisor()
        env = FakeEnv(log)
        sup.adopt(make_trial(env))
        await env.start(force_build=False)

        signal.raise_signal(signal.SIGTERM)
        await _until(lambda: stops(log))
        await asyncio.sleep(0.05)

        assert signal.getsignal(signal.SIGTERM) == opaque.sighandler
        assert seen == []
    finally:
        opaque.close()


def test_max_starts_comes_from_the_environment(monkeypatch):
    monkeypatch.setattr(supervisor_mod, "_supervisor", None)
    monkeypatch.setenv("OPENENV_HARBOR_MAX_SANDBOX_STARTS", "lots")
    with pytest.raises(ValueError, match="OPENENV_HARBOR_MAX_SANDBOX_STARTS"):
        supervisor_mod.get_supervisor()

    monkeypatch.setenv("OPENENV_HARBOR_MAX_SANDBOX_STARTS", "3")
    assert supervisor_mod.get_supervisor()._limit._limit == 3
    assert supervisor_mod.get_supervisor() is supervisor_mod.get_supervisor()


@pytest.mark.parametrize("retry_via_close", [False, True])
async def test_failed_stop_retains_ownership_and_can_be_retried(retry_via_close):
    sup = SandboxSupervisor()
    attempts = 0

    class FlakyEnv(FakeEnv):
        async def stop(self, delete: bool) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary backend failure")
            await super().stop(delete)

    log = []
    env = FlakyEnv(log)
    sup.adopt(make_trial(env))
    await env.start(force_build=False)
    with pytest.raises(RuntimeError, match="temporary backend failure"):
        await env.stop(delete=True)
    retained = sup.owned

    if retry_via_close:
        await sup.aclose()
    else:
        await env.stop(delete=True)
    assert retained == 1
    assert attempts == 2
    assert sup.owned == 0
    assert stops(log) == [("env", "stop", True)]
    await env.stop(delete=True)
    assert attempts == 2


async def test_cancelled_stop_attempt_can_be_retried():
    sup = SandboxSupervisor()
    attempts = 0

    class CancelledEnv(FakeEnv):
        async def stop(self, delete: bool) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise asyncio.CancelledError
            await super().stop(delete)

    env = CancelledEnv([])
    sup.adopt(make_trial(env))
    await env.start(force_build=False)
    with pytest.raises(asyncio.CancelledError):
        await env.stop(delete=True)
    retained = sup.owned
    await env.stop(delete=True)
    assert retained == 1
    assert attempts == 2
    assert sup.owned == 0


async def test_retry_shares_cleanup_when_a_waiter_is_cancelled():
    sup = SandboxSupervisor()
    entered = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    class BlockingRetryEnv(FakeEnv):
        async def stop(self, delete: bool) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary backend failure")
            entered.set()
            await release.wait()
            await super().stop(delete)

    log = []
    env = BlockingRetryEnv(log)
    sup.adopt(make_trial(env))
    await env.start(force_build=False)
    with pytest.raises(RuntimeError, match="temporary backend failure"):
        await env.stop(delete=True)

    waiter = asyncio.create_task(env.stop(delete=True))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        # Cancelling a waiter must not cancel or forget the shared cleanup.
        assert sup.owned == 1
        await sup.aclose(timeout=0)
        assert attempts == 2
        assert sup.owned == 1
    finally:
        release.set()
        await asyncio.gather(waiter, return_exceptions=True)
    await asyncio.gather(sup.aclose(), sup.aclose(), env.stop(delete=True))
    assert attempts == 2
    assert stops(log) == [("env", "stop", True)]
    assert sup.owned == 0
