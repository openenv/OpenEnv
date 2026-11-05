"""Own the sandboxes this process starts: cap how many start at once, stop them on SIGINT/SIGTERM.

Harbor already tears a sandbox down at the end of every trial: `Trial._finalize` shields
`agent_environment.stop()`. Three gaps remain once a run is large enough:

- SIGTERM's default action kills the process before any `finally` runs, so every live sandbox is
  stranded. SIGINT cancels, but `asyncio.run` closes its loop right after, which cancels the shielded
  stop as well.
- Nothing caps how many sandboxes are being created at once. Providers throttle creation well below
  the rollout concurrency a trainer wants, and a throttled create reads as a failed rollout.
- Cancelling a rollout while its sandbox is starting abandons the start mid-flight, and teardown
  then races it. Harbor's docker backend leaves `docker compose up` running when `communicate()` is
  cancelled (only its timeout path terminates the child); a remote backend can finish a create
  request whose handle was dropped. Either way the sandbox can appear after `stop()` has run.

`SandboxSupervisor.adopt(trial)` wraps the trial's own `agent_environment.start/stop`, so Harbor's
lifecycle is unchanged and the supervisor only sees sandboxes this process created.
"""

from __future__ import annotations

import asyncio
import collections
import os
import signal
import threading
from typing import Any

# How long a cleanup triggered by a signal may take before the signal is passed on anyway. A wedged
# backend must not turn Ctrl-C into a hang; a second signal skips the wait entirely.
_CLEANUP_TIMEOUT_S = 60.0

# How long `stop()` waits for a start that is still in flight. Stopping first is what leaks, so it
# waits; but a start that never returns must not hold teardown forever.
_START_GRACE_S = 120.0

_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class _StartLimit:
    """At most `limit` concurrent starts across every event loop in the process.

    Not an `asyncio.Semaphore`: rollouts run on several loops (the UI runs each one under its own
    `asyncio.run` in a worker thread), and an asyncio primitive binds to the first loop that waits on
    it. Waiters queue FIFO and are woken on their own loop.
    """

    def __init__(self, limit: int):
        self._limit = limit
        self._active = 0
        self._waiters: collections.deque[asyncio.Future] = collections.deque()
        self._lock = threading.Lock()

    async def acquire(self) -> None:
        with self._lock:
            if self._active < self._limit and not self._waiters:
                self._active += 1
                return
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    raise
            # The permit was handed over just as this waiter was cancelled: pass it on.
            self.release()
            raise

    def release(self) -> None:
        with self._lock:
            if not self._waiters:
                self._active -= 1
                return
            waiter = self._waiters.popleft()
        waiter.get_loop().call_soon_threadsafe(self._grant, waiter)

    def _grant(self, waiter: asyncio.Future) -> None:
        # A waiter cancelled after it was dequeued releases the permit itself, in `acquire`.
        if not waiter.done():
            waiter.set_result(None)


class _Sandbox:
    """One adopted agent environment and the loop its handles belong to."""

    def __init__(self, env: Any, delete: bool):
        self.env = env
        self.delete = delete
        self.loop: asyncio.AbstractEventLoop | None = None
        self.start_task: asyncio.Future | None = None
        self.stop_task: asyncio.Future | None = None


class _LoopHandler:
    """A signal callback that was registered on an event loop with `add_signal_handler`."""

    def __init__(self, loop: asyncio.AbstractEventLoop, callback: Any, args: tuple):
        self.loop = loop
        self.callback = callback
        self.args = args


class SandboxSupervisor:
    """Track the sandboxes this process starts and tear them down on request or on a signal.

    Args:
        max_starts (`int`, *optional*):
            How many sandboxes may be starting at once. `None` or `0` means no limit. This bounds
            creation only; how many rollouts run at once is still `MAX_CONCURRENT_ENVS`, so a
            rollout that has its sandbox no longer counts against it.
    """

    def __init__(self, max_starts: int | None = None):
        self._limit = _StartLimit(max_starts) if max_starts else None
        self._owned: set[_Sandbox] = set()
        self._lock = threading.Lock()
        self._previous: dict[int, Any] = {}
        # Signals whose previous handler the event loop delivers by itself; see `install_signal_handlers`.
        self._loop_delivers: set[int] = set()
        self._handling = False
        self._cleanup: asyncio.Future | None = None
        # Set by `aclose` under `_lock`. A start that has not registered yet must not open a sandbox
        # cleanup will never see, so it fails instead.
        self._closing = False

    @property
    def owned(self) -> int:
        """How many sandboxes this supervisor started and has not yet stopped."""
        with self._lock:
            return len(self._owned)

    def adopt(self, trial: Any) -> Any:
        """Route `trial.agent_environment`'s start and stop through this supervisor.

        Call after `Trial.create` and before `trial.run()`. Harbor's own calls are unchanged; they
        now pass through the creation limit, and `stop()` waits for a pending start. Concurrent
        callers share cleanup; successful cleanup is cached and failed cleanup can be retried.
        """
        # `getattr`, because callers and tests pass their own trial objects; one without an agent
        # environment has no sandbox to supervise.
        env = getattr(trial, "agent_environment", None)
        if env is None or getattr(env, "_openenv_sandbox", None) is not None:
            return trial
        box = _Sandbox(env, delete=trial.config.environment.delete)
        real_start, real_stop = env.start, env.stop

        async def start(force_build: bool) -> None:
            box.loop = asyncio.get_running_loop()
            self.install_signal_handlers()
            if self._limit is not None:
                await self._limit.acquire()
            with self._lock:
                closing = self._closing
                if not closing:
                    self._owned.add(box)
            if closing:
                if self._limit is not None:
                    self._limit.release()
                raise RuntimeError(
                    "sandbox supervisor is shutting down; not starting a new sandbox"
                )
            # Shielded: a caller cancelled mid-start gets its CancelledError, but the start itself
            # runs to completion so that `stop()` has a whole sandbox to delete, not half of one.
            box.start_task = asyncio.ensure_future(real_start(force_build=force_build))
            if self._limit is not None:
                box.start_task.add_done_callback(lambda _: self._limit.release())
            await asyncio.shield(box.start_task)

        async def stop(delete: bool) -> None:
            # Share an in-flight stop and cache success, but allow failed cleanup to retry.
            if box.stop_task is None or (
                box.stop_task.done()
                and (box.stop_task.cancelled() or box.stop_task.exception() is not None)
            ):
                box.stop_task = asyncio.ensure_future(_stop_once(delete))
            await asyncio.shield(box.stop_task)

        async def _stop_once(delete: bool) -> None:
            if box.start_task is not None:
                await asyncio.wait({box.start_task}, timeout=_START_GRACE_S)
                if not box.start_task.done():
                    box.start_task.cancel()
                elif not box.start_task.cancelled():
                    # A start failure is the trial's to report; retrieved so asyncio stays quiet.
                    box.start_task.exception()
            await real_stop(delete=delete)
            with self._lock:
                self._owned.discard(box)

        env.start, env.stop = start, stop
        env._openenv_sandbox = box
        return trial

    async def aclose(self, timeout: float = _CLEANUP_TIMEOUT_S) -> None:
        """Stop every sandbox this supervisor owns. Safe to call more than once, and concurrently.

        From the first call on, new starts are refused, so a rollout still queued for a creation
        permit cannot open a sandbox while cleanup is running.

        Each stop runs on the loop that started the sandbox, since backends hold loop-bound handles
        (docker's are asyncio subprocesses). A sandbox whose loop has already closed cannot be
        reached from here and is left alone.
        """
        with self._lock:
            # Same lock as the registration in `start`, so every sandbox is either in this snapshot
            # or refused: none can be created after cleanup has looked.
            self._closing = True
            boxes = list(self._owned)
        here = asyncio.get_running_loop()
        pending = []
        for box in boxes:
            if box.loop is here:
                pending.append(asyncio.ensure_future(box.env.stop(delete=box.delete)))
            elif box.loop is not None and box.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    box.env.stop(delete=box.delete), box.loop
                )
                pending.append(asyncio.wrap_future(future))
        if not pending:
            return
        done, _ = await asyncio.wait(pending, timeout=timeout)
        for task in done:
            task.exception()  # stop failures belong to the trial; retrieved so asyncio stays quiet

    def install_signal_handlers(self, signals: tuple[int, ...] = _SIGNALS) -> bool:
        """Stop owned sandboxes on these signals, then hand the signal to the previous handler.

        Called automatically when the first sandbox starts. Only the main thread can install signal
        handlers, so this returns `False` elsewhere; a process that starts every sandbox from worker
        threads should call it once from the main thread. Installing late is deliberate: a server
        such as uvicorn replaces handlers when it starts serving, and this one has to wrap its.

        A handler registered with `loop.add_signal_handler` (uvicorn before 0.29) is delivered by
        the loop through the wakeup fd, whatever `signal.signal` holds, so wrapping that would let
        the server start shutting down while cleanup is still running and then see the signal a
        second time. On an asyncio loop the supervisor takes the loop's slot instead and puts the
        callback back when cleanup is done. A loop that does not expose its callbacks (uvloop)
        cannot be wrapped: cleanup then runs alongside the server's shutdown, and the signal is not
        delivered again.
        """
        if threading.current_thread() is not threading.main_thread():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        for sig in signals:
            current = signal.getsignal(sig)
            # Private, but unchanged since asyncio was added: the public API can replace a loop's
            # signal callback, not read it back.
            handle = getattr(loop, "_signal_handlers", {}).get(sig)
            if current == self._on_signal or (
                handle is not None and handle._callback == self._on_loop_signal
            ):
                continue
            if handle is not None:
                self._previous[sig] = _LoopHandler(loop, handle._callback, handle._args)
                loop.add_signal_handler(sig, self._on_loop_signal, sig)
                continue
            if isinstance(
                getattr(current, "__self__", None), asyncio.AbstractEventLoop
            ):
                self._loop_delivers.add(sig)
            self._previous[sig] = current
            signal.signal(sig, self._on_signal)
        return True

    def _on_loop_signal(self, sig: int) -> None:
        # Runs as an ordinary loop callback, so unlike `_on_signal` it may start the cleanup itself.
        if self._handling:
            self._chain(sig)
            return
        self._handling = True
        self._start_cleanup(sig)

    def _on_signal(self, sig: int, frame: Any) -> None:
        if self._handling:
            # A second signal means "stop waiting".
            self._chain(sig)
            return
        self._handling = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # The handler interrupts the loop mid-callback; scheduling is all it may do safely.
            loop.call_soon_threadsafe(self._start_cleanup, sig)
            return
        # No loop on the main thread, so the sandboxes live on other threads' loops and waiting
        # for them here cannot deadlock.
        worker = threading.Thread(
            target=asyncio.run, args=(self.aclose(),), daemon=True
        )
        worker.start()
        worker.join(_CLEANUP_TIMEOUT_S + 5)
        self._chain(sig)

    def _start_cleanup(self, sig: int) -> None:
        async def cleanup_then_chain() -> None:
            try:
                await self.aclose()
            finally:
                if self._handling:
                    self._chain(sig)

        self._cleanup = asyncio.ensure_future(cleanup_then_chain())

    def _chain(self, sig: int) -> None:
        """Put the previous handler back and re-deliver the signal to it.

        Re-raising rather than calling it keeps its semantics exactly: SIG_DFL still terminates,
        the default SIGINT handler still raises KeyboardInterrupt, uvicorn still shuts down.
        """
        self._handling = False
        previous = self._previous.pop(sig, signal.SIG_DFL)
        if isinstance(previous, _LoopHandler):
            if previous.loop.is_closed():
                return
            previous.loop.add_signal_handler(sig, previous.callback, *previous.args)
        else:
            signal.signal(sig, signal.SIG_DFL if previous is None else previous)
            if sig in self._loop_delivers:
                # The loop already ran its callback for this signal.
                self._loop_delivers.discard(sig)
                return
        signal.raise_signal(sig)


_supervisor: SandboxSupervisor | None = None
_supervisor_lock = threading.Lock()


def get_supervisor() -> SandboxSupervisor:
    """The process-wide supervisor that `run_rollout` adopts every trial into.

    `OPENENV_HARBOR_MAX_SANDBOX_STARTS` caps concurrent sandbox creation; unset or `0` leaves it
    uncapped. It is read once, on first use.
    """
    global _supervisor
    with _supervisor_lock:
        if _supervisor is None:
            raw = os.environ.get("OPENENV_HARBOR_MAX_SANDBOX_STARTS", "0") or "0"
            if not raw.isdigit():
                raise ValueError(
                    f"OPENENV_HARBOR_MAX_SANDBOX_STARTS must be a non-negative integer, got {raw!r}"
                )
            _supervisor = SandboxSupervisor(max_starts=int(raw))
        return _supervisor


__all__ = ["SandboxSupervisor", "get_supervisor"]
