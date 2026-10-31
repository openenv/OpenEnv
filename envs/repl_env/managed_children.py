"""Opt-in, process-supervised recursive queries for the REPL environment."""

from __future__ import annotations

import math
import multiprocessing
import os
import pickle
import signal
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from queue import SimpleQueue
from typing import Callable
from uuid import uuid4

from .recursive_backends import BackendLimits, ChildTrace, DirectLMBackend


MANAGED_CHILD_TOOLS_PROMPT = """
rlm_query(prompt, model=None, blocking=True) waits and returns an answer string.
To keep working while a child runs, use child = rlm_query(prompt, model=None, blocking=False).
The blocking option is keyword-only; blocking=False returns an episode-scoped handle.
Keep working, then inspect rlm_status(child), read rlm_result(child), or call
rlm_wait(child, timeout_s=30). Wait timeouts do not cancel work. Results are strings;
unfinished or failed children raise errors. Handles expire on environment reset.
Outstanding children are stopped when you finalize. Concurrency limits reject
new children rather than queueing them, including when nested parents occupy slots.
"""


class ChildCleanupError(RuntimeError):
    """A managed worker's exit could not be confirmed."""


@dataclass(frozen=True)
class ManagedChildConfig:
    """Construct a fresh model client inside each spawned worker.

    Args:
        chat_factory (`Callable`):
            Picklable, importable callable returning a chat function. Live clients
            and closures must not be passed across the process boundary.
        shutdown_grace_s (`float`, *optional*, defaults to `1.0`):
            Time to allow cooperative shutdown before sending SIGTERM.
        terminate_grace_s (`float`, *optional*, defaults to `1.0`):
            Time to wait after SIGTERM before sending SIGKILL.
        kill_timeout_s (`float`, *optional*, defaults to `5.0`):
            Final bounded process-exit wait after SIGKILL.
    """

    chat_factory: Callable[[], Callable[..., str]] = field(repr=False)
    shutdown_grace_s: float = 1.0
    terminate_grace_s: float = 1.0
    kill_timeout_s: float = 5.0

    def __post_init__(self):
        if os.name != "posix":
            raise ValueError("Managed children require POSIX process signals")
        if not callable(self.chat_factory):
            raise ValueError("chat_factory must be callable")
        for duration in (
            self.shutdown_grace_s,
            self.terminate_grace_s,
            self.kill_timeout_s,
        ):
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Shutdown timeouts must be finite and nonnegative")
        if self.kill_timeout_s == 0:
            raise ValueError("kill_timeout_s must be positive")
        try:
            pickle.dumps(self.chat_factory)
        except (TypeError, AttributeError, pickle.PicklingError) as exc:
            raise ValueError("chat_factory must be importable and picklable") from exc


@dataclass
class _Job:
    process: multiprocessing.Process
    channel: Connection
    shutdown: Connection
    prompt: str
    depth: int
    parent: _Job | None = None
    model: str | None = None
    accepting: bool = True
    reported: bool = False
    pid: int = field(init=False)
    exitcode: int | None = None
    started: float = field(default_factory=time.monotonic)
    stop: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    status: str = "running"
    answer: str | None = None
    error: str | None = None
    thread: threading.Thread | None = None

    def __post_init__(self):
        self.pid = self.process.pid


class ManagedChildren:
    """Own all managed workers for one episode, including nested queries.

    Every process is spawned and reaped by this controller, not by another
    worker. This registry does not cover arbitrary subprocesses created by a
    custom client, nor does it guarantee cancellation on the model server.
    """

    def __init__(
        self,
        config: ManagedChildConfig,
        limits: BackendLimits,
        *,
        on_subcall_start=None,
        on_subcall_complete=None,
        **runner_options,
    ):
        if limits.max_depth < 1 or limits.max_batch_workers < 1:
            raise ValueError("Managed recursion depth and concurrency must be positive")
        if limits.per_child_timeout_s is not None:
            if (
                not math.isfinite(limits.per_child_timeout_s)
                or limits.per_child_timeout_s <= 0
            ):
                raise ValueError("per_child_timeout_s must be finite and positive")
        for limit in (
            limits.max_children_total,
            limits.max_children_per_batch,
            limits.result_truncation_limit,
        ):
            if limit is not None and limit < 0:
                raise ValueError("Child count and result limits must be nonnegative")
        self.config = config
        self.limits = limits
        self.runner_options = runner_options
        self.on_subcall_start = on_subcall_start
        self.on_subcall_complete = on_subcall_complete
        self.child_traces: list[ChildTrace] = []
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._callback_queue = SimpleQueue()
        self._callback_thread: threading.Thread | None = None
        self._accepting = True
        self._closed = False
        self._cleanup_failed = False

    def check_ready(self):
        if self._cleanup_failed:
            raise ChildCleanupError(
                "Managed child cleanup failed; environment cannot be reused"
            )
        if self._closed:
            raise RuntimeError("Managed child controller is closed")

    @staticmethod
    def _run_callbacks(notifications):
        while (notification := notifications.get()) is not None:
            callback, args = notification
            with suppress(Exception):
                callback(*args)

    def _notify(self, callback, *args):
        if callback is None:
            return
        with self._lock:
            if self._callback_thread is None:
                try:
                    callback_thread = threading.Thread(
                        target=self._run_callbacks,
                        args=(self._callback_queue,),
                        daemon=True,
                    )
                    callback_thread.start()
                except RuntimeError:
                    return
                self._callback_thread = callback_thread
            self._callback_queue.put((callback, args))

    def spawn(self, prompt: str, model: str | None = None) -> str:
        """Start a child without waiting; return an episode-scoped handle."""
        return self._spawn(prompt, model, parent=None)

    def _spawn(self, prompt: str, model: str | None, *, parent: _Job | None) -> str:
        if not isinstance(prompt, str) or (
            model is not None and not isinstance(model, str)
        ):
            raise TypeError("prompt and model must be strings")
        with self._lock:
            self.check_ready()
            depth = parent.depth if parent is not None else 0
            if parent is not None and (parent.stop.is_set() or not parent.accepting):
                raise RuntimeError("Parent child is finalizing")
            if depth >= self.limits.max_depth:
                raise RuntimeError("max_depth exceeded")
            if not self._accepting:
                raise RuntimeError("Episode is finalizing; new children are disabled")
            if self.limits.max_children_total is not None:
                if len(self._jobs) >= self.limits.max_children_total:
                    raise RuntimeError("max_children_total exceeded")
            if (
                sum(not job.finished.is_set() for job in self._jobs.values())
                >= self.limits.max_batch_workers
            ):
                raise RuntimeError("Managed child concurrency limit exceeded")
            context = multiprocessing.get_context("spawn")
            host_channel, worker_channel = context.Pipe()
            shutdown_reader, shutdown_writer = context.Pipe(duplex=False)
            options = {
                "max_depth": self.limits.max_depth,
                "max_batch_workers": self.limits.max_batch_workers,
                "max_children_per_batch": self.limits.max_children_per_batch,
                "result_truncation_limit": self.limits.result_truncation_limit,
                **self.runner_options,
            }
            process = context.Process(
                target=_run_child,
                args=(
                    self.config.chat_factory,
                    worker_channel,
                    shutdown_reader,
                    prompt,
                    model,
                    depth + 1,
                    options,
                ),
            )
            try:
                process.start()
            except BaseException:
                host_channel.close()
                shutdown_writer.close()
                raise
            finally:
                worker_channel.close()
                shutdown_reader.close()
            handle = uuid4().hex
            job = _Job(
                process,
                host_channel,
                shutdown_writer,
                prompt,
                depth + 1,
                parent,
            )
            self._jobs[handle] = job
            job.model = model
            self._notify(
                self.on_subcall_start, job.depth, model or "default", prompt[:80]
            )
            try:
                job.thread = threading.Thread(
                    target=self._watch, args=(job,), daemon=True
                )
                job.thread.start()
            except BaseException as exc:
                job.error = str(exc)
                job.stop.set()
                self._watch(job)
                if job.exitcode is None or job.status == "cleanup_failed":
                    raise ChildCleanupError(
                        "Failed to clean up unmonitored worker"
                    ) from exc
                raise
            return handle

    def _job(self, handle: str) -> _Job:
        with self._lock:
            try:
                return self._jobs[handle]
            except KeyError:
                raise KeyError("Unknown or stale child handle") from None

    def status(self, handle: str) -> str:
        """Return running, stopping, completed, failed, cancelled, or cleanup_failed."""
        job = self._job(handle)
        if not job.finished.is_set():
            return "stopping" if job.stop.is_set() else "running"
        return job.status

    def result(self, handle: str) -> str:
        """Read a completed result without waiting."""
        job = self._job(handle)
        if not job.finished.is_set():
            raise RuntimeError("Child has not finished")
        if job.status != "completed":
            raise RuntimeError(job.error or f"Child {job.status}")
        return job.answer or ""

    def wait(self, handle: str, timeout_s: float = 30.0) -> str:
        """Wait for a bounded period; a wait timeout does not cancel the child."""
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and nonnegative")
        if not self._job(handle).finished.wait(timeout_s):
            raise TimeoutError("Child wait timed out; child is still running")
        return self.result(handle)

    def cancel(self, handle: str) -> None:
        """Request shutdown, escalate if necessary, and confirm process exit."""
        job = self._job(handle)
        job.stop.set()
        self._await_cleanup([job])

    def _signal(self, job: _Job, signum: int):
        if job.exitcode is not None:
            return
        try:
            os.kill(job.pid, signum)
        except ProcessLookupError:
            pass

    def _dispatch(self, job: _Job, method: str, args: tuple):
        try:
            if method == "spawn":
                if job.stop.is_set():
                    raise RuntimeError("Parent child is stopping")
                value = self._spawn(*args, parent=job)
            elif method == "status":
                value = self.status(*args)
            elif method == "result":
                value = self.result(*args)
            elif method == "stop":
                self._stop_descendants(job)
                value = None
            else:
                raise ValueError("Unknown child request")
            response = (True, value)
        except Exception as exc:
            response = (False, str(exc))
        with suppress(BrokenPipeError, ConnectionResetError):
            job.channel.send(response)

    def _stop_descendants(self, job: _Job):
        with self._lock:
            job.accepting = False
            descendants = [
                child for child in self._jobs.values() if child.parent is job
            ]
            for child in descendants:
                child.stop.set()
        self._await_cleanup(descendants)

    def _watch(self, job: _Job):
        shutdown_at = None
        term_at = None
        kill_at = None
        ready_at = None
        terminal_message = None
        channel_ended = job.channel.closed
        try:
            while True:
                now = time.monotonic()
                if not channel_ended:
                    try:
                        if job.channel.poll(0.01):
                            message = job.channel.recv()
                            if message[0] == "ready":
                                ready_at = now
                            elif message[0] == "request":
                                self._dispatch(job, message[1], message[2])
                            else:
                                terminal_message = message
                    except (EOFError, OSError):
                        channel_ended = True
                if channel_ended and terminal_message is None and not job.stop.is_set():
                    job.error = "Worker channel closed before completion"
                    job.stop.set()
                timeout = self.limits.per_child_timeout_s
                if (
                    ready_at is not None
                    and timeout is not None
                    and now - ready_at >= timeout
                ):
                    job.error = "Child timeout exceeded"
                    job.stop.set()
                if ready_at is None and now - job.started >= 30:
                    job.error = "Child startup timeout exceeded"
                    job.stop.set()
                if job.stop.is_set() and shutdown_at is None:
                    job.status = "stopping"
                    job.shutdown.close()
                    shutdown_at = now
                if (
                    shutdown_at is not None
                    and term_at is None
                    and now - shutdown_at >= self.config.shutdown_grace_s
                ):
                    self._signal(job, signal.SIGTERM)
                    term_at = now
                if (
                    term_at is not None
                    and kill_at is None
                    and now - term_at >= self.config.terminate_grace_s
                ):
                    self._signal(job, signal.SIGKILL)
                    kill_at = now
                if job.exitcode is None:
                    job.process.join(timeout=0.01 if channel_ended else 0)
                    job.exitcode = job.process.exitcode
                if job.exitcode is not None:
                    if (
                        terminal_message is not None
                        or channel_ended
                        or not job.channel.poll()
                    ):
                        break
                if kill_at is not None and now - kill_at >= self.config.kill_timeout_s:
                    raise ChildCleanupError("Worker exit not confirmed after SIGKILL")
            if job.stop.is_set():
                job.status = "failed" if job.error else "cancelled"
            elif (
                terminal_message
                and terminal_message[0] == "result"
                and job.exitcode == 0
            ):
                job.answer = terminal_message[1]
                job.status = "completed"
            else:
                job.status = "failed"
                job.error = (
                    terminal_message[1]
                    if terminal_message and terminal_message[0] == "error"
                    else f"Worker exited with code {job.exitcode}"
                )
        except Exception as exc:
            job.status = "cleanup_failed"
            job.error = str(exc)
            self._cleanup_failed = True
        finally:
            try:
                self._stop_descendants(job)
            except ChildCleanupError as exc:
                job.status = "cleanup_failed"
                job.error = str(exc)
                self._cleanup_failed = True
            if job.exitcode is not None:
                job.channel.close()
                job.shutdown.close()
                job.process.close()
            if not job.reported:
                self.child_traces.append(
                    ChildTrace(
                        job.depth,
                        time.monotonic() - job.started,
                        job.prompt[:80],
                        job.answer[:80] if job.answer else None,
                        job.error,
                    )
                )
                self._notify(
                    self.on_subcall_complete,
                    job.depth,
                    job.model or "default",
                    time.monotonic() - job.started,
                    job.error,
                )
                job.reported = True
            job.finished.set()

    def _await_cleanup(self, jobs: list[_Job]):
        deadline = (
            time.monotonic()
            + self.config.shutdown_grace_s
            + self.config.terminate_grace_s
            + self.config.kill_timeout_s
            + 1
        )
        for job in jobs:
            if job.status == "cleanup_failed" and job.finished.is_set():
                job.finished.clear()
                job.thread = threading.Thread(
                    target=self._watch, args=(job,), daemon=True
                )
                job.thread.start()
            if (
                not job.finished.wait(max(0, deadline - time.monotonic()))
                or job.exitcode is None
                or job.status == "cleanup_failed"
            ):
                self._cleanup_failed = True
                raise ChildCleanupError("Managed worker exit could not be confirmed")

    def stop(self):
        """Stop every registered worker, including nested workers, before returning."""
        with self._lifecycle_lock:
            with self._lock:
                self._accepting = False
                jobs = list(self._jobs.values())
                for job in jobs:
                    job.stop.set()
            self._await_cleanup(jobs)
            self._cleanup_failed = False

    def reset(self):
        """Drain the old episode before invalidating its handles and budget."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Managed child controller is closed")
            self.stop()
            with self._lock:
                self._jobs.clear()
                self.child_traces.clear()
                self._accepting = True

    def close(self):
        """Close permanently; a failed cleanup may be retried."""
        with self._lifecycle_lock:
            self.stop()
            if not self._closed:
                self._closed = True
                self._callback_queue.put(None)


class _ChildCancelled(BaseException):
    pass


class _RemoteChildren:
    def __init__(self, channel: Connection, shutdown: Connection):
        self.channel = channel
        self.shutdown = shutdown
        self.lock = threading.Lock()

    def check_ready(self):
        if self.shutdown.poll():
            raise _ChildCancelled()

    def _request(self, method: str, *args):
        with self.lock:
            self.check_ready()
            self.channel.send(("request", method, args))
            while not self.channel.poll(0.01):
                self.check_ready()
            success, value = self.channel.recv()
            if not success:
                raise RuntimeError(value)
            return value

    def spawn(self, prompt: str, model: str | None = None):
        return self._request("spawn", prompt, model)

    def status(self, handle: str):
        return self._request("status", handle)

    def result(self, handle: str):
        return self._request("result", handle)

    def wait(self, handle: str, timeout_s: float = 30.0):
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and nonnegative")
        deadline = time.monotonic() + timeout_s
        while self.status(handle) in ("running", "stopping"):
            if time.monotonic() >= deadline:
                raise TimeoutError("Child wait timed out; child is still running")
            self.check_ready()
            time.sleep(0.01)
        return self.result(handle)

    def stop(self):
        self._request("stop")

    def reset(self):
        self.check_ready()

    def close(self):
        self.stop()


class ManagedChildBackend(DirectLMBackend):
    """Keep existing blocking query APIs and add managed child handles."""

    def __init__(self, chat_fn, *, children, depth=0, limits=None):
        super().__init__(chat_fn, depth=depth, limits=limits)
        self.children = children
        if isinstance(children, ManagedChildren):
            self.child_traces = children.child_traces

    def recursive_query(self, prompt: str, model: str | None = None) -> str:
        if self.depth + 1 >= self.max_depth:
            return self.query(prompt, model)
        try:
            handle = self.children.spawn(prompt, model)
            while self.children.status(handle) in ("running", "stopping"):
                try:
                    return self.children.wait(handle, timeout_s=1)
                except TimeoutError:
                    pass
            return self.children.result(handle)
        except RuntimeError as exc:
            return f"Error: {exc}"

    def recursive_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        if self.limits.max_children_per_batch is not None:
            prompts = prompts[: self.limits.max_children_per_batch]
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=self.limits.max_batch_workers) as executor:
            return list(
                executor.map(
                    lambda prompt: self.recursive_query(prompt, model), prompts
                )
            )


def _run_child(chat_factory, channel, shutdown, prompt, model, depth, options):
    children = _RemoteChildren(channel, shutdown)
    try:
        channel.send(("ready",))
        chat = chat_factory()

        def checked_chat(messages, model=None):
            children.check_ready()
            try:
                result = chat(messages, model)
            except TypeError:
                result = chat(messages)
            children.check_ready()
            return result

        limits = BackendLimits(
            **{
                key: options[key]
                for key in (
                    "max_depth",
                    "max_batch_workers",
                    "max_children_per_batch",
                    "result_truncation_limit",
                )
            }
        )

        def backend_factory(chat_fn, **kwargs):
            return ManagedChildBackend(
                chat_fn, children=children, depth=kwargs["depth"], limits=limits
            )

        if depth >= limits.max_depth:
            answer = DirectLMBackend(checked_chat, limits=limits).query(prompt, model)
        else:
            from .runner import LocalRLMRunner

            result = LocalRLMRunner(
                checked_chat,
                depth=depth,
                backend_factory=backend_factory,
                **{
                    key: options[key]
                    for key in (
                        "max_depth",
                        "max_batch_workers",
                        "system_prompt",
                        "max_iterations",
                        "env_max_iterations_multiplier",
                    )
                },
            ).run(prompt, prompt, model=model)
            answer = result.final_answer or ""
            limit = limits.result_truncation_limit
            if limit is not None:
                answer = answer[:limit]
        with children.lock:
            channel.send(("result", answer))
    except _ChildCancelled:
        pass
    except Exception as exc:
        with children.lock:
            channel.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        channel.close()
        shutdown.close()
