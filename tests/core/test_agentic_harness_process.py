# SPDX-License-Identifier: BSD-3-Clause

"""Subprocess lifecycle tests for HarnessProcess (real processes)."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from openenv.core.harness import (
    HarnessNotRunningError,
    HarnessProcess,
    HarnessStartupError,
)

# Subprocess behaviors live in a real, lintable module (see its docstring
# for the available modes) instead of inline code strings.
SCRIPTED_HARNESS = Path(__file__).parent / "scripted_harness.py"


def make_process(mode: str, **kwargs) -> HarnessProcess:
    defaults = {"startup_timeout_s": 10.0, "terminate_grace_s": 1.0}
    defaults.update(kwargs)
    return HarnessProcess(
        [sys.executable, "-u", str(SCRIPTED_HARNESS), mode], cwd=".", **defaults
    )


def ready(line: str) -> bool:
    return line == "ready"


class TestLifecycle:
    async def test_start_echo_read_stop(self):
        process = make_process("echo")
        assert process.is_running() is False
        await process.start(ready_check=ready)
        assert process.is_running() is True

        await process.write_line("hello")
        assert await process.read_line(timeout_s=10.0) == "echo:hello"

        await process.stop()
        assert process.is_running() is False

    async def test_stop_is_idempotent(self):
        process = make_process("echo")
        await process.start(ready_check=ready)
        await process.stop()
        await process.stop()
        assert process.is_running() is False
        assert process._proc is None
        assert process._reader_threads == []

    async def test_stop_before_start_is_noop(self):
        process = make_process("echo")
        await process.stop()
        assert process.is_running() is False

    async def test_double_start_rejected(self):
        process = make_process("echo")
        await process.start(ready_check=ready)
        try:
            with pytest.raises(HarnessStartupError, match="already running"):
                await process.start(ready_check=ready)
        finally:
            await process.stop()


class TestStartupFailures:
    @pytest.mark.parametrize("cancel", [False, True])
    async def test_interrupted_readiness_cleans_up_process(self, cancel):
        process = make_process("echo")
        checking = asyncio.Event()
        resources = []

        def check(line):
            resources.append((process._proc, list(process._reader_threads)))
            checking.set()
            if not cancel:
                raise ValueError("readiness check failed")
            return False

        start_task = asyncio.create_task(process.start(ready_check=check))
        try:
            await asyncio.wait_for(checking.wait(), timeout=10.0)
            proc, readers = resources[0]
            if cancel:
                start_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await start_task
            else:
                with pytest.raises(ValueError, match="readiness check failed"):
                    await start_task

            assert proc.poll() is not None
            assert all(s.closed for s in (proc.stdin, proc.stdout, proc.stderr))
            assert all(not thread.is_alive() for thread in readers)
            assert process._proc is None
        finally:
            if not start_task.done():
                start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)
            await process.stop()

    async def test_startup_timeout_kills_process(self):
        process = make_process("slow-start", startup_timeout_s=0.5)
        with pytest.raises(HarnessStartupError, match="did not become ready"):
            await process.start(ready_check=ready)
        assert process.is_running() is False

    async def test_immediate_exit_reports_code_and_retains_diagnostics(self):
        process = make_process("exit-now")
        with pytest.raises(HarnessStartupError) as exc_info:
            await process.start(ready_check=ready)
        assert "exited with code 3" in str(exc_info.value)
        assert "dying" in process.drain_stderr()
        assert "dying" not in str(exc_info.value)

    async def test_startup_error_does_not_expose_stderr(self):
        process = make_process("exit-with-secret")
        with pytest.raises(HarnessStartupError) as exc_info:
            await process.start(ready_check=ready)
        assert "exited with code 3" in str(exc_info.value)
        assert "FAKE_HARNESS_SECRET_FOR_TEST" not in str(exc_info.value)
        assert "stderr tail" not in str(exc_info.value)
        assert "FAKE_HARNESS_SECRET_FOR_TEST" in process.drain_stderr()

    async def test_unspawnable_command(self):
        process = HarnessProcess(
            ["/nonexistent/definitely-not-a-binary", "FAKE_HARNESS_SECRET_FOR_TEST"],
            cwd=".",
        )
        with pytest.raises(HarnessStartupError, match="failed to spawn") as exc_info:
            await process.start()
        assert "FAKE_HARNESS_SECRET_FOR_TEST" not in str(exc_info.value)
        assert "/nonexistent" not in str(exc_info.value)


class TestCrashDetection:
    @pytest.mark.parametrize("restart_fails", [False, True])
    async def test_restart_cleans_up_exited_process(self, restart_fails):
        process = make_process("crash-after-echo")
        await process.start(ready_check=ready)
        old_proc = process._proc
        old_readers = list(process._reader_threads)
        try:
            await process.write_line("old turn")
            assert await asyncio.to_thread(old_proc.wait, timeout=10.0) == 1

            if restart_fails:
                process.command = ["/nonexistent/definitely-not-a-binary"]
                with pytest.raises(HarnessStartupError, match="failed to spawn"):
                    await process.start()
            else:
                await process.start(ready_check=ready)

            assert all(
                stream.closed
                for stream in (old_proc.stdin, old_proc.stdout, old_proc.stderr)
            )
            assert all(not thread.is_alive() for thread in old_readers)

            if not restart_fails:
                await process.write_line("new turn")
                assert await process.read_line(timeout_s=10.0) == "echo:new turn"
        finally:
            await process.stop()
            # Also release the original handles if a regression loses them.
            for stream in (old_proc.stdin, old_proc.stdout, old_proc.stderr):
                stream.close()
            for thread in old_readers:
                thread.join(timeout=2.0)

    async def test_crash_mid_session(self):
        process = make_process("crash-after-echo")
        await process.start(ready_check=ready)
        await process.write_line("one")
        assert await process.read_line(timeout_s=10.0) == "echo:one"

        # Process exits after the first echo; wait for it to die
        deadline = time.monotonic() + 10.0
        while process.is_running() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert process.is_running() is False

        with pytest.raises(HarnessNotRunningError):
            await process.write_line("two")
        await process.stop()

    async def test_read_line_returns_none_on_timeout(self):
        process = make_process("echo")
        await process.start(ready_check=ready)
        try:
            assert await process.read_line(timeout_s=0.1) is None
        finally:
            await process.stop()


class TestEOF:
    @pytest.mark.parametrize("timeout_s", [None, 10.0])
    async def test_exit_preserves_buffered_lines_before_repeated_eof(self, timeout_s):
        process = make_process("exit-with-output")
        await process.start()
        try:
            deadline = time.monotonic() + 10.0
            while process.is_running() and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert process.is_running() is False

            for expected in ("first", "second", "last"):
                assert (
                    await asyncio.wait_for(process.read_line(timeout_s), timeout=2.0)
                    == expected
                )
            for _ in range(2):
                assert (
                    await asyncio.wait_for(process.read_line(timeout_s), timeout=2.0)
                    is None
                )
        finally:
            await process.stop()

    @pytest.mark.parametrize("timeout_s", [None, 10.0])
    async def test_read_line_returns_none_when_stdout_closes(self, timeout_s):
        process = make_process("close-stdout")
        await process.start(ready_check=ready)
        try:
            await process.write_line("close")
            for _ in range(2):
                assert (
                    await asyncio.wait_for(process.read_line(timeout_s), timeout=2.0)
                    is None
                )
            assert process.is_running() is True
        finally:
            await process.stop()

    async def test_stop_wakes_pending_read_and_restart_can_read_output(self):
        process = make_process("echo")
        await process.start(ready_check=ready)
        try:
            pending_read = asyncio.create_task(process.read_line())
            await asyncio.sleep(0)
            await process.stop()
            assert await asyncio.wait_for(pending_read, timeout=2.0) is None
            assert await asyncio.wait_for(process.read_line(), timeout=2.0) is None

            await process.start(ready_check=ready)
            await process.write_line("again")
            assert (
                await asyncio.wait_for(process.read_line(), timeout=2.0) == "echo:again"
            )
        finally:
            await process.stop()


class TestTerminateEscalation:
    @pytest.mark.parametrize("leader_exits_first", [False, True])
    async def test_stop_kills_descendants_after_leader_exits(self, leader_exits_first):
        process = make_process("spawn-descendant", terminate_grace_s=0.1)
        await process.start(ready_check=ready)
        proc = process._proc
        readers = list(process._reader_threads)
        stop_task = None
        try:
            descendant_pid = int(await process.read_line(timeout_s=5.0))
            if leader_exits_first:
                await process.write_line("exit")
                assert await asyncio.to_thread(proc.wait, timeout=5.0) == 1

            stop_task = asyncio.create_task(process.stop())
            done, _ = await asyncio.wait([stop_task], timeout=2.0)
            assert stop_task in done, "stop() blocked on a surviving descendant's pipes"
            await stop_task
            assert all(not reader.is_alive() for reader in readers)
            assert all(
                stream.closed for stream in (proc.stdin, proc.stdout, proc.stderr)
            )
            assert process._proc is None

            # This child keeps both output pipes open until it exits. EOF
            # therefore proves teardown without depending on init reaping it.
            assert descendant_pid != proc.pid
            assert await process.read_line(timeout_s=1.0) is None
        finally:
            # A regression must fail promptly without leaking the sleeping child.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if stop_task is not None:
                await asyncio.wait_for(stop_task, timeout=5.0)
            await process.stop()

    async def test_sigterm_ignorer_is_killed_within_grace(self):
        process = make_process("ignore-sigterm", terminate_grace_s=0.5)
        await process.start(ready_check=ready)

        started = time.monotonic()
        await process.stop()
        elapsed = time.monotonic() - started

        assert process.is_running() is False
        assert elapsed < 5.0  # grace (0.5s) + margin, well under the 60s sleep


class TestEncoding:
    async def test_non_ascii_output_survives_the_reader(self):
        # text=True alone decodes with the locale encoding (often ASCII in a
        # container), and UnicodeDecodeError is a ValueError, which the reader
        # thread used to swallow -- killing stdout pumping for the whole turn.
        process = make_process("unicode-echo")
        await process.start(ready_check=ready)
        try:
            await process.write_line("hello")
            line = await process.read_line(timeout_s=10.0)
            assert line == "echo:✓ hello 世界 \U0001f600"

            # The pump is still alive for subsequent turns.
            await process.write_line("again")
            assert await process.read_line(timeout_s=10.0) is not None
        finally:
            await process.stop()
