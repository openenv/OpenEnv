# SPDX-License-Identifier: BSD-3-Clause

"""Scripted subprocess behaviors for the agentic harness tests.

This module is not imported by the tests — it is *spawned* by them
(``python scripted_harness.py <mode>``) to play the role of an external CLI
harness, so the subprocess-facing code lives in a real, lintable file instead
of inline strings.

Modes:
  mcp              - full fake harness: announces readiness, then for each
                     JSON stdin line calls the env tool over the MCP bridge
                     (from the config file at ``$HARNESS_MCP_CONFIG``) and
                     emits JSON-line events ending in ``turn_complete``.
  echo             - print ``ready``, then echo each stdin line.
  slow-start       - sleep past any startup timeout before printing ``ready``.
  exit-now         - print to stderr and exit with code 3.
  exit-with-secret - print a fake credential to stderr, then exit with code 3.
  spawn-descendant - spawn a SIGTERM-ignoring child with inherited pipes.
  descendant       - print its PID, ignore SIGTERM, and sleep.
  exit-with-output - print multiple lines, including an unterminated final line.
  close-stdout     - close stdout on request, then wait on stdin without exiting.
  crash-after-echo - ``ready``, echo one line, then exit with code 1.
  ignore-sigterm   - ignore SIGTERM, print ``ready``, then sleep.
  unicode-echo     - like echo, but the reply carries non-ASCII characters.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time


def mode_mcp() -> None:
    """Fake agentic harness speaking JSON lines over stdio."""
    with open(os.environ["HARNESS_MCP_CONFIG"]) as f:
        config = json.load(f)
    bridge_url = config.get("mcp_url")
    tool_names = config.get("tools", [])

    def emit(payload: dict) -> None:
        print(json.dumps(payload), flush=True)

    emit({"event": "ready", "tools": tool_names})

    async def run_turn(message: str) -> None:
        if bridge_url and "add" in tool_names:
            from fastmcp import Client

            arguments = {"a": 2, "b": 3}
            emit({"event": "tool_call", "tool_name": "add", "arguments": arguments})
            async with Client(bridge_url) as client:
                result = await client.call_tool("add", arguments)
            value = result.content[0].text
            emit({"event": "tool_result", "tool_name": "add", "result": value})
            response = f"sum={value} for: {message}"
        else:
            response = f"no tools for: {message}"
        emit(
            {
                "event": "turn_complete",
                "response": response,
                "done": "finish" in message,
            }
        )

    for line in sys.stdin:
        request = json.loads(line)
        asyncio.run(run_turn(request["message"]))


def mode_echo() -> None:
    print("ready", flush=True)
    for line in sys.stdin:
        print("echo:" + line.strip(), flush=True)


def mode_unicode_echo() -> None:
    """Echo with non-ASCII output, which locale-encoded pipes would fail on."""
    print("ready", flush=True)
    for line in sys.stdin:
        print("echo:\u2713 " + line.strip() + " \u4e16\u754c \U0001f600", flush=True)


def mode_slow_start() -> None:
    time.sleep(60)
    print("ready", flush=True)


def mode_exit_now() -> None:
    print("dying", file=sys.stderr, flush=True)
    sys.exit(3)


def mode_exit_with_secret() -> None:
    print("FAKE_HARNESS_SECRET_FOR_TEST", file=sys.stderr, flush=True)
    sys.exit(3)


def mode_spawn_descendant() -> None:
    print("ready", flush=True)
    subprocess.Popen([sys.executable, "-u", __file__, "descendant"])
    sys.stdin.readline()
    sys.exit(1)


def mode_descendant() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
    time.sleep(60)


def mode_crash_after_echo() -> None:
    print("ready", flush=True)
    line = sys.stdin.readline()
    print("echo:" + line.strip(), flush=True)
    sys.exit(1)


def mode_exit_with_output() -> None:
    print("first")
    print("second")
    print("last", end="", flush=True)


def mode_close_stdout() -> None:
    print("ready", flush=True)
    sys.stdin.readline()
    os.close(sys.stdout.fileno())
    sys.stdin.readline()


def mode_ignore_sigterm() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("ready", flush=True)
    time.sleep(60)


MODES = {
    "mcp": mode_mcp,
    "echo": mode_echo,
    "unicode-echo": mode_unicode_echo,
    "slow-start": mode_slow_start,
    "exit-now": mode_exit_now,
    "exit-with-secret": mode_exit_with_secret,
    "spawn-descendant": mode_spawn_descendant,
    "descendant": mode_descendant,
    "exit-with-output": mode_exit_with_output,
    "close-stdout": mode_close_stdout,
    "crash-after-echo": mode_crash_after_echo,
    "ignore-sigterm": mode_ignore_sigterm,
}


if __name__ == "__main__":
    MODES[sys.argv[1]]()
