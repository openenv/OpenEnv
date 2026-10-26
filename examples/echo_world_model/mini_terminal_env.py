"""A tiny, fully deterministic "terminal" environment for the ECHO demo.

**Scenario: forward-deployed incident triage.** An alert fires; an agent opens a
terminal in the customer's box and investigates — `grep` the logs, `cat` the
config, check service state — to find the root cause. ECHO is about *terminal*
agents, so this mirrors that setting in miniature: a fixed mini-filesystem (a
service that's throwing errors) plus a handful of read-only tools. The tool
outputs are **predictable** (deterministic from the filesystem) but **complex**
(multi-line, structured) — exactly the regime where world-modeling helps and
*generalizes* (you learn what `grep`/`cat`/`wc` *do* on this box, not a memorized
string).

This is OpenEnv-shaped (``reset()`` -> observation, ``step(action)`` ->
(observation, reward, done)) but dependency-free so the example runs anywhere. In
a real deployment this same terminal runs **inside an ACA Sandbox**, see
``backends/aca-sandboxes.md``. The point it demonstrates: every
``step()`` returns an observation the policy is conditioned on — that observation
stream is the free supervision ECHO trains on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- the fixed mini-filesystem (deterministic) ------------------------------
FILES: dict[str, list[str]] = {
    "app.log": [
        "INFO  boot sequence started",
        "INFO  cache warm complete",
        "WARN  retry budget low for shipping-svc",
        "ERROR payment-svc timeout after 3 retries",
        "INFO  request 200 /health",
        "ERROR db connection reset by peer",
        "WARN  disk usage at 81 percent",
        "INFO  request 200 /orders",
    ],
    "server.log": [
        "listen on 0.0.0.0:8080",
        "worker 1 ready",
        "worker 2 ready",
        "ERROR tls handshake failed for 10.1.2.9",
        "worker 3 ready",
    ],
    "config.yaml": [
        "service: orders",
        "port: 8080",
        "replicas: 3",
        "region: westus2",
        "retries: 3",
    ],
    "users.csv": [
        "id,name,plan",
        "1,alice,pro",
        "2,bob,free",
        "3,carol,pro",
    ],
}

TOOLS_DOC = (
    "You are in a read-only shell. Available commands:\n"
    "  ls                      list files\n"
    "  cat <file>              print a file\n"
    "  grep <pattern> <file>   print matching lines\n"
    "  wc <file>               count the lines in a file\n"
    "  head -n <N> <file>      print the first N lines\n"
    "When you know the answer, reply:  answer: <value>\n"
)


def run_command(cmd: str) -> str:
    """Execute one shell command against the fixed filesystem. Deterministic.

    Unknown commands / missing files return a realistic error line — those
    "failed" observations still teach the model what its action *caused*, which
    is exactly the dense signal ECHO harvests from otherwise-zero-reward turns.
    """
    cmd = cmd.strip()
    if cmd == "ls":
        return "  ".join(sorted(FILES))
    m = re.fullmatch(r"cat\s+(\S+)", cmd)
    if m:
        f = m.group(1)
        return "\n".join(FILES[f]) if f in FILES else f"cat: {f}: No such file"
    m = re.fullmatch(r"wc\s+(\S+)", cmd)
    if m:
        f = m.group(1)
        return f"{len(FILES[f])} {f}" if f in FILES else f"wc: {f}: No such file"
    m = re.fullmatch(r"head\s+-n\s+(\d+)\s+(\S+)", cmd)
    if m:
        n, f = int(m.group(1)), m.group(2)
        return "\n".join(FILES[f][:n]) if f in FILES else f"head: {f}: No such file"
    m = re.fullmatch(r"grep\s+(\S+)\s+(\S+)", cmd)
    if m:
        pat, f = m.group(1), m.group(2)
        if f not in FILES:
            return f"grep: {f}: No such file"
        hits = [ln for ln in FILES[f] if pat in ln]
        return "\n".join(hits) if hits else "(no matches)"
    return f"sh: {cmd.split()[0] if cmd else ''}: command not found"


@dataclass(frozen=True)
class Task:
    prompt: str
    solution: str
    oracle: tuple[str, ...]  # a command sequence that solves it (for scripted rollouts)


# Train and held-out tasks share the same *dynamics* AND the same files: training
# exposes each file's contents (via `cat`) so the model can learn "the world,"
# then we test on NEW command-slices of those same files (different grep patterns,
# head counts, counts) — so good held-out CE means it learned tool *semantics*,
# not a memorized string.
TRAIN_TASKS: list[Task] = [
    # expose the full contents of every file (the world state)
    Task("Show app.log", "", ("cat app.log",)),
    Task("Show server.log", "", ("cat server.log",)),
    Task("Show config.yaml", "", ("cat config.yaml",)),
    Task("Show users.csv", "", ("cat users.csv",)),
    # basic tool ops the model can learn the semantics of
    Task("List the files", "4", ("ls",)),
    Task("Find ERROR lines in app.log", "", ("grep ERROR app.log",)),
    Task("Find INFO lines in app.log", "", ("grep INFO app.log",)),
    Task("Count lines in app.log", "8 app.log", ("wc app.log",)),
    Task("Count lines in config.yaml", "5 config.yaml", ("wc config.yaml",)),
    Task("First 2 lines of app.log", "", ("head -n 2 app.log",)),
    Task("First line of server.log", "", ("head -n 1 server.log",)),
    Task("What port is configured?", "8080", ("grep port config.yaml",)),
]
# Held out: NEW commands over the SAME (already-seen) files.
TEST_TASKS: list[Task] = [
    Task("Find WARN lines in app.log", "", ("grep WARN app.log",)),
    Task("Find ERROR lines in server.log", "", ("grep ERROR server.log",)),
    Task("First 3 lines of app.log", "", ("head -n 3 app.log",)),
    Task("Count lines in server.log", "5 server.log", ("wc server.log",)),
    Task("What region is configured?", "westus2", ("grep region config.yaml",)),
    Task("First 2 lines of users.csv", "", ("head -n 2 users.csv",)),
]

SYSTEM = TOOLS_DOC


@dataclass
class MiniTerminalEnv:
    """OpenEnv-shaped deterministic terminal (reset/step)."""

    task: Task | None = None
    _turns: int = field(default=0)

    def reset(self, task: Task) -> str:
        self.task = task
        self._turns = 0
        return f"{SYSTEM}\nTask: {task.prompt}"

    def step(self, action: str) -> tuple[str, float, bool]:
        """One turn. `action` is either `answer: X` (ends episode) or a command."""
        self._turns += 1
        assert self.task is not None
        ans = re.search(r"answer:\s*(.+)", action, re.IGNORECASE)
        if ans:
            got = ans.group(1).strip()
            reward = 1.0 if got == self.task.solution else 0.0
            return f"(final answer: {got})", reward, True
        # otherwise treat the action as a shell command and return real output
        cmd = action.strip().splitlines()[-1] if action.strip() else ""
        cmd = cmd.removeprefix("$").removeprefix(">").strip()
        obs = run_command(cmd)
        done = self._turns >= 6
        return obs, 0.0, done
