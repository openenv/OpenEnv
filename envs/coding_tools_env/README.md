---
title: Coding Tools Environment Server
emoji: 🧰
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - coding
  - tools
  - e2b
short_description: SETA-style multi-tool coding environment backed by E2B
---

# Coding Tools Environment

`coding_tools_env` is an E2B-backed multi-tool coding environment with explicit
filesystem and shell tools. The tool surface is modeled after
[SETA](https://github.com/EnvCommons/SETA).

Each episode creates a fresh E2B sandbox. Reset accepts setup and verify
commands. Verify commands are used by `submit_solution`.

Reward defaults to `passed_verify_commands / total_verify_commands`. A verify
command can override this by writing a float to:

```text
/home/user/logs/verifier/reward.txt
```

The file is deleted before the verify commands run, so only a verify command
can set it; anything the agent writes there earlier is discarded. The value
must be a finite number; a verify command is written by the task author, so
negative rewards and values above 1 are accepted too. A value that is not a
finite number is ignored, the pass rate is used, and the reason is recorded in
`state.reward_override_ignored`.

Verify commands, the clearing of that file and the read of it run as their own
sandbox processes through E2B's process API, as `root` in `/home/user`, rather
than as a `subprocess.run` cell in the code-interpreter kernel. Nothing done to
the kernel, such as a rebound `subprocess.run`, a changed working directory or
an edited `os.environ`, reaches verification. This removes the coupling to the
kernel; it does not isolate verification from the agent. The agent's `bash`
tool runs as root in the same sandbox, so it can still change anything there,
including the files a verify command reads and the startup files of the shell
it runs in. E2B starts each command as a login shell, so a profile the agent
writes decides what a verify command reports: an `exit 0` or an `EXIT` trap in
`/root/.bash_profile` makes a failing command look like a passing one. A test
pins that behaviour, and #1232 tracks verification outside the agent's sandbox,
which is what closes it.

Only a command that ran and exited decides a verify result. If the sandbox
itself fails, because it cannot be reached, the command cannot be started, or
the wait for it ends without an exit status (a verify command that outlives its
120 s timeout, for example), the error is raised instead: verification stops,
no reward is produced, and the episode does not finish. A command with no exit
status may still be running, since E2B's kill signals the command's own process
and not the children it started.

## Tools

- `bash(command, timeout=30)`
- `read(file_path, offset=None, limit=None)`
- `write(file_path, content)`
- `edit(file_path, old_string, new_string, replace_all=False)`
- `multi_edit(file_path, edits)`
- `glob(pattern, path=None)`
- `grep(pattern, path=None, include=None)`
- `ls(path=".", ignore=None)`
- `todo_write(todos)`
- `submit_solution()`

## Quick Start

```python
from coding_tools_env import CodingToolsEnv

with CodingToolsEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(
        setup=["mkdir -p /home/user/work"],
        verify=["test -f /home/user/work/answer.txt"],
    )
    print(env.call_tool("write", file_path="/home/user/work/answer.txt", content="done\n"))
    print(env.call_tool("submit_solution"))
```

## Local Server

```bash
cd envs/coding_tools_env
E2B_API_KEY=e2b_... uv run --project . server
```

## Docker

```bash
cd envs/coding_tools_env
openenv build -t coding-tools-env
docker run -p 8000:8000 -e E2B_API_KEY=e2b_... coding-tools-env
```

## Configuration

- `E2B_API_KEY`: required when resetting an episode.
- `MAX_CONCURRENT_ENVS`: max concurrent sessions (default `4`).

