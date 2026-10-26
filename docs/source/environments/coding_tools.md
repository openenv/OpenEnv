<!-- openenv-source: coding_tools_env -->
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

