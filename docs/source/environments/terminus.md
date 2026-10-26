<!-- openenv-source: terminus_env -->
# Terminus Environment

`terminus_env` is a single-tool coding environment backed by E2B Code
Interpreter. Each OpenEnv episode creates a fresh E2B sandbox, runs optional
setup commands, keeps files isolated for that episode, and runs
optional verify commands when the agent submits a final answer.

The tool shape follows the Terminus-style "one tool" idea: agents do their work
through a single terminal entrypoint rather than a notebook/toolbox surface.

## Tool

- `terminal(command="", final_answer="")`: run a shell command inside the
  session sandbox, or submit a final answer and run verification.

## Quick Start

```python
from openenv.core.env_server.mcp_types import CallToolAction
from terminus_env import TerminusEnv

with TerminusEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(
        setup=["mkdir -p /home/user/work"],
        verify=["test -f /home/user/work/answer.txt"],
    )
    env.step(CallToolAction(tool_name="terminal", arguments={"command": "echo done > /home/user/work/answer.txt"}))
    result = env.step(CallToolAction(tool_name="terminal", arguments={"final_answer": "done"}))
    print(result.reward)
```

## Local Server

```bash
cd envs/terminus_env
E2B_API_KEY=e2b_... uv run --project . server
```

The API and custom terminal web UI are served on port 8000. The UI is mounted
at `/web`.

## Docker

```bash
cd envs/terminus_env
openenv build -t terminus-env
docker run -p 8000:8000 -e E2B_API_KEY=e2b_... terminus-env
```

## Configuration

- `E2B_API_KEY`: required when resetting an episode.
- `MAX_CONCURRENT_ENVS`: maximum concurrent WebSocket sessions. Defaults to `4`.

## Setup and Verify Commands

`reset()` accepts either `setup` / `verify` or `setup_scripts` /
`verify_scripts`.

```python
env.reset(
    setup=["pip install -q pytest"],
    verify=["pytest -q /home/user/work/tests"],
)
```

Setup failure ends the reset response with `done=True` and returns captured
setup results. Verify commands run when `terminal(final_answer="...")` is
called. Reward defaults to `passed_verify_commands / total_verify_commands`. A
verify command can override this by writing a float to:

```text
/home/user/logs/verifier/reward.txt
```

The file is deleted before the verify commands run, discarding earlier
contents. Policy background processes can still write it during verification;
this convention is not a trusted boundary against an adversarial policy. The value
must be a finite number; a verify command is written by the task author, so
negative rewards and values above 1 are accepted too. A value that is not a
finite number is ignored, the pass rate is used, and the reason is recorded in
`state.reward_override_ignored`.

## Episode lifetime and cleanup

Files persist between steps, but each command starts a new shell process: `cd` and exported variables do not carry to the next call. Reset kills the previous sandbox and creates a fresh one. Use `step(CallToolAction(...))` for `done`/reward postprocessing; direct `call_tool()` only invokes the MCP tool.

The SDK's default sandbox timeout is 300 seconds. The orchestrator must budget for batch startup and model generation, manage known sandbox IDs through the E2B control API when a longer lifetime is needed, and treat expiry as an infrastructure failure. Local request cancellation does not prove remote work stopped.

A failed deletion raises to the direct caller and retains ownership for retry. The OpenEnv server may still suppress teardown errors; closing a network client alone is not confirmation of deletion. Reconcile known owned IDs through E2B and report unresolved cleanup. Do not replay uncertain stateful steps or replace an episode silently.

For E2B installation, sandbox operations and integration recipes, see the [E2B documentation](https://docs.e2b.dev/) and [E2B cookbook](https://github.com/e2b-dev/e2b-cookbook).
