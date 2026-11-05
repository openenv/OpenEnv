---
title: REPL Environment Server
emoji: 🎮
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# REPL Environment for OpenEnv

`repl_env` runs model-generated Python in a persistent namespace, with optional
recursive model queries, following the [RLM paradigm](https://huggingface.co/papers/2512.24601).

- `REPLEnv`: remote async client, or synchronous client via `.sync()`.
- `LocalREPLEnv`: the environment in process.
- `LocalRLMRunner`: the model prompting, code-execution, and recursion loop.

Try it on the [`openenv/repl`](https://huggingface.co/spaces/openenv/repl) Space, or connect a client to `https://openenv-repl.hf.space`.

## Quick Start

Start a server:

```bash
PYTHONPATH=src:envs uvicorn envs.repl_env.server.app:app --host 127.0.0.1 --port 8000
```

Async:

```python
import asyncio
from repl_env import REPLEnv


async def main():
    async with REPLEnv(base_url="http://127.0.0.1:8000") as env:
        result = await env.reset(
            context="alpha beta gamma",
            task_prompt="Count the words",
        )
        result = await env.execute("count = len(context.split())")
        result = await env.execute("print(FINAL(count))")
        print(result.done)


asyncio.run(main())
```

For synchronous remote use, enter `with REPLEnv(...).sync() as env:` and omit
`await`. Both clients also provide `submit_final_answer()`, `get_variable()`, and
`list_variables()`.

### In Process

```python
from repl_env import LocalREPLEnv

with LocalREPLEnv() as env:
    result = env.reset(
        context="The quick brown fox jumps over the lazy dog",
        task_prompt="Count the words",
    )
    result = env.execute("count = len(context.split())")
    result = env.execute("print(FINAL(count))")
    print(env.state().final_answer)
```

## Server Configuration

Environment variables read by [`server/app.py`](server/app.py):

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | unset | Enables `llm_query` on the server. Without it, a client can pass `hf_token` to `reset()` |
| `LLM_MODEL` | `Qwen/Qwen3.5-9B` | Default model for `llm_query`. A client can pass `llm_model` to `reset()` |
| `REPL_MAX_ITERATIONS` | `30` | Maximum steps per episode |
| `REPL_MAX_OUTPUT_LENGTH` | `8192` | Maximum captured output per step |
| `REPL_CONTEXT_PREVIEW_LENGTH` | `500` | Length of `context_preview` in observations |
| `REPL_RLM_MAX_DEPTH` | `2` | Maximum recursion depth for `rlm_query` |
| `REPL_RLM_MAX_ITERATIONS` | `30` | Maximum iterations for recursive child runs |
| `MAX_CONCURRENT_ENVS` | `8` | Maximum concurrent sessions |

## Reward

The default `REPLRubric` gives 1.0 for a matching `expected_answer`, 0.0 for a
mismatch or successful intermediate step, -0.05 for execution errors, and -0.1
when iterations run out without an answer. Set the expected answer at reset:

```python
with LocalREPLEnv() as env:
    env.reset(expected_answer="42")
    result = env.execute("print(FINAL(42))")
    print(result.reward)
```

Customize `LocalREPLEnv(rubric=...)` with the match, partial-credit, custom-metric,
or execution rubrics in [`rubrics.py`](rubrics.py). See the
[rubric guide](https://huggingface.co/docs/openenv/guides/rewards) for composition.

## Running an RLM Locally

`LocalRLMRunner` accepts any `chat_fn(messages, model=None) -> str`, including
adapters for hosted or local model servers. For HF inference, save this reusable
factory in `my_model.py` (uses `HF_TOKEN` or your cached login):

```python
from huggingface_hub import InferenceClient


def make_chat():
    client = InferenceClient(model="Qwen/Qwen3.5-9B", timeout=300)

    def chat(messages, model=None):
        response = client.chat.completions.create(
            model=model or "Qwen/Qwen3.5-9B",
            messages=messages,
            max_tokens=2048,
            temperature=0.6,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content or ""

    return chat
```

```python
from my_model import make_chat
from repl_env import LocalRLMRunner

runner = LocalRLMRunner(make_chat(), max_iterations=30, max_depth=2)
result = runner.run("The answer is 42", "What number is mentioned?")
print(result.final_answer)
```

The runner supports child-count, batch, timeout, and result-length limits, plus
`on_subcall_start` / `on_subcall_complete` callbacks and child traces. For different
outer/inner models, supply a `backend_factory`; see
[`recursive_backends.py`](recursive_backends.py).

### Managed child lifecycle (opt-in)

Managed children let the parent keep working while queries run in supervised
processes. Each worker builds its own client; the model server stays shared.
Use an importable factory such as `my_model.make_chat` above, not a lambda or live
client, and guard the entry point for Python's `spawn` start method:

```python
from my_model import make_chat
from repl_env import LocalRLMRunner, ManagedChildConfig

if __name__ == "__main__":
    runner = LocalRLMRunner(
        make_chat(),
        managed_children=ManagedChildConfig(chat_factory=make_chat),
        max_depth=3,
        max_iterations=10,
        max_children_total=8,
        max_batch_workers=4,
        per_child_timeout_s=60,
    )
    print(runner.run("alpha beta gamma", "Count the words").final_answer)
```

Inside the model's REPL:

```python
child = rlm_query("Count the words: alpha beta gamma", blocking=False)
parent_count = len(context.split())
child_answer = rlm_wait(child, timeout_s=30)
print(FINAL(parent_count))
```

- `rlm_query(prompt, model=None, *, blocking=True)` waits for an answer by default.
  With `blocking=False`, it returns an episode-scoped handle instead and requires
  managed mode. Use `rlm_status` to inspect the handle,
  `rlm_result` for a completed answer, or `rlm_wait` for a bounded wait. Failed or
  unfinished results raise errors; a wait timeout does **not** cancel the child.
- Nested workers share child-count and concurrency limits. Full concurrency
  rejects new spawns rather than queuing them. Child timeouts interrupt even
  in-flight model calls; deeper explicit spawns are rejected at the depth limit.
- Finalization, reset, and close stop admission, request cooperative shutdown,
  then escalate to SIGTERM and SIGKILL. Cleanup waits for every registered worker
  to exit; failure raises `ChildCleanupError` and blocks reuse. Reset invalidates
  handles and replenishes the budget; close is permanent and idempotent.
- Controller lifecycle transitions are serialized. Callbacks are asynchronous,
  ordered notifications: blocked or failing observers do not delay cleanup, and
  notifications may run after reset/close. REPL execution itself is not thread-safe.

`ManagedChildConfig` exposes `shutdown_grace_s`, `terminate_grace_s`, and
`kill_timeout_s` (defaults: 1, 1, and 5 seconds). This POSIX-only backend does not
contain arbitrary subprocesses or guarantee server-side generation abort.
User-facing cancellation and follow-up messages are deferred.

For manual/server use, pass `managed_children` to
[`create_server_recursive_controller`](recursive_controller.py), then supply that
controller to `LocalREPLEnv` or `REPLEnvironment` via `recursive_controller=`.
Configure its credentials and limits at construction, not through reset overrides;
infrastructure can cancel a handle with `controller.children.cancel(handle)`.
`managed_children` cannot be combined with a custom `backend_factory`.

## Actions and Observations

`REPLAction` carries `code`, or `is_final=True` with a `final_answer` for explicit
submission. `REPLObservation` includes execution results (stdout/stderr, success,
exceptions, and timing), context/iteration metadata, `done`, and `reward`.
See [`models.py`](models.py) for all fields.

## REPL Helpers

When configured, the REPL namespace exposes:

- `llm_query(prompt, model=None)` and `llm_query_batched(prompts, model=None)`
- `rlm_query(prompt, model=None, *, blocking=True)` and `rlm_query_batched(prompts, model=None)`: recursive child runs that wait for their answers. Blocking queries fall back to direct model calls at the recursion limit. In managed mode, `rlm_query(..., blocking=False)` starts a background child and returns a handle.
- `SHOW_VARS()` lists variables in the namespace.

Finish by executing `print(FINAL(value))`, `print(FINAL_VAR("variable_name"))`, or
`answer.update({"content": value, "ready": True})`.

## Prompts and Examples

See [`prompts.py`](prompts.py) for system prompts and message/code-block helpers.

Examples, which default to `Qwen/Qwen3.5-9B` through Hugging Face inference (needs `HF_TOKEN`):

- [`examples/repl_with_llm.py`](../../examples/repl_with_llm.py)
- [`examples/repl_oolong_simple.py`](../../examples/repl_oolong_simple.py)

## References

- [RLM Paper (arXiv:2512.24601)](https://huggingface.co/papers/2512.24601)
- [RLM Implementation](https://github.com/alexzhang13/rlm)
- [Alex Zhang's RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Prime Intellect RLM Blog](https://www.primeintellect.ai/blog/rlm)
