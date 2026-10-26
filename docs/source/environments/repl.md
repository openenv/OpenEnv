<!-- openenv-source: repl_env -->
# REPL Environment for OpenEnv

`repl_env` is a Python REPL environment for [Recursive Language Model](https://huggingface.co/papers/2512.24601) (RLM) style execution. The model writes code that runs in a persistent namespace and can:

- inspect `context`
- execute Python across multiple turns with persistent state
- call `llm_query(...)` and `llm_query_batched(...)` to query a language model
- call `rlm_query(...)` and `rlm_query_batched(...)` for recursive child runs, when configured
- finish with `FINAL(...)`, `FINAL_VAR(...)`, or `answer = {"content": ..., "ready": True}`

The package provides:

- `REPLEnv`: the async client for a remote server (`.sync()` for synchronous code), with `execute(code)`, `submit_final_answer(answer)`, `get_variable(name)` and `list_variables()` on top of `reset()`/`step()`
- `LocalREPLEnv`: the same environment, in process
- `LocalRLMRunner`: a local RLM loop that prompts a model, runs its code and handles recursion

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

Sync:

```python
from repl_env import REPLEnv

with REPLEnv(base_url="http://127.0.0.1:8000").sync() as env:
    result = env.reset(
        context="alpha beta gamma",
        task_prompt="Count the words",
    )
    result = env.execute("count = len(context.split())")
    result = env.execute("print(FINAL(count))")
    print(result.observation.result.stdout)
```

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

Environment variables read by [`server/app.py`](https://github.com/huggingface/OpenEnv/blob/main/envs/repl_env/server/app.py):

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

Rewards use the OpenEnv [rubric system](https://huggingface.co/docs/openenv/guides/rewards). The default `REPLRubric` combines:

- **Outcome reward** (on terminal steps): compares `final_answer` against
  `expected_answer` if provided. Returns 1.0 for match, 0.0 otherwise.
- **Process reward** (on non-terminal steps): returns -0.05 for code
  execution errors, 0.0 for successful steps.
- **Failure reward**: returns -0.1 when max iterations exhausted without an answer.

For RL training (GRPO, etc.), pass `expected_answer` at reset time:

```python
with LocalREPLEnv() as env:
    env.reset(
        context="...",
        task_prompt="...",
        expected_answer="42",  # ground truth for rubric scoring
    )
    result = env.execute("print(FINAL(42))")
    print(result.reward)  # 1.0 (correct)
```

Other rubrics: `ExactMatchRubric` (binary match), `FuzzyMatchRubric` (1.0 for an exact match, 0.5 when the expected answer is contained in the final answer), `CustomMetricRubric` (your `metric(expected, predicted) -> float`) and `CodeExecutionRubric` (per-step error penalty). Pass one at construction:

```python
from repl_env import LocalREPLEnv, CustomMetricRubric, REPLRubric

def my_metric(expected, predicted):
    return 1.0 if expected.strip() == predicted.strip() else 0.0

env = LocalREPLEnv(rubric=REPLRubric(outcome=CustomMetricRubric(my_metric)))
```

## Running an RLM Locally

`LocalRLMRunner` takes any `chat_fn(messages, model=None) -> str`. It works
with HF Inference API, vLLM, SGLang, Ollama, or any OpenAI-compatible server.

With HF Inference API:

```python
from huggingface_hub import InferenceClient
from repl_env import LocalRLMRunner, RLM_SYSTEM_PROMPT

client = InferenceClient(model="Qwen/Qwen3.5-9B", timeout=300)

def chat_fn(messages, model=None):
    response = client.chat.completions.create(
        model=model or "Qwen/Qwen3.5-9B",
        messages=messages,
        max_tokens=2048,
        temperature=0.6,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content

runner = LocalRLMRunner(chat_fn, max_iterations=30, max_depth=2)
result = runner.run("The answer is 42", "What number is mentioned?")
print(result.final_answer)
```

With a local vLLM server:

```python
from openai import OpenAI
from repl_env import LocalRLMRunner

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

def chat_fn(messages, model=None):
    response = client.chat.completions.create(
        model=model or "Qwen/Qwen3.5-9B",
        messages=messages,
        max_tokens=2048,
        temperature=0.6,
    )
    return response.choices[0].message.content

runner = LocalRLMRunner(chat_fn, max_iterations=30, max_depth=2)
result = runner.run(context, task)
```

### Different Models for Outer and Inner Loops

The outer loop (code generation) can use a large model while inner
`llm_query`/`rlm_query` calls use a smaller, faster model. Pass a
custom `backend_factory` to the runner:

```python
from openai import OpenAI
from huggingface_hub import InferenceClient
from repl_env import LocalRLMRunner
from repl_env.recursive_backends import BackendLimits, LocalChildRLMBackend

# Outer loop: large local model via vLLM
vllm = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

def outer_chat(messages, model=None):
    r = vllm.chat.completions.create(
        model="Qwen/Qwen3-32B", messages=messages, max_tokens=2048,
    )
    return r.choices[0].message.content

# Inner calls (llm_query/rlm_query): smaller HF-hosted model
hf = InferenceClient(model="Qwen/Qwen3.5-9B")

def inner_chat(messages, model=None):
    r = hf.chat.completions.create(
        model=model or "Qwen/Qwen3.5-9B", messages=messages, max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return r.choices[0].message.content

def my_backend_factory(llm_chat_fn, **kwargs):
    return LocalChildRLMBackend(
        inner_chat,  # inner calls use the smaller model
        runner_factory=LocalRLMRunner,
        system_prompt=kwargs["system_prompt"],
        max_iterations=kwargs["max_iterations"],
        env_max_iterations_multiplier=kwargs["env_max_iterations_multiplier"],
        depth=kwargs["depth"],
        limits=BackendLimits(max_depth=2),
    )

runner = LocalRLMRunner(
    outer_chat,                        # outer loop: large model
    backend_factory=my_backend_factory, # inner calls: small model
    max_iterations=30,
    max_depth=2,
)
result = runner.run(context, task)
```

`LocalRLMRunner` also takes recursion limits (`max_children_total`, `max_children_per_batch`, `per_child_timeout_s`, `result_truncation_limit`) and lifecycle callbacks (`on_subcall_start(depth, model, prompt_preview)`, `on_subcall_complete(depth, model, duration, error_or_none)`). Its results carry lightweight child trace metadata.

## Actions and Observations

`REPLAction`

```python
code: str = ""
is_final: bool = False
final_answer: str | None = None
```

`REPLObservation`

```python
result: CodeBlockResult
context_preview: str | None
context_length: int
available_variables: list[str]
iteration: int
max_iterations: int
done: bool
reward: float | None
metadata: dict
```

## REPL Helpers

When configured, the REPL namespace exposes:

- `llm_query(prompt, model=None)` and `llm_query_batched(prompts, model=None)`
- `rlm_query(prompt, model=None)` and `rlm_query_batched(prompts, model=None)`: recursive child runs. At the maximum depth they fall back to direct model calls.
- `FINAL(value)`, `FINAL_VAR(name)` and `SHOW_VARS()`

## Finalization Patterns

### `FINAL(...)`

```python
result = env.execute("answer = 42")
result = env.execute("print(FINAL(answer))")
```

### `FINAL_VAR(...)`

```python
result = env.execute("my_answer = '42'")
result = env.execute('print(FINAL_VAR("my_answer"))')
```

### `answer` dict

```python
result = env.execute("answer['content'] = '42'")
result = env.execute("answer['ready'] = True")
```

## Prompts and Examples

[`prompts.py`](https://github.com/huggingface/OpenEnv/blob/main/envs/repl_env/prompts.py) has the system prompts and helpers used by the runner: `RLM_SYSTEM_PROMPT`, `RLM_SYSTEM_PROMPT_QWEN`, `QueryMetadata`, `build_rlm_system_prompt(...)`, `build_user_prompt(...)`, `extract_code_blocks(...)` and `format_observations(...)`.

Examples, which default to `Qwen/Qwen3.5-9B` through Hugging Face inference (needs `HF_TOKEN`):

- [`examples/repl_with_llm.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/repl_with_llm.py)
- [`examples/repl_oolong_simple.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/repl_oolong_simple.py)

## References

- [RLM Paper (arXiv:2512.24601)](https://huggingface.co/papers/2512.24601)
- [RLM Implementation](https://github.com/alexzhang13/rlm)
- [Alex Zhang's RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Prime Intellect RLM Blog](https://www.primeintellect.ai/blog/rlm)
