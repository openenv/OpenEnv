# Run an RLM with a model server

These examples run `LocalRLMRunner` in your Python process. The runner sends
chat requests to your model server and executes model-generated code locally.
Run the commands from the OpenEnv repository root.

## One model with vLLM

Start a [vLLM server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
with a model that supports chat completions. For example, on a machine with
vLLM installed and hardware that can serve the model:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000
```

In a second terminal, run the example. Use the model name exposed by your
server and include `/v1` in the URL.

```bash
PYTHONPATH=src:envs uv run --with 'smolagents>=1.22,<2' --with openai \
  python examples/repl_model_servers/vllm_runner.py \
  --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
```

The task asks the model to count words in a short string. The expected answer
is `3`. If your server requires authentication, set `REPL_API_KEY` to its key.

## Separate parent and child models

Keep the parent server from the first recipe running. For the child model,
start a second server with sufficient free accelerator memory, for example on
another GPU or machine:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8001
```

The smaller model handles `llm_query()` and `rlm_query()` calls inside the REPL.
Replace the model names and URLs below with those of your servers.

```bash
PYTHONPATH=src:envs uv run --with 'smolagents>=1.22,<2' --with openai \
  python examples/repl_model_servers/separate_models.py \
  --parent-url http://127.0.0.1:8000/v1 --parent-model Qwen/Qwen2.5-7B-Instruct \
  --child-url http://127.0.0.1:8001/v1 --child-model Qwen/Qwen2.5-1.5B-Instruct
```

The task asks the parent to delegate counting the words in each of two lines
and add the results. The expected total is `5`. Set `REPL_PARENT_API_KEY` or
`REPL_CHILD_API_KEY` if the corresponding server requires authentication.

The parent model is configured in `LocalRLMRunner(parent_chat)`. The
`backend_factory` configures the model for calls made from its REPL. This
recipe uses the ordinary blocking recursive backend.
