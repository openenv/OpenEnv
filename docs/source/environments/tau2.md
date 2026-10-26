<!-- openenv-source: tau2_env -->
# τ²-bench Environment

[τ²-bench](https://github.com/sierra-research/tau2-bench) as an OpenEnv environment. Each episode is a customer-service conversation: an LLM-simulated user comes with a request, and the agent has to solve it with the domain's tools while following the domain's policy. The user replies to what the agent says, changes their mind, or pushes for things the policy does not allow. When the conversation ends, τ²-bench's own evaluator scores it, mostly from the final state of the database.

That makes it multi-turn (a real back-and-forth with a user), multi-step (several tool calls per turn), and verifiable. It works for evaluation and as an RL environment.

| Domain | Tasks (train / test) | Tools | Reward |
|---|---|---|---|
| `airline` | 30 / 20 | book, cancel, change flights, baggage, certificates… | database state + information given to the user |
| `retail` | 74 / 40 | orders, returns, exchanges, addresses, payments… | database state + natural-language assertions (judged by an LLM) |
| `telecom` | 74 / 40 | account and line management; the user also has tools on their phone | assertions on the final state of the account and the phone |

`telecom-workflow`, `banking_knowledge` and `mock` are also available (`banking_knowledge` needs τ²-bench's `knowledge` extra).

## Web UI

The Space opens on a τ²-bench tab. The domain, the split and the simulated customer's model can be changed at the top, and there are four views:

- **Tasks**: an explorer of the split's tasks, with search and filters (changes the database or read-only). A task page shows what the agent does not see (why the customer calls, what they know, how they behave) and what the conversation is scored on.
- **Run a model**: pick an agent model on Hugging Face Inference Providers (the list shows each model's providers and price) and watch the conversation, with each tool call as a collapsible row and τ²-bench's reward breakdown at the end. A run can be stopped.
- **Play as the agent**: talk to the simulated customer yourself and call the domain's tools from a form, then see your reward.
- **Runs**: the runs of your session. Open one to read it again or download it as JSON, or compare two to four side by side.

On a Space, visitors sign in with Hugging Face, and their conversations run on their own [Inference Providers](https://huggingface.co/docs/inference-providers) credits (the Space asks for the `inference-api` scope). Running a conversation requires signing in, and browsing the tasks doesn't.

The runs live in the browser session and are not kept across restarts. The generic OpenEnv playground stays available as a second tab.

## Tools

The agent acts through MCP tools:

- **The domain's tools**, e.g. `get_user_details`, `search_direct_flight`, `book_reservation`. They read and write the task's database.
- **`respond_to_user(message)`**: says something to the user and returns their reply.
- **`done()`**: ends the conversation from the agent's side.

The episode ends when the user is done (or the agent calls `done`), and the last observation carries `done=True`, the reward, and τ²-bench's breakdown in `metadata["reward_info"]`.

## The simulated user

The user (and the judge for retail's natural-language assertions) is an LLM. It runs on [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers) by default, with `deepseek-ai/DeepSeek-V4.1-Flash`:

| `TAU2_USER_PROVIDER` | Credential | Default `TAU2_USER_MODEL` |
|---|---|---|
| `hf` (default) | `HF_TOKEN` | `deepseek-ai/DeepSeek-V4.1-Flash` |
| `openai` | `OPENAI_API_KEY` | `gpt-4.1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |

Any model of the provider works through `TAU2_USER_MODEL` (for `hf`, any [Inference Providers chat model](https://huggingface.co/inference/models)). Non-reasoning models answer fastest, which matters when the user is called thousands of times during training.

> [!NOTE]
> τ²-bench's published results use `gpt-4.1` as the user. Scores with a different user model are not comparable to the leaderboard.

## Quick start

```python
from openenv.core.env_server.mcp_types import CallToolAction
from tau2_env import Tau2Env

async with Tau2Env(base_url="http://localhost:8000") as env:
    result = await env.reset(task_id="2")
    print(result.observation.metadata["user_message"])  # the user's opening message
    policy = result.observation.metadata["policy"]  # what the agent must follow

    tools = await env.list_tools()
    reply = await env.call_tool(
        "respond_to_user", message="Happy to help. What's your user id?"
    )
    user = await env.call_tool("get_user_details", user_id="noah_muller_9847")

    # `call_tool` returns the tool's output only. `step` also returns whether the
    # episode is over and, once it is, the reward.
    result = await env.step(CallToolAction(tool_name="done", arguments={}))
    print(result.done, result.reward, result.observation.metadata["reward_info"])
```

`reset()` takes `task_id` to run a given task, or picks one of the split at random (`seed` makes it reproducible). Give the agent the policy as its system prompt. With the default `hf` provider, `reset(hf_token=...)` makes the session's user and judge run on your token, which is how to use a server that has none, like the public Space (`base_url="https://sergiopaniego-tau2-env.hf.space"`).

## Running the server

τ²-bench's tasks, databases and policies live in its repository rather than in the package, so outside Docker fetch them once (the commit matches the `tau2` pin in `pyproject.toml`) and point `TAU2_DATA_DIR` at them:

```bash
git clone --filter=blob:none --no-checkout https://github.com/sierra-research/tau2-bench
git -C tau2-bench sparse-checkout set --no-cone data/tau2/domains data/tau2/user_simulator
git -C tau2-bench checkout 5bfa7e37b36656b37dc6d022156be6563c1007f3

cd envs/tau2_env
TAU2_DATA_DIR=../../tau2-bench/data HF_TOKEN=hf_... ENABLE_WEB_INTERFACE=true uv run server
```

The Docker image does this at build time.

| Variable | Default | |
|---|---|---|
| `TAU2_DOMAIN` | `airline` | τ²-bench domain |
| `TAU2_SPLIT` | `test` | `train`, `test` or `base` (all tasks) |
| `TAU2_USER_PROVIDER` | `hf` | `hf`, `openai` or `anthropic` |
| `TAU2_USER_MODEL` | the provider's default | model for the user and the judge |
| `TAU2_DATA_DIR` | set in the Docker image | τ²-bench's `data` folder |
| `MAX_CONCURRENT_ENVS` | `8` | API sessions at once |

With Docker:

```bash
docker build -t tau2-env -f envs/tau2_env/server/Dockerfile envs/tau2_env
docker run -p 8000:8000 -e HF_TOKEN=hf_... tau2-env
```

On a Space, the UI's runs use each visitor's own token once they sign in, so the Space needs no secret. They always run on Inference Providers, whatever `TAU2_USER_PROVIDER` is. API clients pass their own with `reset(hf_token=...)`, so an `HF_TOKEN` secret is only a fallback for clients that don't, and the UI never uses it. Without credentials the server and the task explorer still run, and `reset()` explains what is missing.

The image starts from a Python 3.12 base rather than `openenv-base`, because τ²-bench requires Python 3.12+.

## Notes

- τ²-bench is installed from GitHub, pinned to a commit. The `tau2` package on PyPI is an unrelated project.
- To evaluate Claude Code on τ²-bench, see [Evaluate Claude Code in an Environment](https://huggingface.co/docs/openenv/tutorials/claude-code-harness).
- Train on the `train` split and evaluate on `test`, so training does not leak into the benchmark.
- litellm has no prices for Inference Providers models, so τ²-bench reports the cost of those runs as $0.
