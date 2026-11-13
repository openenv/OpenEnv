---
title: τ-Emotion-Bench Environment Server
emoji: 💬
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# τ-Emotion-Bench Environment

[τ-Emotion-Bench](https://github.com/sahsaeedi/tau-emotion-bench) as an OpenEnv environment. Each episode is a customer-service conversation with a simulated user who is in an emotional state: anxious, frustrated, grieving, rushed, suspicious… The user comes with a request, and the agent has to solve it with the domain's tools while following the domain's policy, *and* handle the user's emotions well enough that they stay in the conversation. A user who feels ignored or mishandled gives up, and that run scores -1. When the conversation ends, τ-Emotion-Bench's own evaluator scores it from the final state of the database.

That makes it multi-turn (a real back-and-forth with a user), multi-step (several tool calls per turn), and verifiable. It works for evaluation and as an RL environment. The benchmark and its data-generation pipeline, TRACER, are described in the [paper](https://arxiv.org/abs/2610.19560).

| Domain | Tasks (train / test) | Tools |
|---|---|---|
| `airline` | 143 / 70 | book, cancel and change reservations, baggage, passengers, certificates… |
| `retail` | 585 / 70 | orders, returns, exchanges, addresses, payments… |
| `telecom` | 186 / 70 | services, devices, billing, payments, support tickets… |
| `telehealth` | 584 / 70 | appointments, medical records, prescriptions, drug interactions, telemetry… |
| `bank` | 228 / 50 | accounts, cards, beneficiaries, transfers, loan payments… |
| `basketball` | 346 / 50 | players, clubs, schedules, scores, box scores… |
| `movie` | 231 / 50 | shows, seats, bookings… |
| `restaurant` | 163 / 50 | dishes, tickets, tips, tables, drop-offs… |
| `school` | 133 / 50 | courses, registrations, payments… |
| `travel` | 124 / 50 | packages, bookings, travelers, rooming, add-ons… |

Every task comes with an emotion instruction for the user, from a structured emotion space of families and subfamilies (e.g. 5 families and 16 subfamilies in `airline`). The agent never sees it: it only sees how the user writes.

## Tools

The agent acts through MCP tools:

- **The domain's tools**, e.g. `get_user_details`, `search_direct_flight`, `book_reservation`. They read and write the task's database.
- **`respond_to_user(message)`**: says something to the user and returns their reply.
- **`done()`**: ends the conversation from the agent's side.

The episode ends when the user is done (or gives up), when the agent calls `done` or transfers them to a human, or at the step limit. The last observation carries `done=True`, the reward, and in `metadata` the `termination_reason` (`user_stop`, `user_unsatisfied`, `agent_stop`, `max_steps`…) and τ-Emotion-Bench's breakdown as `reward_info`.

## Reward

| Reward | Meaning |
|---|---|
| `1.0` | The database ends in the expected state (the task is solved). |
| `0.0` | It does not. |
| `-1.0` | The user gave up on the agent (`termination_reason` is `user_unsatisfied`). |
| `null` | The conversation ended unscored because the simulation failed, usually the user's model (`termination_reason` is `infrastructure_error`, and the observation carries the error). Not an agent failure: rerun the task. |

## The simulated user

The user is an LLM. It runs on [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers) by default, with the benchmark's reference user model:

| `TAU_EMOTION_USER_PROVIDER` | Credential | Default `TAU_EMOTION_USER_MODEL` |
|---|---|---|
| `hf` (default) | `HF_TOKEN` | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| `openai` | `OPENAI_API_KEY` | `gpt-4.1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |

Any model of the provider works through `TAU_EMOTION_USER_MODEL`. With `openai`, `TAU_EMOTION_USER_API_BASE` points litellm at any OpenAI-compatible server instead, e.g. a vLLM server hosting the reference model.

> [!NOTE]
> τ-Emotion-Bench's published results use `Qwen3-235B-A22B-Instruct-2507` at temperature 0 as the user. Scores with a different user model are not comparable to the [leaderboard](https://tau-emotion-bench.github.io).

## Quick start

```python
from openenv.core.env_server.mcp_types import CallToolAction
from tau_emotion_env import TauEmotionEnv

async with TauEmotionEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task_id="airline_test_0000")
    print(result.observation.metadata["user_message"])  # the user's opening message
    policy = result.observation.metadata["policy"]  # what the agent must follow

    tools = await env.list_tools()
    reply = await env.call_tool(
        "respond_to_user", message="I'm sorry to hear that. What's your user id?"
    )
    user = await env.call_tool("get_user_details", user_id="ethan_nguyen_6045")

    # `call_tool` returns the tool's output only. `step` also returns whether the
    # episode is over and, once it is, the reward.
    result = await env.step(CallToolAction(tool_name="done", arguments={}))
    print(result.done, result.reward, result.observation.metadata["reward_info"])
```

`reset()` takes `task_id` to run a given task (ids look like `airline_test_0003` or `airline_train_original_0014`), or picks one of the split at random (`seed` makes it reproducible). Give the agent the policy as its system prompt. With the default `hf` provider, `reset(hf_token=...)` makes the session's user run on your token, which is how to use a server that has none.

## Running the server

τ-Emotion-Bench's tasks, databases and policies live in its repository rather than in the package, so outside Docker fetch them once (the commit matches the `tau-emotion` pin in `pyproject.toml`) and point `TAU_EMOTION_DATA_DIR` at them:

```bash
git clone --filter=blob:none --no-checkout https://github.com/sahsaeedi/tau-emotion-bench
git -C tau-emotion-bench sparse-checkout set --no-cone data/tau_emotion/domains
git -C tau-emotion-bench checkout 287ebfce6181b4e141f778661b3455c2c3f03859

cd envs/tau_emotion_env
TAU_EMOTION_DATA_DIR=../../tau-emotion-bench/data HF_TOKEN=hf_... uv run server
```

The Docker image does this at build time.

| Variable | Default | |
|---|---|---|
| `TAU_EMOTION_DOMAIN` | `airline` | τ-Emotion-Bench domain |
| `TAU_EMOTION_SPLIT` | `test` | `train` or `test` |
| `TAU_EMOTION_USER_PROVIDER` | `hf` | `hf`, `openai` or `anthropic` |
| `TAU_EMOTION_USER_MODEL` | the provider's default | model for the user |
| `TAU_EMOTION_USER_API_BASE` | | with `openai`, an OpenAI-compatible server |
| `TAU_EMOTION_DATA_DIR` | set in the Docker image | τ-Emotion-Bench's `data` folder |
| `MAX_CONCURRENT_ENVS` | `8` | API sessions at once |

With Docker:

```bash
docker build -t tau-emotion-env -f envs/tau_emotion_env/server/Dockerfile envs/tau_emotion_env
docker run -p 8000:8000 -e HF_TOKEN=hf_... tau-emotion-env
```

API clients can pass their own token with `reset(hf_token=...)`, so an `HF_TOKEN` on the server is only a fallback for clients that don't. Without credentials the server still runs, and `reset()` explains what is missing.

The image starts from a Python 3.12 base rather than `openenv-base`, because τ-Emotion-Bench requires Python 3.12+.

## Notes

- τ-Emotion-Bench is installed from GitHub, pinned to a commit.
- Train on the `train` split and evaluate on `test`, so training does not leak into the benchmark. The train tasks run against the domain's train database, which the environment loads for them.
- Rewards are in `[-1, 1]`, not `[0, 1]`: the `-1` for a user who gives up is the benchmark's own signal for emotionally mishandled conversations. Clip or shift it if a trainer expects non-negative rewards.
- litellm has no prices for Inference Providers models, so τ-Emotion-Bench reports the cost of those runs as $0.
