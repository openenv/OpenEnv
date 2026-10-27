# Evaluating Claude Code in an OpenEnv environment (RFC 005)

This recipe runs [Claude Code](https://docs.anthropic.com/en/docs/claude-code) *inside* an OpenEnv environment with [`HarnessEnvironment`](../../src/openenv/core/harness/environment.py), the agentic harness wrapper from [RFC 005](../../rfcs/005-agentic-harnesses.md), and evaluates it on [τ²-bench](https://github.com/sierra-research/tau2-bench) through [`tau2_env`](../../envs/tau2_env/README.md). Claude Code is the customer service agent: it gets the domain's tools and policy over MCP, τ²-bench's simulated customer talks to it over several turns, and τ²-bench scores the conversation.

The walkthrough is in the docs: [Evaluate Claude Code in an Environment](https://huggingface.co/docs/openenv/tutorials/claude-code-harness).

It is an evaluation recipe. RFC 005 does not capture token ids, so this is not a training path. To train a harness's policy on task-and-verifier episodes, use [Harbor](https://huggingface.co/docs/openenv/environments/harbor).

| File | |
|---|---|
| [`claude_code_adapter.py`](claude_code_adapter.py) | `AgenticHarnessAdapter` for Claude Code's headless stream-json mode |
| [`tau2_harness.py`](tau2_harness.py) | `Tau2Harness`: a `HarnessEnvironment` for one τ²-bench task, with the simulated customer between turns and τ²-bench's score as its rubric |
| [`run_eval.py`](run_eval.py) | evaluates Claude Code on τ²-bench tasks and reports pass^1 |
| [`serve.py`](serve.py), [`chat.py`](chat.py) | serve the harness in production mode (`WS /harness`) and talk to it as the customer |
| [`Dockerfile`](Dockerfile) | runs `serve.py` with Claude Code inside the container |

## Run it

You need:

- the `claude` CLI, logged in or with `ANTHROPIC_API_KEY` set;
- Python 3.12 or later, which τ²-bench requires;
- `tau2_env` installed (`pip install -e envs/tau2_env`) and `TAU2_DATA_DIR` set, see [its README](../../envs/tau2_env/README.md#running-the-server);
- `HF_TOKEN`, because the simulated customer runs on [Inference Providers](https://huggingface.co/docs/inference-providers).

From the repository root:

```bash
PYTHONPATH=src:envs:examples/claude_code_harness_eval \
    python examples/claude_code_harness_eval/run_eval.py --domain airline --task-ids 8 16 19 26 --model haiku
```

Tasks 8, 16 and 19 book, change and cancel a reservation, and in task 26 the policy makes the agent refuse a cancellation:

```
=== airline task 26
customer: Hi, I need to cancel my flights from Orlando to Charlotte. I'd like to get a refund for them, please.
[...]
  -> get_user_details({'user_id': 'amelia_sanchez_4739'})
  -> get_reservation_details({'reservation_id': '3FRNFB'})
[...]
agent: I'm sorry, but I can't cancel reservation 3FRNFB. It was booked on May 6, more than 24 hours ago. [...]
reward: 1.00 {'DB': 1.0, 'COMMUNICATE': 1.0}
...
pass^1: 4/4
```

In production mode, you are the customer:

```bash
PYTHONPATH=src:envs:examples/claude_code_harness_eval \
    python examples/claude_code_harness_eval/serve.py --domain airline --task-id 2 --port 8000
python examples/claude_code_harness_eval/chat.py ws://localhost:8000/harness \
    "Hi, I'm Noah Muller, user id noah_muller_9847. What reservations do I have?"
```

The [`Dockerfile`](Dockerfile) puts Claude Code, `tau2_env` and `serve.py` in one image, as the tutorial describes. The server has no authentication, so keep it on localhost or behind your own auth.

[`tests/scripts/test_claude_code_harness_eval_example.py`](../../tests/scripts/test_claude_code_harness_eval_example.py) covers the adapter offline: a fake `claude` replays the same stream-json events and calls the environment's tools over the real bridge.
