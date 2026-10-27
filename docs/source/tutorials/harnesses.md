# Harnesses in OpenEnv

A harness is the loop around a model: it sends the conversation to the model, runs the tools the model calls, and decides when the episode ends. There are three ways to put a harness and an OpenEnv environment together. They differ in who owns the loop, what the environment provides and who is on the other side, and that decides what each one is for. "Black-box" is the term [RFC 006](https://github.com/huggingface/OpenEnv/blob/main/rfcs/006-agentic-rl-harness-interception.md) uses, and "white-box" the one `openenv.core.harness` uses.

| | White-box | Black-box | Agent inside the environment |
|---|---|---|---|
| Who owns the loop | The trainer (for example TRL's `environment_factory`) | A real agent (OpenCode, Claude Code, Codex, …) | A real agent (Claude Code in the tutorial) |
| What the environment provides | Tools and a reward for the model being trained | Tasks: an instruction, a sandbox, and a verifier | Its own tools and state, injected into the agent over MCP |
| Who is on the other side | Nobody, the model works through the task | Nobody, the agent runs to completion | A user, simulated or real, over several turns |
| Trains a model | Yes | Yes, from the captured token ids and logprobs in a `TrainingTrace` (against vLLM or SGLang) | No, nothing is captured |
| Serves real users | No | No | Yes, over `WS /harness` |
| Example | Web tasks in BrowserGym | "Fix this repository", "analyze these files" | Customer service on τ²-bench |
| Start here | [BrowserGym](browsergym-harness) | [Harbor](harbor-harness) | [Claude Code on τ²-bench](claude-code-harness) |

## Which One to Use

- **You want to train a model on your environment and the loop can be simple.** Use the white-box path. The trainer runs the tool loop, so token ids and logprobs come for free, and the environment only supplies tools and a reward. The BrowserGym tutorial does it with TRL's `GRPOTrainer`, and [Training with OpenEnv](../guides/training) lists the other frameworks that train on OpenEnv environments.
- **The job is a task that an agent does on its own, and a verifier checks the result.** Use Harbor, to train the model behind the agent or to evaluate it. The agent keeps its own planner, tools and context management in a sandbox, and a proxy records every model call it makes. The resulting `TrainingTrace` is plain data that doesn't depend on a trainer, and TRL's `AsyncGRPOTrainer` has a worked example.
- **The job is a conversation, and the agent has to use your environment's tools.** Use `HarnessEnvironment` from [RFC 005](https://github.com/huggingface/OpenEnv/blob/main/rfcs/005-agentic-harnesses.md), to evaluate the agent against a simulated user or to serve it to real ones. The environment injects its tools, answers between turns and scores the conversation with a rubric.

### Harbor or `HarnessEnvironment`?

Both run a real agent, and the same agent can go either way: Claude Code is one of Harbor's agents and the agent in the RFC 005 tutorial. What decides is the shape of the job. If an instruction goes in and a verifier checks what comes out, it is a Harbor task. If someone answers the agent between turns, or the agent works on your environment's own state (a database, an API), use `HarnessEnvironment`. τ²-bench is the second kind: a simulated customer replies after every agent turn, which a Harbor task has no place for. To train, only Harbor records the tokens.

[The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) goes deeper into the white-box and black-box paths and why training across several agents matters.

> [!NOTE]
> `openenv.core.harness` also has a session runtime (`ResourceSession`, `MCPHarnessAdapter`, `build_harness_rollout_func`), where OpenEnv runs a small tool loop and a model generates each turn. [`openenv collect`](sft-warmup) uses it to collect rollouts for SFT. Its `HarnessAdapter` is a different class from RFC 005's `AgenticHarnessAdapter`.
