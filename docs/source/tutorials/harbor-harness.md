# Black-Box: Train Real Agents with Harbor

Some agents can't be reimplemented as a trainer's tool loop. A coding agent such as OpenCode, Claude Code or Codex has its own planner, tools, context management and stop condition, and the point is to train the model that drives *that* agent. This is the **black-box** harness path: the agent owns its loop, and OpenEnv records every model call it makes. Episodes are tasks: an instruction goes in, the agent works on its own, and a verifier checks the result. If the job is a conversation instead, or the agent has to use your environment's own tools, see [Evaluate Claude Code in an Environment](claude-code-harness). [Harnesses in OpenEnv](harnesses) compares the paths.

OpenEnv does this through [Harbor](../environments/harbor), which supplies the tasks, the sandboxes, the agents and the verifiers. One `harbor_env` server runs any of 16 validated agents on any Harbor dataset, in any of Harbor's sandbox backends, and its capture proxy returns the token ids and logprobs of each model call together with the task's reward.

## Try One Rollout

With an OpenAI-compatible endpoint in `$LLM` and Python 3.12 or later:

```bash
pip install "openenv[harbor]"

openenv harbor rollout \
  --llm-url $LLM \
  --dataset AdithyaSK/data_agent_rl_environment_eval \
  --task-index 0 --harness opencode --sandbox e2b \
  --out rollout.json
```

Against a hosted provider you get an evaluation rollout: the reward and the full trace. Against vLLM or SGLang you also get the token ids and logprobs that training needs. `openenv harbor info` checks first which agents, sandboxes and capture level this machine can use.

## Train on the Captures

Each rollout comes back as a `TrainingTrace`: the token ids, logprobs and loss masks of every model call, plus the task's reward. It doesn't depend on a trainer, so any framework can train on it. TRL has a worked example: `AsyncGRPOTrainer` with a `HarnessRolloutWorker` runs each task to completion, reads back the trace, and syncs the new weights into the same vLLM server.

- [TRL's guide to training on harnesses](https://huggingface.co/docs/trl/main/en/openenv#training-on-harnesses-training-real-coding-agents-harbor) explains how it works and how to wire it.
- [`examples/async_grpo_harbor`](https://github.com/huggingface/trl/tree/main/examples/async_grpo_harbor) is the complete script, with a Hugging Face Jobs launcher.
- [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) trains one model across four agents (OpenCode, Claude Code, Codex and Mini-SWE-Agent) and covers why the white-box loop isn't enough, how the capture works and what the runs showed.

## Learn More

- [The Harbor environment](../environments/harbor): supported agents and sandboxes, the web UI, deploying to Spaces and troubleshooting.
- [TRL's Harbor integration](https://huggingface.co/docs/trl/harbor) for the white-box path on Harbor tasks, where TRL's own loop works through a Harbor task suite.
- [Training with OpenEnv](../guides/training) for the other ways to train and the frameworks that support them.
