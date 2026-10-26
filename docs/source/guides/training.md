# Training with OpenEnv

OpenEnv defines the contract between environments and the code that uses them. Training stays in your framework. An environment exposes `reset()`, `step()` and `state()` (and its tools over MCP), runs as a service anywhere, and computes its own reward, so any training framework that can call it can train on it. This page maps the ways to train on OpenEnv environments and the frameworks that support it.

## Ways to Train

| You want to | What the environment gives | Worked examples |
|---|---|---|
| Train a model with RL, with the trainer running the episode | Observations, tools and a reward. The trainer generates every turn, so it has the tokens and logprobs. This is the white-box path | [Wordle](../tutorials/wordle-grpo), [2048](../tutorials/rl-training-2048), [a reasoning model](../tutorials/end-to-end-walkthrough) and [web tasks in BrowserGym](../tutorials/browsergym-harness) with TRL, [2048 with gpt-oss](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb) with Unsloth, and the other frameworks' examples under [Integrations](#integrations) |
| Train the model behind a real agent (OpenCode, Claude Code, Codex, …) | Harbor runs the agent on a task, captures every model call, and returns a `TrainingTrace` with token ids, logprobs and the verifier's reward. This is the black-box path | [Harbor](../tutorials/harbor-harness), trained with TRL's `AsyncGRPOTrainer` |
| Warm-start a model with supervised data | `openenv collect` runs a teacher in the environment and saves reward-labeled rollouts as a dataset | [Collecting rollouts for SFT](../tutorials/sft-warmup) |
| Evaluate, not train | Scores from the environment's rubric | [Inspect AI](../tutorials/evaluation-inspect), [an agent inside the environment](../tutorials/claude-code-harness) |

None of these paths ties the environment to a framework. `openenv.core` does not depend on any trainer, and a `TrainingTrace` or a collected dataset is plain data that any trainer can read. The worked examples use the frameworks named above because that is where the examples exist today. [Harnesses in OpenEnv](../tutorials/harnesses) compares the white-box and black-box paths in more detail.

## Integrations

These frameworks and platforms train on OpenEnv environments. If your project supports OpenEnv, open a PR to add it here and to the [README](https://github.com/huggingface/OpenEnv#integrations).

| Framework | Example |
|---|---|
| [ART](https://art.openpipe.ai) | [OpenEnv integration](https://art.openpipe.ai/integrations/openenv-integration) |
| [Lightning AI](https://lightning.ai) | [OpenEnv templates](https://lightning.ai/templates?section=featured&query=openenv) |
| [Miles](https://github.com/radixark/miles) | [GRPO on Terminal-Bench 2](https://github.com/radixark/miles/tree/main/examples/experimental/openenv) |
| [Oumi](https://github.com/oumi-ai/oumi) | [OpenEnv GRPO notebook](https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20OpenEnv%20GRPO%20with%20trl.ipynb) |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | [Training on OpenEnv environments](https://skyrl.readthedocs.io/en/latest/examples/openenv.html) |
| [torchforge](https://meta-pytorch.org/torchforge/) | [GRPO on BlackJack](https://github.com/huggingface/OpenEnv/tree/main/examples/grpo_blackjack) |
| [TRL](https://huggingface.co/docs/trl) | [OpenEnv integration guide](https://huggingface.co/docs/trl/openenv): `environment_factory` with `GRPOTrainer`, several environments at once, and harness training through Harbor |
| [Unsloth](https://unsloth.ai) | [2048 with gpt-oss](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb) |

## Your Own Training Loop

Every integration above comes down to the same calls. To plug OpenEnv into a framework that has no integration yet, drive the client from its rollout code:

```python
from openenv import AutoAction, AutoEnv

env = AutoEnv.from_env("my-env")
Action = AutoAction.from_env("my-env")  # the environment's action class

with env.sync() as client:
    for episode in range(num_episodes):
        result = client.reset()
        while not result.done:
            action = Action(**policy(result.observation))  # your model picks the next action
            result = client.step(action)
            # result.reward is the environment's reward for this step
```

The [Task API](task-api) lets a trainer list an environment's tasks and pick which one each episode runs, and [Rewards](rewards) covers how environments compute the reward.
