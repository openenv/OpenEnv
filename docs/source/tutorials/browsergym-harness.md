# White-Box: Train a Web Agent on BrowserGym

This tutorial trains a model to complete web tasks in [BrowserGym](../environments/browsergym) with TRL's `GRPOTrainer`. It is the **white-box** harness path: the trainer, here TRL, owns the agent loop. It samples every turn, parses the tool calls, runs them against the environment and feeds the results back, so it has the token ids and logprobs of everything the model generated without capturing anything. [Harnesses in OpenEnv](harnesses) compares it to the other paths, and [Training with OpenEnv](../guides/training) lists the other frameworks that train on OpenEnv environments.

The full script is TRL's [`examples/grpo_browsergym/grpo_browsergym.py`](https://github.com/huggingface/trl/tree/main/examples/grpo_browsergym). This page walks through its parts.

## What You'll Build

- An environment class whose methods are the browser actions the model can call.
- A reward function that reads the task's result from the environment.
- A `GRPOTrainer` that runs multi-turn episodes against the hosted BrowserGym Space.

## Install Dependencies

```bash
pip install -U "trl[vllm]" trackio
pip install "openenv-browsergym-env @ git+https://huggingface.co/spaces/openenv/browsergym_env"
```

The script connects to the hosted Space at `https://openenv-browsergym-env.hf.space`, so the browser runs there and not on your machine.

## The Environment Class

`environment_factory` takes a class with no constructor arguments. TRL creates one instance per rollout and calls `reset()` at the start of each episode. Every other public method becomes a tool, and its docstring's `Args:` section gives the tool's schema. Each method here sends one BrowserGym action and returns what the page looks like afterwards:

```python
from browsergym_env import BrowserGymAction, BrowserGymEnv


class BrowserGymVLMEnv:
    def __init__(self):
        self.client = BrowserGymEnv(base_url="https://openenv-browsergym-env.hf.space")
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        self.done = False
        result = self.client.reset(task_name="click-test")
        return self._format_observation(result.observation)  # the goal and the page's accessibility tree

    def click(self, bid: str) -> list:
        """Click an element on the page.

        Args:
            bid: The BrowserGym ID of the element to click.

        Returns:
            The updated page observation with screenshot.
        """
        return self._do_action(f"click('{bid}')")

    # fill, send_keys, scroll and noop follow the same pattern

    def _do_action(self, action_str: str) -> list:
        result = self.client.step(BrowserGymAction(action_str=action_str))
        step_reward = float(result.reward or 0.0)
        self.done = result.done
        if self.done:
            self.reward = 1.0 if step_reward > 0 else 0.0
        else:
            self.reward = step_reward
        return self._format_observation_multimodal(result.observation)  # screenshot + text
```

The string returned by `reset()` is appended to the prompt. The tools return a list of content blocks, a screenshot and the page's text, so a vision-language model sees the page after each action. The full script also caps the episode length and resizes the screenshots.

## Reward and Trainer

The reward function receives the environment instances, so it reads the result the episode left behind:

```python
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


def reward_completion(completions, environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]


dataset = Dataset.from_dict({"prompt": [[
    {"role": "system", "content": SYSTEM_PROMPT},  # how to read the page and which tools exist
    {"role": "user", "content": "Complete the web task successfully."},
]] * 1000})

trainer = GRPOTrainer(
    model="Qwen/Qwen3.5-2B",
    reward_funcs=reward_completion,
    train_dataset=dataset,
    args=GRPOConfig(
        num_generations=4,
        max_completion_length=1024,
        chat_template_kwargs={"enable_thinking": False},
        report_to="trackio",
    ),
    environment_factory=BrowserGymVLMEnv,
)
trainer.train()
```

## Run It

```bash
# Transformers generation, one GPU
python examples/grpo_browsergym/grpo_browsergym.py

# vLLM on the training GPU
python examples/grpo_browsergym/grpo_browsergym.py --use-vllm

# vLLM server on one GPU, training on another
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-2B --host 0.0.0.0 --port 8000 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1
CUDA_VISIBLE_DEVICES=1 python examples/grpo_browsergym/grpo_browsergym.py --use-vllm --vllm-mode server
```

`--task-name` picks another MiniWoB task, and `--model-id` another model.

## Learn More

- TRL's OpenEnv guide: [how `environment_factory` works](https://huggingface.co/docs/trl/openenv#how-environmentfactory-works), [training on several environments at once](https://huggingface.co/docs/trl/openenv#multi-environment-training), and [when to write a `rollout_func` instead](https://huggingface.co/docs/trl/openenv#environmentfactory-vs-rolloutfunc).
- [Wordle with TRL](wordle-grpo), the same pattern on a text game.
- [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl), on when the white-box loop is enough and when to train a real agent instead.
