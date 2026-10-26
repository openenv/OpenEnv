# Train a Reasoning Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/end_to_end_walkthrough.ipynb)

In this tutorial you'll take a small open-weight model, an OpenEnv environment, and TRL, and run the full training pipeline end-to-end:

1. Connect to a hosted environment.
2. Wire it into TRL via the `environment_factory` pattern.
3. Fine-tune with **GRPO** (Group Relative Policy Optimization).
4. Read the reward delta from the training logs to see how much the policy improved.
5. Publish the trained model to the Hub.

The goal is to see the whole pipeline as one coherent narrative — model, environment, training, metric — rather than three separate articles.

## Why this shape

We pair **GRPO** with a **procedural** task on purpose. GRPO is a value-free RL method that ranks several rollouts of the same prompt against each other, so the only signal it needs is a per-rollout scalar reward — exactly what an environment can return after a `step`. Procedural means the env generates a fresh question every episode rather than serving a fixed dataset, so the model has to *generalize* over the family of problems instead of memorizing specific items.

## What you'll use

- **Model**: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) — fits a single A100 (40 GB) at the settings below and is large enough for GRPO to move the needle. For smaller GPUs, swap to [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B).
- **Environment**: [`reasoning_gym_env`](https://github.com/huggingface/OpenEnv/tree/main/envs/reasoning_gym_env), an OpenEnv wrapper around the [Reasoning Gym](https://github.com/open-thought/reasoning-gym) library. Each episode is a single Q→A.
- **Dataset**: `chain_sum` from Reasoning Gym — chains of integer additions like `Compute 17 + 4 + 22 + 9`. Procedurally generated, so every rollout sees a fresh problem.
- **Trainer**: [TRL `GRPOTrainer`](https://huggingface.co/docs/trl/main/en/grpo_trainer) with `environment_factory`.

> [!NOTE]
> This tutorial runs through training; on a single A100 (40 GB) the recipe completes in roughly an hour at the suggested settings, peaking around ~38 GB of VRAM. T4 (16 GB) won't fit Qwen3-1.7B at these settings — see the model bullet above for the smaller-GPU swap. The exact reward numbers you see will vary with seed and budget — the point is to watch the reward curve climb and report the delta.

---

## 1. Install dependencies

This tutorial connects to [`sergiopaniego/reasoning_gym`](https://huggingface.co/spaces/sergiopaniego/reasoning_gym). For your own training runs, deploy your own copy first by running `openenv push --repo-id <your-username>/reasoning_gym` inside `envs/reasoning_gym_env/` of the OpenEnv repo, then replace `sergiopaniego` with your username in the install line and the `base_url=` strings further down.

Install pip dependencies — keep them as separate cells (don't merge into one `pip install`):

```python
!pip install -q trl
!pip install -q openenv
!pip install -q --no-deps git+https://huggingface.co/spaces/sergiopaniego/reasoning_gym
!pip install -Uq "transformers>=5.3.0"  # 5.3+ has the `environment_factory` integration TRL needs
!pip install -q trackio jmespath
```

---

## 2. Log in to Hugging Face

You'll need to be logged in to download the base model and (optionally) push the trained checkpoint.

```python
from huggingface_hub import notebook_login

notebook_login()
```

---

## 3. Define the prompt

The model will be asked to use a single tool, `answer`, to submit its final number. The prompt makes that explicit. The environment wrapper below prepends it to every question.

```python
prompt = """You are a careful arithmetic assistant.

You will be given a chain of integer additions. Compute the result and submit it as a single number.

Rules:
1. Read the question carefully.
2. Use the tool `answer` exactly once with your final number.
3. The answer must be a single integer with no units or explanation.
"""
```

---

## 4. Define the environment class

The `environment_factory` pattern asks for a Python class that the trainer can instantiate per rollout. It needs:

- An `__init__` that opens a connection to the underlying environment.
- A `reset(**kwargs)` method that starts a new episode and returns the initial observation as a string (the prompt and question, in our case).
- One or more *tool methods* — public methods with docstrings — that the trainer auto-discovers and exposes as tools to the model. Each call corresponds to one `env.step` on the underlying environment.
- A `get_reward()` method that returns the episode's reward. The environment owns the reward, so no separate reward function is needed.

Because Reasoning Gym episodes are **single-step** (one question → one answer → done), the wrapper is small.

```python
import random

from reasoning_gym_env import ReasoningGymAction, ReasoningGymEnv


class ReasoningGymTrainEnv:
    """Environment wrapper for GRPO training on chain_sum.

    Each rollout episode = one question → one `answer` tool call → done.
    """

    DATASET_NAME = "chain_sum"
    DATASET_SIZE = 1000
    DATASET_CONFIG = {
        "min_terms": 2,
        "max_terms": 3,
        "min_digits": 2,
        "max_digits": 2,
    }

    def __init__(self):
        # `EnvClient` subclasses are async by default; `.sync()` returns a
        # synchronous wrapper so the trainer can call our tool methods directly.
        self.client = ReasoningGymEnv(base_url="https://sergiopaniego-reasoning-gym.hf.space").sync()
        # Random seed per instance so the parallel envs the trainer creates
        # don't all iterate over the same question sequence.
        self._dataset_seed = random.randint(0, 2**31 - 1)
        self._initialized = False
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str:
        if not self._initialized:
            # First reset: configure the dataset (name + config + seed + size).
            result = self.client.reset(
                dataset_name=self.DATASET_NAME,
                dataset_config=self.DATASET_CONFIG,
                seed=self._dataset_seed,
                size=self.DATASET_SIZE,
            )
            self._initialized = True
        else:
            # Subsequent resets: no args → server returns the next question
            # from the same dataset iterator. Re-sending the config would
            # rebuild the dataset and rewind to question 0.
            result = self.client.reset()
        self.reward = 0.0
        self.done = False
        return prompt + result.observation.question

    def answer(self, answer: str) -> str:
        """Submit the final answer for the current question.

        Args:
            answer: The agent's answer (will be parsed as a number server-side).

        Returns:
            A short feedback string with the score and the correct answer.
        """
        if self.done:
            raise ValueError("Episode is already finished.")
        # The model often emits `answer` as a JSON int (e.g. `7`) even though
        # the tool schema declares string — coerce so pydantic validation on
        # `ReasoningGymAction` doesn't reject the rollout.
        result = self.client.step(ReasoningGymAction(answer=str(answer)))
        self.reward = float(result.observation.score or 0.0)
        self.done = True
        return f"score={self.reward} correct={result.observation.correct_answer}"

    def get_reward(self) -> float:
        return self.reward
```

> [!NOTE]
> Replace the `base_url` with your own deployment if you've pushed `reasoning_gym_env` to your own Space — the hosted versions have limited concurrency and are intended for tutorials and small experiments.

### What the trainer does with this class

It helps to picture the runtime loop. The trainer keeps a pool of `ReasoningGymTrainEnv` instances, one per rollout in a generation batch (`per_device_train_batch_size × gradient_accumulation_steps` with the settings below). Instances are created on first use and reused across batches, so each one keeps its WebSocket session. Per generation batch it then does, **for each rollout in parallel**:

1. `env.reset(**row)` starts a new episode and returns the prompt and question, which become the user message.
2. The model generates a completion, and the trainer parses any `<tool_call>` blocks out of it. Each prompt is rolled out `num_generations` times, on separate instances.
3. For each parsed call it dispatches to the matching tool method (here, `answer(...)`) and feeds the return value back to the model as a `<tool_response>`.
4. When the rollout ends, the trainer calls `env.get_reward()` to score it.
5. GRPO computes one advantage per completion (relative to the group's mean reward), updates the policy, and the cycle repeats.

That's why the wrapper only needs four things: a connection in `__init__`, a `reset` that returns the initial observation, one or more tool methods, and a `get_reward` that reads the score back.

---

## 5. No dataset needed

The environment generates every question and returns it from `reset()`, so `GRPOTrainer` doesn't need a `train_dataset`. Without one, `max_steps` sets the length of the run. When your task does come from a dataset, pass it as `train_dataset`: each row's columns are forwarded to `reset(**row)` and its `prompt` is prepended to what `reset()` returns.

---

## 6. Set the GRPO config

These settings mirror the [Wordle GRPO tutorial](wordle-grpo) and are tuned for a single A100 (40 GB). Bigger GPUs can raise `per_device_train_batch_size` and `num_generations`; smaller GPUs should drop to Qwen3-0.6B and shrink `max_completion_length`.

```python
from trl import GRPOConfig

output_dir = "reasoning-gym-chain-sum-Qwen3-1.7B"

grpo_config = GRPOConfig(
    num_train_epochs=1,
    max_steps=150,
    learning_rate=1e-6,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=1,
    warmup_steps=10,
    optim="adamw_torch",
    max_grad_norm=1.0,
    num_generations=2,
    max_completion_length=256,
    log_completions=True,
    num_completions_to_print=2,
    chat_template_kwargs={"enable_thinking": False},
    output_dir=output_dir,
    report_to="trackio",
    trackio_space_id=output_dir,
    logging_steps=10,
    gradient_checkpointing=True,
    save_strategy="no",
    push_to_hub=True,
)
```

A few of the choices above worth flagging: `max_steps=150` caps the run before saturation (see *Reading the dashboard* below). `gradient_accumulation_steps=4` keeps the environment pool at `1 × 4 = 4` instances, well under the server's default concurrency limit. `save_strategy="no"` skips intermediate checkpoints so the run stays quiet — we push the final model explicitly in section 8. `use_vllm` is left at its default (`False`); enabling it speeds up rollouts on bare-metal but its distributed init breaks under IPython.

> [!NOTE]
> `chat_template_kwargs={"enable_thinking": False}` disables Qwen3's thinking mode so the model emits tool calls directly instead of reasoning tokens first. For a pure tool-use task like this one that's what you want; for harder math you may benefit from re-enabling it and growing `max_completion_length`.

---

## 7. Create the `GRPOTrainer` and start training

`environment_factory=ReasoningGymTrainEnv` is the only piece wiring our wrapper into the training loop. It supplies the prompts, the tools and the reward.

```python
from trl import GRPOTrainer

MODEL_NAME = "Qwen/Qwen3-1.7B"

trainer = GRPOTrainer(
    model=MODEL_NAME,
    args=grpo_config,
    environment_factory=ReasoningGymTrainEnv,
)

trainer.train()
```

### Reading the trackio dashboard while it runs

Open the Trackio Space linked in the trainer logs to follow the run live. A healthy GRPO trajectory looks roughly like this:

- **`reward`** (also logged per source as `rewards/ReasoningGymTrainEnv/mean`) climbs from your baseline toward `1.0` over the first ~100 steps. A flat line near 0 means the task is too hard for the base model; a flat line near 1 means it's too easy — adjust `DATASET_CONFIG` in either case.
- **`reward_std`** starts moderate and *drops* as the policy converges (most rollouts succeed). Persistent zero means every rollout in the group gives the same score → no advantage signal → no learning. Bump `num_generations` or task difficulty.
- **`frac_reward_zero_std`** is the fraction of groups where every rollout has the same reward — when it climbs toward 1.0 you've saturated.
- **`entropy`** stays low while the model is learning. Once `reward` saturates, `entropy` typically rises again: most groups have zero advantage, so the remaining updates come from the few groups that still differ and from optimizer momentum — at that point further training is net-negative. Stop with a kernel interrupt or trust `max_steps`.
- **`grad_norm`** decays toward zero as gradients become uninformative; same saturation signal.

Once training finishes, the model in the running process has been fine-tuned in place.

---

## 8. Publish the trained model to the Hub

`save_strategy="no"` means the trainer didn't write any intermediate checkpoints. Push the final model explicitly so others can reuse it (and so the experiment is reproducible from the Hub):

```python
trainer.push_to_hub(commit_message="GRPO fine-tune on reasoning_gym chain_sum")
```

The repo is derived automatically from `output_dir` (or `hub_model_id` if set in `GRPOConfig`). After this completes, the model lives at `https://huggingface.co/<your-username>/reasoning-gym-chain-sum-Qwen3-1.7B` and anyone can load it with `AutoModelForCausalLM.from_pretrained(...)`.

---

## 9. Read the training reward delta

Every rollout the trainer ran left a `reward` entry in `trainer.state.log_history`. Comparing the first few logged rewards (the model's starting capability) to the last few (after training) gives a clean before/after number — same metric, same distribution, no second eval pass required.

```python
import statistics

rewards = [log["reward"] for log in trainer.state.log_history if "reward" in log]

if len(rewards) < 5:
    print(f"Only {len(rewards)} reward entries logged — train for a few more `logging_steps` and re-run.")
else:
    initial = statistics.mean(rewards[:5])
    final = statistics.mean(rewards[-5:])
    print(f"Initial reward (first 5 logs avg): {initial:.2%}")
    print(f"Final reward   (last 5 logs avg):  {final:.2%}")
    print(f"Delta:                             {(final - initial) * 100:+.2f} pp")
```

A delta of **+10 to +30 pp** is what you should expect at this difficulty; outside that range:

- **Δ ≈ 0 pp, initial already high (≥90%)** — `DATASET_CONFIG` is too easy; the model already solves it before training. Bump `min_terms` / `min_digits`.
- **Δ ≈ 0 pp, initial very low (≤20%)** — task is too hard for the base model to ever stumble onto a correct answer, so GRPO has no positive rollouts to learn from. Lower `min_terms` / `min_digits`. If the reward stays near zero even at minimum difficulty, the bottleneck is likely **format compliance** rather than task difficulty — the model never produces a valid `<tool_call>` so the env cannot score it. See the [SFT warm-up tutorial](sft-warmup) for how to fix this before returning to GRPO.
- **Δ negative** — you trained past saturation: once `reward` plateaus, the few groups that still carry signal move the policy without improving it. Reduce `max_steps` so training stops while it's still net-improving.

> [!NOTE]
> This delta is measured *during training* — same prompt format, same env, same procedural distribution that produced each rollout. It's the most direct way to ask "did the policy improve over the run?". A more rigorous protocol — generating completions on a held-out split with a separate evaluation harness — is what frameworks like [Inspect AI](https://inspect.aisi.org.uk/) are designed for; that's a follow-up rather than part of this walkthrough.

---

## 10. Where to go next

- **Swap the dataset.** `chain_sum` is one of ~100 datasets in [Reasoning Gym](https://github.com/open-thought/reasoning-gym) — try `simple_equations`, `letter_counting`, or `propositional_logic` by changing `DATASET_NAME` and re-running the same recipe.
- **Try a different environment.** The same `environment_factory` shape works for any OpenEnv environment with a small tool surface — browse the [environment catalog](../environments) for ideas.
- **Use SFT as a warm-start.** If format compliance is the bottleneck (initial reward near zero regardless of difficulty), the [SFT warm-up tutorial](sft-warmup) shows how to collect teacher rollouts, filter by reward, and fine-tune a student model — so GRPO starts with non-zero `reward_std` from the first batch.
- **Read the other tutorials.** [Wordle GRPO](wordle-grpo) covers the multi-step variant; the full list is in the [tutorials index](index).
