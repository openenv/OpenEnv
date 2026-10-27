# Play Wordle with GRPO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/grpo_wordle/grpo_wordle.ipynb)

![trl banner](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_banner_dark.png)


With [**Transformers Reinforcement Learning (TRL)**](https://github.com/huggingface/trl), you can train a model that learns to **play Wordle**, a word-guessing game, through interaction and reinforcement.

- [TRL GitHub Repository](https://github.com/huggingface/trl) -- star us to support the project!
- [Official TRL Examples](https://huggingface.co/docs/trl/example_overview)
- [Community Tutorials](https://huggingface.co/docs/trl/community_tutorials)
- [OpenEnv](https://github.com/huggingface/OpenEnv)


An **agentic environment** is a setting where a model can take actions, observe outcomes, and adjust its behavior based on feedback, similar to how humans learn from trial and error.
In this case, the agent interacts with the **Wordle** environment through the [**OpenEnv**](https://github.com/huggingface/OpenEnv) framework, which standardizes multi-agent and RL-style text environments.

[Wordle](https://en.wikipedia.org/wiki/Wordle) is a popular word puzzle where the player must guess a secret five-letter word within six tries.
After each guess, feedback indicates whether each letter is:
- **GREEN (G)**: Correct and in the right position
- **YELLOW (Y)**: Present but in the wrong position
- **GRAY (X)**: Not in the word

This feedback loop makes Wordle a perfect environment for **RL with LLMs**, where the goal is to maximize the probability of guessing the correct word efficiently.


We'll fine-tune a model using **GRPO** (Group Relative Policy Optimization) via TRL.
Using `environment_factory`, the trainer automatically handles:
1. Creating environment instances for each rollout.
2. Generating model completions and parsing tool calls.
3. Stepping through the environment with the model's actions.
4. Collecting rewards and managing the interaction loop.

This means you only need to define the environment class and reward function -- the trainer takes care of the rest.


## Install dependencies

We'll start by installing **TRL** (with vLLM support), the **OpenEnv** Wordle environment, and **trackio** for logging.

```bash
pip install -Uq trl[vllm] git+https://huggingface.co/spaces/openenv/wordle trackio
```

### Log in to Hugging Face

Log in to your **Hugging Face** account to save your fine-tuned model, track your experiment results directly on the Hub or access gated models. You can find your **access token** on your [account settings page](https://huggingface.co/settings/tokens).

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Define the system prompt

This prompt instructs the model on how to play Wordle. It includes the game rules, feedback format, and importantly, tells the model to use the `guess` tool to submit guesses. The `environment_factory` pattern uses tool calling to interact with the environment, so the model needs to know which tool to call.

```python
prompt = """You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

Follow these rules to play Wordle:

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN (G): Letter is correct and in the correct position
   - YELLOW (Y): Letter is in the word but in the wrong position
   - GRAY (X): Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed
6. Use the tool `guess` to make a guess.
"""
```

## Define the environment

The `WordleEnv` class wraps the OpenEnv TextArena Wordle environment into the interface expected by `environment_factory`.

When you pass `environment_factory=WordleEnv` to the trainer, it will:
1. Create a new `WordleEnv()` instance for each rollout episode.
2. Call `reset()` to start a new game (returns the initial observation or `None`).
3. Automatically generate model completions, parse tool calls, and invoke the corresponding methods (e.g., `guess(...)`).
4. Repeat until the environment signals `done=True` or the max completion length is reached.

The environment exposes its public methods as tools. Any public method (other than `reset`) with a docstring is automatically discovered and exposed as a callable tool. Here, the `guess` method lets the model submit a Wordle guess and receive feedback.

For this example, we connect to the hosted environment at [openenv/wordle](https://huggingface.co/spaces/openenv/wordle).
For production use, we recommend duplicating the Space to your own account or running it locally via Docker, as the hosted versions have limited concurrency.

For more information, refer to the [TRL-OpenEnv documentation](https://huggingface.co/docs/trl/main/en/openenv).

```python
from textarena_env import TextArenaAction, TextArenaEnv


class WordleEnv:
    def __init__(self):
        self.client = TextArenaEnv(base_url="https://openenv-wordle.hf.space")

    def reset(self, **kwargs) -> None | str:
        result = self.client.reset()
        # The game returns cumulative feedback each turn (new text appended at the end), so
        # we store the previous full response and slice out only the newly appended part.
        self._last_full_feedback = result.observation.messages[0].content
        self.reward = 0.0
        self.done = False
        return self._last_full_feedback

    def guess(self, guess: str) -> str:
        """
        Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            raise ValueError("Game over.")
        result = self.client.step(TextArenaAction(message=guess))
        _full_feedback = result.observation.messages[0].content
        # Just take the new feedback since the last guess
        feedback = _full_feedback[len(self._last_full_feedback):]
        self._last_full_feedback = _full_feedback
        # Penalize invalid moves
        if "You attempted an invalid move" in feedback:
            self.reward = 0.0
        else:
            self.reward = result.reward
        self.done = result.done
        return feedback
```

## Define the reward function

The reward function receives the list of environment instances after each episode completes. Since the `WordleEnv` tracks its own reward (updated after each `guess` call), we simply read it out.

This is much simpler than defining multiple reward functions manually -- the environment already knows the game outcome.

```python
def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]
```

## Create the dataset

We create a dataset with repeated prompts to control the number of training episodes.
Each entry triggers one rollout episode during training. The prompt is formatted as a chat message.

```python
from datasets import Dataset

dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}] for _ in range(3000)]})
```

## Set GRPO Config

Next, we define the **GRPOConfig**, which controls all key training parameters.
This configuration specifies how the model interacts with vLLM, manages memory, and logs results.

Note the `chat_template_kwargs={"enable_thinking": False}` parameter -- this disables Qwen3's thinking mode so the model responds directly with tool calls instead of generating internal reasoning tokens first.

```python
from trl import GRPOConfig

model_name = "Qwen/Qwen3-1.7B"
output_dir = "wordle-grpo-Qwen3-1.7B"

grpo_config = GRPOConfig(
    # Training schedule / optimization
    num_train_epochs=1,
    learning_rate=1e-6,
    gradient_accumulation_steps=64,
    per_device_train_batch_size=1,
    warmup_steps=10,
    optim="adamw_torch",
    max_grad_norm=1.0,

    # GRPO configuration
    num_generations=2,
    max_completion_length=1024,
    log_completions=True,
    num_completions_to_print=2,
    chat_template_kwargs={"enable_thinking": False},

    # vLLM configuration
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.15,
    vllm_max_model_length=3072,

    # Logging / reporting
    output_dir=output_dir,
    report_to="trackio",
    trackio_space_id=output_dir,
    logging_steps=1,
    save_steps=10,
    save_total_limit=1,

    # Memory optimization
    gradient_checkpointing=True,

    # Hub integration
    push_to_hub=True,
)
```

## Create the `GRPOTrainer` and start training

Now we initialize the `GRPOTrainer` with `environment_factory=WordleEnv`.

This tells the trainer to automatically handle the entire interaction loop:
- It creates a `WordleEnv` instance for each episode.
- It generates model completions, parses tool calls (like `guess`), and steps through the environment.
- It collects rewards and manages the `tool_mask` (which tokens are model-generated vs environment-generated) automatically.

No need to write a custom `rollout_func` or manage tokenization manually.

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=reward_func,
    train_dataset=dataset,
    args=grpo_config,
    environment_factory=WordleEnv,
)
```

Show memory stats before training

```python
import torch

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

And train!

```python
trainer_stats = trainer.train()
```

Show memory stats after training

```python
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_training = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
training_memory_percentage = round(used_memory_for_training / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {training_memory_percentage} %.")
```

## Save and push to Hub

```python
trainer.save_model(output_dir)
trainer.push_to_hub()
```

## Load the fine-tuned model and run inference

Now let's test our fine-tuned model by loading it and playing a game of Wordle.
We use the same `WordleEnv` class to interact with the environment, and generate model responses with standard Transformers inference.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sergiopaniego/wordle-grpo-Qwen3-1.7B"  # Replace with your HF username or organization

fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float32", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

```python
import json


def play_wordle(model, tokenizer):
    env = WordleEnv()
    initial_observation = env.reset()

    print("Initial observation:")
    print(initial_observation)
    print()

    messages = [{"role": "user", "content": prompt}]
    if initial_observation:
        messages.append({"role": "user", "content": initial_observation})

    for turn in range(6):
        if env.done:
            break

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"Turn {turn + 1} - Model output: {generated_text}")

        # Try to parse tool call from the generated text
        try:
            # Try to extract a guess from tool call format or bracket format
            if "guess" in generated_text:
                # Parse JSON tool call
                start = generated_text.index("{")
                end = generated_text.rindex("}") + 1
                args = json.loads(generated_text[start:end])
                if "arguments" in args:
                    args = args["arguments"]
                guess_word = args.get("guess", "")
            else:
                # Fallback: extract from brackets [word]
                import re
                match = re.search(r"\[([a-zA-Z]{5})\]", generated_text)
                guess_word = match.group(1) if match else generated_text.strip()[:5]

            feedback = env.guess(f"[{guess_word}]")
            print(f"         Guess: {guess_word} | Reward: {env.reward}")
            print(f"         Feedback: {feedback.strip()}")
            print()

            messages.append({"role": "assistant", "content": generated_text})
            messages.append({"role": "user", "content": feedback})
        except Exception as e:
            print(f"         Error: {e}")
            break

    print(f"Game finished! Final reward: {env.reward}")
    print(f"Done: {env.done}")
```

Let's play the game!

```python
play_wordle(fine_tuned_model, tokenizer)
```
