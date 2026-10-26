# OpenEnv Wordle with GRPO using TRL

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/huggingface/trl/blob/main/examples/grpo_wordle/grpo_wordle.ipynb)

![trl banner](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_banner_dark.png)

With [**Transformers Reinforcement Learning (TRL)**](https://github.com/huggingface/trl), you can train a model that learns to **play Wordle**, a word-guessing game, through interaction and reinforcement.

- [TRL GitHub Repository](https://github.com/huggingface/trl)
- [Official TRL Examples](https://huggingface.co/docs/trl/example_overview)
- [Community Tutorials](https://huggingface.co/docs/trl/community_tutorials)
- [OpenEnv](https://github.com/huggingface/OpenEnv)

An **agentic environment** is a setting where a model can take actions, observe outcomes, and adjust its behavior based on feedback, similar to how humans learn from trial and error.
In this case, the agent interacts with the **Wordle** environment through the [**OpenEnv**](https://github.com/huggingface/OpenEnv) framework, which standardizes multi-agent and RL-style text environments.

[Wordle](https://en.wikipedia.org/wiki/Wordle) is a popular word puzzle where the player must guess a secret five-letter word within six tries.
After each guess, feedback indicates whether each letter is:

- 🟩 **Correct and in the right position**
- 🟨 **Present but in the wrong position**
- ⬛ **Not in the word**

This feedback loop makes Wordle a perfect environment for **RL with LLMs**, where the goal is to maximize the probability of guessing the correct word efficiently.

We will fine-tune a model using **GRPO** (Group Relative Policy Optimization) via TRL.
The agent will:

1. Generate guesses based on the game state and feedback.
2. Receive structured feedback from the environment after each guess.
3. Learn to improve its guessing strategy over time through reward signals.

---

## Install dependencies

We will start by installing **TRL**, which automatically includes the main dependencies like **Transformers**.
We will also install the **OpenEnv** framework (for the environment), **trackio** (for logging and monitoring training runs), and **vLLM** (for efficient generation).

```python
!pip install -Uq git+https://github.com/huggingface/trl.git git+https://github.com/huggingface/OpenEnv.git trackio vllm==0.10.2 bitsandbytes
```

---

## Log in to Hugging Face

Log in to your **Hugging Face** account to save your fine-tuned model, track your experiment results directly on the Hub or access gated models. You can find your **access token** on your [account settings page](https://huggingface.co/settings/tokens).

```python
from huggingface_hub import notebook_login

notebook_login()
```

---

## Initialize the Environment

Let us begin by setting up the environment that will be used during training.
For this task, we will rely on the **TextArena** environment from **OpenEnv**, which exposes a familiar Gymnasium-style API (`reset()`, `step()`, etc.) to simplify interaction.

In this example, we will connect to the hosted environment at [burtenshaw/textarena](https://huggingface.co/spaces/burtenshaw/textarena).
For production use or custom configurations, we **strongly recommend** running the environment locally via Docker. The hosted versions on the Hub currently have limited concurrency support, so duplicating the Space to your own account is the preferred approach in those cases.

For more information, refer to the [TRL-OpenEnv documentation](https://huggingface.co/docs/trl/main/en/openenv).

```python
from envs.textarena_env import TextArenaEnv

textarena_url = "https://burtenshaw-textarena.hf.space" # Duplicate the Space and update this!
# GRPOTrainer calls rollout_func synchronously; EnvClient calls block automatically
# in synchronous code.
env = TextArenaEnv(base_url=textarena_url)
```

---

## Init model and tokenizer

We will use [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), a lightweight instruction-tuned model that works well for quick experiments.
Despite its small size, it can still learn interesting strategies during fine-tuning.
If you have stronger hardware, you can easily scale up to larger models.

```python
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

---

## Rollout function with helpers

The **rollout function** defines how the agent interacts with the environment during GRPO training.
It is responsible for generating model completions, collecting feedback (rewards), and returning all necessary information for optimization.

In this setup:

- The function is called automatically by the **GRPOTrainer** during each training step.
- It uses the trainer's built-in `generate_rollout_completions()` method for efficient generation with vLLM in colocate mode.
- Each rollout represents a full interaction loop. The model guesses, receives feedback from Wordle, and updates based on reward signals.

### System Prompt

First, we define the `system_prompt` that guides the model's behavior as an expert Wordle solver with strategic reasoning and structured responses.

```python
system_prompt = """
You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

## GAME RULES

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN: Letter is correct and in the correct position
   - YELLOW: Letter is in the word but in the wrong position
   - GRAY: Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed

## RESPONSE FORMAT

Only respond with your next guess in square brackets, e.g., [crane].

## STRATEGIC APPROACH

Do not repeat the same guess twice.

### Opening Strategy
- Start with words rich in common vowels (A, E, I, O, U) and consonants (R, S, T, L, N)
- Optimal starters: CRANE, SLATE, STARE, AROSE, IRATE

### Mid-Game Strategy
- Use confirmed GREEN letters in their correct positions
- Place YELLOW letters in different positions than where they appeared
- Eliminate GRAY letters from consideration

## YOUR GOAL

Solve the Wordle in as few guesses as possible by strategically using feedback to eliminate impossible words and narrow down the solution space efficiently.
"""
```

### Rollout Function

```python
def rollout_func(prompts, trainer=None):
    """
    Rollout function for GRPO training with environment interaction.
    """
    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    correctness_rewards = []
    green_rewards = []
    yellow_rewards = []
    repetition_rewards = []

    for prompt_text in prompts:
        episode = rollout_once(
            trainer=trainer,
            env=env,
            tokenizer=tokenizer,
            dataset_prompt=prompt_text,
            system_prompt=system_prompt,
            max_turns=6,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        correctness_rewards.append(episode["correct_reward"])
        green_rewards.append(episode["green_reward"])
        yellow_rewards.append(episode["yellow_reward"])
        repetition_rewards.append(episode["repetition_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "correct_reward": correctness_rewards,
        "green_reward": green_rewards,
        "yellow_reward": yellow_rewards,
        "repetition_reward": repetition_rewards,
    }
```

---

## Define rollout_once

The `rollout_once` function runs **one full interaction loop** between the model and the Wordle environment using the trainer's generation method.

```python
from collections import defaultdict
from envs.textarena_env import TextArenaAction
from envs.textarena_env.rewards import extract_feedback_counts, extract_guess, extract_wordle_feedback
from trl.experimental.openenv import generate_rollout_completions


def rollout_once(trainer, env, tokenizer, dataset_prompt, system_prompt, max_turns):
    """
    Execute one full Wordle episode with the model.
    """
    result = env.reset()
    observation = result.observation

    prompt_ids = []
    completion_ids = []
    logprobs = []
    raw_rewards = []
    green_scores = []
    yellow_scores = []
    repetition_scores = []
    correct_scores = []
    guess_counts = defaultdict(int)

    for _turn in range(max_turns):
        if result.done:
            break

        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_user_prompt(base_prompt, observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        guess = extract_guess(completion_text)
        result = env.step(TextArenaAction(message=guess))
        raw_rewards.append(float(result.reward or 0.0))
        observation = result.observation
        correct_score = float(result.reward or 0.0)
        feedback = extract_wordle_feedback(observation)

        previous_occurrences = guess_counts[guess]
        repetition_score = scale_repetition_score(previous_occurrences, len(guess_counts))
        guess_counts[guess] += 1

        if not feedback:
            green_score = 0.0
            yellow_score = 0.0
        else:
            green_count, yellow_count = extract_feedback_counts(feedback)
            green_score = green_count / 5.0
            yellow_score = yellow_count / 5.0

        repetition_scores.append(repetition_score)
        green_scores.append(green_score)
        yellow_scores.append(yellow_score)
        correct_scores.append(correct_score)

    correct_reward_value = correct_scores[-1] if correct_scores else (raw_rewards[-1] if raw_rewards else 0.0)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "raw_rewards": raw_rewards,
        "correct_reward": correct_reward_value,
        "green_reward": green_scores[-1] if green_scores else 0.0,
        "yellow_reward": yellow_scores[-1] if yellow_scores else 0.0,
        "repetition_reward": repetition_scores[-1] if repetition_scores else 0.0,
    }
```

---

## Helper functions

```python
def make_user_prompt(prompt_text, messages):
    """Builds a structured user prompt combining the task description and message history"""
    history = format_history(messages)
    prompt_section = prompt_text.strip() if prompt_text.strip() else "Wordle-v0"
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return (
        f"Game prompt:\n{prompt_section}\n\n"
        f"Conversation so far:\n{history_section}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )

def format_history(messages):
    """Formats the message history with tags for clear conversational context"""
    lines = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)

def scale_repetition_score(previous_occurrences, max_occurrences):
    """Scale the repetition score based on the number of previous occurrences from 0 to 1"""
    if max_occurrences == 0:
        return 0.0
    return (max_occurrences - previous_occurrences) / max_occurrences
```

---

## Define reward functions

```python
def reward_correct(completions, **kwargs):
    rewards = kwargs.get("correct_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_greens(completions, **kwargs):
    rewards = kwargs.get("green_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_yellows(completions, **kwargs):
    rewards = kwargs.get("yellow_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_repetition(completions, **kwargs):
    rewards = kwargs.get("repetition_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]
```

---

## Create dataset

```python
from datasets import Dataset

dataset_size = 1000
dataset_prompt = "Play Wordle like an expert."

dataset = Dataset.from_dict({"prompt": [dataset_prompt] * dataset_size})
```

---

## Set GRPO Config

```python
from trl import GRPOConfig

output_dir = "wordle-grpo-Qwen3-1.7B"

grpo_config = GRPOConfig(
    num_train_epochs = 1,
    learning_rate = 5e-6,
    gradient_accumulation_steps = 64,
    per_device_train_batch_size = 1,
    warmup_steps = 20,
    num_generations = 2,
    max_completion_length = 8,
    max_prompt_length = 1400,
    use_vllm = True,
    vllm_mode = "colocate",
    vllm_gpu_memory_utilization = 0.1,
    output_dir = output_dir,
    report_to="trackio",
    trackio_space_id = output_dir,
    logging_steps = 1,
    save_steps = 10,
    gradient_checkpointing = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    push_to_hub = True,
)
```

---

## Create GRPOTrainer and start training

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model_name,
    processing_class=tokenizer,
    reward_funcs=[
        reward_correct,
        reward_greens,
        reward_yellows,
        reward_repetition,
    ],
    train_dataset=dataset,
    args=grpo_config,
    rollout_func=rollout_func,
)
```

### Memory stats before training

```python
import torch
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

**Output:**
```
GPU = NVIDIA A100-SXM4-40GB. Max memory = 39.557 GB.
10.516 GB of memory reserved.
```

### Train!

```python
trainer_stats = trainer.train()
```

**Training Progress:**

| Step | Training Loss |
|------|---------------|
| 1 | 0.008300 |
| 2 | 0.001900 |
| 3 | 0.015100 |
| 4 | 0.008700 |
| 5 | 0.009800 |
| 6 | 0.006700 |
| 7 | 0.006100 |
| 8 | 0.004400 |
| 9 | -0.002100 |
| 10 | 0.007500 |
| 11 | 0.008400 |
| 12 | 0.008000 |
| 13 | 0.007800 |
| 14 | -0.002400 |
| 15 | -0.003200 |
| 16 | -0.006000 |
| 17 | -0.008300 |
| 18 | -0.011000 |
| 19 | -0.004200 |
| 20 | -0.001700 |
| 21 | -0.004100 |
| 22 | -0.011600 |
| 23 | -0.006400 |
| 24 | -0.009100 |
| 25 | 0.003200 |
| 26 | 0.005100 |
| 27 | -0.002800 |
| 28 | 0.001400 |
| 29 | 0.011500 |
| 30 | -0.010500 |
| 31 | -0.006400 |

### Memory stats after training

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

**Output:**
```
5231.7046 seconds used for training.
87.2 minutes used for training.
Peak reserved memory = 36.68 GB.
Peak reserved memory for training = 26.164 GB.
Peak reserved memory % of max memory = 92.727 %.
Peak reserved memory for training % of max memory = 66.143 %.
```

### Save and push to Hub

```python
env.close()
trainer.save_model(output_dir)
trainer.push_to_hub()
```

---

## Load the Fine-Tuned Model and Run Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sergiopaniego/wordle-grpo-Qwen3-1.7B" # Replace with your HF username

fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

```python
MAX_TURNS=6

def play_wordle(env, model, tokenizer):
    result = env.reset()
    observation = result.observation

    print("Initial Prompt:\n" + observation.prompt)

    for turn in range(MAX_TURNS):
        if result.done:
            break

        user_prompt = make_user_prompt(observation.prompt, observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        guess = extract_guess(generated_text)

        print(f"\nTurn {turn}: model replied with -> {generated_text}")
        print(f"   Parsed guess: {guess}")

        result = env.step(TextArenaAction(message=guess))
        observation = result.observation

        print("   Feedback messages:")
        for message in observation.messages:
            print(f"     [{message.category}] {message.content}")

    print("\nGame finished")
    print(f"   Reward: {result.reward}")
    print(f"   Done: {result.done}")
```

### Let us play the game!

```python
try:
    play_wordle(env, fine_tuned_model, tokenizer)
finally:
    env.close()
```

**Output:**
```
Initial Prompt:
You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be given as follows:
  - G (green): correct letter in the correct position
  - Y (yellow): letter exists in the word but in the wrong position
  - X (wrong): letter is not in the word
Enter your guess to begin.

Turn 0: model replied with -> [crane]
   Parsed guess: [crane]
   Feedback messages:
     [MESSAGE] [crane]
     [MESSAGE] Player 0 submitted [crane].
Feedback:
C R A N E
X Y X X X

You have 5 guesses left.

Turn 1: model replied with -> [spare]
   Parsed guess: [spare]
   Feedback messages:
     [MESSAGE] [spare]
     [MESSAGE] Player 0 submitted [spare].
Feedback:
C R A N E
X Y X X X

S P A R E
G X X G X

You have 4 guesses left.

...

Game finished
   Reward: 0.0
   Done: True
```

> **Note:** The model has learned some good opening strategies (starting with "crane", then "spare"), but still tends to repeat guesses. This is a common challenge in RL training that can be improved with:
>
> - Longer training runs
> - Stronger repetition penalties
> - Better reward shaping
> - Larger models
