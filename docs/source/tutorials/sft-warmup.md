# Collect Rollouts for SFT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/sft_warmup.ipynb)

OpenEnv environments are not only useful for RL training — they are also a natural tool for **collecting
rollouts that become supervised training data**. The environment handles episode management, automatic scoring,
and reproducibility, so you get a reward-labeled dataset without writing any of that infrastructure yourself.

This tutorial shows the full pipeline:

1. Run a strong teacher model inside an OpenEnv environment to collect rollouts.
2. Use the environment's reward signal to filter out incorrect examples automatically.
3. Train a smaller student model on the filtered rollouts with TRL's `SFTTrainer`.

As a concrete application, the resulting checkpoint is used as a warm-start for GRPO: once the student
reliably produces valid tool calls, GRPO's `reward_std` is non-zero from the first batch and the reward
curve climbs immediately.

## Why use an environment to collect training data

Building a supervised dataset usually means writing a custom collection loop, a scorer, and episode
bookkeeping. An OpenEnv environment gives you all three out of the box:

- **Automatic scoring** — every `step()` returns a reward. Filter by `reward == 1.0` and you have a
  clean, correct dataset with no manual labelling.
- **Reproducible episodes** — `reset(seed=42, size=N)` produces the same sequence of problems every
  run. Anyone can regenerate the exact dataset.
- **Configurable difficulty** — adjust `DATASET_CONFIG` to control problem complexity without changing
  any collection code.
- **Portable across environments** — the same collect → filter → train pipeline works for any OpenEnv
  environment. Swap the env and the tool definition; everything else stays the same.

## What you'll use

| | |
|---|---|
| **Student model** | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) |
| **Teacher model** | `gpt-5-mini` via the OpenAI API |
| **Environment** | [`reasoning_gym_env`](https://github.com/huggingface/OpenEnv/tree/main/envs/reasoning_gym_env) / `chain_sum` |
| **SFT trainer** | [TRL `SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer) |
| **Next step** | [End-to-end walkthrough with GRPO](end-to-end-walkthrough) |

---

## 1. Install dependencies

```python
!pip install -q openai trl
!pip install -q openenv
!pip install -q --no-deps git+https://huggingface.co/spaces/sergiopaniego/reasoning_gym
!pip install -Uq "transformers>=5.3.0"
```

---

## 2. Set your credentials

```python
import getpass, os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
```

You'll also need a Hugging Face login to download the base model and push both the collected dataset
and the fine-tuned checkpoint:

```python
from huggingface_hub import notebook_login

notebook_login()
```

```python
YOUR_HF_USERNAME = "your-username"  # replace with your Hugging Face username
assert YOUR_HF_USERNAME != "your-username", "Replace YOUR_HF_USERNAME with your Hugging Face username"
```

---

## 3. Define the system prompt

Use the same prompt as the [GRPO tutorial](end-to-end-walkthrough)
so the SFT-trained model is a drop-in replacement when you continue with GRPO.

```python
SYSTEM_PROMPT = """You are a careful arithmetic assistant.

You will be given a chain of integer additions. Compute the result and submit it as a single number.

Rules:
1. Read the question carefully.
2. Use the tool `answer` exactly once with your final number.
3. The answer must be a single integer with no units or explanation.
"""
```

---

## 4. Configure data collection

`DATASET_CONFIG` controls the difficulty of the `chain_sum` problems the environment generates:
`min_terms`/`max_terms` set how many integers are added together, and `min_digits`/`max_digits` set
how many digits each integer has. At these settings each problem is a sum of 2–3 two-digit numbers
— easy enough for `gpt-5-mini` to answer correctly ~90% of the time, which gives a clean training
signal after filtering.

`N_EPISODES` is the number of problems to collect. 300 is enough to get ~270 correct examples after
filtering, which is sufficient for format compliance training.

```python
DATASET_CONFIG = {
    "min_terms": 2,
    "max_terms": 3,
    "min_digits": 2,
    "max_digits": 2,
}

N_EPISODES = 300
```

---

## 5. Collect rollouts with `openenv collect`

`openenv collect` runs the teacher model inside the environment and records every episode — the
environment's `step()` reward is written alongside the messages, so filtering by correctness requires
no additional scoring code. By default it drops rollouts with a negative reward (`--keep-losses` keeps
them). Reasoning Gym scores a wrong answer `0.0`, so wrong answers are kept here and filtered in section 6.

```python
import json, shlex

dataset_config_arg = shlex.quote(json.dumps(DATASET_CONFIG))
system_prompt_arg = shlex.quote(SYSTEM_PROMPT)
hub_repo_arg = shlex.quote(f"{YOUR_HF_USERNAME}/chain-sum-rollouts")

!openenv collect reasoning_gym:chain_sum \
  --base-url https://sergiopaniego-reasoning-gym.hf.space \
  --provider openai \
  --model gpt-5-mini \
  --num-episodes {N_EPISODES} \
  --max-tokens 1024 \
  --dataset-config {dataset_config_arg} \
  --system-prompt {system_prompt_arg} \
  --push-to-hub {hub_repo_arg} \
  --output-dir ./rollouts
```

To use a self-hosted teacher instead of a hosted provider, point `--llm-endpoint` at any
OpenAI-compatible server (vLLM, TGI, Ollama) and pass the model id it serves:

```bash
openenv collect reasoning_gym:chain_sum \
  --base-url https://sergiopaniego-reasoning-gym.hf.space \
  --llm-endpoint http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --num-episodes 300 \
  --output-dir ./rollouts
```

The teacher drives the environment through tool calls, so the server must have tool calling
enabled (for vLLM: `--enable-auto-tool-choice --tool-call-parser <parser for your model>`).

The command prints a live progress summary and pushes the collected episodes to the Hub as
`{YOUR_HF_USERNAME}/chain-sum-rollouts`. Pull them back to start filtering:

```python
from datasets import load_dataset

ds = load_dataset(f"{YOUR_HF_USERNAME}/chain-sum-rollouts", split="train")
raw_rollouts = list(ds)
print(f"Collected {len(raw_rollouts)} episodes")
```

The `messages` field stores the full conversation in standard OpenAI format (assistant messages have
a `tool_calls` list). Convert to Qwen3's `<tool_call>` text format before training — GRPOTrainer
produces this same format during RL, so the SFT checkpoint becomes a direct drop-in:

```python
def to_qwen3_messages(record):
    converted = []
    for msg in record["messages"]:
        if msg["role"] == "tool":
            continue  # strip environment responses; SFT only needs the assistant turn
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            answer_str = args.get("answer", "")
            tool_call_text = (
                "<tool_call>\n"
                + json.dumps({"name": "answer", "arguments": {"answer": answer_str}})
                + "\n</tool_call>"
            )
            converted.append({"role": "assistant", "content": tool_call_text})
        else:
            converted.append(msg)
    return {"messages": converted, "reward": record["reward"]}

rollouts = [to_qwen3_messages(r) for r in raw_rollouts]
```

---

## 6. Filter the dataset

Keep only episodes where the teacher answered correctly. `openenv collect` already dropped negative
rewards, and this filter also removes the wrong answers, which score `0.0`. The environment's reward
signal does the labelling — no manual annotation needed.

```python
correct = [r for r in rollouts if r["reward"] == 1.0]
print(f"Correct: {len(correct)} / {len(rollouts)} ({len(correct)/len(rollouts):.1%})")
```

`gpt-5-mini` typically scores above 90% on `chain_sum` at this difficulty, so you should end up with
~270 examples from 300 rollouts.

---

## 7. Inspect the dataset before training

Always look at your data before training. Automated collection can introduce unexpected patterns that the
student model will learn to imitate.

```python
import random

for row in random.sample(correct, 3):
    question = row["messages"][0]["content"]
    response = row["messages"][1]["content"]
    print(f"Q: {question}")
    print(f"A: {response}")
    print()
```

Things to check:

- Does every response contain a valid `<tool_call>` block?
- Are the answers integers with no extra text?
- Is there any reasoning in the assistant message that you don't want the student to learn?
  (For example: an internal monologue, disclaimers, or repeated phrasing that the teacher leaked
  from its own system prompt.)

---

## 8. Measure token lengths

Set `max_length` in `SFTConfig` to cover nearly all examples without wasting GPU memory on padding.
The 99th percentile is a good target: you truncate fewer than 1% of examples while keeping batches tight.

```python
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

lengths = []
for row in correct:
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )
    ids = tokenizer.encode(text)
    lengths.append(len(ids))

lengths = np.array(lengths)
MAX_SEQ_LEN = int(np.percentile(lengths, 99)) + 16

print(
    f"p50={np.percentile(lengths, 50):.0f}  "
    f"p95={np.percentile(lengths, 95):.0f}  "
    f"p99={np.percentile(lengths, 99):.0f}  "
    f"max={lengths.max()}"
)
print(f"Setting MAX_SEQ_LEN = {MAX_SEQ_LEN}")
```

---

## 9. Fine-tune with SFTTrainer

`assistant_only_loss=True` in `SFTConfig` masks the prompt tokens so the loss is computed only on the
assistant response — the `<tool_call>` block. This is more efficient than full-sequence training and avoids
accidentally reinforcing the system prompt wording.

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

dataset = Dataset.from_list([{"messages": r["messages"]} for r in correct])

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")

sft_config = SFTConfig(
    output_dir="reasoning-gym-chain-sum-Qwen3-1.7B-sft",
    max_length=MAX_SEQ_LEN,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="no",
    assistant_only_loss=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
)

trainer.train()
trainer.push_to_hub(commit_message="SFT warm-up on reasoning_gym chain_sum")
```

> [!NOTE]
> Training ~270 examples for 3 epochs takes around 5 minutes on a single A100 (40 GB). The goal is format
> compliance, not task mastery — a handful of epochs is enough. Mastery comes from GRPO.

---

## 10. Evaluate: before vs after

Run both the base model and the SFT checkpoint on a held-out set and compare. The key metric for a
warm-up evaluation is **format compliance** — how often the model uses `<tool_call>` correctly — as
well as overall accuracy.

```python
import re
from transformers import pipeline
from reasoning_gym_env.client import ReasoningGymEnv
from reasoning_gym_env.models import ReasoningGymAction


async def evaluate_model(model_name, n_eval=50, seed=999):
    gen = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device_map="auto",
        dtype="auto",
    )
    gen.model.generation_config.max_length = None
    tok = AutoTokenizer.from_pretrained(model_name)
    env = ReasoningGymEnv(base_url="https://sergiopaniego-reasoning-gym.hf.space")

    obs = await env.reset(
        dataset_name="chain_sum",
        dataset_config=DATASET_CONFIG,
        seed=seed,
        size=n_eval,
    )

    rewards, format_hits = [], 0

    for i in range(n_eval):
        if i > 0:
            obs = await env.reset()

        question = obs.observation.question
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion = gen(prompt, max_new_tokens=64)[0]["generated_text"][len(prompt):]

        m = re.search(r'"answer"\s*:\s*"?(\d+)"?', completion)
        if m:
            format_hits += 1
            answer = m.group(1)
        else:
            nums = re.findall(r"\b(\d+)\b", completion)
            answer = nums[-1] if nums else "0"

        result = await env.step(ReasoningGymAction(answer=answer))
        rewards.append(float(result.observation.score or 0.0))

    await env.close()
    del gen  # free GPU memory before loading the next model

    return {
        "accuracy": sum(rewards) / len(rewards),
        "format_compliance": format_hits / n_eval,
    }


base_metrics = await evaluate_model("Qwen/Qwen3-1.7B")
sft_metrics = await evaluate_model(f"{YOUR_HF_USERNAME}/reasoning-gym-chain-sum-Qwen3-1.7B-sft")

print(f"\n{'Metric':<25} {'Base model':>12} {'After SFT':>12} {'Delta':>10}")
print("-" * 62)
for key, label in [("format_compliance", "Format compliance"), ("accuracy", "Accuracy")]:
    b, s = base_metrics[key], sft_metrics[key]
    print(f"{label:<25} {b:>12.1%} {s:>12.1%} {(s - b) * 100:>+9.1f} pp")
```

A successful warm-up looks like this:

| Metric | Base model | After SFT | Delta |
|---|---|---|---|
| Format compliance | ~0% | ~68% | +68 pp |
| Accuracy | ~4% | ~68% | +64 pp |

Format compliance should jump sharply from near-zero — that's the primary goal. `Qwen3-1.7B` produces
essentially no valid `<tool_call>` blocks out of the box. After SFT on ~270 examples, the model reliably
uses the format, and accuracy rises in lockstep because correct format is a prerequisite for the
environment's scorer to award any credit.

---

## 11. Where to go next: GRPO

The SFT checkpoint is ready to use as the starting model for GRPO. In the
[end-to-end walkthrough](end-to-end-walkthrough),
change one line in section 7:

```python
# Before (cold-start from the base model):
MODEL_NAME = "Qwen/Qwen3-1.7B"

# After (warm-start from your SFT checkpoint):
MODEL_NAME = f"{YOUR_HF_USERNAME}/reasoning-gym-chain-sum-Qwen3-1.7B-sft"
```

With most completions already in the right format, GRPO's `reward_std` will be non-zero from the very first
batch and the reward curve will climb immediately — no cold-start stall.

**Other directions:**

- **Harder tasks.** Increase `max_terms` or `max_digits` in `DATASET_CONFIG` and collect a new SFT set.
  Once the student handles easier examples reliably, a harder GRPO phase can push further.
- **Different environments.** The same pipeline — teacher collects → filter → SFT → GRPO — applies to
  any OpenEnv environment. Swap `reasoning_gym_env` and the `answer` tool definition for your env's
  tool surface.
- **Larger teacher.** A stronger model, such as `claude-opus-5-5` (`--provider anthropic`) or
  `Qwen/Qwen3-32B` served behind `--llm-endpoint`, yields higher-quality examples on tasks where
  `gpt-5-mini` struggles.
