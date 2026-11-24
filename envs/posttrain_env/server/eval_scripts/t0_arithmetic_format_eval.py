# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.14.1", "transformers==5.19.0"]
#
# [tool.uv]
# exclude-newer = "2026-10-10T00:00:00Z"
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
# SPDX-License-Identifier: BSD-3-Clause

"""Score `t0_arithmetic_format`: exact-match accuracy of `#### {a+b}` replies.

The split is drawn from `POSTTRAIN_SEED`. The dev seed is public; the test seed
is held by the environment server and only reaches score jobs.

Only the weights come from the checkpoint. The tokenizer, chat template, model
config and generation settings are the base model's, which the broker mounts at
`/base_model` at a pinned revision: a chat template is a program that sees the
question, so the agent's own template could write the answer into the prompt.

The job runs on CPU, so torch comes from the CPU-only index: the default wheel
pulls several GB of CUDA libraries, and that download alone can exceed the
job's timeout.
"""

import json
import os
import random
import re

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

BASE_MODEL = "/base_model"
N_PROMPTS = {"dev": 50, "test": 100}


def emit(result):
    print("POSTTRAIN_SCORE " + json.dumps(result), flush=True)


def invalid(feedback):
    emit({"status": "invalid", "score": 0.0, "feedback": feedback})
    raise SystemExit(0)


split = os.environ["POSTTRAIN_SPLIT"]
rng = random.Random(int(os.environ["POSTTRAIN_SEED"]))
pairs = [(rng.randint(0, 99), rng.randint(0, 99)) for _ in range(N_PROMPTS[split])]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
config = AutoConfig.from_pretrained(BASE_MODEL)
generation_config = GenerationConfig.from_pretrained(BASE_MODEL)
generation_config.update(max_new_tokens=16, do_sample=False)

try:
    # The checkpoint is agent output: never unpickle it or run its code, and
    # ignore its config so the weights must fit the base architecture.
    model, info = AutoModelForCausalLM.from_pretrained(
        "/model",
        config=config,
        dtype=torch.float32,
        use_safetensors=True,
        output_loading_info=True,
    )
except Exception as e:  # Report a bad checkpoint as a score, don't crash.
    invalid(f"cannot load /model: {type(e).__name__}")
if info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"]:
    invalid("/model weights do not match the base model")
# from_pretrained also reads /model/generation_config.json; generate() would fill
# unset fields from it, so the checkpoint could change or break decoding.
model.generation_config = generation_config
model.eval()

correct = formatted = 0
for a, b in pairs:
    messages = [{"role": "user", "content": f"What is {a} + {b}?"}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=generation_config)
    reply = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    match = re.fullmatch(r"#### (\d+)", reply)
    formatted += match is not None
    correct += match is not None and int(match.group(1)) == a + b

accuracy = correct / len(pairs)
emit(
    {
        "status": "ok",
        "score": accuracy,
        "metrics": {"accuracy": accuracy, "format_rate": formatted / len(pairs)},
    }
)
