# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers"]
# ///
# SPDX-License-Identifier: BSD-3-Clause

"""Score `t0_arithmetic_format`: exact-match accuracy of `#### {a+b}` replies.

The split is drawn from `POSTTRAIN_SEED`. The dev seed is public; the test seed
is held by the environment server and only reaches score jobs.
"""

import json
import os
import random
import re

N_PROMPTS = {"dev": 50, "test": 100}


def emit(result):
    print("POSTTRAIN_SCORE " + json.dumps(result), flush=True)


split = os.environ["POSTTRAIN_SPLIT"]
rng = random.Random(int(os.environ["POSTTRAIN_SEED"]))
pairs = [(rng.randint(0, 99), rng.randint(0, 99)) for _ in range(N_PROMPTS[split])]

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/model")
    # The checkpoint is agent output: never unpickle it or run its code.
    model = AutoModelForCausalLM.from_pretrained(
        "/model", torch_dtype=torch.float32, use_safetensors=True
    )
    model.eval()
except Exception as e:  # Report a bad checkpoint as a score, don't crash.
    emit(
        {
            "status": "invalid",
            "score": 0.0,
            "feedback": f"cannot load /model: {type(e).__name__}",
        }
    )
    raise SystemExit(0)

correct = formatted = 0
for a, b in pairs:
    messages = [{"role": "user", "content": f"What is {a} + {b}?"}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
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
