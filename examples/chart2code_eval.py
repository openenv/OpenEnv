"""Score vision-language models on the chart-to-code environment.

Each model sees the target chart, writes matplotlib code, may render drafts
(seeing what its code draws) and then submits. The environment does all the
scoring; this script only relays messages and averages the rewards per level.

Example:

```bash
# Against a local server
PYTHONPATH=src:envs uvicorn chart2code_env.server.app:app --port 8000 &
python examples/chart2code_eval.py --env-url http://127.0.0.1:8000 \\
    --models Qwen/Qwen3-VL-8B-Instruct --tasks 20
```

Models are called through Hugging Face Inference Providers (`HF_TOKEN`).
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from envs.chart2code_env import Chart2CodeAction, Chart2CodeEnv
from huggingface_hub import get_token, InferenceClient

SYSTEM = (
    "You write Python code with matplotlib. Reply with one ```python code block "
    "and nothing else."
)
FEEDBACK = (
    "This is what your code drew{error}. Fix the differences from the target and "
    "reply with the complete corrected code."
)


def extract_code(reply: str) -> str:
    """The last fenced code block of a reply, or the whole reply."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, re.DOTALL)
    return blocks[-1] if blocks else reply


def image_part(png_base64: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{png_base64}"},
    }


def play(client, model, env, level, index, renders, max_tokens) -> float:
    """One episode: draft, render up to `renders` times, then submit."""
    observation = env.reset(split="test", level=level, task_index=index).observation
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                image_part(observation.target_png_base64),
                {"type": "text", "text": observation.prompt},
            ],
        },
    ]
    code = ""
    for attempt in range(renders + 1):
        reply = client.chat_completion(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0
        )
        content = reply.choices[0].message.content or ""
        code = extract_code(content)
        if attempt == renders:
            break
        draft = env.step(Chart2CodeAction(code=code, mode="render")).observation
        messages.append({"role": "assistant", "content": content})
        error = f" (it failed: {draft.error.splitlines()[-1]})" if draft.error else ""
        parts = (
            [image_part(draft.rendered_png_base64)] if draft.rendered_png_base64 else []
        )
        messages.append(
            {
                "role": "user",
                "content": parts
                + [{"type": "text", "text": FEEDBACK.format(error=error)}],
            }
        )
    result = env.step(Chart2CodeAction(code=code, mode="submit"))
    return result.reward or 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score VLMs on chart2code_env.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-url", default="http://127.0.0.1:8000")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-VL-8B-Instruct"])
    parser.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--tasks", type=int, default=20, help="Test tasks per level.")
    parser.add_argument(
        "--renders", type=int, default=2, help="Drafts before submitting."
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    client = InferenceClient(api_key=get_token(), timeout=300)
    with Chart2CodeEnv(base_url=args.env_url).sync() as env:
        for model in args.models:
            for level in args.levels:
                rewards = [
                    play(
                        client, model, env, level, index, args.renders, args.max_tokens
                    )
                    for index in range(args.tasks)
                ]
                print(
                    f"{model}  {level:<6}  mean {statistics.mean(rewards):.3f}  "
                    f"solved {sum(r >= 0.99 for r in rewards)}/{len(rewards)}"
                )


if __name__ == "__main__":
    main()
