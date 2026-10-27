# /// script
# dependencies = [
#     "transformers",
#     "torch",
#     "openenv==0.10.0",
#     "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env",
# ]
# ///
"""
Play Wordle with a model and save every game, to compare models or make a GIF with `make_gif.py`.

The model plays as in `grpo.py`: the same prompt and the same `guess` tool. Pass `--thinking` for models trained with
thinking on. With `--secrets`, the environment runs in-process and the secret words are fixed, so every model plays the
same games. Without it, games are played against `--env-url`.

```sh
uv run evaluate.py --model Qwen/Qwen3-1.7B --max-new-tokens 256 --secrets "$(cat secrets30.txt)" --out base.jsonl
uv run evaluate.py --model openenv/Qwen3-1.7B-Wordle-SFT-GRPO --thinking --max-new-tokens 512 \
    --secrets "$(cat secrets30.txt)" --out grpo.jsonl
```
"""

import argparse
import json
import re
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from huggingface_hub import get_token
from textarena_env import TextArenaAction, TextArenaEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema


PROMPT = """You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

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


def guess(guess: str) -> str:
    """
    Make a guess in the Wordle environment.

    Args:
        guess: The guessed word, formatted as '[abcde]'

    Returns:
        The feedback message from the environment.
    """


TOOLS = [get_json_schema(guess)]
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
THINK = re.compile(r"<think>(.*?)</think>", re.S)
FEEDBACK = re.compile(r"Feedback:\n([A-Z ]+)\n([GYX ]+)")


def play(model, tokenizer, env, args):
    result = env.reset()
    full = result.observation.messages[0].content
    # Same as TRL: the reset text is appended to the prompt with no separator.
    messages = [{"role": "user", "content": PROMPT + full}]
    rows, reward, done = [], 0.0, False
    legal_guesses = 0
    # Allow one invalid warning per legal guess, but bound nonterminating servers.
    while not done and legal_guesses < 6 and len(rows) < 12:
        text = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=args.thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        sampling = (
            {"do_sample": True, "temperature": args.temperature}
            if args.temperature > 0
            else {"do_sample": False}
        )
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, **sampling
        )
        reply = tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        call = TOOL_CALL.search(reply)
        if call is None:
            break  # the model answered without calling the tool
        word = json.loads(call.group(1)).get("arguments", {}).get("guess", "")
        result = env.step(TextArenaAction(message=word))
        new_full = result.observation.messages[0].content
        feedback, full = new_full[len(full) :], new_full
        invalid = "invalid move" in feedback.lower()
        legal_guesses += not invalid
        reward, done = (
            (0.0 if invalid else result.reward or 0.0),
            result.done,
        )
        match = FEEDBACK.search(feedback)
        thinking = THINK.search(reply)
        reasoning = thinking.group(1).strip() if thinking else ""
        rows.append(
            {
                "guess": word,
                "letters": match.group(1).replace(" ", "")
                if match
                else word.strip("[]").upper(),
                "colors": match.group(2).replace(" ", "")
                if match
                else "G" * 5
                if reward >= 1.0
                else None,
                "reasoning": reasoning,
            }
        )
        messages += [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "guess", "arguments": {"guess": word}},
                    }
                ],
            },
            {"role": "tool", "name": "guess", "content": feedback},
        ]
    secret = env.state().raw_state.get("game_state", {}).get("secret_word")
    return {"rows": rows, "reward": reward, "won": reward >= 1.0, "secret": secret}


class LocalWordle:
    """The Wordle server class run in-process, with the secret word set by us, so every model plays the same words."""

    def __init__(self, secrets):
        from textarena_env.server.environment import TextArenaEnvironment

        self.env, self.secrets = TextArenaEnvironment("Wordle-v0"), iter(secrets)

    def reset(self):
        observation = self.env.reset()
        game = self.env._ta_env
        while hasattr(game, "env"):
            game = game.env
        game.state.game_state["secret_word"] = next(self.secrets)
        return SimpleNamespace(observation=observation)

    def step(self, action):
        observation = self.env.step(action)
        return SimpleNamespace(
            observation=observation, reward=observation.reward, done=observation.done
        )

    def state(self):
        return self.env.state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--env-url", default="https://openenv-wordle.hf.space")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--secrets",
        help="comma-separated words: play them in-process instead of against --env-url",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # T4 GPUs (free Colab) have no bf16 support.
    dtype = (
        torch.float16
        if device == "cuda" and not torch.cuda.is_bf16_supported()
        else torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    # A private Space needs your token. A public one ignores it.
    headers = {"Authorization": f"Bearer {get_token()}"} if get_token() else None

    if args.secrets:
        secrets = args.secrets.split(",")
        args.games = len(secrets)
        env_context = nullcontext(LocalWordle(secrets))
    else:
        env_context = TextArenaEnv(base_url=args.env_url, headers=headers).sync()
    with env_context as env, open(args.out, "w") as f:
        for i in range(args.games):
            game = play(model, tokenizer, env, args)
            f.write(json.dumps({"model": args.model, **game}) + "\n")
            print(
                f"game {i + 1}: {'won' if game['won'] else 'lost'} in {len(game['rows'])} guesses, reward {game['reward']}"
            )


if __name__ == "__main__":
    main()
