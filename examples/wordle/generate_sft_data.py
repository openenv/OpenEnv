# /// script
# dependencies = [
#     "openenv==0.10.0",
#     "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env",
#     "transformers",
#     "datasets",
# ]
# ///
"""
Generate Wordle games for a short SFT warmup before GRPO.

A solver plays real games against the TextArena Wordle environment (run in-process) and writes each one as a
tool-calling conversation: a short reasoning and a `guess` call per turn, then the environment feedback. The format
matches what TRL's `GRPOTrainer` renders with `environment_factory`, so the model plays the same way during RL.

```sh
uv run generate_sft_data.py --games 1000 --push-to-hub <user>/wordle-tool-calls-sft
```
"""

import argparse
import json
import random
from collections import Counter

from datasets import Dataset
from textarena_env import TextArenaAction
from textarena_env.server.environment import TextArenaEnvironment
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

OPENERS = [
    "crane",
    "slate",
    "raise",
    "trace",
    "stare",
    "crate",
    "arise",
    "least",
    "alert",
    "roast",
    "saint",
    "stale",
    "snare",
    "trade",
]
NAMES = {"G": "green", "Y": "yellow", "X": "gray"}
ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}


def guess(guess: str) -> str:
    """
    Make a guess in the Wordle environment.

    Args:
        guess: The guessed word, formatted as '[abcde]'

    Returns:
        The feedback message from the environment.
    """


def score(word, secret):
    feedback, rest = ["X"] * 5, list(secret)
    for i in range(5):
        if word[i] == secret[i]:
            feedback[i], rest[i] = "G", None
    for i in range(5):
        if feedback[i] == "X" and word[i] in rest:
            feedback[i] = "Y"
            rest[rest.index(word[i])] = None
    return "".join(feedback)


def pick(candidates, rng):
    # The guess that leaves the fewest candidates on average, at random among near-ties.
    def cost(word):
        return sum(
            n * n
            for n in Counter(score(word, secret) for secret in candidates).values()
        )

    ranked = sorted(rng.sample(candidates, min(len(candidates), 150)), key=cost)
    return rng.choice(
        [word for word in ranked[:3] if cost(word) <= cost(ranked[0]) + 2]
    )


def reasoning(history, candidates, choice, rng):
    if not history:
        letters = ", ".join(c.upper() for c in choice)
        vowels = " and ".join(c.upper() for c in choice if c in "aeiou")
        return rng.choice(
            [
                f"No feedback yet, so I want an opener with common letters. {choice.upper()} tests {letters}, including the vowels {vowels}.",
                f"First guess. I'll start with {choice.upper()}: five distinct, frequent letters ({letters}) that should narrow the word down quickly.",
                f"Nothing to go on yet. A good opener covers frequent letters and vowels, so I'll try {choice.upper()}.",
            ]
        )
    word, feedback = history[-1]
    lines = [
        f"{word.upper()} gave {' '.join(feedback)}: "
        + ", ".join(f"{c.upper()} {NAMES[f]}" for c, f in zip(word, feedback))
        + "."
    ]
    greens, misplaced, found = ["_"] * 5, {}, set()
    for word, feedback in history:
        for i, (c, f) in enumerate(zip(word, feedback)):
            if f == "G":
                greens[i] = c.upper()
            if f in "GY":
                found.add(c)
            if f == "Y":
                misplaced.setdefault(c, set()).add(i + 1)
    grays = {
        c
        for word, feedback in history
        for c, f in zip(word, feedback)
        if f == "X" and c not in found
    }
    known = []
    if greens != ["_"] * 5:
        known.append(f"Pattern so far: {' '.join(greens)}.")
    if misplaced:
        known.append(
            "In the word but misplaced: "
            + ", ".join(
                f"{c.upper()} (not {'/'.join(ORDINALS[p] for p in sorted(ps))})"
                for c, ps in sorted(misplaced.items())
            )
            + "."
        )
    if grays:
        known.append(f"Ruled out: {', '.join(sorted(c.upper() for c in grays))}.")
    lines.append(" ".join(known))
    if len(candidates) == 1:
        lines.append(
            rng.choice(
                [
                    f"Only {choice.upper()} fits all of that, so it should be the answer.",
                    f"The one word that fits every clue is {choice.upper()}.",
                ]
            )
        )
    elif len(candidates) == 2:
        other = next(w for w in candidates if w != choice)
        lines.append(
            f"Two words fit: {choice.upper()} and {other.upper()}. I'll try {choice.upper()} first."
        )
    else:
        shown = [choice] + [w for w in candidates if w != choice][: rng.randint(2, 4)]
        rng.shuffle(shown)
        lines.append(
            f"Words that fit: {', '.join(w.upper() for w in shown)}"
            + (", and a few more." if len(candidates) > len(shown) else ".")
        )
        lines.append(
            rng.choice(
                [
                    f"{choice.upper()} fits the clues and its letters split the remaining options best, so I'll guess it.",
                    f"I'll go with {choice.upper()}: it is consistent with everything and tests the letters that separate these words.",
                    f"{choice.upper()} is consistent with all feedback and tells the others apart, so it's my next guess.",
                ]
            )
        )
    return "\n".join(lines)


def play(env, rng):
    full = env.reset().messages[0].content
    game = env._ta_env
    while hasattr(game, "env"):
        game = game.env
    secret = game.state.game_state["secret_word"]
    # TRL appends the reset text to the prompt with no separator.
    messages = [{"role": "user", "content": PROMPT + full}]
    candidates, history, observation = list(game.word_list), [], None
    while observation is None or not observation.done:
        choice = rng.choice(OPENERS) if not history else pick(candidates, rng)
        call = [
            {
                "type": "function",
                "function": {"name": "guess", "arguments": {"guess": f"[{choice}]"}},
            }
        ]
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning(history, candidates, choice, rng),
                "tool_calls": call,
            }
        )
        observation = env.step(TextArenaAction(message=f"[{choice}]"))
        new_full = observation.messages[0].content
        messages.append(
            {"role": "tool", "name": "guess", "content": new_full[len(full) :]}
        )
        full = new_full
        history.append((choice, score(choice, secret)))
        candidates = [
            w for w in candidates if score(choice, w) == history[-1][1] and w != choice
        ]
    word = secret.upper()
    closing = rng.choice(
        [
            f"The game says [{secret}] was correct, so the game is over.",
            f"{word} was the word: the game congratulated me and there are no guesses left to make.",
            f"Correct on guess {len(history)}. Nothing more to do.",
        ]
    )
    messages.append(
        {
            "role": "assistant",
            "content": f"Solved: the word is {word}.",
            "reasoning_content": closing,
        }
    )
    return {
        "messages": messages,
        "tools": json.dumps([get_json_schema(guess)]),
    }, observation.reward >= 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="wordle_sft.jsonl")
    parser.add_argument(
        "--push-to-hub",
        help="dataset repo to push to, e.g. <user>/wordle-tool-calls-sft",
    )
    args = parser.parse_args()

    random.seed(
        args.seed
    )  # TextArena draws the secret word from the global random state
    rng, env, rows = random.Random(args.seed), TextArenaEnvironment("Wordle-v0"), []
    while len(rows) < args.games:
        row, won = play(env, rng)
        if won:
            rows.append(row)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    print(f"wrote {len(rows)} games to {args.out}")
    if args.push_to_hub:
        Dataset.from_list(rows).push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
