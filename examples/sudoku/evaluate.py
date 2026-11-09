# /// script
# dependencies = [
#     "transformers",
#     "torch",
#     "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env",
# ]
# ///
"""
Play Sudoku with a model on a fixed set of puzzles (from `make_puzzles.py`) and save every game.

The environment runs in-process with the puzzle set by us, so every model plays the same boards. `--hints` appends the
cells that have a single candidate to every board, like `--difficulty easy` in the TRL `grpo_sudoku` example.

```sh
uv run evaluate.py --model Qwen/Qwen3-1.7B --puzzles puzzles_c65.json --out base_c65.jsonl
```
"""

import argparse
import copy
import json
import re

import torch
from textarena_env import TextArenaAction
from textarena_env.server.environment import TextArenaEnvironment
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema


PROMPT = """You are playing Sudoku. Fill every empty cell (shown as '.') of the 9x9 grid with a digit from 1 to 9 so that \
each row, each column and each 3x3 box contains every digit exactly once.

Use the tool `place` to fill one cell at a time. Rows and columns are numbered from 1 to 9. A digit that is not the \
solution of that cell is a mistake. Two mistakes in a row end the game.

"""


def place(row: int, col: int, number: int) -> str:
    """
    Place a number in an empty cell of the Sudoku grid.

    Args:
        row: Row number (1-9).
        col: Column number (1-9).
        number: Number to place (1-9).

    Returns:
        The updated grid, or the reason the move was rejected.
    """


TOOLS = [get_json_schema(place)]
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
THINK = re.compile(r"<think>(.*?)</think>", re.S)


def single_candidates(board):
    """Empty cells that only one digit can fill, as `[row col number]`."""
    hints = []
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                box = [
                    board[i][j]
                    for i in range(r - r % 3, r - r % 3 + 3)
                    for j in range(c - c % 3, c - c % 3 + 3)
                ]
                used = set(board[r]) | {board[i][c] for i in range(9)} | set(box)
                candidates = set(range(1, 10)) - used
                if len(candidates) == 1:
                    hints.append(f"[{r + 1} {c + 1} {candidates.pop()}]")
    return hints


class LocalSudoku:
    """The Sudoku server class run in-process, with the puzzle set by us."""

    def __init__(self):
        self.env = TextArenaEnvironment("Sudoku-v0")
        self.game = self.env._ta_env
        while hasattr(self.game, "env"):
            self.game = self.game.env

    def reset(self, puzzle, solution):
        self.full = self.env.reset().messages[0].content
        self.game.full_grid, self.game.game_board = solution, copy.deepcopy(puzzle)
        self.game.state.game_state["board"] = copy.deepcopy(puzzle)
        return self.render()

    def step(self, move):
        """Returns the new text of the game log (the server sends the whole log every step), the reward and done."""
        observation = self.env.step(TextArenaAction(message=move))
        full = observation.messages[0].content
        feedback, self.full = full[len(self.full) :], full
        return feedback, observation.reward or 0.0, observation.done

    def board(self):
        return self.game.state.game_state["board"]

    def render(self):
        return self.game._get_grid_string_with_indices()


def play(model, tokenizer, env, puzzle, solution, args):
    def with_hints(text):
        hints = single_candidates(env.board()) if args.hints else []
        return text + (
            f"\n\nCells with a single candidate: {', '.join(hints[:5])}"
            if hints
            else ""
        )

    messages = [
        {"role": "user", "content": PROMPT + with_hints(env.reset(puzzle, solution))}
    ]
    empty = sum(row.count(0) for row in puzzle)
    moves, reward, done, last_reply = [], 0.0, False, None
    # The same budget as GRPO (max_tool_calling_iterations=40). A game ends on two mistakes in a row, not on the total.
    while not done and len(moves) < 40:
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
            last_reply = reply  # the model answered without calling the tool
            break
        try:
            arguments = json.loads(call.group(1)).get("arguments", {})
            row, col, number = (int(arguments[k]) for k in ("row", "col", "number"))
        except (ValueError, KeyError, TypeError):
            last_reply = reply  # malformed tool call
            break
        feedback, reward, done = env.step(f"[{row} {col} {number}]")
        valid = "invalid" not in feedback.lower()
        tool_result = (
            with_hints(env.render()) if valid else feedback.split("[GAME]")[-1].strip()
        )
        thinking = THINK.search(reply)
        reasoning = thinking.group(1).strip() if thinking else ""
        moves.append(
            {
                "row": row,
                "col": col,
                "number": number,
                "valid": valid,
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
                        "function": {
                            "name": "place",
                            "arguments": {"row": row, "col": col, "number": number},
                        },
                    }
                ],
            },
            {"role": "tool", "name": "place", "content": tool_result},
        ]
    board = env.board()
    filled = sum(
        board[r][c] == solution[r][c]
        for r in range(9)
        for c in range(9)
        if puzzle[r][c] == 0
    )
    return {
        "puzzle": puzzle,
        "moves": moves,
        "reward": reward,
        "solved": filled == empty,
        "filled": filled,
        "empty": empty,
        "mistakes": sum(not m["valid"] for m in moves),
        "last_reply": last_reply,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--revision",
        help="branch or commit of the model repo, e.g. a training checkpoint",
    )
    parser.add_argument("--puzzles", required=True)
    parser.add_argument("--n", type=int, help="only play the first n puzzles")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--hints", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype
    ).to(device)

    puzzles = json.load(open(args.puzzles))[: args.n]
    env = LocalSudoku()
    with open(args.out, "w") as f:
        for i, p in enumerate(puzzles):
            game = play(model, tokenizer, env, p["puzzle"], p["solution"], args)
            f.write(
                json.dumps({"model": args.model, "hints": args.hints, **game}) + "\n"
            )
            f.flush()
            print(
                f"puzzle {i + 1}: {'solved' if game['solved'] else 'unsolved'}, {game['filled']}/{game['empty']} cells, "
                f"{game['mistakes']} mistakes, {len(game['moves'])} moves"
            )


if __name__ == "__main__":
    main()
