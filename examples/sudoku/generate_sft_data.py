# /// script
# dependencies = [
#     "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env",
#     "transformers",
#     "datasets",
# ]
# ///
"""
Generate Sudoku games for a short SFT warmup before GRPO.

A solver plays real games against the TextArena Sudoku environment (run in-process) and writes each one as a
tool-calling conversation: a short reasoning and a `place` call per move, then the updated grid. The solver only uses
single candidates (the one digit a cell can take, or the one cell a digit can take in a row, column or box), so every
reasoning step can be checked on the grid. The format matches `evaluate.py` (same prompt, tool and grid) and what
`GRPOTrainer` renders with `environment_factory`.

```sh
uv run generate_sft_data.py --games 1000 --clues 50,55,60,65 --push-to-hub <user>/sudoku-tool-calls-sft
```
"""

import argparse
import copy
import json
import random

from datasets import Dataset
from textarena_env import TextArenaAction
from textarena_env.server.environment import TextArenaEnvironment
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


def units(r, c):
    box = [
        (i, j)
        for i in range(r - r % 3, r - r % 3 + 3)
        for j in range(c - c % 3, c - c % 3 + 3)
    ]
    return [(i, c) for i in range(9)], [(r, j) for j in range(9)], box


def candidates(board, r, c):
    return set(range(1, 10)) - {board[i][j] for unit in units(r, c) for i, j in unit}


def digits(board, cells):
    return (
        ", ".join(str(d) for d in sorted({board[i][j] for i, j in cells} - {0}))
        or "nothing"
    )


def next_move(board, rng):
    """A (row, col, number, reasoning) move found with single candidates, or None if there is none."""
    empty = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
    naked = [(r, c) for r, c in empty if len(candidates(board, r, c)) == 1]
    if naked:
        r, c = rng.choice(naked)
        n = candidates(board, r, c).pop()
        column, row, box = units(r, c)
        return (
            r,
            c,
            n,
            (
                f"Cell ({r + 1}, {c + 1}) is empty. Row {r + 1} has {digits(board, row)}, column {c + 1} has "
                f"{digits(board, column)} and its box has {digits(board, box)}. The only digit missing from all three is {n}."
            ),
        )
    hidden = []
    for name, unit_index in (("row", 1), ("column", 0), ("box", 2)):
        for k in range(9):
            r0, c0 = (
                (k, 0)
                if name == "row"
                else (0, k)
                if name == "column"
                else (3 * (k // 3), 3 * (k % 3))
            )
            cells = units(r0, c0)[unit_index]
            for n in set(range(1, 10)) - {board[i][j] for i, j in cells}:
                spots = [
                    (i, j)
                    for i, j in cells
                    if board[i][j] == 0 and n in candidates(board, i, j)
                ]
                if len(spots) == 1:
                    hidden.append((name, k, n, spots[0]))
    if hidden:
        name, k, n, (r, c) = rng.choice(hidden)
        where = f"box {k + 1}" if name == "box" else f"{name} {k + 1}"
        return (
            r,
            c,
            n,
            (
                f"{n} is missing from {where}, and ({r + 1}, {c + 1}) is the only empty cell there that can take it "
                f"without repeating {n} in its row, column or box."
            ),
        )
    return None


def play(env, puzzle, solution, rng):
    messages = [{"role": "user", "content": PROMPT + env.reset(puzzle, solution)}]
    done = False
    while not done:
        move = next_move(env.board(), rng)
        if move is None:
            return None  # needs more than single candidates
        r, c, n, reasoning = move
        call = {"row": r + 1, "col": c + 1, "number": n}
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "place", "arguments": call},
                    }
                ],
            }
        )
        feedback, reward, done = env.step(f"[{r + 1} {c + 1} {n}]")
        assert "invalid" not in feedback.lower(), feedback
        messages.append({"role": "tool", "name": "place", "content": env.render()})
    messages.append(
        {
            "role": "assistant",
            "content": "Solved: every cell is filled.",
            "reasoning_content": "The grid is complete and the game says the puzzle is solved, so there are no moves left.",
        }
    )
    return {"messages": messages, "tools": json.dumps(TOOLS)}, reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument(
        "--clues",
        default="50,55,60,65",
        help="comma-separated clue counts, one is drawn per game",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="sudoku_sft.jsonl")
    parser.add_argument(
        "--push-to-hub",
        help="dataset repo to push to, e.g. <user>/sudoku-tool-calls-sft",
    )
    args = parser.parse_args()

    clues = [int(c) for c in args.clues.split(",")]
    # TextArena's Sudoku draws boards from the global `random`. The evaluation sets use seeds 42-71.
    random.seed(1000 + args.seed)
    rng, env, rows, skipped = random.Random(args.seed), LocalSudoku(), [], 0
    while len(rows) < args.games:
        env.game.clues = rng.choice(clues)
        solution, puzzle = env.game._generate_board()
        result = play(env, puzzle, solution, rng)
        if result is None:
            skipped += 1
            continue
        row, reward = result
        assert reward == 1.0
        rows.append(row)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    print(
        f"wrote {len(rows)} games to {args.out} ({skipped} skipped: not solvable with single candidates)"
    )
    if args.push_to_hub:
        Dataset.from_list(rows).push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
