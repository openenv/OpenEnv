# /// script
# dependencies = [
#     "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env",
# ]
# ///
"""
Write a fixed set of Sudoku puzzles (and their solutions) with TextArena's own generator, so every model plays the same
boards.

```sh
uv run make_puzzles.py --clues 65 --n 30 --out puzzles_c65.json
```
"""

import argparse
import json
import random

import textarena as ta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clues", type=int, default=30)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    game = ta.make("Sudoku-v0", clues=args.clues)
    while hasattr(game, "env"):
        game = game.env
    puzzles = []
    for i in range(args.n):
        # TextArena's Sudoku draws boards from the global `random`, not from reset(seed).
        random.seed(args.seed + i)
        solution, puzzle = game._generate_board()
        puzzles.append({"puzzle": puzzle, "solution": solution})
    with open(args.out, "w") as f:
        json.dump(puzzles, f)
    empty = [sum(row.count(0) for row in p["puzzle"]) for p in puzzles]
    print(
        f"{args.n} puzzles, {min(empty)}-{max(empty)} empty cells (mean {sum(empty) / len(empty):.1f})"
    )


if __name__ == "__main__":
    main()
