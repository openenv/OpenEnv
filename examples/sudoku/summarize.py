"""Print one line per evaluation file: puzzles solved, cells filled, mistakes, how games ended."""

import json
import sys


for path in sys.argv[1:]:
    games = [json.loads(line) for line in open(path)]
    n = len(games)
    solved = sum(g["solved"] for g in games)
    filled = sum(g["filled"] for g in games) / sum(g["empty"] for g in games)
    # A game ends solved, on two mistakes in a row, or because the model stopped making valid tool calls.
    two_mistakes = sum(
        len(g["moves"]) >= 2
        and not g["moves"][-1]["valid"]
        and not g["moves"][-2]["valid"]
        for g in games
    )
    no_call = n - solved - two_mistakes
    first_given = sum(
        bool(g["moves"])
        and 1 <= g["moves"][0]["row"] <= 9
        and 1 <= g["moves"][0]["col"] <= 9
        and g["puzzle"][g["moves"][0]["row"] - 1][g["moves"][0]["col"] - 1] != 0
        for g in games
    )
    print(
        f"{path:32s} solved {solved:2d}/{n}  cells {filled:6.1%}  mistakes/game {sum(g['mistakes'] for g in games) / n:4.1f}  "
        f"two mistakes {two_mistakes:2d}  stopped calling {no_call:2d}  first move on a given cell {first_given:2d}"
    )
