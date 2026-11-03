# /// script
# dependencies = ["pillow"]
# ///
"""
Turn games saved by `evaluate.py` into a side-by-side GIF, one column per model.

Given digits are gray, digits the model placed are green, and a rejected move shows its digit in a red cell.

```sh
uv run make_gif.py base.jsonl sft.jsonl grpo.jsonl --labels "Qwen3-1.7B" "+ SFT" "+ SFT + GRPO" --games 5 --out progress.gif
```
"""

import argparse
import json

from PIL import Image, ImageDraw, ImageFont


CELL, PAD = 34, 24
GIVEN, PLACED, PLACED_BG, WRONG_BG = (
    (60, 60, 60),
    (46, 125, 50),
    (220, 237, 200),
    (244, 199, 195),
)
LINE, BG = (40, 40, 40), (250, 250, 250)


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def grid(draw, x0, y0, game, shown):
    digits = font(20)
    placed = {
        (m["row"] - 1, m["col"] - 1): m["number"]
        for m in game["moves"][:shown]
        if m["valid"]
    }
    last = game["moves"][shown - 1] if shown else None
    wrong = (last["row"] - 1, last["col"] - 1) if last and not last["valid"] else None
    for r in range(9):
        for c in range(9):
            x, y = x0 + c * CELL, y0 + r * CELL
            given = game["puzzle"][r][c]
            fill = (
                WRONG_BG
                if (r, c) == wrong
                else PLACED_BG
                if (r, c) in placed
                else (255, 255, 255)
            )
            draw.rectangle(
                [x, y, x + CELL, y + CELL], fill=fill, outline=(200, 200, 200)
            )
            if (r, c) == wrong:
                text, color = str(last["number"]), (180, 40, 40)
            elif given:
                text, color = str(given), GIVEN
            elif (r, c) in placed:
                text, color = str(placed[(r, c)]), PLACED
            else:
                continue
            draw.text(
                (x + CELL / 2, y + CELL / 2), text, font=digits, anchor="mm", fill=color
            )
    for k in range(0, 10, 3):
        draw.line([x0 + k * CELL, y0, x0 + k * CELL, y0 + 9 * CELL], fill=LINE, width=3)
        draw.line([x0, y0 + k * CELL, x0 + 9 * CELL, y0 + k * CELL], fill=LINE, width=3)


def frames(columns, labels):
    size = 9 * CELL
    W, H = PAD + len(columns) * (size + PAD), 3 * PAD + 40 + size + 50
    out = []
    for g in range(min(len(c) for c in columns)):
        solved = [sum(x["solved"] for x in column[:g]) for column in columns]
        moves = max(1, *(len(column[g]["moves"]) for column in columns))
        for shown in range(moves + 1):
            img = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(img)
            for i, (column, label) in enumerate(zip(columns, labels)):
                game, x0 = column[g], PAD + i * (size + PAD)
                done = shown >= len(game["moves"])
                draw.text(
                    (x0 + size / 2, PAD),
                    label,
                    font=font(20),
                    anchor="mt",
                    fill=(30, 30, 30),
                )
                draw.text(
                    (x0 + size / 2, PAD + 26),
                    f"solved {solved[i] + (done and game['solved'])}/{g + 1}",
                    font=font(15),
                    anchor="mt",
                    fill=(90, 90, 90),
                )
                grid(draw, x0, 2 * PAD + 40, game, min(shown, len(game["moves"])))
                if done:
                    verdict = (
                        "solved"
                        if game["solved"]
                        else f"stopped at {game['filled']}/{game['empty']} cells"
                    )
                    draw.text(
                        (x0 + size / 2, H - PAD - 6),
                        verdict,
                        font=font(18),
                        anchor="mm",
                        fill=PLACED if game["solved"] else (180, 60, 60),
                    )
            out.append(img)
        out += [out[-1]] * 4  # hold the final board
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "games", nargs="+", help="jsonl files written by evaluate.py, one per model"
    )
    parser.add_argument("--labels", nargs="+")
    parser.add_argument(
        "--games", type=int, dest="n", help="only show the first n games"
    )
    parser.add_argument("--out", default="progress.gif")
    args = parser.parse_args()

    columns = []
    for path in args.games:
        with open(path) as games_file:
            columns.append([json.loads(line) for line in games_file][: args.n])
    labels = args.labels or [c[0]["model"].split("/")[-1] for c in columns]
    imgs = frames(columns, labels)
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:], duration=300, loop=0)
    print(f"wrote {args.out} ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
