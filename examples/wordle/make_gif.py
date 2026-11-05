# /// script
# dependencies = ["pillow"]
# ///
"""
Turn games saved by `evaluate.py` into a side-by-side GIF, one column per model.

```sh
uv run make_gif.py base.jsonl sft.jsonl grpo.jsonl --labels "Qwen3-1.7B" "+ SFT" "+ SFT + GRPO" --out progress.gif
```
"""

import argparse
import json

from PIL import Image, ImageDraw, ImageFont


TILE, GAP, PAD = 52, 6, 24
COLORS = {"G": (106, 170, 100), "Y": (201, 180, 88), "X": (120, 124, 126)}
EMPTY, BORDER, TEXT, BG = (
    (255, 255, 255),
    (211, 214, 218),
    (255, 255, 255),
    (250, 250, 250),
)


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def board(draw, x0, y0, game, rows_shown):
    big = font(28)
    # Rejected submissions have no colors and do not consume a Wordle row.
    rows = [row for row in game["rows"][:rows_shown] if row["colors"]]
    for r in range(6):
        row = rows[r] if r < len(rows) else None
        for c in range(5):
            x, y = x0 + c * (TILE + GAP), y0 + r * (TILE + GAP)
            colors = row and row["colors"]
            fill = COLORS[colors[c]] if colors else EMPTY
            draw.rectangle(
                [x, y, x + TILE, y + TILE],
                fill=fill,
                outline=None if colors else BORDER,
                width=2,
            )
            if row and c < len(row["letters"]):
                draw.text(
                    (x + TILE / 2, y + TILE / 2),
                    row["letters"][c],
                    font=big,
                    anchor="mm",
                    fill=TEXT if colors else (60, 60, 60),
                )


def frames(columns, labels):
    width = 5 * TILE + 4 * GAP
    height = 6 * TILE + 5 * GAP
    W = PAD + len(columns) * (width + PAD)
    H = 3 * PAD + 40 + height + 70
    games = min(len(c) for c in columns)
    out = []
    for g in range(games):
        wins = [sum(x["won"] for x in column[:g]) for column in columns]
        # One frame per submission, including invalids; preserve legacy pacing.
        submissions = max(6, *(len(column[g]["rows"]) for column in columns))
        for shown in range(1, submissions + 1):
            img = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(img)
            for i, (column, label) in enumerate(zip(columns, labels)):
                game, x0 = column[g], PAD + i * (width + PAD)
                done = shown >= len(game["rows"])
                won_so_far = wins[i] + (done and game["won"])
                draw.text(
                    (x0 + width / 2, PAD),
                    label,
                    font=font(20),
                    anchor="mt",
                    fill=(30, 30, 30),
                )
                draw.text(
                    (x0 + width / 2, PAD + 26),
                    f"wins {won_so_far}/{g + 1}",
                    font=font(15),
                    anchor="mt",
                    fill=(90, 90, 90),
                )
                board(draw, x0, 2 * PAD + 40, game, shown)
                visible = game["rows"][:shown]
                if visible and not visible[-1]["colors"]:
                    # Keep rejected input out of the board and verdict area.
                    letters = visible[-1]["letters"]
                    letters = letters[:13] + "..." if len(letters) > 16 else letters
                    draw.text(
                        (x0 + width / 2, 2 * PAD + 40 + height + 14),
                        f"invalid: {letters}",
                        font=font(15),
                        anchor="mm",
                        fill=(180, 60, 60),
                    )
                if done:
                    verdict = (
                        "solved"
                        if game["won"]
                        else f"missed: {(game['secret'] or '?').upper()}"
                    )
                    draw.text(
                        (x0 + width / 2, H - PAD - 10),
                        verdict,
                        font=font(18),
                        anchor="mm",
                        fill=COLORS["G"] if game["won"] else (180, 60, 60),
                    )
            out.append(img)
        out += [out[-1]] * 3  # hold the final board
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "games", nargs="+", help="jsonl files written by wordle_eval.py, one per model"
    )
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out", default="progress.gif")
    args = parser.parse_args()

    columns = []
    for path in args.games:
        with open(path) as games_file:
            columns.append([json.loads(line) for line in games_file])
    labels = args.labels or [c[0]["model"].split("/")[-1] for c in columns]
    imgs = frames(columns, labels)
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:], duration=450, loop=0)
    print(f"wrote {args.out} ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
