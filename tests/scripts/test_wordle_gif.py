# SPDX-License-Identifier: BSD-3-Clause
"""Render synthetic evaluation traces without inference or a server."""

import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("PIL")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def renderer(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "wordle_gif", ROOT / "examples/wordle/make_gif.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    texts = []
    original = module.ImageDraw.Draw

    def record(image):
        draw = original(image)
        text = draw.text
        labels = []
        texts.append(labels)

        def capture(xy, value, **kwargs):
            labels.append(value)
            return text(xy, value, **kwargs)

        draw.text = capture
        return draw

    monkeypatch.setattr(module.ImageDraw, "Draw", record)
    return module, texts


def game(rows, won=False):
    return {"rows": rows, "won": won, "secret": "apple"}


def row(colors="XXXXX"):
    return {"letters": "APPLE", "colors": colors}


@pytest.mark.parametrize("alternating", [False, True])
def test_late_win_uses_six_legal_rows(renderer, alternating):
    module, texts = renderer
    rows = (
        [item for _ in range(6) for item in (row(None), row())]
        if alternating
        else [row(None)] + [row() for _ in range(6)]
    )
    rows[-1] = row("GGGGG")
    images = module.frames([[game(rows, True)]], ["model"])
    assert "solved" in texts[-1]
    assert "wins 1/1" in texts[-1]
    assert all("wins 0/1" in frame for frame in texts[:-1])
    assert len(images) == len(rows) + 3
    # The winning sixth legal guess is visible, after the rejected submission(s).
    x = module.PAD + 4
    y = 2 * module.PAD + 40 + 5 * (module.TILE + module.GAP) + 4
    assert images[-1].getpixel((x, y)) == module.COLORS["G"]
    assert "invalid: APPLE" in texts[0]


def test_invalid_only_trace_finishes_and_leaves_board_empty(renderer):
    module, texts = renderer
    images = module.frames([[game([row(None)] * 12)]], ["model"])
    assert len(images) == 15
    assert "missed: APPLE" in texts[-1]
    assert "wins 0/1" in texts[-1]
    assert "invalid: APPLE" in texts[-1]
    assert images[-1].getpixel((module.PAD + 4, 2 * module.PAD + 44)) == module.EMPTY


def test_legacy_short_games_and_running_wins(renderer):
    module, texts = renderer
    early = game([row("GGGGG")], True)
    late = game([row()] * 5 + [row("GGGGG")], True)
    module.frames([[early, late], [late, game([row()] * 6)]], ["a", "b"])
    assert texts[0].count("solved") == 1
    assert "wins 1/1" in texts[0] and "wins 0/1" in texts[0]
    assert texts[5].count("wins 1/1") == 2
    assert texts[6].count("wins 1/2") == 2
    assert "wins 2/2" in texts[-1] and "wins 1/2" in texts[-1]
    assert "solved" in texts[-1] and "missed: APPLE" in texts[-1]


def test_empty_trace_has_verdict(renderer):
    module, texts = renderer
    images = module.frames([[game([])]], ["model"])
    assert images
    assert "missed: APPLE" in texts[-1]
