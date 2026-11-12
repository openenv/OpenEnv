"""Draw a SPaRC board as SVG for the web playground, in the colours of the SPaRC paper's figures.

Grid positions (x, y) follow `puzzle_array`: nodes sit where x and y are even, rule cells where both
are odd, and the lanes between nodes where exactly one is odd.
"""

import math
from html import escape
from typing import Dict, List, Optional, Sequence

BOARD, TRACK, CELL, LINE, INK = "#132e2a", "#2e3d3b", "#00a07a", "#ffffff", "#1f2523"
COLORS = {
    "K": "#1f2523",
    "R": "#c8553d",
    "P": "#8a63c2",
    "W": "#f4f1ea",
    "Y": "#ecc94b",
    "B": "#3f6fa8",
    "G": "#3e8e5e",
    "O": "#e08a3c",
}
U = 32  # pixels per grid position (half a cell)
LANE = 0.42 * U
MARGIN = 1.3 * U
NUB = 0.62 * U


def _xy(x: float, y: float) -> str:
    return f"{MARGIN + x * U:.1f},{MARGIN + y * U:.1f}"


def _symbol(
    cx: float, cy: float, code: str, polyshapes: Dict[str, List[List[int]]]
) -> str:
    parts = code.split("-")
    kind = parts[0]
    color = COLORS.get(parts[1], INK) if len(parts) > 1 else INK
    px, py = MARGIN + cx * U, MARGIN + cy * U
    if kind == "o":
        s = 0.6 * U
        return f'<rect x="{px - s / 2:.1f}" y="{py - s / 2:.1f}" width="{s:.1f}" height="{s:.1f}" rx="{0.14 * U:.1f}" fill="{color}"/>'
    if kind == "*":
        pts = []
        for i in range(16):
            r = (0.36 if i % 2 == 0 else 0.19) * U
            a = math.pi / 8 * i - math.pi / 2
            pts.append(f"{px + r * math.cos(a):.1f},{py + r * math.sin(a):.1f}")
        return f'<polygon points="{" ".join(pts)}" fill="{color}"/>'
    if kind in "ABCD" and len(kind) == 1:
        n = "ABCD".index(kind) + 1
        w, g = 0.34 * U, 0.076 * U
        x0 = px - (n * w + (n - 1) * g) / 2
        out = []
        for i in range(n):
            tx = x0 + i * (w + g)
            out.append(
                f'<polygon points="{tx:.1f},{py + 0.15 * U:.1f} {tx + w / 2:.1f},{py - 0.17 * U:.1f} {tx + w:.1f},{py + 0.15 * U:.1f}" fill="{color}"/>'
            )
        return "".join(out)
    if kind in ("P", "Y") and len(parts) > 2 and parts[2] in polyshapes:
        grid = polyshapes[parts[2]]
        filled = [(i, j) for j, row in enumerate(grid) for i, v in enumerate(row) if v]
        if not filled:
            return ""
        mi, ma = min(i for i, _ in filled), max(i for i, _ in filled)
        mj, mb = min(j for _, j in filled), max(j for _, j in filled)
        s, g = 0.21 * U, 0.048 * U
        bw, bh = (ma - mi + 1) * s + (ma - mi) * g, (mb - mj + 1) * s + (mb - mj) * g
        out = []
        for i, j in filled:
            rx, ry = px - bw / 2 + (i - mi) * (s + g), py - bh / 2 + (j - mj) * (s + g)
            if kind == "P":
                out.append(
                    f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{s:.1f}" height="{s:.1f}" fill="{color}"/>'
                )
            else:
                out.append(
                    f'<rect x="{rx + 1:.1f}" y="{ry + 1:.1f}" width="{s - 2:.1f}" height="{s - 2:.1f}" fill="none" stroke="{color}" stroke-width="1.6"/>'
                )
        return "".join(out)
    return ""


def board_svg(
    board: Sequence[Sequence[str]],
    path: Sequence[Sequence[int]],
    polyshapes: Optional[Dict[str, List[List[int]]]] = None,
    label: str = "SPaRC puzzle",
) -> str:
    """SVG of the puzzle in `board` (the original puzzle_array) with the line `path` drawn on it."""
    polyshapes = polyshapes or {}
    h, w = len(board), len(board[0])
    marks = {board[y][x]: (x, y) for y in range(h) for x in range(w)}
    start, end = marks.get("S"), marks.get("E")
    width, height = 2 * MARGIN + (w - 1) * U, 2 * MARGIN + (h - 1) * U
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" height="{height:.0f}" role="img" aria-label="{escape(label)}">',
        f'<rect width="{width:.0f}" height="{height:.0f}" rx="{0.6 * U:.0f}" fill="{BOARD}"/>',
    ]
    lanes, gaps = [], []
    for y in range(h):
        for x in range(w):
            if x % 2 == y % 2:
                continue
            a, b = ((x - 1, y), (x + 1, y)) if x % 2 else ((x, y - 1), (x, y + 1))
            lanes.append(f"M{_xy(*a)}L{_xy(*b)}")
            if board[y][x] == "G":
                # A gap is a cut in the lane: board colour over its middle.
                along, across = 0.5 * U, LANE + 2
                gw, gh = (along, across) if x % 2 else (across, along)
                gaps.append(
                    f'<rect x="{MARGIN + x * U - gw / 2:.1f}" y="{MARGIN + y * U - gh / 2:.1f}" width="{gw:.1f}" height="{gh:.1f}" fill="{BOARD}"/>'
                )
    if end is not None:
        ex, ey = end
        dx, dy = (
            (-1, 0)
            if ex == 0
            else (1, 0)
            if ex == w - 1
            else (0, -1)
            if ey == 0
            else (0, 1)
        )
        nub = f"{MARGIN + ex * U + dx * NUB:.1f},{MARGIN + ey * U + dy * NUB:.1f}"
        lanes.append(f"M{_xy(ex, ey)}L{nub}")
    out.append(
        f'<path d="{"".join(lanes)}" stroke="{TRACK}" stroke-width="{LANE:.1f}" stroke-linecap="round" fill="none"/>'
    )
    out.extend(gaps)
    for y in range(1, h, 2):
        for x in range(1, w, 2):
            out.append(
                f'<rect x="{MARGIN + (x - 0.7) * U:.1f}" y="{MARGIN + (y - 0.7) * U:.1f}" width="{1.4 * U:.1f}" height="{1.4 * U:.1f}" rx="{0.08 * U:.1f}" fill="{CELL}"/>'
            )
            out.append(_symbol(x, y, board[y][x], polyshapes))
    if len(path) > 1:
        pts = "L".join(_xy(px, py) for px, py in path)
        out.append(
            f'<path d="M{pts}" stroke="{LINE}" stroke-width="{LANE:.1f}" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
        )
    if start is not None:
        cx, cy = start
        out.append(
            f'<circle cx="{MARGIN + cx * U:.1f}" cy="{MARGIN + cy * U:.1f}" r="{0.55 * U:.1f}" fill="{LINE if path else TRACK}"/>'
        )
    for y in range(h):
        for x in range(w):
            if board[y][x] == ".":
                cx, cy, r = MARGIN + x * U, MARGIN + y * U, 0.18 * U
                pts = " ".join(
                    f"{cx + r * math.cos(math.pi / 3 * i + math.pi / 6):.1f},{cy + r * math.sin(math.pi / 3 * i + math.pi / 6):.1f}"
                    for i in range(6)
                )
                out.append(f'<polygon points="{pts}" fill="#111111"/>')
    out.append("</svg>")
    return "".join(out)
