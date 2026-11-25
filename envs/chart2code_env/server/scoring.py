# SPDX-License-Identifier: BSD-3-Clause

"""Compares the figure a submission drew with the target's.

Both figures arrive as summaries from
[`~envs.chart2code_env.server.runner`]: per axes, the series it holds (kind,
label, points, colors), its scales and its words. No pixels are compared, so
pasting the target as an image earns nothing, and matching is F1-style, so
extra bars or labels cost as much as missing ones.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

WEIGHTS = {"structure": 0.2, "data": 0.45, "text": 0.2, "color": 0.15}
# Relative error at or below FULL earns full credit, at or above ZERO none.
FULL, ZERO = 0.02, 0.10
COLOR_FULL, COLOR_ZERO = 0.1, 0.4


def _credit(error: float, full: float = FULL, zero: float = ZERO) -> float:
    if error <= full:
        return 1.0
    if error >= zero:
        return 0.0
    return (zero - error) / (zero - full)


def _f1(target: Counter, submitted: Counter) -> float:
    if not target and not submitted:
        return 1.0
    matched = sum((target & submitted).values())
    if not matched:
        return 0.0
    precision = matched / sum(submitted.values())
    recall = matched / sum(target.values())
    return 2 * precision * recall / (precision + recall)


def _rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _color_credit(target: list[str], submitted: list[str]) -> float:
    if not target:
        return 1.0
    total = 0.0
    for want, got in zip(target, submitted):
        distance = math.dist(_rgb(want), _rgb(got)) / math.sqrt(3)
        total += _credit(distance, COLOR_FULL, COLOR_ZERO)
    return total / len(target)


def _scales(axes: dict[str, Any], kind: str) -> tuple[float, float, bool]:
    """Ranges that point errors are measured against, from the target axes."""
    values = [p[-1] for s in axes["series"] if s["kind"] == kind for p in s["points"]]
    log = (axes["xscale"] if kind == "barh" else axes["yscale"]) == "log"
    if log:
        logs = [math.log10(v) for v in values if v > 0]
        value_scale = max(max(logs) - min(logs), 1.0) if logs else 1.0
    else:
        value_scale = max([abs(v) for v in values] + [1e-9])
    xs = [p[0] for s in axes["series"] if s["kind"] == kind for p in s["points"]]
    x_scale = max(max(xs) - min(xs), 1e-9) if xs else 1.0
    return x_scale, value_scale, log


def _series_credit(target: dict, submitted: dict, scales: tuple) -> float:
    """Mean point credit, over the longer of the two series."""
    x_scale, value_scale, log = scales
    total = 0.0
    for want, got in zip(target["points"], submitted["points"]):
        if target["kind"] == "pie":
            error = abs(want[0] - got[0])
        else:
            a, b = want[-1], got[-1]
            if log:
                a, b = math.log10(max(a, 1e-12)), math.log10(max(b, 1e-12))
            error = abs(a - b) / value_scale
            if target["kind"] == "scatter":
                error = max(error, abs(want[0] - got[0]) / x_scale)
        total += _credit(error)
    return total / max(len(target["points"]), len(submitted["points"]))


def _pair(target_axes: dict, submitted_axes: dict) -> list[tuple[dict, dict, float]]:
    """Greedily match each target series to a submitted one of the same kind."""
    candidates = []
    for t in target_axes["series"]:
        scales = _scales(target_axes, t["kind"])
        for s in submitted_axes["series"]:
            if s["kind"] == t["kind"]:
                candidates.append((_series_credit(t, s, scales), id(t), id(s), t, s))
    pairs, used = [], set()
    for credit, tid, sid, t, s in sorted(candidates, key=lambda c: -c[0]):
        if tid in used or sid in used:
            continue
        used.update((tid, sid))
        pairs.append((t, s, credit))
    return pairs


def _words(figure: dict[str, Any]) -> Counter:
    texts = list(figure["suptitle"])
    for axes in figure["axes"]:
        texts += axes["text"]
    return Counter(" ".join(t.lower().split()) for t in texts)


def _shape(figure: dict[str, Any]) -> Counter:
    shape = Counter()
    for index, axes in enumerate(figure["axes"]):
        shape.update((index, s["kind"]) for s in axes["series"])
        shape.update(
            (index, f"{axis}:log")
            for axis in ("x", "y")
            if axes[f"{axis}scale"] == "log"
        )
    return shape


def score_figure(
    target: dict[str, Any], submitted: dict[str, Any] | None
) -> dict[str, Any]:
    """Score a submitted figure summary against the target's.

    Args:
        target (`dict`):
            Summary of the target figure.
        submitted (`dict`, *optional*):
            Summary of the submission's figure; `None` when the code failed.

    Returns:
        `dict` with keys:
            - `gate_passed` (`bool`): The code drew a figure with data and no
              raster image.
            - `reason` (`str`): Why the gate failed, empty when it passed.
            - `structure`, `data`, `text`, `color` (`float`): Component
              scores in [0, 1], zero when the gate failed.
            - `total` (`float`): The weighted sum, zero when the gate failed.
    """
    zero = {name: 0.0 for name in WEIGHTS}
    if submitted is None:
        return {"gate_passed": False, "reason": "no figure", **zero, "total": 0.0}
    if submitted["raster"]:
        return {"gate_passed": False, "reason": "raster image", **zero, "total": 0.0}
    if not any(axes["series"] for axes in submitted["axes"]):
        return {"gate_passed": False, "reason": "no data drawn", **zero, "total": 0.0}

    data_credit, color_credit = 0.0, 0.0
    n_target = sum(len(axes["series"]) for axes in target["axes"])
    n_submitted = sum(len(axes["series"]) for axes in submitted["axes"])
    for target_axes, submitted_axes in zip(target["axes"], submitted["axes"]):
        for t, s, credit in _pair(target_axes, submitted_axes):
            data_credit += credit
            color_credit += _color_credit(t["colors"], s["colors"])
    scores = {
        "structure": _f1(_shape(target), _shape(submitted)),
        "data": data_credit / max(n_target, n_submitted),
        "text": _f1(_words(target), _words(submitted)),
        "color": color_credit / n_target,
    }
    total = sum(WEIGHTS[name] * value for name, value in scores.items())
    return {"gate_passed": True, "reason": "", **scores, "total": round(total, 6)}
