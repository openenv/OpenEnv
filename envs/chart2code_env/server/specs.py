# SPDX-License-Identifier: BSD-3-Clause

"""Seeded chart tasks and the matplotlib code that draws them.

A task is a [`ChartSpec`] sampled from a seed. The target image is whatever
[`reference_code`] draws for it, so the spec, the target and the reference
solution can never disagree.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

LEVELS = ("easy", "medium", "hard")
SPLITS = ("train", "test")
# Test tasks use seeds from this offset up, so a train seed never serves one.
TEST_SEED_OFFSET = 1_000_000_000
TEST_SIZE = 500

PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
CATEGORIES = (
    ("North", "South", "East", "West", "Central", "Coast", "Valley"),
    ("Apples", "Pears", "Plums", "Grapes", "Cherries", "Figs", "Limes"),
    ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
    ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
    ("Alpha", "Beta", "Gamma", "Delta", "Omega", "Sigma", "Kappa"),
)
SUBJECTS = ("Sales", "Rainfall", "Visitors", "Revenue", "Orders", "Downloads")
SERIES = ("2023", "2024", "2025", "Online", "Retail", "Target")
UNITS = ("units", "mm", "people", "USD", "count", "k")
FIGSIZES = ((6.0, 4.0), (7.0, 4.5), (8.0, 5.0))
DPI = 100


@dataclass
class Series:
    """One data series of a chart."""

    label: str
    values: list[float]
    color: str
    x: list[float] = field(default_factory=list)


@dataclass
class Panel:
    """One axes: a chart type and its data."""

    kind: str
    title: str
    xlabel: str
    ylabel: str
    categories: list[str]
    series: list[Series]
    value_labels: bool = False
    log_y: bool = False


@dataclass
class ChartSpec:
    """A chart to reproduce, fully determined by its level, split and seed."""

    task_id: str
    level: str
    figsize: tuple[float, float]
    panels: list[Panel]
    dpi: int = DPI


def _values(rng: random.Random, n: int, log: bool = False) -> list[float]:
    if log:
        return [float(round(10 ** rng.uniform(0, 4))) for _ in range(n)]
    if rng.random() < 0.5:
        return [float(rng.randint(5, 95)) for _ in range(n)]
    return [round(rng.uniform(1, 50), 1) for _ in range(n)]


def _panel(
    rng: random.Random,
    kind: str,
    n_series: int,
    value_labels: bool,
    log_y: bool = False,
) -> Panel:
    subject = rng.choice(SUBJECTS)
    unit = rng.choice(UNITS)
    colors = rng.sample(PALETTE, n_series if kind != "pie" else 0)
    categories = list(rng.choice(CATEGORIES)[: rng.randint(4, 6)])
    labels = rng.sample(SERIES, n_series)
    series = []
    for index in range(n_series):
        if kind == "scatter":
            n = rng.randint(8, 14)
            series.append(
                Series(
                    label=labels[index],
                    values=[round(rng.uniform(0, 100), 1) for _ in range(n)],
                    x=[round(rng.uniform(0, 10), 1) for _ in range(n)],
                    color=colors[index],
                )
            )
        else:
            series.append(
                Series(
                    label=labels[index],
                    values=_values(rng, len(categories), log=log_y),
                    color=colors[index] if colors else "",
                )
            )
    if kind == "pie":
        # A pie's colors belong to its wedges, one per category.
        series[0].color = ",".join(rng.sample(PALETTE, len(categories)))
    return Panel(
        kind=kind,
        title=f"{subject} by {rng.choice(('region', 'month', 'product', 'group'))}",
        xlabel="" if kind == "pie" else rng.choice(("Category", "Group", "Item")),
        ylabel="" if kind == "pie" else f"{subject} ({unit})",
        categories=categories,
        series=series,
        value_labels=value_labels,
        log_y=log_y,
    )


def task_seed(split: str, index: int) -> int:
    """The seed of the `index`-th task of a split."""
    if split == "test":
        if not 0 <= index < TEST_SIZE:
            raise ValueError(f"test has {TEST_SIZE} tasks per level, not index {index}")
        return TEST_SEED_OFFSET + index
    if not 0 <= index < TEST_SEED_OFFSET:
        raise ValueError(f"train seeds are in [0, {TEST_SEED_OFFSET}), not {index}")
    return index


def sample_spec(level: str, split: str, index: int) -> ChartSpec:
    """Build the task with this level, split and index.

    Args:
        level (`str`):
            `"easy"`: one series with its values drawn on the chart.
            `"medium"`: two or three series with a legend and no value labels.
            `"hard"`: two side-by-side charts of different types, one with a
            log axis.
        split (`str`):
            `"train"` or `"test"`. Their seeds never overlap.
        index (`int`):
            Task number within the split; the train split takes any seed.

    Returns:
        [`ChartSpec`]: The sampled task.

    Examples:

    ```python
    spec = sample_spec("easy", "test", 0)
    print(reference_code(spec))
    ```
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, not {level!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    seed = task_seed(split, index)
    rng = random.Random(f"{level}:{seed}")
    if level == "easy":
        panels = [_panel(rng, rng.choice(("bar", "barh", "line", "pie")), 1, True)]
    elif level == "medium":
        kind = rng.choice(("grouped_bar", "line", "scatter"))
        panels = [_panel(rng, kind, rng.randint(2, 3), False)]
    else:
        first, second = rng.sample(("grouped_bar", "line", "scatter", "bar"), 2)
        # Every hard task has one log y-axis, never on the scatter panel.
        log_panel = rng.choice(
            [i for i, k in enumerate((first, second)) if k != "scatter"]
        )
        panels = [
            _panel(rng, kind, 1 if kind == "bar" else 2, False, log_y=i == log_panel)
            for i, kind in enumerate((first, second))
        ]
    return ChartSpec(
        task_id=f"{level}-{split}-{index}",
        level=level,
        figsize=FIGSIZES[1] if level == "hard" else rng.choice(FIGSIZES),
        panels=panels,
    )


def _panel_code(panel: Panel, ax: str) -> list[str]:
    lines = []
    categories = panel.categories
    if panel.kind == "bar":
        s = panel.series[0]
        lines.append(
            f"bars = {ax}.bar({categories!r}, {s.values!r}, color={s.color!r}, label={s.label!r})"
        )
        if panel.value_labels:
            lines.append(f"{ax}.bar_label(bars)")
    elif panel.kind == "barh":
        s = panel.series[0]
        lines.append(
            f"bars = {ax}.barh({categories!r}, {s.values!r}, color={s.color!r}, label={s.label!r})"
        )
        if panel.value_labels:
            lines.append(f"{ax}.bar_label(bars)")
    elif panel.kind == "grouped_bar":
        width = round(0.8 / len(panel.series), 4)
        lines.append(f"x = list(range({len(categories)}))")
        for i, s in enumerate(panel.series):
            offset = round((i - (len(panel.series) - 1) / 2) * width, 4)
            lines.append(
                f"{ax}.bar([p + {offset} for p in x], {s.values!r}, width={width}, "
                f"color={s.color!r}, label={s.label!r})"
            )
        lines.append(f"{ax}.set_xticks(x, {categories!r})")
    elif panel.kind == "line":
        for s in panel.series:
            lines.append(
                f"{ax}.plot({categories!r}, {s.values!r}, marker='o', color={s.color!r}, label={s.label!r})"
            )
            if panel.value_labels:
                lines.append(
                    f"for c, v in zip({categories!r}, {s.values!r}):\n"
                    f"    {ax}.annotate(str(v), (c, v), textcoords='offset points', "
                    "xytext=(0, 6), ha='center')"
                )
    elif panel.kind == "scatter":
        for s in panel.series:
            lines.append(
                f"{ax}.scatter({s.x!r}, {s.values!r}, color={s.color!r}, label={s.label!r})"
            )
    elif panel.kind == "pie":
        s = panel.series[0]
        lines.append(
            f"{ax}.pie({s.values!r}, labels={categories!r}, colors={s.color.split(',')!r}, "
            "autopct='%1.1f%%', startangle=90)"
        )
    if panel.log_y:
        lines.append(f"{ax}.set_yscale('log')")
    lines.append(f"{ax}.set_title({panel.title!r})")
    if panel.xlabel:
        xlabel, ylabel = panel.xlabel, panel.ylabel
        if panel.kind == "barh":
            xlabel, ylabel = ylabel, xlabel
        lines.append(f"{ax}.set_xlabel({xlabel!r})")
        lines.append(f"{ax}.set_ylabel({ylabel!r})")
    if len(panel.series) > 1:
        lines.append(f"{ax}.legend()")
    return lines


def reference_code(spec: ChartSpec) -> str:
    """matplotlib code that draws the target of `spec`.

    The environment draws every target by running this code, so it scores 1.0
    by construction and doubles as the task's reference solution.
    """
    width, height = spec.figsize
    lines = ["import matplotlib.pyplot as plt", ""]
    if len(spec.panels) == 1:
        lines.append(f"fig, ax = plt.subplots(figsize=({width}, {height}))")
        lines += _panel_code(spec.panels[0], "ax")
    else:
        lines.append(
            f"fig, axes = plt.subplots(1, {len(spec.panels)}, figsize=({width}, {height}))"
        )
        for i, panel in enumerate(spec.panels):
            lines += _panel_code(panel, f"axes[{i}]")
    lines.append("fig.tight_layout()")
    return "\n".join(lines) + "\n"
