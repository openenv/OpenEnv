# SPDX-License-Identifier: BSD-3-Clause

"""Runs chart code in its own process and reports what the figure contains.

[`~envs.chart2code_env.server.executor.run_code`] starts this file as a
script and sends `{"code", "dpi", "png", "nonce"}` on stdin. It prints one line,
the nonce followed by JSON with the figure's artists, or the error. Scoring
reads that summary, never the code or the pixels, so any library that draws
through matplotlib's artists scores the same.
"""

import base64
import contextlib
import io
import json
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.container import BarContainer  # noqa: E402
from matplotlib.image import (  # noqa: E402
    AxesImage,
    BboxImage,
    FigureImage,
    NonUniformImage,
    PcolorImage,
)
from matplotlib.offsetbox import OffsetImage  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

RASTERS = (AxesImage, BboxImage, FigureImage, NonUniformImage, PcolorImage, OffsetImage)


def is_number(text):
    try:
        float(text.replace("−", "-").replace(",", "").rstrip("%"))
    except ValueError:
        return False
    return True


def words(texts, keep_numbers=False):
    return [
        t.strip()
        for t in texts
        if t.strip() and (keep_numbers or not is_number(t.strip()))
    ]


def summarize_axes(ax):
    series = []
    for container in ax.containers:
        if not isinstance(container, BarContainer) or not container.patches:
            continue
        vertical = container.orientation == "vertical"
        points = []
        for rect in container.patches:
            if vertical:
                points.append(
                    [
                        float(rect.get_x() + rect.get_width() / 2),
                        float(rect.get_height()),
                    ]
                )
            else:
                points.append(
                    [
                        float(rect.get_y() + rect.get_height() / 2),
                        float(rect.get_width()),
                    ]
                )
        series.append(
            {
                "kind": "bar" if vertical else "barh",
                "label": container.get_label() or "",
                "points": sorted(points),
                "colors": [to_hex(container.patches[0].get_facecolor())],
            }
        )
    for line in ax.get_lines():
        xy = line.get_xydata()
        if len(xy) < 2:
            continue
        series.append(
            {
                "kind": "line",
                "label": line.get_label(),
                "points": sorted([float(x), float(y)] for x, y in xy),
                "colors": [to_hex(line.get_color())],
            }
        )
    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue
        offsets = collection.get_offsets()
        if len(offsets) == 0:
            continue
        colors = collection.get_facecolors()
        series.append(
            {
                "kind": "scatter",
                "label": collection.get_label(),
                "points": sorted([float(x), float(y)] for x, y in offsets),
                "colors": [to_hex(colors[0])] if len(colors) else [],
            }
        )
    wedges = [patch for patch in ax.patches if isinstance(patch, Wedge)]
    if wedges:
        series.append(
            {
                "kind": "pie",
                "label": "",
                "points": [[float(w.theta2 - w.theta1) / 360.0] for w in wedges],
                "colors": [to_hex(w.get_facecolor()) for w in wedges],
            }
        )
    legend = ax.get_legend()
    return {
        "series": series,
        "xscale": ax.get_xscale(),
        "yscale": ax.get_yscale(),
        "xlim": [float(v) for v in ax.get_xlim()],
        "ylim": [float(v) for v in ax.get_ylim()],
        # Titles, axis labels and legend entries count even when numeric (a "2024"
        # series); numeric ticks and value labels are data, scored as such.
        "text": words(
            [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
            + [t.get_text() for t in (legend.get_texts() if legend else [])],
            keep_numbers=True,
        )
        + words(
            [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
            + [t.get_text() for t in ax.texts]
        ),
    }


def main():
    request = json.loads(sys.stdin.read())
    if request["memory_limit"] and os.name == "posix":
        # Set here, in the child, rather than with preexec_fn, which is unsafe in a
        # server that runs other threads.
        import resource

        limit = request["memory_limit"]
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    nonce = request["nonce"]
    result = {"error": "", "figure": None, "png": None}
    namespace = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(request["code"], "<chart code>", "exec"), namespace)
        if not plt.get_fignums():
            raise RuntimeError("the code drew no figure")
        fig = plt.gcf()
        fig.canvas.draw()
        result["figure"] = {
            "raster": bool(fig.findobj(lambda artist: isinstance(artist, RASTERS))),
            "axes": [summarize_axes(ax) for ax in fig.axes if ax.get_visible()],
            "suptitle": words([fig.get_suptitle()]),
        }
        if request["png"]:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=request["dpi"])
            result["png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    except BaseException:
        result["error"] = traceback.format_exc(limit=-3)
    sys.__stdout__.write(nonce + json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
