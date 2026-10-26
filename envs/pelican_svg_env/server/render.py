# SPDX-License-Identifier: BSD-3-Clause

"""Rasterise SVG source and measure how much of the canvas got drawn on.

Rendering uses ``resvg`` rather than a Cairo binding: it ships self-contained
wheels on every platform the env targets, so the container needs no system
libraries, it is deterministic across runs, and it refuses to resolve external
entities or fetch remote hrefs.

The one measurement here answers one question, "did the model put a drawing on
the canvas at all?", and is deliberately blind to what the drawing depicts.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import resvg_py
from PIL import Image

# Rendering happens at a fixed size so scores are comparable across submissions
# regardless of the viewBox the model chose.
DEFAULT_RENDER_SIZE = 512

# Euclidean RGB distance from the modal (background) colour above which a pixel
# counts as ink. Anti-aliased edges of a thin stroke sit well above this.
INK_DISTANCE_THRESHOLD = 32.0


class RenderError(Exception):
    """Raised when the SVG source cannot be rasterised."""


@dataclass(frozen=True)
class ImageStats:
    """Measurements taken from a rasterised submission.

    Attributes:
        ink_fraction (`float`):
            Share of pixels differing from the background colour.
    """

    ink_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the stats."""
        return asdict(self)


def render_png(svg_source: str, size: int = DEFAULT_RENDER_SIZE) -> bytes:
    """Rasterise SVG source to PNG bytes.

    Args:
        svg_source (`str`):
            The SVG document to render.
        size (`int`, *optional*, defaults to `512`):
            Width and height of the output raster in pixels.

    Returns:
        `bytes`: PNG-encoded image data.

    Raises:
        RenderError: If the source is not renderable SVG.

    Examples:

    ```python
    png = render_png('<svg xmlns="http://www.w3.org/2000/svg"/>')
    ```
    """
    try:
        raw = resvg_py.svg_to_bytes(svg_string=svg_source, width=size, height=size)
    except Exception as exc:  # resvg raises bare ValueError with a parse message
        raise RenderError(str(exc)) from exc
    return bytes(raw)


def _flatten_on_white(png_bytes: bytes) -> np.ndarray:
    """Composite a possibly-transparent PNG onto white and return an RGB array."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.alpha_composite(img)
    # float32, not an integer dtype: squared channel differences reach 65025,
    # which overflows int16 and turns black-on-white pixels into NaN distances
    # that then fail every comparison.
    return np.asarray(canvas.convert("RGB"), dtype=np.float32)


def _modal_color(rgb: np.ndarray) -> np.ndarray:
    """Return the most common colour after coarse quantisation.

    Using the modal colour rather than transparency means a submission that
    paints a full-canvas background rect is measured the same way as one that
    leaves the canvas transparent.
    """
    bins = (rgb.astype(np.uint16) >> 4).astype(np.uint32)
    packed = (bins[..., 0] << 16) | (bins[..., 1] << 8) | bins[..., 2]
    values, counts = np.unique(packed, return_counts=True)
    mask = packed == values[int(np.argmax(counts))]
    return rgb[mask].mean(axis=0).astype(np.float32)


def image_stats(png_bytes: bytes) -> ImageStats:
    """Measure how much of a rasterised submission differs from its background.

    Args:
        png_bytes (`bytes`):
            PNG data as produced by [`render_png`].

    Returns:
        [`ImageStats`]: The measurement taken from the raster.

    Examples:

    ```python
    stats = image_stats(render_png(svg_source))
    print(stats.ink_fraction)
    ```
    """
    rgb = _flatten_on_white(png_bytes)
    distance = np.sqrt(((rgb - _modal_color(rgb)) ** 2).sum(axis=-1))
    ink = distance > INK_DISTANCE_THRESHOLD
    return ImageStats(ink_fraction=float(ink.sum() / ink.size))
