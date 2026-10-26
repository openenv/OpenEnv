# SPDX-License-Identifier: BSD-3-Clause

"""Flatten an SVG document into measurable shapes.

Shapes become polylines in viewBox coordinates with transforms applied, then
normalised by the viewBox extent so every downstream threshold is scale-free.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Sequence

from .svg_source import strip_namespace

# Points used to flatten one bezier segment or one elliptical arc.
_CURVE_SAMPLES = 16

# Shapes whose outline is shorter than this fraction of the canvas diagonal are
# detail work (an eye, a rivet) and are ignored by the structural detectors.
MIN_SHAPE_EXTENT = 0.04

_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_TRANSFORM = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")

Point = tuple[float, float]
# Affine transform as (a, b, c, d, e, f), mapping (x, y) to
# (a*x + c*y + e, b*x + d*y + f), matching the SVG matrix() argument order.
Matrix = tuple[float, float, float, float, float, float]

IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _numbers(text: str | None) -> list[float]:
    return [float(m) for m in _NUMBER.findall(text or "")]


def multiply(outer: Matrix, inner: Matrix) -> Matrix:
    """Compose two affine transforms, applying `inner` first."""
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def apply(matrix: Matrix, point: Point) -> Point:
    """Map a point through an affine transform."""
    a, b, c, d, e, f = matrix
    x, y = point
    return (a * x + c * y + e, b * x + d * y + f)


def parse_transform(value: str | None) -> Matrix:
    """Parse an SVG ``transform`` attribute into a single affine matrix.

    Args:
        value (`str` or `None`):
            The attribute value, for example `"translate(10 20) rotate(45)"`.

    Returns:
        `Matrix`: The composed transform, identity when `value` is empty.
    """
    result = IDENTITY
    if not value:
        return result
    for name, raw_args in _TRANSFORM.findall(value):
        args = _numbers(raw_args)
        if name == "matrix" and len(args) >= 6:
            step: Matrix = tuple(args[:6])  # type: ignore[assignment]
        elif name == "translate" and args:
            step = (1.0, 0.0, 0.0, 1.0, args[0], args[1] if len(args) > 1 else 0.0)
        elif name == "scale" and args:
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            step = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate" and args:
            angle = math.radians(args[0])
            cos, sin = math.cos(angle), math.sin(angle)
            step = (cos, sin, -sin, cos, 0.0, 0.0)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                step = multiply(
                    (1.0, 0.0, 0.0, 1.0, cx, cy),
                    multiply(step, (1.0, 0.0, 0.0, 1.0, -cx, -cy)),
                )
        elif name == "skewX" and args:
            step = (1.0, 0.0, math.tan(math.radians(args[0])), 1.0, 0.0, 0.0)
        elif name == "skewY" and args:
            step = (1.0, math.tan(math.radians(args[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            continue
        result = multiply(result, step)
    return result


def _sample_cubic(p0: Point, p1: Point, p2: Point, p3: Point) -> list[Point]:
    pts = []
    for i in range(1, _CURVE_SAMPLES + 1):
        t = i / _CURVE_SAMPLES
        u = 1.0 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _sample_quadratic(p0: Point, p1: Point, p2: Point) -> list[Point]:
    pts = []
    for i in range(1, _CURVE_SAMPLES + 1):
        t = i / _CURVE_SAMPLES
        u = 1.0 - t
        x = u**2 * p0[0] + 2 * u * t * p1[0] + t**2 * p2[0]
        y = u**2 * p0[1] + 2 * u * t * p1[1] + t**2 * p2[1]
        pts.append((x, y))
    return pts


def _sample_arc(
    start: Point,
    rx: float,
    ry: float,
    rotation: float,
    large_arc: bool,
    sweep: bool,
    end: Point,
) -> list[Point]:
    """Flatten an elliptical arc using the endpoint-to-centre conversion."""
    if rx == 0 or ry == 0 or start == end:
        return [end]
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(rotation)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    dx2 = (start[0] - end[0]) / 2.0
    dy2 = (start[1] - end[1]) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    # Scale radii up if they are too small to span the endpoints.
    lam = (x1p**2) / (rx**2) + (y1p**2) / (ry**2)
    if lam > 1:
        scale = math.sqrt(lam)
        rx, ry = rx * scale, ry * scale

    denom = rx**2 * y1p**2 + ry**2 * x1p**2
    if denom == 0:
        return [end]
    numer = max(0.0, rx**2 * ry**2 - denom)
    coef = math.sqrt(numer / denom)
    if large_arc == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx

    cx = cos_phi * cxp - sin_phi * cyp + (start[0] + end[0]) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (start[1] + end[1]) / 2.0

    def angle_of(ux: float, uy: float) -> float:
        return math.atan2(uy, ux)

    theta1 = angle_of((x1p - cxp) / rx, (y1p - cyp) / ry)
    theta2 = angle_of((-x1p - cxp) / rx, (-y1p - cyp) / ry)
    delta = theta2 - theta1
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    pts = []
    for i in range(1, _CURVE_SAMPLES + 1):
        theta = theta1 + delta * (i / _CURVE_SAMPLES)
        x = cos_phi * rx * math.cos(theta) - sin_phi * ry * math.sin(theta) + cx
        y = sin_phi * rx * math.cos(theta) + cos_phi * ry * math.sin(theta) + cy
        pts.append((x, y))
    return pts


def _tokenize_path(d: str) -> list[str | float]:
    """Split path data into command letters and numbers.

    Needs command context rather than one regex, because arc flags are single
    digits that may run straight into the next number: in ``A 25 25 0 0175 50``
    the spec reads ``0``, ``1``, ``75``. Positions three and four of an arc's
    argument group therefore consume exactly one character.
    """
    tokens: list[str | float] = []
    command = ""
    arg_index = 0
    i = 0
    while i < len(d):
        char = d[i]
        if char in " \t\n\r,":
            i += 1
        elif char.isalpha():
            command = char
            arg_index = 0
            tokens.append(char)
            i += 1
        elif command in "Aa" and arg_index % 7 in (3, 4):
            if char not in "01":
                break
            tokens.append(float(char))
            arg_index += 1
            i += 1
        else:
            match = _NUMBER.match(d, i)
            if match is None:
                break
            tokens.append(float(match.group(0)))
            arg_index += 1
            i = match.end()
    return tokens


def flatten_path(d: str) -> list[tuple[list[Point], bool]]:
    """Flatten a path ``d`` attribute into subpaths of straight segments.

    Args:
        d (`str`):
            The path data.

    Returns:
        `list[tuple[list[Point], bool]]`: One entry per subpath, holding its
            points and whether the subpath was explicitly closed with `Z`.
    """
    tokens = _tokenize_path(d or "")

    subpaths: list[tuple[list[Point], bool]] = []
    current: list[Point] = []
    cursor: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    last_cubic: Point | None = None
    last_quad: Point | None = None
    command = ""
    index = 0

    def flush(closed: bool) -> None:
        nonlocal current
        if len(current) >= 2:
            subpaths.append((current, closed))
        current = []

    while index < len(tokens):
        token = tokens[index]
        if isinstance(token, str):
            command = token
            index += 1
            if command in "Zz":
                if current:
                    current.append(start)
                flush(True)
                cursor = start
                continue
        elif not command:
            index += 1
            continue

        def take(count: int) -> list[float] | None:
            nonlocal index
            if index + count > len(tokens):
                return None
            chunk = tokens[index : index + count]
            if any(isinstance(v, str) for v in chunk):
                return None
            index += count
            return [float(v) for v in chunk]  # type: ignore[arg-type]

        relative = command.islower()
        upper = command.upper()

        if upper == "M":
            args = take(2)
            if args is None:
                break
            point = (
                cursor[0] + args[0] if relative else args[0],
                cursor[1] + args[1] if relative else args[1],
            )
            flush(False)
            cursor = start = point
            current = [point]
            # Subsequent coordinate pairs after a moveto are implicit linetos.
            command = "l" if relative else "L"
        elif upper == "L":
            args = take(2)
            if args is None:
                break
            cursor = (
                cursor[0] + args[0] if relative else args[0],
                cursor[1] + args[1] if relative else args[1],
            )
            current.append(cursor)
        elif upper == "H":
            args = take(1)
            if args is None:
                break
            cursor = (cursor[0] + args[0] if relative else args[0], cursor[1])
            current.append(cursor)
        elif upper == "V":
            args = take(1)
            if args is None:
                break
            cursor = (cursor[0], cursor[1] + args[0] if relative else args[0])
            current.append(cursor)
        elif upper in {"C", "S"}:
            args = take(6 if upper == "C" else 4)
            if args is None:
                break
            if upper == "C":
                c1 = (
                    cursor[0] + args[0] if relative else args[0],
                    cursor[1] + args[1] if relative else args[1],
                )
                c2 = (
                    cursor[0] + args[2] if relative else args[2],
                    cursor[1] + args[3] if relative else args[3],
                )
                end = (
                    cursor[0] + args[4] if relative else args[4],
                    cursor[1] + args[5] if relative else args[5],
                )
            else:
                c1 = (
                    (
                        2 * cursor[0] - last_cubic[0],
                        2 * cursor[1] - last_cubic[1],
                    )
                    if last_cubic
                    else cursor
                )
                c2 = (
                    cursor[0] + args[0] if relative else args[0],
                    cursor[1] + args[1] if relative else args[1],
                )
                end = (
                    cursor[0] + args[2] if relative else args[2],
                    cursor[1] + args[3] if relative else args[3],
                )
            if not current:
                current = [cursor]
            current.extend(_sample_cubic(cursor, c1, c2, end))
            last_cubic, last_quad, cursor = c2, None, end
            continue
        elif upper in {"Q", "T"}:
            args = take(4 if upper == "Q" else 2)
            if args is None:
                break
            if upper == "Q":
                c1 = (
                    cursor[0] + args[0] if relative else args[0],
                    cursor[1] + args[1] if relative else args[1],
                )
                end = (
                    cursor[0] + args[2] if relative else args[2],
                    cursor[1] + args[3] if relative else args[3],
                )
            else:
                c1 = (
                    (
                        2 * cursor[0] - last_quad[0],
                        2 * cursor[1] - last_quad[1],
                    )
                    if last_quad
                    else cursor
                )
                end = (
                    cursor[0] + args[0] if relative else args[0],
                    cursor[1] + args[1] if relative else args[1],
                )
            if not current:
                current = [cursor]
            current.extend(_sample_quadratic(cursor, c1, end))
            last_quad, last_cubic, cursor = c1, None, end
            continue
        elif upper == "A":
            args = take(7)
            if args is None:
                break
            end = (
                cursor[0] + args[5] if relative else args[5],
                cursor[1] + args[6] if relative else args[6],
            )
            if not current:
                current = [cursor]
            current.extend(
                _sample_arc(
                    cursor, args[0], args[1], args[2], bool(args[3]), bool(args[4]), end
                )
            )
            cursor = end
        else:
            index += 1
            continue
        last_cubic = last_quad = None

    flush(False)
    return subpaths


def _ellipse_points(cx: float, cy: float, rx: float, ry: float) -> list[Point]:
    steps = 48
    return [
        (
            cx + rx * math.cos(2 * math.pi * i / steps),
            cy + ry * math.sin(2 * math.pi * i / steps),
        )
        for i in range(steps + 1)
    ]


@dataclass(frozen=True)
class Viewport:
    """The reference lengths percentages and relative units resolve against.

    Attributes:
        width (`float`):
            Viewport width in user units, from the viewBox.
        height (`float`):
            Viewport height in user units.
    """

    width: float
    height: float

    @property
    def diagonal(self) -> float:
        """`float`: The normalised diagonal the SVG spec uses for lengths that
        are neither horizontal nor vertical, such as a circle's `r`."""
        return math.sqrt((self.width**2 + self.height**2) / 2.0)


# CSS absolute units in user units, with 1in = 96px. `em` and `ex` depend on
# font size, for which the SVG default of 16px is assumed.
_UNIT_SCALE = {
    "": 1.0,
    "px": 1.0,
    "pt": 96.0 / 72.0,
    "pc": 16.0,
    "in": 96.0,
    "mm": 96.0 / 25.4,
    "cm": 96.0 / 2.54,
    "q": 96.0 / 101.6,
    "em": 16.0,
    "rem": 16.0,
    "ex": 8.0,
    "ch": 8.0,
}
_LENGTH = re.compile(
    r"^\s*([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)\s*"
    r"(%|px|pt|pc|in|mm|cm|q|em|rem|ex|ch)?\s*$",
    re.IGNORECASE,
)


def length(value: str | None, reference: float, default: float = 0.0) -> float:
    """Resolve an SVG length attribute to user units.

    Percentages and CSS units are legal in every geometry attribute and models
    use them freely, `width="100%"` most of all, so the raw attribute cannot be
    passed to `float()`.

    Args:
        value (`str`, *optional*):
            The raw attribute value.
        reference (`float`):
            The length a percentage is relative to. Per the SVG spec that is
            the viewport width for horizontal attributes, the height for
            vertical ones, and the normalised diagonal otherwise.
        default (`float`, *optional*, defaults to `0.0`):
            Returned when the value is missing or unparseable.

    Returns:
        `float`: The length in user units.

    Examples:

    ```python
    length("100%", reference=400.0)  # 400.0
    length("12pt", reference=400.0)  # 16.0
    ```
    """
    if value is None:
        return default
    match = _LENGTH.match(value)
    if match is None:
        return default
    magnitude = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit == "%":
        return magnitude / 100.0 * reference
    return magnitude * _UNIT_SCALE.get(unit, 1.0)


def element_polylines(
    element: ET.Element, viewport: Viewport | None = None
) -> list[tuple[list[Point], bool]]:
    """Convert one SVG element into untransformed polylines.

    Args:
        element (`xml.etree.ElementTree.Element`):
            A drawable SVG element.
        viewport ([`Viewport`], *optional*):
            Reference lengths for resolving percentages. Defaults to a
            100 by 100 viewport.

    Returns:
        `list[tuple[list[Point], bool]]`: Subpaths with their closed flags. An
            empty list for elements that draw nothing.
    """
    tag = strip_namespace(element.tag)
    get = element.get
    view = viewport or Viewport(100.0, 100.0)
    horizontal, vertical, diagonal = view.width, view.height, view.diagonal

    if tag in {"circle", "ellipse"}:
        cx = length(get("cx"), horizontal)
        cy = length(get("cy"), vertical)
        if tag == "circle":
            rx = ry = length(get("r"), diagonal)
        else:
            rx = length(get("rx"), horizontal)
            ry = length(get("ry"), vertical)
        return [(_ellipse_points(cx, cy, rx, ry), True)] if rx > 0 and ry > 0 else []
    if tag == "rect":
        x, y = length(get("x"), horizontal), length(get("y"), vertical)
        w = length(get("width"), horizontal)
        h = length(get("height"), vertical)
        if w <= 0 or h <= 0:
            return []
        return [([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)], True)]
    if tag == "line":
        return [
            (
                [
                    (length(get("x1"), horizontal), length(get("y1"), vertical)),
                    (length(get("x2"), horizontal), length(get("y2"), vertical)),
                ],
                False,
            )
        ]
    if tag in {"polyline", "polygon"}:
        coords = _numbers(get("points"))
        pts = list(zip(coords[0::2], coords[1::2]))
        if len(pts) < 2:
            return []
        if tag == "polygon":
            pts = pts + [pts[0]]
        return [(pts, tag == "polygon")]
    if tag == "path":
        return flatten_path(get("d", ""))
    return []


@dataclass(frozen=True)
class Shape:
    """A flattened outline in normalised canvas coordinates.

    Coordinates are divided by the larger viewBox dimension, so a value of 1.0
    spans the canvas regardless of the units the model chose.

    Attributes:
        tag (`str`):
            Originating SVG tag.
        points (`list[Point]`):
            The flattened outline.
        closed (`bool`):
            Whether the outline forms a closed loop.
        centroid (`Point`):
            Mean of the outline points.
        area (`float`):
            Absolute shoelace area of the implicitly closed outline.
        perimeter (`float`):
            Summed segment lengths.
        extent (`float`):
            Diagonal of the bounding box.
        circularity (`float`):
            Isoperimetric quotient, `4*pi*area / perimeter**2`. A circle scores
            1.0, a square 0.785, a thin sliver near 0.
        radius_cv (`float`):
            Coefficient of variation of the centroid-to-outline distance. A
            circle scores 0.0.
    """

    tag: str
    points: list[Point]
    closed: bool
    centroid: Point
    radius: float
    area: float
    perimeter: float
    extent: float
    bbox: tuple[float, float, float, float]
    circularity: float
    radius_cv: float

    @property
    def aspect(self) -> float:
        """`float`: Bounding-box width over height, clamped away from zero.

        A wheel seen from the side is close to 1.0. Rejecting shapes far from
        square is what keeps a long ellipse (a wing, a cloud) from being
        counted as a wheel on circularity alone.
        """
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        if width <= 0 or height <= 0:
            return 0.0
        return width / height


def _densify(points: Sequence[Point], target_segments: int = 64) -> list[Point]:
    """Subdivide long segments so shape statistics do not depend on vertex count.

    A rectangle has four segments whose midpoints are all equidistant from the
    centre, which makes its radius variation read as zero. Splitting segments
    to a common maximum length makes a coarse polygon and a finely sampled
    curve measurable on the same scale.
    """
    total = sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )
    if total <= 0:
        return list(points)
    max_length = total / target_segments
    dense: list[Point] = [points[0]]
    for i in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[i], points[i + 1]
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, math.ceil(length / max_length)) if max_length > 0 else 1
        for step in range(1, steps + 1):
            t = step / steps
            dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return dense


def _build_shape(tag: str, raw_points: Sequence[Point], closed: bool) -> Shape | None:
    if len(raw_points) < 2:
        return None
    points = _densify(raw_points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    extent = math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
    if extent <= 0:
        return None

    # Centroid and radii are weighted by segment length rather than taken as a
    # mean over vertices. Vertex means are biased by uneven sampling: a closing
    # vertex counted twice, or a path that spends twenty bezier samples on one
    # side of a wheel and two on the other, both shift the centre.
    midpoints: list[Point] = []
    weights: list[float] = []
    for i in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[i], points[i + 1]
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0:
            continue
        midpoints.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        weights.append(length)

    perimeter = sum(weights)
    if perimeter > 0:
        cx = sum(m[0] * w for m, w in zip(midpoints, weights)) / perimeter
        cy = sum(m[1] * w for m, w in zip(midpoints, weights)) / perimeter
    else:
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

    area = (
        abs(
            sum(
                points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1]
                for i in range(len(points) - 1)
            )
            + points[-1][0] * points[0][1]
            - points[0][0] * points[-1][1]
        )
        / 2.0
    )

    circularity = (4 * math.pi * area / perimeter**2) if perimeter > 0 else 0.0

    if perimeter > 0:
        radii = [math.hypot(m[0] - cx, m[1] - cy) for m in midpoints]
        mean_r = sum(r * w for r, w in zip(radii, weights)) / perimeter
        if mean_r > 0:
            variance = (
                sum(w * (r - mean_r) ** 2 for r, w in zip(radii, weights)) / perimeter
            )
            radius_cv = math.sqrt(variance) / mean_r
        else:
            radius_cv = 1.0
    else:
        mean_r, radius_cv = 0.0, 1.0

    return Shape(
        tag=tag,
        points=list(points),
        closed=closed,
        centroid=(cx, cy),
        radius=mean_r,
        area=area,
        perimeter=perimeter,
        extent=extent,
        bbox=bbox,
        circularity=min(circularity, 1.0),
        radius_cv=radius_cv,
    )


def viewbox_scale(root: ET.Element) -> tuple[float, float, float]:
    """Return the `(min_x, min_y, size)` used to normalise coordinates."""
    box = _numbers(root.get("viewBox"))
    if len(box) == 4 and box[2] > 0 and box[3] > 0:
        return box[0], box[1], max(box[2], box[3])
    width = _numbers(root.get("width"))
    height = _numbers(root.get("height"))
    size = max(width[0] if width else 0.0, height[0] if height else 0.0)
    return 0.0, 0.0, size if size > 0 else 100.0


def root_viewport(root: ET.Element) -> Viewport:
    """Return the viewport percentages resolve against.

    The viewBox wins when present, since that is the coordinate system the
    drawing is authored in. Otherwise the root width and height are used, and
    failing both a square 100 by 100 viewport is assumed.
    """
    box = _numbers(root.get("viewBox"))
    if len(box) == 4 and box[2] > 0 and box[3] > 0:
        return Viewport(box[2], box[3])
    width = _numbers(root.get("width"))
    height = _numbers(root.get("height"))
    return Viewport(
        width[0] if width and width[0] > 0 else 100.0,
        height[0] if height and height[0] > 0 else 100.0,
    )


# Content in these containers is never painted where it sits: it is a template
# that only becomes visible through a `<use>`, a clip or a mask. Descending into
# them would report shapes at the template's own coordinates, where nothing is
# drawn, and miss the instances that are.
NON_RENDERED = frozenset({"defs", "symbol", "clipPath", "mask", "marker", "pattern"})

# A referenced subtree may itself contain a `<use>`. Bounded so a document that
# references itself cannot recurse forever.
MAX_USE_DEPTH = 4


def _presentation(element: ET.Element, name: str) -> str | None:
    """A presentation property, with any inline `style` declaration winning."""
    for declaration in (element.get("style") or "").split(";"):
        prop, _, value = declaration.partition(":")
        if prop.strip() == name:
            return value.strip()
    return element.get(name)


def _use_target(element: ET.Element) -> str:
    """The id a `<use>` points at, from either `href` or the legacy
    `xlink:href`. Only same-document fragment references are resolvable."""
    for key, value in element.attrib.items():
        if strip_namespace(key) == "href" and value.startswith("#"):
            return value[1:]
    return ""


def extract_shapes(root: ET.Element) -> list[Shape]:
    """Walk an SVG tree and return every drawable outline in normalised space.

    Transforms are accumulated down the tree, so a shape nested inside
    translated groups lands where it visually appears. `<use>` references are
    resolved to the geometry they instantiate, and template containers such as
    `<defs>` contribute nothing on their own. Geometry that never reaches the
    canvas is excluded: hidden or fully transparent subtrees, outlines whose
    fill and stroke both resolve to `none`, and shapes entirely outside the
    viewBox.

    Args:
        root (`xml.etree.ElementTree.Element`):
            The `<svg>` root element.

    Returns:
        `list[Shape]`: Flattened outlines, largest first.

    Examples:

    ```python
    shapes = extract_shapes(parse_svg(source))
    wheels = [s for s in shapes if s.circularity > 0.85]
    ```
    """
    min_x, min_y, size = viewbox_scale(root)
    scale = 1.0 / size if size else 1.0
    normalise: Matrix = (scale, 0.0, 0.0, scale, -min_x * scale, -min_y * scale)
    viewport = root_viewport(root)
    canvas = (viewport.width * scale, viewport.height * scale)

    shapes: list[Shape] = []
    by_id = {el.get("id"): el for el in root.iter() if el.get("id")}

    def on_canvas(shape: Shape) -> bool:
        return (
            shape.bbox[2] >= 0
            and shape.bbox[3] >= 0
            and shape.bbox[0] <= canvas[0]
            and shape.bbox[1] <= canvas[1]
        )

    def walk(
        element: ET.Element,
        matrix: Matrix,
        fill: str | None,
        stroke: str | None,
        depth: int = 0,
    ) -> None:
        if _presentation(element, "display") == "none":
            return
        if _presentation(element, "visibility") in {"hidden", "collapse"}:
            return
        opacity = _numbers(_presentation(element, "opacity"))
        if opacity and opacity[0] == 0:
            return
        fill = _presentation(element, "fill") or fill
        stroke = _presentation(element, "stroke") or stroke

        tag = strip_namespace(element.tag)
        local = multiply(matrix, parse_transform(element.get("transform")))

        if tag == "use":
            target = by_id.get(_use_target(element))
            if target is None or depth >= MAX_USE_DEPTH:
                return
            # x and y on a `<use>` are shorthand for a translate applied to the
            # instantiated copy, in the user space the `<use>` itself sits in.
            offset = (
                1.0,
                0.0,
                0.0,
                1.0,
                length(element.get("x"), viewport.width),
                length(element.get("y"), viewport.height),
            )
            walk(target, multiply(local, offset), fill, stroke, depth + 1)
            return

        # Fill defaults to black, so an outline paints unless the fill is an
        # explicit `none` with no stroke to fall back on.
        if fill != "none" or (stroke or "none") != "none":
            for subpath, closed in element_polylines(element, viewport):
                transformed = [apply(local, point) for point in subpath]
                shape = _build_shape(tag, transformed, closed)
                if shape is not None and on_canvas(shape):
                    shapes.append(shape)
        for child in element:
            if strip_namespace(child.tag) in NON_RENDERED:
                continue
            walk(child, local, fill, stroke, depth)

    walk(root, normalise, None, None)
    shapes.sort(key=lambda s: s.extent, reverse=True)
    return shapes


def significant_shapes(shapes: Sequence[Shape]) -> list[Shape]:
    """Filter out detail work too small to carry structural meaning."""
    return [s for s in shapes if s.extent >= MIN_SHAPE_EXTENT]


def to_dict(shape: Shape) -> dict[str, Any]:
    """Return a compact JSON-serialisable summary of a shape."""
    return {
        "tag": shape.tag,
        "centroid": [round(c, 4) for c in shape.centroid],
        "radius": round(shape.radius, 4),
        "extent": round(shape.extent, 4),
        "aspect": round(shape.aspect, 3),
        "circularity": round(shape.circularity, 3),
        "radius_cv": round(shape.radius_cv, 3),
        "closed": shape.closed,
    }
