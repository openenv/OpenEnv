# SPDX-License-Identifier: BSD-3-Clause

"""Structural checks: is there a vehicle, and is something riding it?

Deliberately shape-level. These ask whether two round things of similar size sit
level and apart, whether something spans between them, and whether a mass rests
above them. Whether that mass is a pelican is the judge's problem, because
structure is free and reproducible and the judge is neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .geometry import Shape, significant_shapes

# A shape counts as a wheel when it is round by area-to-perimeter ratio, has
# little variation in radius, and is not stretched. All three are needed: a
# square passes none, a long ellipse passes only the first.
WHEEL_MIN_CIRCULARITY = 0.90
WHEEL_MAX_RADIUS_CV = 0.12
WHEEL_MIN_ASPECT = 0.70
WHEEL_MAX_ASPECT = 1.43

# Two detected circles closer than this multiple of their radius are the same
# wheel drawn twice, typically a rim plus an inner rim or a tyre outline.
WHEEL_MERGE_DISTANCE = 0.60

# A drawing is full of small circles that are not wheels: an eye, a hub, a
# bottom bracket, the animal's head. Wheels are the big round shapes and are
# similar in size to each other, so anything well under the largest circle is
# discarded before counting.
WHEEL_MIN_RELATIVE_RADIUS = 0.50

# A round body drawn as a near-circular ellipse passes every roundness test, so
# roundness alone cannot separate a torso from a wheel. What separates them is
# position: the wheels of a ground vehicle sit level with one another. Only
# round shapes within this many radii of the largest one's height are
# considered. The tolerance is deliberately looser than the `wheels_level`
# check so that check still measures something.
WHEEL_ROW_TOLERANCE = 1.20

# A rider must enclose at least this fraction of the canvas area, which rules
# out a saddle or a lamp being mistaken for a passenger.
RIDER_MIN_AREA = 0.008

# And no more than this, which rules out a full-canvas background rectangle.
RIDER_MAX_AREA = 0.35

# A shape this large whose bounding box swallows the whole vehicle is scenery.
# Normalisation divides by the longer viewBox side, so a full-canvas rect on a
# 4:3 document measures 1.0 by 0.75 and a fixed coverage threshold would miss
# it. Enclosing every wheel is the reliable signal.
BACKGROUND_MIN_AREA = 0.30

# How far above the axle line the rider's centroid must sit, in wheel radii.
RIDER_MIN_HEIGHT = 0.80

# Decimal places the row height is rounded to before candidate rows are compared.
# Two rows at the same height must compare equal so the tie falls through to the
# wheel radius, rather than being decided by the last bits of a float.
ROW_HEIGHT_PRECISION = 2


@dataclass(frozen=True)
class Check:
    """One structural question and its answer.

    Attributes:
        name (`str`):
            Stable identifier, used as a metric key.
        passed (`bool`):
            Whether the check succeeded.
        detail (`str`):
            The measurement behind the verdict, for debugging and for showing
            the model why it lost the point.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class StructureReport:
    """Outcome of the structural analysis.

    Attributes:
        checks (`list[Check]`):
            Every check that was run, in a stable order.
        wheels (`list[Shape]`):
            Shapes classified as wheels, left to right.
        rider (`Shape` or `None`):
            The shape taken to be the rider's body, if one was found.
        score (`float`):
            Fraction of checks passed, in [0, 1].
    """

    checks: list[Check] = field(default_factory=list)
    wheels: list[Shape] = field(default_factory=list)
    rider: Shape | None = None
    round_candidates: int = 0
    shapes_considered: int = 0

    @property
    def score(self) -> float:
        """`float`: Fraction of structural checks passed."""
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    @property
    def passed_names(self) -> list[str]:
        """`list[str]`: Names of the checks that passed."""
        return [c.name for c in self.checks if c.passed]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary.

        The detected geometry is reported, not just the verdicts. "Found 1 wheel"
        is unauditable on its own: it does not say *which* shape was taken to be
        the wheel, and the whole rider check hangs off that choice, since the axle
        line comes from the wheel's centre. Reporting the shape makes a
        disagreement between two runs a fact rather than a guess.
        """

        def geometry(shape: Shape | None) -> dict[str, Any] | None:
            if shape is None:
                return None
            return {
                "tag": shape.tag,
                "cx": round(shape.centroid[0], 4),
                "cy": round(shape.centroid[1], 4),
                "radius": round(shape.radius, 4),
                "area": round(shape.area, 4),
                "circularity": round(shape.circularity, 3),
                "radius_cv": round(shape.radius_cv, 3),
                "aspect": round(shape.aspect, 3),
            }

        return {
            "score": round(self.score, 4),
            "wheel_count": len(self.wheels),
            "has_rider": self.rider is not None,
            "checks": {c.name: c.passed for c in self.checks},
            "details": {c.name: c.detail for c in self.checks},
            "wheels": [geometry(w) for w in self.wheels],
            "rider": geometry(self.rider),
            "round_candidates": self.round_candidates,
            "shapes_considered": self.shapes_considered,
        }


def is_wheel(shape: Shape) -> bool:
    """Whether a shape looks like a wheel seen from the side."""
    return (
        shape.circularity >= WHEEL_MIN_CIRCULARITY
        and shape.radius_cv <= WHEEL_MAX_RADIUS_CV
        and WHEEL_MIN_ASPECT <= shape.aspect <= WHEEL_MAX_ASPECT
    )


def _merge_concentric(row: Sequence[Shape]) -> list[Shape]:
    """Collapse a wheel drawn as several concentric circles into its outermost."""
    wheels: list[Shape] = []
    for candidate in sorted(row, key=lambda s: s.radius, reverse=True):
        if not any(
            math.dist(candidate.centroid, kept.centroid)
            < WHEEL_MERGE_DISTANCE * max(kept.radius, candidate.radius)
            for kept in wheels
        ):
            wheels.append(candidate)
    return wheels


def find_wheels(shapes: Sequence[Shape], expected: int = 2) -> list[Shape]:
    """Collect wheel-like shapes, merging concentric duplicates.

    Wheels are found as a *row* rather than by picking the largest round shape.
    Anchoring on the largest inverts the detection whenever the rider is drawn
    bigger than the wheels, because the rider becomes the anchor and the real
    wheels are then discarded for not being level with it. A row follows the
    physical fact the detector actually relies on: the wheels of a ground
    vehicle are level with each other and they are the lowest round things in
    the picture.

    Args:
        shapes (`Sequence[Shape]`):
            Candidate shapes, already filtered for significance.
        expected (`int`, *optional*, defaults to `2`):
            How many wheels the requested vehicle should have. Used only to
            prefer one candidate row over another, never to invent wheels.

    Returns:
        `list[Shape]`: One shape per distinct wheel, ordered left to right.
    """
    round_shapes = [s for s in shapes if is_wheel(s)]
    if not round_shapes:
        return []

    # Every round shape gets a turn as the seed of a row. A row holds shapes of
    # similar size sitting at similar height, which is what a pair of wheels is
    # and what a body plus a wheel is not.
    best: list[Shape] = []
    best_key: tuple[int, float, float] | None = None
    for seed in round_shapes:
        row = [
            s
            for s in round_shapes
            if s.radius >= WHEEL_MIN_RELATIVE_RADIUS * seed.radius
            and s.radius <= seed.radius / WHEEL_MIN_RELATIVE_RADIUS
            and abs(s.centroid[1] - seed.centroid[1])
            <= WHEEL_ROW_TOLERANCE * max(seed.radius, s.radius)
        ]
        merged = _merge_concentric(row)
        # Rank rows by the *bottom edge* first, centre plus radius, which is
        # what touches the ground. Only among rows reaching equally low does
        # the expected wheel count get a vote, and radius settles what is
        # left. The order matters: where a row sits is a fact about the
        # drawing, while the expected count is only our preference, and
        # letting the preference outrank the ground let a lone pedal beat two
        # real wheels whenever a bicycle was scored as a unicycle. Bottom
        # edges also break the hub-versus-rim tie by geometry, where centre
        # heights tied exactly and the winner fell to floating-point noise,
        # scoring the same drawing 1.000 on one machine and 0.333 on a
        # deployed Space. The height is still quantised so that rows touching
        # the same ground line compare as equals.
        key = (
            round(
                sum(s.centroid[1] + s.radius for s in merged) / len(merged),
                ROW_HEIGHT_PRECISION,
            ),
            1 if len(merged) == expected else 0,
            sum(s.radius for s in merged) / len(merged),
        )
        if best_key is None or key > best_key:
            best, best_key = merged, key

    candidates = sorted(best, key=lambda s: s.radius, reverse=True)
    wheels: list[Shape] = []
    for candidate in candidates:
        duplicate = False
        for kept in wheels:
            distance = math.dist(candidate.centroid, kept.centroid)
            if distance < WHEEL_MERGE_DISTANCE * max(kept.radius, candidate.radius):
                duplicate = True
                break
        if not duplicate:
            wheels.append(candidate)
    return sorted(wheels, key=lambda s: s.centroid[0])


def _is_background(shape: Shape, wheels: Sequence[Shape] = ()) -> bool:
    """Whether a shape is a backdrop rather than a subject."""
    if shape.area < BACKGROUND_MIN_AREA:
        return False
    if not wheels:
        return True
    x0, y0, x1, y1 = shape.bbox
    return all(x0 <= w.centroid[0] <= x1 and y0 <= w.centroid[1] <= y1 for w in wheels)


def _frame_spans_wheels(
    shapes: Sequence[Shape], wheels: Sequence[Shape]
) -> tuple[bool, str]:
    """Whether geometry bridges the gap between the outermost hubs.

    The span runs between the outermost pair, matching `wheels_apart`, since
    taking the two leftmost measured a short inner gap whenever a third
    wheel-like shape shared the row. Only the two span wheels are excluded
    from the coverage count: anything between them, including another wheel
    candidate, genuinely bridges the gap.
    """
    left, right = wheels[0], wheels[-1]
    mean_r = (left.radius + right.radius) / 2.0
    hub_y = (left.centroid[1] + right.centroid[1]) / 2.0
    x0, x1 = left.centroid[0], right.centroid[0]
    if x1 <= x0:
        return False, "wheels are not horizontally separated"

    wheel_ids = {id(left), id(right)}
    # The frame sits above the axle line but not as high as the rider, so allow
    # a generous band upward and only a little below.
    top = hub_y - 2.5 * mean_r
    bottom = hub_y + 0.5 * mean_r

    bins = 10
    covered = [False] * bins
    for shape in shapes:
        if id(shape) in wheel_ids or _is_background(shape, wheels):
            continue
        for x, y in shape.points:
            if not (top <= y <= bottom):
                continue
            if not (x0 <= x <= x1):
                continue
            index = min(bins - 1, int((x - x0) / (x1 - x0) * bins))
            covered[index] = True

    hits = sum(covered)
    return hits >= 8, f"{hits}/{bins} of the hub-to-hub span carries frame geometry"


def _find_rider(
    shapes: Sequence[Shape], wheels: Sequence[Shape]
) -> tuple[Shape | None, str]:
    """Largest closed mass sitting above the axle line and between the wheels."""
    if not wheels:
        return None, "no wheels, so no axle line to sit above"
    wheel_ids = {id(w) for w in wheels}
    mean_r = sum(w.radius for w in wheels) / len(wheels)
    hub_y = sum(w.centroid[1] for w in wheels) / len(wheels)
    ceiling = hub_y - RIDER_MIN_HEIGHT * mean_r
    left = min(w.centroid[0] for w in wheels) - mean_r
    right = max(w.centroid[0] for w in wheels) + mean_r

    best: Shape | None = None
    for shape in shapes:
        if id(shape) in wheel_ids or not shape.closed:
            continue
        if not RIDER_MIN_AREA <= shape.area <= RIDER_MAX_AREA:
            continue
        if _is_background(shape, wheels):
            continue
        cx, cy = shape.centroid
        if cy > ceiling or not (left <= cx <= right):
            continue
        if best is None or shape.area > best.area:
            best = shape

    if best is None:
        return None, f"no closed shape of area >= {RIDER_MIN_AREA} above the axle line"
    return (
        best,
        f"body of area {best.area:.4f} centred at ({best.centroid[0]:.2f}, {best.centroid[1]:.2f})",
    )


def analyse_structure(
    shapes: Sequence[Shape], expected_wheels: int = 2
) -> StructureReport:
    """Run every structural check against the extracted shapes.

    Args:
        shapes (`Sequence[Shape]`):
            Shapes from
            [`~envs.pelican_svg_env.server.geometry.extract_shapes`].
        expected_wheels (`int`, *optional*, defaults to `2`):
            How many wheels the requested vehicle should have. A unicycle
            takes 1, a bicycle 2.

    Returns:
        [`StructureReport`]: Checks, detected wheels and rider.

    Examples:

    ```python
    report = analyse_structure(extract_shapes(root), expected_wheels=2)
    print(report.score, report.passed_names)
    ```
    """
    major = significant_shapes(shapes)
    wheels = find_wheels(major, expected=expected_wheels)
    round_candidates = sum(1 for s in major if is_wheel(s))
    checks: list[Check] = []

    checks.append(
        Check(
            "wheel_count",
            len(wheels) == expected_wheels,
            f"found {len(wheels)} wheel-like shapes, expected {expected_wheels}",
        )
    )

    # The wheel-pair checks are meaningless for a single-wheeled vehicle, and
    # scoring them as failures punished a perfectly good unicycle for not
    # being a bicycle. When only one wheel is asked for they are dropped from
    # the denominator rather than failed.
    if expected_wheels >= 2 and len(wheels) >= 2:
        radii = [w.radius for w in wheels]
        ratio = min(radii) / max(radii)
        checks.append(
            Check(
                "wheels_similar_size",
                ratio >= 0.65,
                f"smallest/largest wheel radius = {ratio:.2f}",
            )
        )

        mean_r = sum(radii) / len(radii)
        ys = [w.centroid[1] for w in wheels]
        skew = (max(ys) - min(ys)) / mean_r if mean_r else 99.0
        checks.append(
            Check(
                "wheels_level",
                skew <= 0.5,
                f"vertical offset between hubs = {skew:.2f} wheel radii",
            )
        )

        gap = (
            math.dist(wheels[0].centroid, wheels[-1].centroid) / mean_r
            if mean_r
            else 0.0
        )
        checks.append(
            Check(
                "wheels_apart",
                1.8 <= gap <= 8.0,
                f"hub separation = {gap:.2f} wheel radii",
            )
        )

        spans, detail = _frame_spans_wheels(major, wheels)
        checks.append(Check("frame_spans_wheels", spans, detail))
    elif expected_wheels >= 2:
        for name in (
            "wheels_similar_size",
            "wheels_level",
            "wheels_apart",
            "frame_spans_wheels",
        ):
            checks.append(Check(name, False, "fewer than two wheels detected"))

    rider, rider_detail = _find_rider(major, wheels)
    checks.append(Check("rider_present", rider is not None, rider_detail))

    if rider is not None and wheels:
        mean_r = sum(w.radius for w in wheels) / len(wheels)
        span = max(w.centroid[0] for w in wheels) - min(w.centroid[0] for w in wheels)
        reference = span if span > 0 else mean_r * 2
        proportion = rider.extent / reference if reference else 0.0
        checks.append(
            Check(
                "rider_proportionate",
                0.25 <= proportion <= 1.6,
                f"rider extent is {proportion:.2f} of the wheelbase",
            )
        )
    else:
        checks.append(Check("rider_proportionate", False, "no rider to measure"))

    return StructureReport(
        checks=checks,
        wheels=wheels,
        rider=rider,
        round_candidates=round_candidates,
        shapes_considered=len(major),
    )
