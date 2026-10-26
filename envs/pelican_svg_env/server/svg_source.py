# SPDX-License-Identifier: BSD-3-Clause

"""Source-level inspection of a submitted SVG.

Extracts the SVG from whatever prose the model wrapped around it, then decides
whether the submission is a good-faith attempt to *draw* with vector primitives.
That second job cannot be delegated to a vision model: a bitmap smuggled in
through a ``data:`` URI renders as a convincing picture and only the source gives
it away.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, Sequence

SVG_NAMESPACE = "http://www.w3.org/2000/svg"

# A submission larger than this is refused before parsing. Real drawings from
# current models land two orders of magnitude below it.
MAX_SOURCE_BYTES = 200_000

# Upper bound on parsed elements, so a pathological document cannot make the
# geometry pass quadratic.
MAX_ELEMENTS = 5_000

# Below this many drawable elements there is no drawing to speak of.
MIN_DRAWABLE_ELEMENTS = 3

# Total characters of rendered text tolerated before the submission is treated
# as writing rather than drawing.
MAX_TEXT_CHARS = 40

DRAWABLE_TAGS = frozenset(
    {
        "circle",
        "ellipse",
        "line",
        "path",
        "polygon",
        "polyline",
        "rect",
        "text",
        "image",
        "use",
    }
)

# Tags that carry execution or foreign-content semantics. Kept to these two:
# they are the ones with a reason, and banning legitimate SVG on suspicion
# (animate, set) would reject valid drawings for no observed benefit.
FORBIDDEN_TAGS = frozenset({"script", "foreignObject"})

_TEXT_TAGS = frozenset({"text", "tspan", "textPath"})
_INLINE_TEXT_TAGS = frozenset({"tspan", "textPath"})

_HREF_ATTRS = ("href", "{http://www.w3.org/1999/xlink}href", "src", "xlink:href")
_REMOTE_SCHEME = re.compile(r"^\s*(https?|ftp|file)://", re.IGNORECASE)
_DATA_IMAGE = re.compile(r"data:\s*image/", re.IGNORECASE)
_DOCTYPE = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_SVG_BLOCK = re.compile(r"<svg\b.*?</svg\s*>", re.IGNORECASE | re.DOTALL)
_SVG_OPEN = re.compile(r"<svg\b", re.IGNORECASE)


class SvgParseError(Exception):
    """Raised when the submission cannot be parsed as an SVG document."""


class TruncatedSvgError(SvgParseError):
    """Raised when a submission opens an `<svg>` element but never closes it.

    Almost always a generation that ran into its token limit. Worth its own
    type because "the model refused" and "the harness cut the model off" are
    different facts, and a benchmark that reports them as one number is
    measuring the harness as much as the model.
    """


@dataclass(frozen=True)
class SourceViolation:
    """A single reason the submission is not a good-faith vector drawing.

    Attributes:
        code (`str`):
            Stable machine-readable identifier, for example `embedded_raster`.
        detail (`str`):
            Human-readable explanation, safe to show in an observation.
    """

    code: str
    detail: str


@dataclass(frozen=True)
class SourceReport:
    """Result of inspecting the SVG source.

    Attributes:
        violations (`list[SourceViolation]`):
            Everything wrong with the submission. Empty means it passed.
        element_counts (`dict[str, int]`):
            Count of each tag encountered, namespace stripped.
        drawable_count (`int`):
            Number of elements that put marks on the canvas.
        text_content (`str`):
            Concatenated text nodes, used for the label-instead-of-drawing check.
    """

    violations: list[SourceViolation] = field(default_factory=list)
    element_counts: dict[str, int] = field(default_factory=dict)
    drawable_count: int = 0
    text_content: str = ""

    @property
    def ok(self) -> bool:
        """`bool`: Whether the submission cleared every source-level check."""
        return not self.violations

    @property
    def codes(self) -> list[str]:
        """`list[str]`: Violation codes, in the order they were raised."""
        return [v.code for v in self.violations]


def extract_svg(response: str) -> str:
    """Pull the SVG document out of a raw model response.

    Models routinely wrap the answer in a fenced code block or bracket it with
    prose. The last complete ``<svg>...</svg>`` block wins, since models often
    narrate a first attempt before committing to a final one.

    Args:
        response (`str`):
            Raw text returned by the model.

    Returns:
        `str`: The extracted SVG source, stripped of surrounding whitespace.

    Raises:
        TruncatedSvgError: If an `<svg>` element is opened but never closed.
        SvgParseError: If the response contains no SVG element at all.

    Examples:

    ```python
    svg = extract_svg("Here you go:\\n```svg\\n<svg ...></svg>\\n```")
    ```
    """
    text = response or ""
    matches = _SVG_BLOCK.findall(text)
    if not matches:
        if _SVG_OPEN.search(text):
            raise TruncatedSvgError(
                "response opens an <svg> element but never closes it, which "
                "usually means generation hit its token limit"
            )
        raise SvgParseError("response contains no <svg> element")
    return matches[-1].strip()


def strip_namespace(tag: str) -> str:
    """Return an element tag without its namespace prefix."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_svg(source: str) -> ET.Element:
    """Parse SVG source into an element tree.

    Args:
        source (`str`):
            The SVG document.

    Returns:
        `xml.etree.ElementTree.Element`: The root `<svg>` element.

    Raises:
        SvgParseError: If the source is oversized, declares a DTD, is not
            well-formed XML, or is not rooted at `<svg>`.
    """
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise SvgParseError(f"source exceeds {MAX_SOURCE_BYTES} bytes")
    # ElementTree expands internal entities, so a document declaring its own
    # entities is a billion-laughs vector. No legitimate model-authored SVG
    # needs a DTD, so refuse them outright rather than taking a dependency on
    # a hardened parser.
    if _DOCTYPE.search(source):
        raise SvgParseError("DTD and entity declarations are not accepted")
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise SvgParseError(f"not well-formed XML: {exc}") from exc
    if strip_namespace(root.tag) != "svg":
        raise SvgParseError(f"root element is <{strip_namespace(root.tag)}>, not <svg>")
    return root


def _iter_hrefs(element: ET.Element) -> Iterable[str]:
    for attr in _HREF_ATTRS:
        value = element.get(attr)
        if value:
            yield value


def _mentions_any(text: str, terms: Sequence[str]) -> str | None:
    """Return the first term appearing as a whole word in `text`, if any."""
    lowered = text.lower()
    for term in terms:
        if re.search(rf"\b{re.escape(term.lower())}\b", lowered):
            return term
    return None


def inspect_source(source: str, forbidden_terms: Sequence[str] = ()) -> SourceReport:
    """Check an SVG submission for degenerate and bad-faith constructions.

    Args:
        source (`str`):
            SVG document, already extracted from any surrounding prose.
        forbidden_terms (`Sequence[str]`, *optional*):
            Words that must not appear in rendered text, typically the nouns
            naming the requested subject and vehicle. Writing the answer in
            words instead of drawing it is the cheapest way to pass a
            careless vision judge.

    Returns:
        [`SourceReport`]: Violations found, plus an inventory of the document.

    Raises:
        SvgParseError: If the source cannot be parsed at all.

    Examples:

    ```python
    report = inspect_source(svg, forbidden_terms=["pelican", "bicycle"])
    if not report.ok:
        print(report.codes)
    ```
    """
    root = parse_svg(source)

    violations: list[SourceViolation] = []
    counts: dict[str, int] = {}
    drawable = 0
    text_parts: list[str] = []

    elements = list(root.iter())
    if len(elements) > MAX_ELEMENTS:
        violations.append(
            SourceViolation(
                "too_many_elements",
                f"{len(elements)} elements exceeds the {MAX_ELEMENTS} limit",
            )
        )

    for element in elements:
        tag = strip_namespace(element.tag)
        counts[tag] = counts.get(tag, 0) + 1
        if tag in DRAWABLE_TAGS:
            drawable += 1

        if tag in FORBIDDEN_TAGS:
            violations.append(
                SourceViolation("forbidden_element", f"<{tag}> is not allowed")
            )

        if tag == "image":
            violations.append(
                SourceViolation(
                    "embedded_raster",
                    "<image> embeds a bitmap; the drawing must be vector primitives",
                )
            )

        for href in _iter_hrefs(element):
            if _DATA_IMAGE.search(href):
                violations.append(
                    SourceViolation(
                        "embedded_raster",
                        f"<{tag}> references a data: image URI",
                    )
                )
            elif _REMOTE_SCHEME.match(href):
                violations.append(
                    SourceViolation(
                        "external_reference",
                        f"<{tag}> references external resource {href[:60]!r}",
                    )
                )

        if tag in _TEXT_TAGS and element.text and element.text.strip():
            text_parts.append(element.text.strip())
        # A tspan's tail sits inside the enclosing <text> and is rendered; a
        # <text> element's own tail sits outside it and is not.
        if tag in _INLINE_TEXT_TAGS and element.tail and element.tail.strip():
            text_parts.append(element.tail.strip())

    text_content = " ".join(text_parts)
    if len(text_content) > MAX_TEXT_CHARS:
        violations.append(
            SourceViolation(
                "text_heavy",
                f"{len(text_content)} characters of text exceeds the "
                f"{MAX_TEXT_CHARS} limit; draw it, do not write it",
            )
        )
    named = _mentions_any(text_content, forbidden_terms)
    if named is not None:
        violations.append(
            SourceViolation(
                "text_label",
                f"rendered text names the subject ({named!r}) instead of drawing it",
            )
        )

    if drawable < MIN_DRAWABLE_ELEMENTS:
        violations.append(
            SourceViolation(
                "too_few_elements",
                f"{drawable} drawable elements, need at least {MIN_DRAWABLE_ELEMENTS}",
            )
        )

    # Deduplicate by code so one repeated mistake is reported once.
    seen: set[str] = set()
    unique: list[SourceViolation] = []
    for violation in violations:
        if violation.code not in seen:
            seen.add(violation.code)
            unique.append(violation)

    return SourceReport(
        violations=unique,
        element_counts=counts,
        drawable_count=drawable,
        text_content=text_content,
    )
