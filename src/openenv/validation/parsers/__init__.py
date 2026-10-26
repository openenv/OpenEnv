"""Parser protocol and registry.

A parser owns everything format-specific; after `parse()` returns, the format has left
the pipeline. The registry is the only place a [`~openenv.validation.types.SignatureKind`]
selects behavior.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..manifest import NormalizedManifest
from ..signature import UnsupportedPackageError
from ..types import SignatureKind


@runtime_checkable
class Parser(Protocol):
    """
    Parses one package format into the normalized manifest.

    A parse is a pure read: it never imports or executes package code.
    """

    signature: SignatureKind

    def parse(self, package_root: Path) -> NormalizedManifest: ...


class ParserRegistry:
    """
    Maps detected signatures to registered parsers.

    A detected signature with no registered parser raises
    [`~openenv.validation.signature.UnsupportedPackageError`] with category
    `"parser-not-implemented"` — recognized, never guessed.
    """

    def __init__(self) -> None:
        self._parsers: dict[SignatureKind, Parser] = {}

    def register(self, parser: Parser) -> None:
        """
        Register a parser for its signature.

        Args:
            parser ([`~openenv.validation.parsers.Parser`]):
                The parser to register. Replaces any parser previously registered for
                the same signature.
        """
        self._parsers[parser.signature] = parser

    def parser_for(self, signature: SignatureKind) -> Parser:
        """
        Return the parser registered for a detected signature.

        Args:
            signature ([`~openenv.validation.types.SignatureKind`]):
                The detected package signature.

        Returns:
            [`~openenv.validation.parsers.Parser`]: the registered parser.
        """
        parser = self._parsers.get(signature)
        if parser is None:
            raise UnsupportedPackageError(
                "parser-not-implemented",
                f"no parser registered for signature {signature.value!r}",
            )
        return parser
