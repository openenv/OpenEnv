# SPDX-License-Identifier: BSD-3-Clause

"""A deterministic lexical baseline and exact, read-only record inspection."""

from __future__ import annotations

import re

from .errors import CatalogError
from .models import (
    CatalogChanges,
    CatalogSnapshot,
    DiscoveryEntry,
    SearchMatch,
    SupersededEntry,
)
from .serialization import require_complete

SUPPORTED_FILTERS = frozenset(
    {"license", "provider", "artifact_availability", "name", "tags", "type"}
)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", value.casefold()))


def _filter_values(entry: DiscoveryEntry, key: str) -> list[str]:
    if key == "provider":
        return [entry.data.source.provider]
    if key == "tags":
        return entry.tags
    if key == "type":
        return [entry.type]
    return [getattr(entry.data, key)]


def search_catalog(
    snapshot: CatalogSnapshot,
    query: str,
    *,
    filters: dict[str, str] | None = None,
    limit: int = 20,
) -> list[SearchMatch]:
    """
    Find candidate metadata by task without contacting candidate environments.

    Args:
        snapshot ([`~openenv.discovery.CatalogSnapshot`]):
            Complete configured snapshot.
        query (`str`):
            Task-oriented words matched against authored descriptions and hints.
        filters (`dict[str, str]`, *optional*):
            Exact declared-field filters.
        limit (`int`, *optional*, defaults to `20`):
            Maximum returned entries, from 1 to 1000.
    """
    require_complete(snapshot)
    if not 1 <= limit <= 1000:
        raise CatalogError("limit must be between 1 and 1000")
    if len(query) > 8192:
        raise CatalogError("Query exceeds the metadata query limit")
    chosen = filters or {}
    unsupported = chosen.keys() - SUPPORTED_FILTERS
    if unsupported:
        raise CatalogError(f"Unsupported filter: {', '.join(sorted(unsupported))}")
    query_tokens = _tokens(query)
    matches = []
    for entry in snapshot.entries:
        if any(
            value not in _filter_values(entry, key) for key, value in chosen.items()
        ):
            continue
        text = " ".join(
            [
                entry.display_name,
                entry.data.description,
                *entry.tags,
                *entry.capabilities,
                *entry.representative_queries,
            ]
        )
        shared = query_tokens & _tokens(text)
        if query_tokens and not shared:
            continue
        score = round(100 * len(shared) / len(query_tokens)) if query_tokens else 0
        matches.append(SearchMatch(entry=entry, score=score))
    return sorted(matches, key=lambda match: (-match.score, match.entry.identifier))[
        :limit
    ]


def resolve_entry(snapshot: CatalogSnapshot, identifier: str) -> DiscoveryEntry:
    """Inspect an exact identifier in the configured snapshot, without URL inference."""
    require_complete(snapshot)
    for entry in snapshot.entries:
        if entry.identifier == identifier:
            return entry
    raise CatalogError("Entry not found in the configured catalog snapshot")


def compare_catalogs(
    previous: CatalogSnapshot, current: CatalogSnapshot
) -> CatalogChanges:
    """Distinguish new revision cards from withdrawn listings after a complete refresh."""
    require_complete(previous)
    require_complete(current)
    if (
        previous.publisher,
        previous.source.uri,
        previous.inventory.root,
    ) != (current.publisher, current.source.uri, current.inventory.root):
        raise CatalogError(
            "Catalog refresh must preserve publisher and inventory scope"
        )
    old = {entry.data.source.path: entry for entry in previous.entries}
    new = {entry.data.source.path: entry for entry in current.entries}
    return CatalogChanges(
        added=[new[key] for key in sorted(new.keys() - old.keys())],
        withdrawn=[old[key] for key in sorted(old.keys() - new.keys())],
        superseded=[
            SupersededEntry(previous=old[key], current=new[key])
            for key in sorted(old.keys() & new.keys())
            if old[key].identifier != new[key].identifier
        ],
        corrected=[
            SupersededEntry(previous=old[key], current=new[key])
            for key in sorted(old.keys() & new.keys())
            if old[key].identifier == new[key].identifier and old[key] != new[key]
        ],
    )
