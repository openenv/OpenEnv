#!/usr/bin/env python3
"""
Hugging Face Collection Manager for OpenEnv.

This script manages two collection flows:
1) Versioned release collection (for deployed suffix spaces)
2) Global OpenEnv collection (validated first-party canonicals by default)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

from huggingface_hub import HfApi, list_spaces
from huggingface_hub.utils import HfHubHTTPError

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


DEFAULT_COLLECTION_NAMESPACE = "openenv"
DEFAULT_COLLECTION_SLUG = "openenv/environment-hub-68f16377abea1ea114fa0743"
DEFAULT_VERSIONED_COLLECTION_TITLE_PREFIX = "OpenEnv Environment Hub"
DEFAULT_GLOBAL_COLLECTION_TITLE = "OpenEnv Environment Hub"
DEFAULT_TAG_FILTER = "openenv"
DEFAULT_GLOBAL_SCOPE = "canonical"
FALLBACK_DEFAULT_VERSION = "0.2.0"
VERSION_SUFFIX_PATTERN = re.compile(r"-(?:\d+\.\d+\.\d+|v\d+-\d+-\d+)$")
DUAL_MODE_FLAGS = frozenset(
    {
        "--skip-versioned-collection",
        "--skip-global-collection",
        "--global-collection-title",
        "--global-collection-slug",
        "--private-global-collection",
        "--public-global-collection",
        "--private-versioned-collection",
        "--public-versioned-collection",
        "--global-scope",
    }
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_default_version() -> str:
    """Load default version from repository pyproject.toml."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject_path.exists():
        return FALLBACK_DEFAULT_VERSION

    try:
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)
        version = data.get("project", {}).get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"Could not parse default version from pyproject.toml: {exc}")

    return FALLBACK_DEFAULT_VERSION


DEFAULT_VERSION = load_default_version()


def setup_api() -> HfApi:
    """Initialize and authenticate the Hugging Face API client."""
    hf_token = os.environ.get("HF_TOKEN")
    logger.info("Authenticating with Hugging Face...")
    api = HfApi(token=hf_token) if hf_token else HfApi()

    try:
        whoami = api.whoami()
        logger.info(f"✓ Authenticated as: {whoami['name']}")
    except Exception as exc:
        logger.error(f"Failed to authenticate with Hugging Face: {exc}")
        logger.error("Set HF_TOKEN or run 'hf auth login' before running this script.")
        sys.exit(1)

    return api


def normalize_version(version: str) -> str:
    """Normalize version text for display."""
    normalized = version.strip()
    return normalized[1:] if normalized.startswith("v") else normalized


def build_versioned_collection_title(prefix: str, version: str) -> str:
    """Build predictable versioned collection title."""
    return f"{prefix} {normalize_version(version)}"


def synthetic_slug(namespace: str, title: str) -> str:
    """Build synthetic slug used only for dry-run output."""
    slug_base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{namespace}/dry-run-{slug_base}"


def find_collection_by_title(api: HfApi, namespace: str, title: str):
    """Find collection object by exact title within a namespace."""
    try:
        collections = api.list_collections(owner=namespace)
    except Exception as exc:
        logger.error(f"Failed to list collections for '{namespace}': {exc}")
        return None

    try:
        iterator = iter(collections)
    except TypeError:
        logger.warning(
            "list_collections returned non-iterable response; ignoring for compatibility"
        )
        return None

    for collection in iterator:
        if getattr(collection, "title", None) == title:
            return collection
    return None


def ensure_collection_metadata(
    api: HfApi,
    collection_slug: str,
    *,
    title: Optional[str],
    description: Optional[str],
    private: bool,
    dry_run: bool,
) -> None:
    """Ensure collection metadata matches the desired state."""
    state = "private" if private else "public"
    if dry_run:
        logger.info(
            f"[DRY RUN] Would set collection metadata to visibility={state}: {collection_slug}"
        )
        return

    try:
        api.update_collection_metadata(
            collection_slug=collection_slug,
            title=title,
            description=description,
            private=private,
        )
        logger.info(f"✓ Collection metadata synced ({state}): {collection_slug}")
    except TypeError:
        # Backward compatibility for older hub versions (best effort).
        try:
            api.update_collection_metadata(collection_slug=collection_slug, private=private)
            logger.info(
                f"✓ Collection visibility set to {state}: {collection_slug}"
            )
        except Exception as exc:
            logger.warning(f"Could not set collection visibility for {collection_slug}: {exc}")
    except Exception as exc:
        logger.warning(f"Could not sync collection metadata for {collection_slug}: {exc}")


def resolve_explicit_collection_slug(api: HfApi, explicit_slug: str) -> str:
    """Resolve a user-facing collection slug alias to the canonical Hub slug."""
    try:
        collection = api.get_collection(explicit_slug)
    except Exception as exc:
        logger.warning(
            f"Could not resolve explicit collection slug '{explicit_slug}'; using it as-is: {exc}"
        )
        return explicit_slug

    resolved_slug = getattr(collection, "slug", None) or explicit_slug
    if resolved_slug != explicit_slug:
        logger.info(f"Resolved collection slug alias '{explicit_slug}' -> '{resolved_slug}'")
    return resolved_slug


def resolve_collection_slug(
    api: HfApi,
    namespace: str,
    title: str,
    description: str,
    explicit_slug: Optional[str],
    private: bool,
    dry_run: bool,
) -> str:
    """Resolve, create, and/or enforce visibility for a collection."""
    if explicit_slug:
        resolved_slug = resolve_explicit_collection_slug(api, explicit_slug)
        ensure_collection_metadata(
            api,
            resolved_slug,
            title=title,
            description=description,
            private=private,
            dry_run=dry_run,
        )
        return resolved_slug

    existing = find_collection_by_title(api, namespace, title)
    if existing:
        slug = existing.slug
        logger.info(f"Using existing collection: {slug}")
        ensure_collection_metadata(
            api,
            slug,
            title=title,
            description=description,
            private=private,
            dry_run=dry_run,
        )
        return slug

    if dry_run:
        slug = synthetic_slug(namespace, title)
        logger.info(f"[DRY RUN] Collection '{title}' not found; would create '{slug}'")
        return slug

    try:
        collection = api.create_collection(
            title=title,
            namespace=namespace,
            description=description,
            private=private,
            exists_ok=True,
        )
    except TypeError:
        # Backward compatibility if create_collection(private=...) is unavailable.
        collection = api.create_collection(
            title=title,
            namespace=namespace,
            description=description,
            exists_ok=True,
        )
        ensure_collection_metadata(
            api,
            collection.slug,
            title=title,
            description=description,
            private=private,
            dry_run=False,
        )

    logger.info(f"✓ Created collection: {collection.slug}")
    return collection.slug


def get_collection_items(api: HfApi, collection_slug: str = DEFAULT_COLLECTION_SLUG):
    """Retrieve collection items currently present for Spaces."""
    logger.info(f"Fetching current collection contents: {collection_slug}")

    try:
        collection = api.get_collection(collection_slug)
        items = [item for item in collection.items if item.item_type == "space"]
        logger.info(f"✓ Found {len(items)} spaces in collection")
        return items
    except HfHubHTTPError as exc:
        if exc.response.status_code == 404:
            logger.error(f"Collection not found: {collection_slug}")
        else:
            logger.error(f"Error fetching collection: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Unexpected error fetching collection: {exc}")
        sys.exit(1)


def get_collection_spaces(
    api: HfApi, collection_slug: str = DEFAULT_COLLECTION_SLUG
) -> Set[str]:
    """Retrieve space IDs currently in collection."""
    return {
        item.item_id for item in get_collection_items(api, collection_slug=collection_slug)
    }


def discover_openenv_spaces(
    api: HfApi, tag_filter: str = DEFAULT_TAG_FILTER
) -> List[str]:
    """Discover Docker spaces that include the requested tag."""
    logger.info(f"Searching for Docker Spaces with tag '{tag_filter}'...")

    try:
        spaces = list(
            list_spaces(
                search=tag_filter,
                full=False,
                sort="trending_score",
                direction=-1,
            )
        )
    except Exception as exc:
        logger.error(f"Error discovering spaces: {exc}")
        sys.exit(1)

    discovered: List[str] = []
    for space in spaces:
        try:
            info = api.space_info(space.id)
            runtime = getattr(info, "runtime", None)
            runtime_stage = getattr(runtime, "stage", None)
            tags = getattr(info, "tags", []) or []
            if (
                getattr(info, "sdk", None) == "docker"
                and tag_filter in tags
                and runtime_stage != "RUNTIME_ERROR"
            ):
                discovered.append(space.id)
        except Exception as exc:
            logger.warning(f"Could not inspect space {space.id}: {exc}")

    deduped = dedupe_preserve_order(discovered)
    logger.info(f"✓ Discovered {len(deduped)} Docker spaces with tag '{tag_filter}'")
    return deduped


def is_version_suffixed_space(space_id: str) -> bool:
    """Detect whether a space name ends with a version suffix."""
    name = space_id.split("/", 1)[-1]
    return bool(VERSION_SUFFIX_PATTERN.search(name))


def discover_canonical_openenv_spaces(
    api: HfApi, namespace: str, tag_filter: str
) -> List[str]:
    """Discover canonical OpenEnv spaces owned by a namespace."""
    logger.info(
        f"Discovering canonical spaces for namespace '{namespace}' tagged '{tag_filter}'..."
    )

    discovered: List[str] = []
    try:
        spaces = list(
            list_spaces(
                author=namespace,
                full=False,
                sort="last_modified",
                direction=-1,
                limit=1000,
            )
        )
    except Exception as exc:
        logger.error(f"Error listing spaces for namespace '{namespace}': {exc}")
        sys.exit(1)

    for space in spaces:
        sid = space.id
        if is_version_suffixed_space(sid):
            continue
        try:
            info = api.space_info(sid)
            tags = getattr(info, "tags", []) or []
            if tag_filter in tags:
                discovered.append(sid)
        except Exception as exc:
            logger.warning(f"Could not inspect space {sid}: {exc}")

    deduped = dedupe_preserve_order(discovered)
    logger.info(f"✓ Discovered {len(deduped)} canonical spaces")
    return deduped


def discover_global_target_spaces(
    api: HfApi,
    namespace: str,
    tag_filter: str,
    scope: str,
) -> List[str]:
    """Resolve global collection targets for the requested discovery scope."""
    if scope == "canonical":
        return discover_canonical_openenv_spaces(api, namespace, tag_filter)
    return discover_openenv_spaces(api, tag_filter)


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    """Deduplicate while preserving insertion order."""
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def add_spaces_to_collection(
    api: HfApi,
    collection_slug: str,
    space_ids: List[str],
    version: str = DEFAULT_VERSION,
    dry_run: bool = False,
    note: Optional[str] = None,
) -> int:
    """Add spaces to collection, returning count of added/would-add."""
    if not space_ids:
        logger.info("No new spaces to add")
        return 0

    added_count = 0
    failed_count = 0
    note_text = note if note is not None else f"OpenEnv release {normalize_version(version)}"

    for space_id in space_ids:
        if dry_run:
            logger.info(f"[DRY RUN] Would add space to {collection_slug}: {space_id}")
            added_count += 1
            continue

        try:
            logger.info(f"Adding space to collection: {space_id}")
            api.add_collection_item(
                collection_slug=collection_slug,
                item_id=space_id,
                item_type="space",
                note=note_text,
                exists_ok=True,
            )
            logger.info(f"✓ Added: {space_id}")
            added_count += 1
        except HfHubHTTPError as exc:
            if exc.response.status_code == 409:
                logger.warning(f"Space already in collection: {space_id}")
            else:
                logger.error(f"Failed to add {space_id}: {exc}")
                failed_count += 1
        except Exception as exc:
            logger.error(f"Unexpected error adding {space_id}: {exc}")
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"Failed to add {failed_count} spaces")
    return added_count


def remove_spaces_from_collection(
    api: HfApi,
    collection_slug: str,
    current_items,
    target_space_ids: List[str],
    dry_run: bool = False,
) -> int:
    """Remove spaces that are not part of the target set."""
    target_ids = set(target_space_ids)
    items_to_remove = [item for item in current_items if item.item_id not in target_ids]
    if not items_to_remove:
        logger.info("No stale spaces to remove")
        return 0

    removed_count = 0
    failed_count = 0
    for item in items_to_remove:
        if dry_run:
            logger.info(f"[DRY RUN] Would remove space from {collection_slug}: {item.item_id}")
            removed_count += 1
            continue

        try:
            logger.info(f"Removing stale space from collection: {item.item_id}")
            api.delete_collection_item(
                collection_slug=collection_slug,
                item_object_id=item.item_object_id,
                missing_ok=True,
            )
            logger.info(f"✓ Removed: {item.item_id}")
            removed_count += 1
        except Exception as exc:
            logger.error(f"Failed to remove {item.item_id}: {exc}")
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"Failed to remove {failed_count} spaces")
    return removed_count


def should_skip_fetch(collection_slug: str) -> bool:
    """Skip fetch for dry-run synthetic slugs."""
    return "/dry-run-" in collection_slug


def should_use_dual_mode(argv: List[str]) -> bool:
    """Enable dual-collection behavior only when dual-mode flags are explicitly passed."""
    return any(flag in argv for flag in DUAL_MODE_FLAGS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage OpenEnv Hugging Face collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update versioned + global collections after a deployment
  python scripts/manage_hf_collection.py --version 0.2.1 \\
      --space-id openenv/echo_env-0.2.1 --space-id openenv/coding_env-0.2.1

  # Sync only the curated public OpenEnv collection from canonical discovery
  python scripts/manage_hf_collection.py --skip-versioned-collection
        """,
    )

    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Remove stale collection items that are not part of the resolved target set.",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Version label for release collection (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--space-id",
        action="append",
        default=[],
        help="Space to add to versioned collection (repeatable). If omitted, discovery mode is used.",
    )
    parser.add_argument(
        "--collection-namespace",
        default=DEFAULT_COLLECTION_NAMESPACE,
        help=f"Namespace owning collections (default: {DEFAULT_COLLECTION_NAMESPACE})",
    )
    parser.add_argument(
        "--collection-title-prefix",
        default=DEFAULT_VERSIONED_COLLECTION_TITLE_PREFIX,
        help="Title prefix for versioned collections.",
    )
    parser.add_argument(
        "--collection-slug",
        default=None,
        help="Explicit slug for versioned collection.",
    )
    parser.add_argument(
        "--global-collection-title",
        default=DEFAULT_GLOBAL_COLLECTION_TITLE,
        help=f"Global collection title (default: {DEFAULT_GLOBAL_COLLECTION_TITLE})",
    )
    parser.add_argument(
        "--global-collection-slug",
        default=None,
        help="Explicit slug for global OpenEnv collection.",
    )
    parser.add_argument(
        "--tag-filter",
        default=DEFAULT_TAG_FILTER,
        help=f"Tag used for discovery (default: {DEFAULT_TAG_FILTER})",
    )
    parser.add_argument(
        "--global-scope",
        choices=["canonical", "tagged"],
        default=DEFAULT_GLOBAL_SCOPE,
        help=(
            "Scope for global collection discovery: "
            "'canonical' (namespace-owned, non-versioned) or "
            "'tagged' (all Hub spaces with the tag)."
        ),
    )
    parser.add_argument(
        "--skip-versioned-collection",
        action="store_true",
        help="Skip versioned collection updates.",
    )
    parser.add_argument(
        "--skip-global-collection",
        action="store_true",
        help="Skip global collection updates.",
    )

    versioned_visibility = parser.add_mutually_exclusive_group()
    versioned_visibility.add_argument(
        "--private-versioned-collection",
        dest="private_versioned_collection",
        action="store_true",
        help="Make versioned collection private (default).",
    )
    versioned_visibility.add_argument(
        "--public-versioned-collection",
        dest="private_versioned_collection",
        action="store_false",
        help="Make versioned collection public.",
    )
    parser.set_defaults(private_versioned_collection=True)

    global_visibility = parser.add_mutually_exclusive_group()
    global_visibility.add_argument(
        "--private-global-collection",
        dest="private_global_collection",
        action="store_true",
        help="Make global collection private.",
    )
    global_visibility.add_argument(
        "--public-global-collection",
        dest="private_global_collection",
        action="store_false",
        help="Make global collection public (default).",
    )
    parser.set_defaults(private_global_collection=False)

    raw_argv = sys.argv[1:]
    args = parser.parse_args()

    if args.skip_versioned_collection and args.skip_global_collection:
        logger.error("Nothing to do: both collection flows are skipped.")
        sys.exit(1)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("=" * 60)

    api = setup_api()

    # Backward-compatible single-collection flow (default behavior unless
    # dual-collection flags are explicitly requested).
    if not should_use_dual_mode(raw_argv):
        collection_title = build_versioned_collection_title(
            args.collection_title_prefix, args.version
        )
        collection_slug = resolve_collection_slug(
            api=api,
            namespace=args.collection_namespace,
            title=collection_title,
            description=f"OpenEnv spaces for release {normalize_version(args.version)}",
            explicit_slug=args.collection_slug,
            private=args.private_versioned_collection,
            dry_run=args.dry_run,
        )

        current_items = (
            []
            if args.dry_run and should_skip_fetch(collection_slug)
            else get_collection_items(api, collection_slug)
        )
        current_spaces = {item.item_id for item in current_items}

        if args.space_id:
            resolved_spaces = dedupe_preserve_order(args.space_id)
        else:
            resolved_spaces = discover_global_target_spaces(
                api,
                namespace=args.collection_namespace,
                tag_filter=args.tag_filter,
                scope=args.global_scope,
            )

        new_spaces = [space_id for space_id in resolved_spaces if space_id not in current_spaces]

        logger.info("=" * 60)
        logger.info("Summary:")
        logger.info(f"  Version: {normalize_version(args.version)}")
        logger.info(f"  Collection: {collection_slug}")
        logger.info(f"  Total spaces in collection: {len(current_spaces)}")
        logger.info(f"  Total spaces resolved: {len(resolved_spaces)}")
        logger.info(f"  New spaces to add: {len(new_spaces)}")
        logger.info(
            f"  Stale spaces to remove: "
            f"{len([item for item in current_items if item.item_id not in set(resolved_spaces)]) if args.reconcile else 0}"
        )
        logger.info("=" * 60)

        add_spaces_to_collection(
            api=api,
            collection_slug=collection_slug,
            space_ids=new_spaces,
            version=args.version,
            dry_run=args.dry_run,
        )
        if args.reconcile:
            remove_spaces_from_collection(
                api=api,
                collection_slug=collection_slug,
                current_items=current_items,
                target_space_ids=resolved_spaces,
                dry_run=args.dry_run,
            )
        logger.info(f"Collection URL: https://huggingface.co/collections/{collection_slug}")
        return

    explicit_spaces = dedupe_preserve_order(args.space_id)
    discovered_spaces: List[str] = []
    if (not args.skip_global_collection) or (
        not args.skip_versioned_collection and len(explicit_spaces) == 0
    ):
        discovered_spaces = discover_global_target_spaces(
            api,
            namespace=args.collection_namespace,
            tag_filter=args.tag_filter,
            scope=args.global_scope,
        )

    versioned_targets = explicit_spaces if explicit_spaces else discovered_spaces
    global_targets = discovered_spaces

    if args.verbose:
        logger.debug(f"Explicit spaces: {explicit_spaces}")
        logger.debug(f"Discovered spaces: {discovered_spaces}")
        logger.debug(f"Versioned targets: {versioned_targets}")
        logger.debug(f"Global targets: {global_targets}")

    if not args.skip_versioned_collection:
        versioned_title = build_versioned_collection_title(
            args.collection_title_prefix, args.version
        )
        versioned_slug = resolve_collection_slug(
            api=api,
            namespace=args.collection_namespace,
            title=versioned_title,
            description=f"OpenEnv spaces for release {normalize_version(args.version)}",
            explicit_slug=args.collection_slug,
            private=args.private_versioned_collection,
            dry_run=args.dry_run,
        )

        current_versioned_items = (
            []
            if args.dry_run and should_skip_fetch(versioned_slug)
            else get_collection_items(api, versioned_slug)
        )
        current_versioned = {item.item_id for item in current_versioned_items}
        new_versioned = [s for s in versioned_targets if s not in current_versioned]

        logger.info("=" * 60)
        logger.info("Versioned Collection Summary:")
        logger.info(f"  Version: {normalize_version(args.version)}")
        logger.info(f"  Collection: {versioned_slug}")
        logger.info(
            f"  Visibility: {'private' if args.private_versioned_collection else 'public'}"
        )
        logger.info(f"  Current spaces: {len(current_versioned)}")
        logger.info(f"  Resolved spaces: {len(versioned_targets)}")
        logger.info(f"  New spaces to add: {len(new_versioned)}")
        logger.info(
            f"  Stale spaces to remove: "
            f"{len([item for item in current_versioned_items if item.item_id not in set(versioned_targets)]) if args.reconcile else 0}"
        )
        logger.info("=" * 60)

        add_spaces_to_collection(
            api=api,
            collection_slug=versioned_slug,
            space_ids=new_versioned,
            version=args.version,
            dry_run=args.dry_run,
            note=f"OpenEnv release {normalize_version(args.version)}",
        )
        if args.reconcile:
            remove_spaces_from_collection(
                api=api,
                collection_slug=versioned_slug,
                current_items=current_versioned_items,
                target_space_ids=versioned_targets,
                dry_run=args.dry_run,
            )
        logger.info(f"Versioned collection URL: https://huggingface.co/collections/{versioned_slug}")

    if not args.skip_global_collection:
        global_description = (
            "Validated first-party canonical OpenEnv environments on Hugging Face Hub"
            if args.global_scope == "canonical"
            else "All OpenEnv-tagged environments on Hugging Face Hub"
        )
        global_note = (
            "OpenEnv canonical sync"
            if args.global_scope == "canonical"
            else f"OpenEnv tag sync ({args.tag_filter})"
        )
        discovery_label = (
            "Canonical spaces discovered"
            if args.global_scope == "canonical"
            else "Tagged spaces discovered"
        )
        global_slug = resolve_collection_slug(
            api=api,
            namespace=args.collection_namespace,
            title=args.global_collection_title,
            description=global_description,
            explicit_slug=args.global_collection_slug,
            private=args.private_global_collection,
            dry_run=args.dry_run,
        )

        current_global_items = (
            []
            if args.dry_run and should_skip_fetch(global_slug)
            else get_collection_items(api, global_slug)
        )
        current_global = {item.item_id for item in current_global_items}
        new_global = [s for s in global_targets if s not in current_global]

        logger.info("=" * 60)
        logger.info("Global Collection Summary:")
        logger.info(f"  Collection: {global_slug}")
        logger.info(
            f"  Visibility: {'private' if args.private_global_collection else 'public'}"
        )
        logger.info(f"  Current spaces: {len(current_global)}")
        logger.info(f"  {discovery_label}: {len(global_targets)}")
        logger.info(f"  New spaces to add: {len(new_global)}")
        logger.info(
            f"  Stale spaces to remove: "
            f"{len([item for item in current_global_items if item.item_id not in set(global_targets)]) if args.reconcile else 0}"
        )
        logger.info("=" * 60)

        add_spaces_to_collection(
            api=api,
            collection_slug=global_slug,
            space_ids=new_global,
            version=args.version,
            dry_run=args.dry_run,
            note=global_note,
        )
        if args.reconcile:
            remove_spaces_from_collection(
                api=api,
                collection_slug=global_slug,
                current_items=current_global_items,
                target_space_ids=global_targets,
                dry_run=args.dry_run,
            )
        logger.info(f"Global collection URL: https://huggingface.co/collections/{global_slug}")


if __name__ == "__main__":
    main()
