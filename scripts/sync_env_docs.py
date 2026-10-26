#!/usr/bin/env python3
"""Sync environment documentation with the envs/ directory.

Ensures every environment in envs/ with a README.md has a corresponding
doc stub in docs/source/environments/<slug>.md, and that existing stubs
stay in sync with their source README.

Also detects orphaned stubs that reference envs which no longer exist,
and stubs that are not listed in docs/source/_toctree.yml (sidebar) or
docs/source/environments.md (HTML catalog).

Modes:
  --check   : Exit non-zero if out of sync (for CI)
  --fix     : Auto-create missing stubs, refresh stale ones, delete orphans
  --dry-run : Preview what --fix would do without writing anything

Note: entries in docs/source/environments.md (HTML catalog) and
docs/source/_toctree.yml are managed manually. This script only
writes the per-environment stub files; missing toctree/catalog
entries are reported so they can be added by hand.
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS_DIR = os.path.join(ROOT, "envs")
DOCS_ENVS_DIR = os.path.join(ROOT, "docs", "source", "environments")
TOCTREE_PATH = os.path.join(ROOT, "docs", "source", "_toctree.yml")
CATALOG_PATH = os.path.join(ROOT, "docs", "source", "environments.md")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/huggingface/OpenEnv/main"
GITHUB_BASE = "https://github.com/huggingface/OpenEnv"

SKIP_DIRS = {"README.md"}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def get_env_dirs():
    """Return sorted list of environment directory names under envs/."""
    return sorted(
        d
        for d in os.listdir(ENVS_DIR)
        if os.path.isdir(os.path.join(ENVS_DIR, d)) and d not in SKIP_DIRS
    )


def get_existing_stub_mapping():
    """Build slug → env_dir mapping by reading stubs in docs/source/environments/.

    Supports two formats:
    - New: ``<!-- openenv-source: env_dir -->`` comment at top of file
    - Legacy: ``{include} ../../../envs/<env_dir>/README.md`` directive
    """
    mapping = {}
    for fname in os.listdir(DOCS_ENVS_DIR):
        if not fname.endswith(".md"):
            continue
        slug = fname[:-3]
        stub_path = os.path.join(DOCS_ENVS_DIR, fname)
        with open(stub_path) as f:
            content = f.read()
        # New format: <!-- openenv-source: env_dir -->
        match = re.search(r"<!--\s*openenv-source:\s*(\S+)\s*-->", content)
        if match:
            mapping[slug] = match.group(1)
            continue
        # Legacy format: {include}
        match = re.search(
            r"\{include\}\s+\.\./\.\./\.\./envs/([^/]+)/README\.md", content
        )
        if match:
            mapping[slug] = match.group(1)
    return mapping


def env_dir_to_slug(env_dir):
    """Convert an env directory name to a doc slug (best-effort default)."""
    slug = env_dir
    if slug.endswith("_env"):
        slug = slug[:-4]
    return slug


# ---------------------------------------------------------------------------
# README helpers
# ---------------------------------------------------------------------------


def _strip_frontmatter(text):
    """Remove YAML frontmatter (--- ... ---) from the start of text."""
    if text.startswith("---"):
        try:
            end = text.index("---", 3)
            return text[end + 3 :].lstrip("\n")
        except ValueError:
            pass
    return text


def _rewrite_relative_links(text, env_dir):
    """Keep README links working after moving their content into the docs site."""
    root = Path(ROOT).resolve()
    source_dir = Path(ENVS_DIR) / env_dir
    pattern = re.compile(
        r"(?P<code>`+).*?(?P=code)"
        r"|(?P<image>!\[[^\]\n]*\]\()(?P<image_url>[^)\s]+)"
        r"|(?P<link>\]\()(?P<link_url><[^>\n]+>|[^)\s]+)"
        r"|(?P<attribute>(?:href|src)=[\"'])(?P<attribute_url>[^\"'\n]+)"
    )

    def is_nested_image_destination(match):
        """Return whether a generic link match belongs to an unsupported image."""
        prefix = match.string[: match.start()]
        image_start = prefix.rfind("![")
        if image_start < 0:
            return False

        depth = 1
        escaped = False
        for character in prefix[image_start + 2 :]:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "[":
                depth += 1
            elif character == "]":
                depth -= 1
                if depth == 0:
                    return False
        return depth == 1

    def rewrite(match):
        if match.group("code"):
            return match.group(0)
        kind = next(k for k in ("image", "link", "attribute") if match.group(k))
        if kind == "link" and is_nested_image_destination(match):
            return match.group(0)
        original = match.group(f"{kind}_url")
        url = original.strip("<>")
        try:
            parsed = urlsplit(url)
        except ValueError:
            return match.group(0)
        if parsed.scheme or parsed.netloc or not parsed.path or url.startswith("/"):
            return match.group(0)
        try:
            target = (source_dir / unquote(parsed.path)).resolve()
        except (OSError, ValueError):
            return match.group(0)
        try:
            relative = target.relative_to(root)
        except ValueError:
            return match.group(0)
        if not target.exists():
            return match.group(0)
        is_image = kind == "image" or match.group(kind).startswith("src=")
        if is_image:
            base = GITHUB_RAW_BASE
        else:
            base = f"{GITHUB_BASE}/{'tree' if target.is_dir() else 'blob'}/main"
        destination = f"{base}/{quote(relative.as_posix())}"
        destination = urlunsplit(
            (*urlsplit(destination)[:3], parsed.query, parsed.fragment)
        )
        if original.startswith("<"):
            destination = f"<{destination}>"
        return match.group(kind) + destination

    lines = []
    fence = None
    list_content_indents = []
    for line in text.splitlines(keepends=True):
        marker = re.match(
            r"^(?:[ ]{0,3}>[ \t]?)*(?P<indent>[ \t]*)"
            r"(?P<fence>`{3,}|~{3,})(?P<rest>.*)$",
            line,
        )
        if marker:
            fence_indent = len(marker.group("indent").expandtabs(4))
            in_list = any(
                content_indent <= fence_indent <= content_indent + 3
                for content_indent in list_content_indents
            )
            if fence_indent > 3 and not in_list:
                marker = None
        if fence:
            if (
                marker
                and marker.group("fence")[0] == fence[0]
                and len(marker.group("fence")) >= len(fence)
                and not marker.group("rest").strip()
            ):
                fence = None
            lines.append(line)
        elif marker:
            fence = marker.group("fence")
            lines.append(line)
        else:
            indentation = re.match(r"^[ \t]*", line)[0]
            indent_width = len(indentation.expandtabs(4))
            list_marker = re.match(
                r"^(?P<indent>[ \t]*)(?:[-+*]|\d{1,9}[.)])(?P<spacing>[ \t]+)",
                line,
            )

            if list_marker:
                while list_content_indents and indent_width < list_content_indents[-1]:
                    list_content_indents.pop()
                content_indent = len(list_marker.group(0).expandtabs(4))
                if (
                    not list_content_indents
                    or content_indent != list_content_indents[-1]
                ):
                    list_content_indents.append(content_indent)
                lines.append(pattern.sub(rewrite, line))
            elif not line.strip():
                lines.append(line)
            else:
                while list_content_indents and indent_width < list_content_indents[-1]:
                    list_content_indents.pop()
                is_list_continuation = (
                    bool(list_content_indents)
                    and indent_width < list_content_indents[-1] + 4
                )
                if line.startswith(("    ", "\t")) and not is_list_continuation:
                    lines.append(line)
                else:
                    if not indentation:
                        list_content_indents.clear()
                    lines.append(pattern.sub(rewrite, line))
    return "".join(lines)


def generate_stub(env_dir):
    """Return the doc stub content for an environment.

    Inlines the README content (without HF Spaces YAML frontmatter) and
    prepends a ``<!-- openenv-source: env_dir -->`` comment so the script
    can identify the source env when re-reading the file later.
    """
    readme_path = os.path.join(ENVS_DIR, env_dir, "README.md")
    with open(readme_path) as f:
        content = f.read()
    content = _strip_frontmatter(content)
    content = _rewrite_relative_links(content, env_dir)
    return f"<!-- openenv-source: {env_dir} -->\n{content}"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(env_dirs, stub_mapping):
    """Return (missing, orphaned, stale, no_readme) lists."""
    reverse_map = {v: k for k, v in stub_mapping.items()}
    documented_env_dirs = set(stub_mapping.values())

    missing = []
    stale = []
    no_readme = []

    for env_dir in env_dirs:
        readme = os.path.join(ENVS_DIR, env_dir, "README.md")
        if not os.path.exists(readme):
            no_readme.append(env_dir)
            continue

        if env_dir in documented_env_dirs:
            slug = reverse_map[env_dir]
            # Check if existing stub is stale (content drifted from README)
            stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
            if os.path.exists(stub_path):
                expected = generate_stub(env_dir)
                with open(stub_path) as f:
                    actual = f.read()
                if actual != expected:
                    stale.append((env_dir, slug))
        else:
            slug = env_dir_to_slug(env_dir)
            missing.append((env_dir, slug))

    orphaned = []
    env_dir_set = set(env_dirs)
    for slug, env_dir in stub_mapping.items():
        if env_dir not in env_dir_set:
            orphaned.append((env_dir, slug))

    return missing, orphaned, stale, no_readme


def find_unlisted(stub_mapping, orphaned):
    """Return (unlisted_toctree, unlisted_catalog) slug lists.

    A stub is unlisted when docs/source/_toctree.yml has no
    ``- local: environments/<slug>`` entry (the page is unreachable from the
    sidebar) or docs/source/environments.md has no ``environments/<slug>``
    link (the env has no card in the catalog). Both files are managed
    manually, so these are reported rather than auto-fixed. Orphaned stubs
    are skipped: the fix there is deleting the stub, not listing it.
    """
    with open(TOCTREE_PATH) as f:
        toctree = f.read()
    with open(CATALOG_PATH) as f:
        catalog = f.read()

    orphaned_slugs = {slug for _, slug in orphaned}
    unlisted_toctree = []
    unlisted_catalog = []
    for slug in sorted(stub_mapping):
        if slug in orphaned_slugs:
            continue
        if not re.search(
            rf"^\s*-\s*local:\s*environments/{re.escape(slug)}\s*$", toctree, re.M
        ):
            unlisted_toctree.append(slug)
        if not re.search(rf"environments/{re.escape(slug)}\b", catalog):
            unlisted_catalog.append(slug)
    return unlisted_toctree, unlisted_catalog


# ---------------------------------------------------------------------------
# Reporting and fixing
# ---------------------------------------------------------------------------


def run_check(missing, orphaned, stale, no_readme, unlisted_toctree, unlisted_catalog):
    ok = True

    if no_readme:
        print(
            "⚠️  The following environments have no README.md and will not appear on the docs site:\n"
        )
        for env_dir in no_readme:
            print(f"  envs/{env_dir}/")
        print()
        print("  This is a warning only — it will not block your PR.\n")

    if missing:
        ok = False
        print("❌ Missing stubs for the following environments:\n")
        for env_dir, slug in missing:
            print(f"  envs/{env_dir}/  →  docs/source/environments/{slug}.md")
        print()
        print("  Run:  python scripts/sync_env_docs.py --fix\n")

    if stale:
        ok = False
        print("⚠️  The following stubs are out of date with their source README:\n")
        for env_dir, slug in stale:
            print(f"  docs/source/environments/{slug}.md  ←  envs/{env_dir}/README.md")
        print()
        print("  Run:  python scripts/sync_env_docs.py --fix\n")

    if orphaned:
        ok = False
        print("⚠️  Orphaned stubs (env directory no longer exists):\n")
        for env_dir, slug in orphaned:
            print(f"  docs/source/environments/{slug}.md  (was envs/{env_dir}/)")
        print()
        print("  Run:  python scripts/sync_env_docs.py --fix\n")

    if unlisted_toctree:
        ok = False
        print("❌ Stubs missing from the docs sidebar (docs/source/_toctree.yml):\n")
        for slug in unlisted_toctree:
            print(f"  docs/source/environments/{slug}.md")
        print()
        print(
            "  Add an entry to the Environments section of docs/source/_toctree.yml:\n"
            "    - local: environments/<slug>\n"
            "      title: <Env Name>\n"
        )

    if unlisted_catalog:
        ok = False
        print("❌ Stubs missing from the HTML catalog (docs/source/environments.md):\n")
        for slug in unlisted_catalog:
            print(f"  docs/source/environments/{slug}.md")
        print()
        print(
            "  Add a card to docs/source/environments.md (copy an existing\n"
            '  <div class="border ..."> card and link to environments/<slug>).\n'
        )

    if ok:
        print("✅ All environment stubs are present and up to date.")
    return 0 if ok else 1


def run_fix(missing, orphaned, stale, dry_run=False):
    def label(action):
        return f"[dry-run] Would {action}" if dry_run else action

    def write_stub(stub_path, env_dir):
        content = generate_stub(env_dir)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=os.path.dirname(stub_path),
                prefix=f".{os.path.basename(stub_path)}.",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                temporary.write(content)
            os.replace(temporary_path, stub_path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    for env_dir, slug in missing:
        stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
        if dry_run:
            print(f"  {label('create')} {os.path.relpath(stub_path, ROOT)}")
        else:
            write_stub(stub_path, env_dir)
            print(f"  ✅ Created {os.path.relpath(stub_path, ROOT)}")

    for env_dir, slug in stale:
        stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
        if dry_run:
            print(f"  {label('refresh')} {os.path.relpath(stub_path, ROOT)}")
        else:
            write_stub(stub_path, env_dir)
            print(f"  🔄 Refreshed {os.path.relpath(stub_path, ROOT)}")

    for env_dir, slug in orphaned:
        stub_path = os.path.join(DOCS_ENVS_DIR, f"{slug}.md")
        if os.path.exists(stub_path):
            if dry_run:
                print(f"  {label('delete')} {os.path.relpath(stub_path, ROOT)}")
            else:
                os.remove(stub_path)
                print(f"  🗑️  Deleted {os.path.relpath(stub_path, ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", action="store_true", help="Check sync status (CI mode)"
    )
    group.add_argument(
        "--fix", action="store_true", help="Fix missing, stale, and orphaned stubs"
    )
    group.add_argument(
        "--dry-run", action="store_true", help="Preview --fix without writing"
    )
    args = parser.parse_args()

    env_dirs = get_env_dirs()
    stub_mapping = get_existing_stub_mapping()
    missing, orphaned, stale, no_readme = analyze(env_dirs, stub_mapping)
    unlisted_toctree, unlisted_catalog = find_unlisted(stub_mapping, orphaned)

    if args.check:
        sys.exit(
            run_check(
                missing, orphaned, stale, no_readme, unlisted_toctree, unlisted_catalog
            )
        )

    # --fix and --dry-run
    if no_readme:
        print("⚠️  Environments without README.md (skipped):\n")
        for env_dir in no_readme:
            print(f"  envs/{env_dir}/")
        print()

    if not missing and not orphaned and not stale:
        print("✅ All stubs are already in sync.")
    else:
        print(
            "Fixing documentation...\n"
            if not args.dry_run
            else "Dry run — no files will be modified:\n"
        )
        run_fix(missing, orphaned, stale, dry_run=args.dry_run)

    # Toctree/catalog entries are managed manually; remind about gaps that
    # --fix cannot write (including stubs it just created).
    manual_toctree = sorted(set(unlisted_toctree) | {slug for _, slug in missing})
    manual_catalog = sorted(set(unlisted_catalog) | {slug for _, slug in missing})
    if manual_toctree:
        print(
            "\n⚠️  Add these to docs/source/_toctree.yml manually (- local: environments/<slug>):"
        )
        for slug in manual_toctree:
            print(f"  {slug}")
    if manual_catalog:
        print("\n⚠️  Add a card for these to docs/source/environments.md manually:")
        for slug in manual_catalog:
            print(f"  {slug}")


if __name__ == "__main__":
    main()
