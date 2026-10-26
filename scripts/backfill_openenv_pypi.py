#!/usr/bin/env python3
"""Backfill historical ``openenv-core`` PyPI releases to ``openenv``.

This script rebuilds existing source releases from the historical PyPI project
under the acquired ``openenv`` project name. It is intentionally manual:
artifacts are built and checked by default, but uploads require ``--upload``.

Examples:

    python scripts/backfill_openenv_pypi.py
    python scripts/backfill_openenv_pypi.py --versions 0.2.0 0.2.1
    python scripts/backfill_openenv_pypi.py --upload --repository pypi
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_PROJECT = "openenv-core"
DEFAULT_TARGET_PROJECT = "openenv"
DEFAULT_FROM_VERSION = "0.2.0"


@dataclass(frozen=True)
class ReleaseFile:
    filename: str
    url: str
    packagetype: str


def numeric_version_key(version: str) -> tuple[int, ...]:
    """Return a sortable key for numeric release versions."""
    if not re.fullmatch(r"\d+(?:\.\d+)*", version):
        raise ValueError(f"Only numeric release versions are supported: {version}")
    return tuple(int(part) for part in version.split("."))


def fetch_pypi_json(project: str) -> dict[str, Any]:
    """Fetch project metadata from PyPI's JSON API."""
    url = f"https://pypi.org/pypi/{project}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"releases": {}}
        raise


def read_current_project_version(repo_root: Path) -> str:
    """Read the current root project version from pyproject.toml."""
    pyproject = repo_root / "pyproject.toml"
    version_re = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
    match = version_re.search(pyproject.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"Could not find project.version in {pyproject}")
    return match.group(1)


def select_versions(
    *,
    source_releases: dict[str, Any],
    target_releases: dict[str, Any],
    explicit_versions: list[str] | None,
    from_version: str,
    before_version: str,
    skip_existing: bool,
) -> list[str]:
    """Select source versions that should be rebuilt for the target project."""
    lower = numeric_version_key(from_version)
    upper = numeric_version_key(before_version)

    if explicit_versions:
        versions = explicit_versions
    else:
        versions = [
            version
            for version in source_releases
            if lower <= numeric_version_key(version) < upper
        ]

    selected: list[str] = []
    for version in sorted(versions, key=numeric_version_key):
        if version not in source_releases:
            raise RuntimeError(f"{version} does not exist on the source PyPI project")
        if version in target_releases:
            if skip_existing:
                print(f"Skipping {version}: target release already exists")
                continue
            raise RuntimeError(
                f"Refusing to rebuild {version}: target release already exists"
            )
        selected.append(version)
    return selected


def find_sdist(release_files: list[dict[str, Any]], version: str) -> ReleaseFile:
    """Return the source distribution file for a PyPI release."""
    for release_file in release_files:
        if release_file.get("packagetype") == "sdist":
            return ReleaseFile(
                filename=str(release_file["filename"]),
                url=str(release_file["url"]),
                packagetype=str(release_file["packagetype"]),
            )
    raise RuntimeError(f"No source distribution found for {version}")


def download_file(url: str, destination: Path) -> None:
    """Download a URL to a local destination."""
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def safe_extract_tarball(tarball: Path, destination: Path) -> Path:
    """Extract a tarball and return the single top-level directory."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(tarball) as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not os.path.commonpath([destination_root, target]) == str(
                destination_root
            ):
                raise RuntimeError(f"Unsafe path in tarball: {member.name}")
        archive.extractall(destination, members=members)

    roots = [
        path
        for path in destination.iterdir()
        if path.is_dir() and not path.name.startswith("__MACOSX")
    ]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one extracted root directory, found {roots}")
    return roots[0]


def _rewrite_text_file(path: Path, source_project: str, target_project: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    rewritten = text.replace(source_project, target_project)
    if rewritten != text:
        path.write_text(rewritten, encoding="utf-8")


def rewrite_source_tree(source_root: Path, source_project: str, target_project: str) -> None:
    """Rewrite a historical source release to publish as the target project."""
    pyproject = source_root / "pyproject.toml"
    if not pyproject.exists():
        raise RuntimeError(f"Missing pyproject.toml in {source_root}")

    for egg_info in source_root.rglob("*.egg-info"):
        if egg_info.is_dir():
            shutil.rmtree(egg_info)

    for path in source_root.rglob("*"):
        if path.is_file():
            _rewrite_text_file(path, source_project, target_project)

    project_text = pyproject.read_text(encoding="utf-8")
    if f'name = "{target_project}"' not in project_text:
        raise RuntimeError(f"Failed to rewrite project name in {pyproject}")


def run_command(command: list[str], cwd: Path) -> None:
    """Run a command with a readable prefix."""
    print(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def build_release(
    *,
    source_root: Path,
    output_dir: Path,
    python: str,
    check: bool,
) -> None:
    """Build a rewritten source release and optionally run twine check."""
    run_command([python, "-m", "build", "--outdir", str(output_dir)], cwd=source_root)
    if check:
        artifacts = sorted(str(path) for path in output_dir.iterdir())
        run_command([python, "-m", "twine", "check", *artifacts], cwd=source_root)


def upload_release(*, output_dir: Path, python: str, repository: str) -> None:
    """Upload built artifacts with twine."""
    artifacts = sorted(str(path) for path in output_dir.iterdir())
    run_command(
        [python, "-m", "twine", "upload", "--repository", repository, *artifacts],
        cwd=output_dir,
    )


def backfill_version(
    *,
    version: str,
    source_file: ReleaseFile,
    args: argparse.Namespace,
    work_root: Path,
) -> None:
    """Download, rewrite, build, and optionally upload one version."""
    version_workdir = work_root / version
    version_workdir.mkdir(parents=True, exist_ok=True)
    downloads_dir = version_workdir / "downloads"
    extract_dir = version_workdir / "source"
    output_dir = args.out_dir / version
    downloads_dir.mkdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==> Backfilling {version}")
    tarball = downloads_dir / source_file.filename
    download_file(source_file.url, tarball)
    source_root = safe_extract_tarball(tarball, extract_dir)
    rewrite_source_tree(source_root, args.source_project, args.target_project)
    build_release(
        source_root=source_root,
        output_dir=output_dir,
        python=args.python,
        check=not args.no_check,
    )

    if args.upload:
        upload_release(
            output_dir=output_dir,
            python=args.python,
            repository=args.repository,
        )
    else:
        print(f"Built {version} artifacts in {output_dir}")
        print("Upload skipped. Re-run with --upload to publish.")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_before_version = read_current_project_version(repo_root)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-project", default=DEFAULT_SOURCE_PROJECT)
    parser.add_argument("--target-project", default=DEFAULT_TARGET_PROJECT)
    parser.add_argument(
        "--from-version",
        default=DEFAULT_FROM_VERSION,
        help="Lowest source version to backfill, inclusive.",
    )
    parser.add_argument(
        "--before-version",
        default=default_before_version,
        help=(
            "Stop before this version. Defaults to the current root package "
            f"version ({default_before_version})."
        ),
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        help="Explicit versions to rebuild. Overrides --from-version range selection.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "dist" / "openenv-backfill",
        help="Directory for rebuilt artifacts.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for build and twine commands.",
    )
    parser.add_argument(
        "--repository",
        default="pypi",
        help="Twine repository name used when --upload is passed.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip versions that already exist on the target PyPI project.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip twine check after building artifacts.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload artifacts with twine. Without this flag the script only builds.",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the temporary unpacked source trees for inspection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()

    source_json = fetch_pypi_json(args.source_project)
    target_json = fetch_pypi_json(args.target_project)
    versions = select_versions(
        source_releases=source_json.get("releases", {}),
        target_releases=target_json.get("releases", {}),
        explicit_versions=args.versions,
        from_version=args.from_version,
        before_version=args.before_version,
        skip_existing=args.skip_existing,
    )

    if not versions:
        print("No versions selected.")
        return 0

    print(f"Selected versions: {', '.join(versions)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="openenv-pypi-backfill-") as temp_dir:
        work_root = Path(temp_dir)
        for version in versions:
            source_file = find_sdist(source_json["releases"][version], version)
            backfill_version(
                version=version,
                source_file=source_file,
                args=args,
                work_root=work_root,
            )

        if args.keep_workdir:
            keep_dir = args.out_dir / "_workdir"
            if keep_dir.exists():
                shutil.rmtree(keep_dir)
            shutil.copytree(work_root, keep_dir)
            print(f"Kept workdir at {keep_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
