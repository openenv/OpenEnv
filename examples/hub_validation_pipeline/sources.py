"""Discover public Hub bundles and acquire pinned task files without executing them."""

import asyncio
import hashlib
import json
import re
import shutil
import tempfile
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, snapshot_download

HARBOR_GIT_REVISION = "9c7a39b991d7e2205019f6be0598cbd22d84bfc3"
HARBOR_INSTALL = (
    "uv pip install --python .venv/bin/python "
    f"'harbor @ git+https://github.com/harbor-framework/harbor.git@{HARBOR_GIT_REVISION}'"
)


def discover_hf(limit: int) -> list[dict]:
    """List the exact Harbor-library AND RL-environment view, in trending order."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if limit == 0:
        return []
    required = {"library:harbor", "rl-environment"}
    found = {}
    # Exhaust the SDK iterator: its limit is a total cap, not a page size.
    for rank, item in enumerate(
        HfApi(token=False).list_datasets(
            filter=("library:harbor", "rl-environment"),
            sort="trending_score",
            expand=["sha", "tags", "gated", "disabled"],
            limit=None,
        )
    ):
        if not required.issubset(item.tags or []) or item.id in found:
            continue
        found[item.id] = {
            "source": "hf",
            "id": item.id,
            "revision": _pin(item.sha, "hf"),
            "metadata": {
                "tags": item.tags,
                "gated": item.gated,
                "disabled": item.disabled,
                "rank": rank,
            },
        }
    return list(found.values())[:limit]


def _harbor_api():
    # The PyPI 0.25.0 wheel lacks listing; the official Git pin provides it.
    try:
        installed = distribution("harbor")
        origin = json.loads(installed.read_text("direct_url.json") or "{}")
        if (
            installed.version != "0.25.0"
            or origin.get("vcs_info", {}).get("commit_id") != HARBOR_GIT_REVISION
        ):
            raise RuntimeError(
                f"Harbor source capability mismatch (installed {installed.version}); "
                f"native Hub listing requires official Git {HARBOR_GIT_REVISION}. "
                f"Install: {HARBOR_INSTALL}"
            )
        from harbor.models.task.id import PackageTaskId
        from harbor.registry.client.package import PackageDatasetClient
        from harbor.tasks.client import TaskClient
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError(
            f"Pinned Harbor Hub package APIs unavailable. Install: {HARBOR_INSTALL}"
        ) from error
    return PackageDatasetClient, PackageTaskId, TaskClient


def discover_harbor(limit: int) -> list[dict]:
    """List current Harbor Hub dataset versions, retaining their package digests."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if limit == 0:
        return []
    registry, _, _ = _harbor_api()
    # This Git-pinned API exhausts pages, as does `harbor dataset list --json`.
    try:
        rows = asyncio.run(registry().list_datasets())
    except NotImplementedError as error:
        raise RuntimeError(
            f"Harbor native Hub listing is unsupported. Install: {HARBOR_INSTALL}"
        ) from error
    found = {}
    for row in sorted(rows, key=lambda row: (row.name, row.version or "")):
        revision = _pin(row.version, "harbor")
        _package_name(row.name)
        found.setdefault(
            (row.name, revision),
            {
                "source": "harbor",
                "id": row.name,
                "revision": revision,
                "metadata": {
                    "kind": "dataset",
                    "description": row.description,
                    "task_count": row.task_count,
                    "harbor_source_revision": HARBOR_GIT_REVISION,
                },
            },
        )
    return list(found.values())[:limit]


def acquire(candidate: dict, cache_dir: Path, max_tasks: int) -> list[dict]:
    """Download at most max_tasks into a fresh revision-addressed directory.

    Errors propagate to the driver for exclusion. Zero selects no tasks. Harbor
    single-task candidates use metadata.kind="task"; discovery returns datasets.
    """
    if max_tasks < 0:
        raise ValueError("max_tasks must be nonnegative")
    if max_tasks == 0:
        return []
    source = candidate["source"]
    revision = _pin(candidate["revision"], source)
    if source == "hf":
        return _acquire_hf(candidate, cache_dir, revision, max_tasks)
    if source == "harbor":
        return asyncio.run(_acquire_harbor(candidate, cache_dir, revision, max_tasks))
    raise ValueError(f"Unknown source: {source}")


def _pin(value: str, source: str) -> str:
    pattern = {"hf": r"[0-9a-f]{40}", "harbor": r"sha256:[0-9a-f]{64}"}
    if source not in pattern or not isinstance(value, str):
        raise ValueError(f"Invalid source or immutable revision: {source}, {value}")
    if not re.fullmatch(pattern[source], value):
        raise ValueError(f"Expected immutable {source} revision, got {value!r}")
    return value


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError(f"Unsafe source path: {value!r}")
    return path


def _package_name(name: str) -> None:
    path = _relative(name)
    if len(path.parts) != 2 or path.as_posix() != name or re.search(r"[@:\s]", name):
        raise ValueError(f"Expected Harbor org/name, got {name!r}")


def _fresh(cache_dir: Path, source: str, identity: str, revision: str) -> Path:
    identity_hash = hashlib.sha256(identity.encode()).hexdigest()[:16]
    parent = cache_dir / source / identity_hash / revision.replace(":", "-")
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="selection-", dir=parent)).resolve()


def _real_files(root: Path) -> None:
    """Reject escaping links and materialize contained links as real files."""
    has_links = False
    for path in root.rglob("*"):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe task symlink: {path}")
            has_links = True
        elif not path.is_dir() and not path.is_file():
            raise ValueError(f"Unsupported task file: {path}")
    if has_links:
        # copytree also rejects dangling links and directory-link cycles.
        with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
            materialized = Path(temporary) / "files"
            shutil.copytree(root, materialized, symlinks=False)
            shutil.rmtree(root)
            materialized.rename(root)


def _record(source, identity, revision, root, task_path, metadata, **extra):
    return {
        "task_id": f"{source}:{identity}@{revision}:{task_path}",
        "path": str((root / task_path).resolve(strict=True)),
        "source": source,
        "source_id": identity,
        "revision": revision,
        "task_path": task_path,
        "acquisition_ref": f"{source}:{identity}@{revision}",
        "metadata": metadata,
        **extra,
    }


def task_roots(files):
    """A nested task.toml inside an existing task is an asset, not another task."""
    roots = {
        PurePosixPath(name).parent
        for name in files
        if PurePosixPath(name).name == "task.toml"
    }
    return sorted(
        str(root)
        for root in roots
        if not any(parent in roots for parent in root.parents)
    )


def _acquire_hf(candidate, cache_dir, revision, max_tasks):
    metadata = candidate.get("metadata", {})
    if metadata.get("gated") or metadata.get("disabled"):
        raise PermissionError(f"HF source is gated or disabled: {candidate['id']}")
    api = HfApi(token=False)
    info = api.dataset_info(candidate["id"], revision=revision)
    if info.sha != revision:
        raise ValueError("HF revision resolution disagrees with the selected commit")
    if info.gated or info.disabled:
        raise PermissionError(f"HF source is gated or disabled: {candidate['id']}")
    files = sorted(
        {
            _relative(name).as_posix()
            for name in api.list_repo_files(
                candidate["id"], repo_type="dataset", revision=revision
            )
        }
    )
    roots = set(task_roots(files))
    selected = sorted(roots)[:max_tasks]
    if not selected:
        raise ValueError(f"No task.toml files in {candidate['id']}@{revision}")
    # Retain shared repository assets while excluding every unselected task tree.
    selected_files = []
    for name in files:
        owner = next(
            (
                str(parent)
                for parent in PurePosixPath(name).parents
                if str(parent) in roots
            ),
            None,
        )
        if owner is None or owner in selected:
            selected_files.append(name)
    # Literal filename patterns prevent [brackets], ? and * from widening a subset.
    patterns = [
        "".join({"[": "[[]", "?": "[?]", "*": "[*]"}.get(char, char) for char in name)
        for name in selected_files
    ]
    root = _fresh(cache_dir, "hf", candidate["id"], revision)
    snapshot_download(
        candidate["id"],
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=patterns,
        token=False,
        force_download=True,
    )
    _real_files(root)
    for name in selected_files:
        if not (root / name).is_file():
            raise FileNotFoundError(f"HF download did not materialize {name}")
    return [
        _record(
            "hf",
            candidate["id"],
            revision,
            root,
            task_path,
            metadata,
            download_patterns=patterns,
        )
        for task_path in selected
    ]


async def _acquire_harbor(candidate, cache_dir, revision, max_tasks):
    registry_type, task_id_type, task_client_type = _harbor_api()
    _package_name(candidate["id"])
    registry = registry_type()
    metadata = candidate.get("metadata", {})
    kind = metadata.get("kind", "dataset")
    bundle = None
    if kind == "dataset":
        bundle = await registry.get_dataset_metadata(f"{candidate['id']}@{revision}")
        if bundle.version != revision:
            raise ValueError("Harbor dataset resolution disagrees with its digest")
        tasks = sorted(
            bundle.task_ids, key=lambda task: (task.org, task.name, task.ref)
        )[:max_tasks]
    elif kind == "task":
        org, name = candidate["id"].split("/")
        tasks = [task_id_type(org=org, name=name, ref=revision)]
    else:
        raise ValueError(f"Unknown Harbor package kind: {kind}")
    if not tasks:
        raise ValueError(f"Harbor dataset contains no tasks: {candidate['id']}")
    for task in tasks:
        if not isinstance(task, task_id_type):
            raise ValueError("Current Harbor Hub acquisition requires package task IDs")
        _package_name(f"{task.org}/{task.name}")
        _pin(task.ref, "harbor")
    root = _fresh(cache_dir, "harbor", candidate["id"], revision)
    dataset_files = {}
    if bundle is not None:
        for file in bundle.files:
            _relative(file.path)
        dataset_files = await registry.download_dataset_files(
            bundle, output_dir=root / "dataset", overwrite=True
        )
    downloaded = await task_client_type().download_tasks(
        task_ids=tasks, output_dir=root / "tasks", overwrite=True, export=False
    )
    _real_files(root)
    records = []
    for task, result in zip(tasks, downloaded.results, strict=True):
        if result.content_hash != task.ref.removeprefix("sha256:"):
            raise ValueError("Harbor task resolution disagrees with its digest")
        task_root = result.path.resolve(strict=True)
        if not task_root.is_relative_to(root):
            raise ValueError("Harbor downloaded a task outside the selected cache")
        if not (task_root / "task.toml").is_file():
            raise ValueError(
                f"Harbor package contains no root task.toml: {task.org}/{task.name}"
            )
        records.append(
            _record(
                "harbor",
                f"{task.org}/{task.name}",
                task.ref,
                task_root,
                ".",
                {
                    **metadata,
                    "candidate_id": candidate["id"],
                    "candidate_revision": revision,
                    "dataset_files": {
                        name: str(path) for name, path in dataset_files.items()
                    },
                },
            )
        )
    return records[:max_tasks]
