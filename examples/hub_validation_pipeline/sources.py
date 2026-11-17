"""Discover public Hub bundles and acquire pinned task files without executing them."""

import asyncio
import gzip
import hashlib
import io
import json
import re
import stat
import tarfile
import tempfile
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, RepoFile, RepoFolder, snapshot_download

MAX_FILES = 128
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024

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
        from harbor.db.client import RegistryDB
        from harbor.models.task.id import PackageTaskId
        from harbor.registry.client.package import PackageDatasetClient
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError(
            f"Pinned Harbor Hub package APIs unavailable. Install: {HARBOR_INSTALL}"
        ) from error
    return PackageDatasetClient, PackageTaskId, RegistryDB


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
    files = {}
    for item in api.list_repo_tree(
        candidate["id"],
        repo_type="dataset",
        revision=revision,
        recursive=True,
        token=False,
    ):
        name = _relative(item.path).as_posix()
        if isinstance(item, RepoFolder):
            continue
        if not isinstance(item, RepoFile) or name in files:
            raise ValueError("HF source requires distinct regular files")
        files[name] = item
    roots = set(task_roots(files))
    selected = sorted(roots)[:max_tasks]
    if not selected:
        raise ValueError(f"No task.toml files in {candidate['id']}@{revision}")
    # Retain shared repository assets while excluding every unselected task tree.
    selected_files = []
    for name in sorted(files):
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
    _check_sizes([files[name].size for name in selected_files])
    if any(files[name].lfs for name in selected_files):
        raise ValueError("Only native regular Git source files are supported")
    hashes = {}
    payloads = {}
    with tempfile.TemporaryDirectory(prefix="openenv-hf-acquisition-") as temporary:
        staging = Path(temporary).resolve()
        snapshot_download(
            candidate["id"],
            repo_type="dataset",
            revision=revision,
            local_dir=staging,
            allow_patterns=patterns,
            token=False,
            force_download=True,
        )
        for name in selected_files:
            path = staging / name
            if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(
                    "HF download must materialize a contained regular file"
                )
            if path.stat().st_size != files[name].size:
                raise ValueError("HF source size disagrees with pinned tree")
            with path.open("rb") as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
            if len(data) != files[name].size:
                raise ValueError("HF source byte cap or size mismatch")
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if blob != files[name].blob_id:
                raise ValueError("HF source Git blob hash mismatch")
            payloads[name] = data
            hashes[name] = hashlib.sha256(data).hexdigest()
    root = _fresh(cache_dir, "hf", candidate["id"], revision)
    for name, data in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return [
        _record(
            "hf",
            candidate["id"],
            revision,
            root,
            task_path,
            metadata,
            download_patterns=patterns,
            task_file_hashes={
                PurePosixPath(name).relative_to(task_path).as_posix(): digest
                for name, digest in hashes.items()
                if PurePosixPath(name).is_relative_to(task_path)
            },
        )
        for task_path in selected
    ]


def _check_sizes(sizes):
    if (
        len(sizes) > MAX_FILES
        or any(
            type(size) is not int or not 0 <= size <= MAX_FILE_BYTES for size in sizes
        )
        or sum(sizes) > MAX_TOTAL_BYTES
    ):
        raise ValueError("Source file count or byte cap exceeded")


async def _download_harbor_file(remote_path, destination, byte_limit):
    import httpx
    from harbor.auth.client import create_authenticated_client
    from harbor.storage.supabase import BUCKET

    client = await create_authenticated_client()
    signed = await client.storage.from_(BUCKET).create_signed_url(remote_path, 300)
    async with httpx.AsyncClient(timeout=120, follow_redirects=False) as http:
        async with http.stream("GET", signed["signedURL"]) as response:
            response.raise_for_status()
            if int(response.headers.get("content-length", "0")) > byte_limit:
                raise ValueError("Harbor download byte cap exceeded")
            total = 0
            with destination.open("xb") as stream:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    total += len(chunk)
                    if total > byte_limit:
                        raise ValueError("Harbor download byte cap exceeded")
                    stream.write(chunk)
    return total


def _harbor_inventory(rows):
    files = {}
    for row in rows:
        name = _relative(row["path"]).as_posix()
        digest = row["content_hash"].removeprefix("sha256:")
        if name.casefold() in {key.casefold() for key in files} or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise ValueError("Invalid Harbor file inventory")
        files[name] = dict(row, sha256=digest)
    _check_sizes([row["size_bytes"] for row in files.values()])
    return files


def _read_harbor_archive(archive_path, inventory):
    payloads = {}
    names = set()
    with gzip.open(archive_path, "rb") as stream:
        data = stream.read(MAX_TOTAL_BYTES + 1)
    if len(data) > MAX_TOTAL_BYTES:
        raise ValueError("Harbor archive expanded byte cap exceeded")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive:
            name = _relative(member.name).as_posix()
            if name in names or len(names) >= MAX_FILES:
                raise ValueError(
                    "Harbor archive path collision or file count cap exceeded"
                )
            names.add(name)
            if member.isdir():
                continue
            if not member.isfile() or member.issparse() or name not in inventory:
                raise ValueError("Harbor archive must match its regular file inventory")
            row = inventory[name]
            if member.size != row["size_bytes"]:
                raise ValueError("Harbor archive byte cap or size mismatch")
            with archive.extractfile(member) as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
            if (
                len(data) != member.size
                or hashlib.sha256(data).hexdigest() != row["sha256"]
            ):
                raise ValueError("Harbor archive file hash or size mismatch")
            payloads[name] = (data, member.mode & 0o777)
    if set(payloads) != set(inventory):
        raise ValueError("Harbor archive file inventory is incomplete")
    return payloads


async def _acquire_harbor(candidate, cache_dir, revision, max_tasks):
    registry_type, task_id_type, db_type = _harbor_api()
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
    db = db_type()
    shared = {}
    if bundle is not None and bundle.files:
        shared = _harbor_inventory(
            await db.get_dataset_version_files(bundle.dataset_version_id)
        )
        if set(shared) != {file.path for file in bundle.files}:
            raise ValueError("Harbor shared file inventory is incomplete")
    packages = []
    sizes = [row["size_bytes"] for row in shared.values()]
    for task in tasks:
        resolved = await db.resolve_task_version(task.org, task.name, task.ref)
        if resolved.content_hash != task.ref.removeprefix("sha256:"):
            raise ValueError("Harbor task resolution disagrees with its digest")
        inventory = _harbor_inventory(await db.get_task_version_files(resolved.id))
        if "task.toml" not in inventory:
            raise ValueError("Harbor package contains no root task.toml")
        sizes.extend(row["size_bytes"] for row in inventory.values())
        _check_sizes(sizes)
        packages.append((task, resolved, inventory))
    root = _fresh(cache_dir, "harbor", candidate["id"], revision)
    dataset_files, records = {}, []
    remaining = MAX_TOTAL_BYTES
    with tempfile.TemporaryDirectory(prefix="openenv-harbor-acquisition-") as temporary:
        staging = Path(temporary)
        for index, (name, row) in enumerate(shared.items()):
            path = staging / f"shared-{index}"
            remaining -= await _download_harbor_file(
                row["storage_path"], path, row["size_bytes"]
            )
            data = path.read_bytes()
            if (
                len(data) != row["size_bytes"]
                or hashlib.sha256(data).hexdigest() != row["sha256"]
            ):
                raise ValueError("Harbor shared file hash or size mismatch")
            destination = root / "dataset" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            dataset_files[name] = str(destination)
        for index, (task, resolved, inventory) in enumerate(packages):
            archive = staging / f"task-{index}.tar.gz"
            remaining -= await _download_harbor_file(
                resolved.archive_path, archive, remaining
            )
            payloads = _read_harbor_archive(archive, inventory)
            task_root = root / "tasks" / task.org / task.name / resolved.content_hash
            for name, (data, mode) in payloads.items():
                path = task_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                path.chmod(mode)
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
                        "dataset_files": dataset_files,
                    },
                    task_file_hashes={
                        name: row["sha256"] for name, row in inventory.items()
                    },
                )
            )
    return records
