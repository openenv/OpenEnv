"""Network-free checks for discovery, immutable locators, and task assets."""

import asyncio
import hashlib
import io
import json
import tarfile
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from examples.hub_validation_pipeline import real_tasks, sources
from huggingface_hub import RepoFile

SHA = "a" * 40
DIGEST = "sha256:" + "b" * 64
TASK_DIGEST = "sha256:" + "c" * 64


def hf_candidate(**metadata):
    return {"source": "hf", "id": "org/bundle", "revision": SHA, "metadata": metadata}


def fake_hf(monkeypatch, files):
    api = Mock()
    api.dataset_info.return_value = SimpleNamespace(
        sha=SHA, gated=False, disabled=False
    )
    api.list_repo_files.return_value = list(files)
    api.list_repo_tree.return_value = [
        RepoFile(
            path=name,
            size=len(content.encode()),
            oid=hashlib.sha1(
                f"blob {len(content.encode())}\0".encode() + content.encode()
            ).hexdigest(),
        )
        for name, content in files.items()
    ]
    factory = Mock(return_value=api)
    monkeypatch.setattr(sources, "HfApi", factory)

    def download(repo_id, **kwargs):
        root = kwargs["local_dir"]
        for name, content in files.items():
            if any(fnmatchcase(name, pattern) for pattern in kwargs["allow_patterns"]):
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)
        return str(root)

    downloader = Mock(side_effect=download)
    monkeypatch.setattr(sources, "snapshot_download", downloader)
    return api, factory, downloader


def test_hf_exact_conjunction_trending_and_exhausted_pagination(monkeypatch):
    api = Mock()
    consumed = []

    def pages():
        for number, tags in enumerate(
            [
                ["harbor", "rl-environment"],
                ["library:harbor"],
                ["library:harbor", "rl-environment"],
                ["library:harbor", "rl-environment"],
            ]
        ):
            consumed.append(number)
            yield SimpleNamespace(
                id=f"org/{number}", sha=SHA, tags=tags, gated=False, disabled=False
            )

    api.list_datasets.return_value = pages()
    monkeypatch.setattr(sources, "HfApi", Mock(return_value=api))
    candidates = sources.discover_hf(1)
    assert [candidate["id"] for candidate in candidates] == ["org/2"]
    assert candidates[0]["revision"] == SHA
    assert consumed == [0, 1, 2, 3]
    api.list_datasets.assert_called_once_with(
        filter=("library:harbor", "rl-environment"),
        sort="trending_score",
        expand=["sha", "tags", "gated", "disabled"],
        limit=None,
    )


def test_hf_bounded_assets_pinned_and_fresh(monkeypatch, tmp_path):
    files = {"registry.json": "{}", "shared/data[train].txt": "shared"}
    for number in range(5000):
        prefix = f"environment/tasks/{number:04d}/"
        files[prefix + "task.toml"] = "version = '1.0'"
        files[prefix + "instruction.md"] = "do a task"
        files[prefix + "environment/assets/data.txt"] = "actual asset"
    api, factory, downloader = fake_hf(monkeypatch, files)
    first = sources.acquire(hf_candidate(), tmp_path, max_tasks=1)[0]
    second = sources.acquire(hf_candidate(), tmp_path, max_tasks=1)[0]
    assert first["task_id"] == f"hf:org/bundle@{SHA}:environment/tasks/0000"
    assert first["task_id"] == second["task_id"]
    assert first["path"] != second["path"]
    path = Path(first["path"])
    assert path.is_absolute() and path == path.resolve()
    assert (path / "environment/assets/data.txt").read_text() == "actual asset"
    assert not (path.parent / "0001").exists()
    root = path.parents[2]
    assert (root / "shared/data[train].txt").read_text() == "shared"
    assert len(first["download_patterns"]) == 5
    assert first["revision"] == SHA and first["source_id"] == "org/bundle"
    assert SHA in path.parts
    factory.assert_called_with(token=False)
    api.dataset_info.assert_called_with("org/bundle", revision=SHA)
    api.list_repo_tree.assert_called_with(
        "org/bundle", repo_type="dataset", revision=SHA, recursive=True, token=False
    )
    assert downloader.call_args.kwargs["revision"] == SHA
    assert downloader.call_args.kwargs["repo_type"] == "dataset"
    assert downloader.call_args.kwargs["force_download"] is True


@pytest.mark.parametrize("prefix", ["", "environment/tasks/solo/"])
def test_hf_individual_and_nested_tasks(monkeypatch, tmp_path, prefix):
    fake_hf(monkeypatch, {prefix + "task.toml": "", prefix + "instruction.md": ""})
    [record] = sources.acquire(hf_candidate(), tmp_path, 1)
    assert record["task_path"] == (prefix.rstrip("/") or ".")
    assert (Path(record["path"]) / "instruction.md").is_file()


def test_hf_same_basename_has_distinct_identity(monkeypatch, tmp_path):
    fake_hf(monkeypatch, {"a/same/task.toml": "", "b/same/task.toml": ""})
    records = sources.acquire(hf_candidate(), tmp_path, 2)
    assert len({record["task_id"] for record in records}) == 2
    assert {record["task_path"] for record in records} == {"a/same", "b/same"}


def test_hf_gated_malformed_and_floating_sources_fail(monkeypatch, tmp_path):
    api, factory, download = fake_hf(monkeypatch, {"README.md": "no tasks"})
    with pytest.raises(PermissionError, match="gated"):
        sources.acquire(hf_candidate(gated="manual"), tmp_path, 1)
    factory.assert_not_called()
    with pytest.raises(ValueError, match="No task.toml"):
        sources.acquire(hf_candidate(), tmp_path, 1)
    download.assert_not_called()
    candidate = {**hf_candidate(), "revision": "main"}
    with pytest.raises(ValueError, match="immutable"):
        sources.acquire(candidate, tmp_path, 1)
    api.dataset_info.return_value.sha = "d" * 40
    with pytest.raises(ValueError, match="disagrees"):
        sources.acquire(hf_candidate(), tmp_path, 1)


def test_unsafe_source_paths_and_symlinks_fail(monkeypatch, tmp_path):
    fake_hf(monkeypatch, {"../escape/task.toml": ""})
    with pytest.raises(ValueError, match="Unsafe source path"):
        sources.acquire(hf_candidate(), tmp_path, 1)
    outside = tmp_path / "outside"
    outside.write_text("original")
    _, _, download = fake_hf(monkeypatch, {"task.toml": "original"})

    def link(repo, **kwargs):
        (kwargs["local_dir"] / "task.toml").symlink_to(outside)

    download.side_effect = link
    with pytest.raises(ValueError, match="contained regular file"):
        sources.acquire(hf_candidate(), tmp_path / "cache", 1)
    assert outside.read_text() == "original"


def test_hf_acquisition_rejects_contained_links(monkeypatch, tmp_path):
    _, _, download = fake_hf(monkeypatch, {"task.toml": "original"})

    def link(repo, **kwargs):
        root = kwargs["local_dir"]
        (root / "actual").write_text("original")
        (root / "task.toml").symlink_to(root / "actual")

    download.side_effect = link
    with pytest.raises(ValueError, match="contained regular file"):
        sources.acquire(hf_candidate(), tmp_path, 1)


class PackageTaskId:
    def __init__(self, org, name, ref):
        self.org, self.name, self.ref = org, name, ref


def fake_harbor(monkeypatch):
    tasks = [
        PackageTaskId("a", "same", TASK_DIGEST),
        PackageTaskId("b", "same", "sha256:" + "d" * 64),
    ]
    registry = SimpleNamespace(
        list_datasets=AsyncMock(
            return_value=[
                SimpleNamespace(
                    name="org/dataset",
                    version=DIGEST,
                    description="bundle",
                    task_count=2,
                )
            ]
        ),
        get_dataset_metadata=AsyncMock(
            return_value=SimpleNamespace(version=DIGEST, task_ids=tasks, files=[])
        ),
        download_dataset_files=AsyncMock(return_value={}),
    )

    payloads = {
        "task.toml": b"",
        "instruction.md": b"task",
        "environment/assets/data": b"asset",
    }

    def rows():
        return [
            dict(
                path=name,
                size_bytes=len(data),
                content_hash=hashlib.sha256(data).hexdigest(),
            )
            for name, data in client.payloads.items()
        ]

    async def resolve(org, name, ref):
        return SimpleNamespace(
            id=org + "/" + name,
            content_hash=ref.removeprefix("sha256:"),
            archive_path=org + "/" + name,
        )

    async def bounded_download(remote_path, destination, byte_limit):
        with tarfile.open(destination, "w:gz") as archive:
            for name, data in client.payloads.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            for member, data in client.extra_members:
                archive.addfile(member, io.BytesIO(data))
        return destination.stat().st_size

    client = SimpleNamespace(
        payloads=payloads,
        extra_members=[],
        download_tasks=AsyncMock(
            side_effect=AssertionError("Unbounded task download forbidden")
        ),
        resolve_task_version=AsyncMock(side_effect=resolve),
        get_task_version_files=AsyncMock(side_effect=lambda *args: rows()),
        get_dataset_version_files=AsyncMock(return_value=[]),
        download=AsyncMock(side_effect=bounded_download),
    )
    monkeypatch.setattr(
        sources, "_download_harbor_file", client.download, raising=False
    )
    monkeypatch.setattr(
        sources,
        "_harbor_api",
        lambda: (Mock(return_value=registry), PackageTaskId, Mock(return_value=client)),
    )
    return registry, client


def test_harbor_discovery_and_bounded_digest_acquisition(monkeypatch, tmp_path):
    registry, client = fake_harbor(monkeypatch)
    [candidate] = sources.discover_harbor(1)
    assert candidate["revision"] == DIGEST
    [record] = sources.acquire(candidate, tmp_path, 1)
    registry.get_dataset_metadata.assert_awaited_once_with(f"org/dataset@{DIGEST}")
    client.resolve_task_version.assert_awaited_once_with("a", "same", TASK_DIGEST)
    client.download.assert_awaited_once()
    client.download_tasks.assert_not_awaited()
    assert record["revision"] == TASK_DIGEST
    assert record["task_id"] == f"harbor:a/same@{TASK_DIGEST}:."
    assert record["metadata"]["candidate_id"] == "org/dataset"
    assert record["metadata"]["candidate_revision"] == DIGEST
    assert (Path(record["path"]) / "environment/assets/data").read_text() == "asset"


@pytest.mark.parametrize("missing_root", [False, True])
def test_harbor_nested_fixture_cannot_replace_package_root(
    monkeypatch, tmp_path, missing_root
):
    _, client = fake_harbor(monkeypatch)
    client.payloads["environment/fixture/task.toml"] = b"fixture"
    if missing_root:
        del client.payloads["task.toml"]
    [candidate] = sources.discover_harbor(1)
    if missing_root:
        with pytest.raises(ValueError, match="no root task.toml"):
            sources.acquire(candidate, tmp_path, 1)
    else:
        [record] = sources.acquire(candidate, tmp_path, 1)
        assert record["task_path"] == "."
        assert (Path(record["path"]) / "environment/fixture/task.toml").is_file()


def test_hf_nested_fixture_remains_parent_task_asset(monkeypatch, tmp_path):
    fake_hf(
        monkeypatch,
        {
            "tasks/parent/task.toml": "parent",
            "tasks/parent/environment/fixture/task.toml": "fixture",
            "tasks/sibling/task.toml": "sibling",
        },
    )
    records = sources.acquire(hf_candidate(), tmp_path, 2)
    assert [row["task_path"] for row in records] == ["tasks/parent", "tasks/sibling"]
    assert (
        Path(records[0]["path"]) / "environment/fixture/task.toml"
    ).read_text() == "fixture"


def test_harbor_single_task_and_cross_org_identity(monkeypatch, tmp_path):
    registry, _ = fake_harbor(monkeypatch)
    candidate = {
        "source": "harbor",
        "id": "a/same",
        "revision": TASK_DIGEST,
        "metadata": {"kind": "task"},
    }
    [record] = sources.acquire(candidate, tmp_path, 1)
    registry.get_dataset_metadata.assert_not_awaited()
    assert record["source_id"] == "a/same" and record["revision"] == TASK_DIGEST
    [dataset] = sources.discover_harbor(1)
    records = sources.acquire(dataset, tmp_path, 2)
    assert len({record["task_id"] for record in records}) == 2
    assert {record["source_id"] for record in records} == {"a/same", "b/same"}


def test_harbor_inaccessible_or_floating_tasks_fail(monkeypatch, tmp_path):
    registry, client = fake_harbor(monkeypatch)
    [candidate] = sources.discover_harbor(1)
    registry.get_dataset_metadata.side_effect = PermissionError("unavailable task")
    with pytest.raises(PermissionError, match="unavailable"):
        sources.acquire(candidate, tmp_path, 1)
    client.download_tasks.assert_not_awaited()
    registry.get_dataset_metadata.side_effect = None
    registry.get_dataset_metadata.return_value.task_ids[0].ref = "latest"
    with pytest.raises(ValueError, match="immutable"):
        sources.acquire(candidate, tmp_path, 1)
    client.download_tasks.assert_not_awaited()


def test_harbor_wheel_mismatch_reports_pinned_install(monkeypatch):
    # The PyPI 0.25.0 wheel shares the version string but lacks native listing.
    monkeypatch.setattr(
        sources,
        "distribution",
        lambda name: SimpleNamespace(version="0.25.0", read_text=lambda name: None),
    )
    with pytest.raises(RuntimeError, match="source capability mismatch") as error:
        sources.discover_harbor(1)
    assert sources.HARBOR_INSTALL in str(error.value)
    assert sources.HARBOR_GIT_REVISION in str(error.value)


def test_harbor_unsupported_listing_reports_install(monkeypatch):
    registry, _ = fake_harbor(monkeypatch)
    registry.list_datasets.side_effect = NotImplementedError("not yet supported")
    with pytest.raises(RuntimeError, match="listing is unsupported") as error:
        sources.discover_harbor(1)
    assert sources.HARBOR_INSTALL in str(error.value)


def test_harbor_native_cli_json_and_model_contract(monkeypatch, capsys):
    # Exercise the real Git-pinned API/CLI conversion, mocking only its DB read.
    pytest.importorskip("harbor")
    from harbor.cli.datasets import list_datasets
    from harbor.db.client import RegistryDB

    native_rows = [
        {
            "content_hash": "b" * 64,
            "description": "bundle",
            "package": {"org": {"name": "org"}, "name": "dataset"},
            "dataset_version_task": [{"count": 2}],
        },
        {
            "content_hash": "d" * 64,
            "description": None,
            "package": {"org": {"name": "z"}, "name": "files-only"},
            "dataset_version_task": [],
        },
    ]
    # A wheel without this method must fail this contract, rather than pass mocks.
    monkeypatch.setattr(
        RegistryDB, "list_dataset_versions", AsyncMock(return_value=native_rows)
    )
    list_datasets(as_json=True)
    cli_rows = json.loads(capsys.readouterr().out)
    assert cli_rows == [
        {
            "name": "org/dataset",
            "version": DIGEST,
            "description": "bundle",
            "task_count": 2,
        },
        {
            "name": "z/files-only",
            "version": "sha256:" + "d" * 64,
            "description": "",
            "task_count": 0,
        },
    ]
    [candidate] = sources.discover_harbor(1)
    assert candidate["id"] == cli_rows[0]["name"]
    assert candidate["revision"] == cli_rows[0]["version"]
    assert candidate["metadata"]["task_count"] == cli_rows[0]["task_count"]


def test_harbor_native_acquisition_models(monkeypatch, tmp_path):
    pytest.importorskip("harbor")
    from harbor.db.client import RegistryDB, ResolvedTaskVersion
    from harbor.models.registry import DatasetMetadata
    from harbor.models.task.id import PackageTaskId as NativeTaskId
    from harbor.registry.client.package import PackageDatasetClient
    from harbor.tasks.client import TaskClient

    task = NativeTaskId(org="a", name="same", ref=TASK_DIGEST)
    bundle = DatasetMetadata(
        name="org/dataset", version=DIGEST, description="", task_ids=[task]
    )
    monkeypatch.setattr(
        PackageDatasetClient, "get_dataset_metadata", AsyncMock(return_value=bundle)
    )
    monkeypatch.setattr(
        RegistryDB,
        "resolve_task_version",
        AsyncMock(
            return_value=ResolvedTaskVersion(
                id="version-id",
                archive_path="archive",
                content_hash=TASK_DIGEST.removeprefix("sha256:"),
            )
        ),
    )
    payloads = {"task.toml": b"", "instruction.md": b"native task"}
    monkeypatch.setattr(
        RegistryDB,
        "get_task_version_files",
        AsyncMock(
            return_value=[
                dict(
                    path=name,
                    size_bytes=len(data),
                    content_hash=hashlib.sha256(data).hexdigest(),
                    storage_path=None,
                )
                for name, data in payloads.items()
            ]
        ),
    )
    unbounded = AsyncMock(
        side_effect=AssertionError("Unbounded archive downloader forbidden")
    )
    monkeypatch.setattr(TaskClient, "download_tasks", unbounded)

    async def download(remote_path, destination, byte_limit):
        assert remote_path == "archive" and byte_limit == real_tasks.MAX_TOTAL_BYTES
        with tarfile.open(destination, "w:gz") as archive:
            for name, data in payloads.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return destination.stat().st_size

    monkeypatch.setattr(sources, "_download_harbor_file", download)
    [record] = sources.acquire(
        {"source": "harbor", "id": "org/dataset", "revision": DIGEST}, tmp_path, 1
    )
    assert record["revision"] == TASK_DIGEST
    assert (Path(record["path"]) / "instruction.md").read_text() == "native task"
    assert record["task_file_hashes"] == {
        name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()
    }
    unbounded.assert_not_awaited()


@pytest.mark.parametrize(
    "case", ["file_bytes", "total_bytes", "file_count", "unknown_size"]
)
def test_hf_acquisition_caps_shared_assets_before_download(monkeypatch, tmp_path, case):
    files = {"task/task.toml": "", "shared/large": "x"}
    if case == "file_count":
        files.update({f"shared/{index}": "" for index in range(real_tasks.MAX_FILES)})
    if case == "total_bytes":
        files.update({f"shared/{index}": "" for index in range(5)})
    api, _, download = fake_hf(monkeypatch, files)
    if case == "file_bytes":
        api.list_repo_tree.return_value[1].size = 1024**3
    elif case == "total_bytes":
        for item in api.list_repo_tree.return_value[1:]:
            item.size = real_tasks.MAX_FILE_BYTES
    elif case == "unknown_size":
        api.list_repo_tree.return_value[1].size = None
    with pytest.raises(ValueError, match="cap"):
        sources.acquire(hf_candidate(), tmp_path, 1)
    download.assert_not_called()


def test_hf_acquisition_requires_pinned_blob_bytes(monkeypatch, tmp_path):
    api, _, download = fake_hf(monkeypatch, {"task/task.toml": "original"})
    api.list_repo_tree.return_value[0].blob_id = "0" * 40
    with pytest.raises(ValueError, match="blob"):
        sources.acquire(hf_candidate(), tmp_path, 1)
    download.assert_called_once()


@pytest.mark.parametrize(
    "case", ["file_bytes", "total_bytes", "file_count", "unknown_size"]
)
@pytest.mark.parametrize("scope", ["task", "shared"])
def test_harbor_caps_inventory_before_download(monkeypatch, tmp_path, case, scope):
    registry, client = fake_harbor(monkeypatch)
    rows = [dict(path="task.toml", size_bytes=0, content_hash="a" * 64)]
    if case == "file_bytes":
        rows[0]["size_bytes"] = 1024**3
    elif case == "total_bytes":
        rows = [
            dict(rows[0], path=f"file-{index}", size_bytes=real_tasks.MAX_FILE_BYTES)
            for index in range(5)
        ]
    elif case == "file_count":
        rows = [
            dict(rows[0], path=f"file-{index}")
            for index in range(real_tasks.MAX_FILES + 1)
        ]
    else:
        rows[0]["size_bytes"] = None
    if scope == "task":
        client.get_task_version_files.side_effect = None
        client.get_task_version_files.return_value = rows
    else:
        registry.get_dataset_metadata.return_value.files = [
            SimpleNamespace(path="shared")
        ]
        registry.get_dataset_metadata.return_value.dataset_version_id = "dataset-id"
        client.get_dataset_version_files.return_value = rows
    [candidate] = sources.discover_harbor(1)
    with pytest.raises(ValueError, match="cap"):
        sources.acquire(candidate, tmp_path, 1)
    client.download.assert_not_awaited()
    client.download_tasks.assert_not_awaited()
    registry.download_dataset_files.assert_not_awaited()


@pytest.mark.parametrize(
    "case", ["extra", "link", "oversized", "hash", "missing", "expanded"]
)
def test_harbor_archive_must_match_bounded_inventory(monkeypatch, tmp_path, case):
    _, client = fake_harbor(monkeypatch)
    if case == "expanded":
        member = tarfile.TarInfo("padding")
        member.size = real_tasks.MAX_TOTAL_BYTES + 1
        client.extra_members = [(member, b"x" * member.size)]
    elif case in {"extra", "link"}:
        member = tarfile.TarInfo("unexpected")
        if case == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/outside"
        client.extra_members = [(member, b"")]
    else:
        original = client.get_task_version_files.side_effect
        rows = original("version")
        if case == "oversized":
            client.payloads["instruction.md"] = b"x" * (real_tasks.MAX_FILE_BYTES + 1)
        elif case == "hash":
            client.payloads["instruction.md"] = b"evil"
        else:
            del client.payloads["instruction.md"]
        client.get_task_version_files.side_effect = None
        client.get_task_version_files.return_value = rows
    [candidate] = sources.discover_harbor(1)
    with pytest.raises(ValueError, match="Harbor archive"):
        sources.acquire(candidate, tmp_path, 1)
    assert not list(tmp_path.rglob("instruction.md"))


@pytest.mark.parametrize("declared", [True, False])
def test_harbor_download_caps_headers_and_stream(monkeypatch, tmp_path, declared):
    import httpx
    from harbor.auth import client as auth

    storage = SimpleNamespace(
        create_signed_url=AsyncMock(
            return_value={"signedURL": "https://example.invalid/archive"}
        )
    )
    hub = SimpleNamespace(storage=SimpleNamespace(from_=lambda bucket: storage))
    monkeypatch.setattr(
        auth, "create_authenticated_client", AsyncMock(return_value=hub)
    )
    response = MagicMock()
    response.__aenter__.return_value = response
    response.headers = {"content-length": "11"} if declared else {}
    chunks = []

    async def stream(**kwargs):
        for data in [b"123456", b"78901"]:
            chunks.append(data)
            yield data

    response.aiter_bytes = stream
    http = MagicMock()
    http.__aenter__.return_value = http
    http.stream.return_value = response
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: http)
    target = tmp_path / "archive"
    with pytest.raises(ValueError, match="byte cap"):
        asyncio.run(sources._download_harbor_file("archive-path", target, 10))
    if declared:
        assert chunks == [] and not target.exists()
    else:
        assert target.read_bytes() == b"123456"
