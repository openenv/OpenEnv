"""Network-free checks for discovery, immutable locators, and task assets."""

import json
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from examples.hub_validation_pipeline import sources

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
    root = downloader.call_args.kwargs["local_dir"]
    assert (root / "shared/data[train].txt").read_text() == "shared"
    assert len(first["download_patterns"]) == 5
    assert first["revision"] == SHA and first["source_id"] == "org/bundle"
    assert SHA in path.parts
    factory.assert_called_with(token=False)
    api.dataset_info.assert_called_with("org/bundle", revision=SHA)
    api.list_repo_files.assert_called_with(
        "org/bundle", repo_type="dataset", revision=SHA
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
    root = tmp_path / "tree"
    root.mkdir()
    (root / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe task symlink"):
        sources._real_files(root)
    assert outside.read_text() == "original"


def test_contained_links_materialize_as_real_files(tmp_path):
    root = tmp_path / "tree"
    (root / "assets").mkdir(parents=True)
    (root / "assets/data").write_text("real data")
    (root / "linked-assets").symlink_to("assets", target_is_directory=True)
    sources._real_files(root)
    assert (root / "linked-assets/data").read_text() == "real data"
    assert not any(path.is_symlink() for path in root.rglob("*"))


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

    async def download(*, task_ids, output_dir, overwrite, export):
        results = []
        for task in task_ids:
            digest = task.ref.removeprefix("sha256:")
            path = output_dir / task.org / task.name / digest
            (path / "environment/assets").mkdir(parents=True)
            (path / "task.toml").write_text("")
            (path / "instruction.md").write_text("task")
            (path / "environment/assets/data").write_text("asset")
            results.append(SimpleNamespace(path=path, content_hash=digest))
        return SimpleNamespace(results=results)

    task_client = SimpleNamespace(download_tasks=AsyncMock(side_effect=download))
    monkeypatch.setattr(
        sources,
        "_harbor_api",
        lambda: (
            Mock(return_value=registry),
            PackageTaskId,
            Mock(return_value=task_client),
        ),
    )
    return registry, task_client


def test_harbor_discovery_and_bounded_digest_acquisition(monkeypatch, tmp_path):
    registry, client = fake_harbor(monkeypatch)
    [candidate] = sources.discover_harbor(1)
    assert candidate["revision"] == DIGEST
    [record] = sources.acquire(candidate, tmp_path, 1)
    registry.get_dataset_metadata.assert_awaited_once_with(f"org/dataset@{DIGEST}")
    assert len(client.download_tasks.call_args.kwargs["task_ids"]) == 1
    assert client.download_tasks.call_args.kwargs["export"] is False
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
    download = client.download_tasks.side_effect

    async def with_nested_fixture(**kwargs):
        result = await download(**kwargs)
        root = result.results[0].path
        nested = root / "environment/fixture"
        nested.mkdir()
        (nested / "task.toml").write_text("fixture")
        if missing_root:
            (root / "task.toml").unlink()
        return result

    client.download_tasks.side_effect = with_nested_fixture
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
    from harbor.models.registry import DatasetMetadata
    from harbor.models.task.id import PackageTaskId as NativeTaskId
    from harbor.registry.client.package import PackageDatasetClient
    from harbor.tasks.client import BatchDownloadResult, TaskClient, TaskDownloadResult

    task = NativeTaskId(org="a", name="same", ref=TASK_DIGEST)
    bundle = DatasetMetadata(
        name="org/dataset", version=DIGEST, description="", task_ids=[task]
    )
    monkeypatch.setattr(
        PackageDatasetClient, "get_dataset_metadata", AsyncMock(return_value=bundle)
    )

    async def download(self, *, task_ids, output_dir, overwrite, export):
        assert task_ids == [task] and overwrite and not export
        bare_digest = task.ref.removeprefix("sha256:")
        path = output_dir / task.org / task.name / bare_digest
        path.mkdir(parents=True)
        (path / "task.toml").write_text("")
        (path / "instruction.md").write_text("native task")
        return BatchDownloadResult(
            results=[
                TaskDownloadResult(
                    path=path,
                    download_time_sec=0,
                    cached=False,
                    content_hash=bare_digest,
                )
            ],
            total_time_sec=0,
        )

    monkeypatch.setattr(TaskClient, "download_tasks", download)
    [record] = sources.acquire(
        {"source": "harbor", "id": "org/dataset", "revision": DIGEST}, tmp_path, 1
    )
    assert record["revision"] == TASK_DIGEST
    assert (Path(record["path"]) / "instruction.md").read_text() == "native task"
