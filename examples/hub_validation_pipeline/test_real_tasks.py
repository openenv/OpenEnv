"""CPU-only operator-panel, acquisition and SDK transfer-contract checks."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from examples.coding_audit.hf import HFTaskBackend
from examples.hub_validation_pipeline import cloud, real_tasks
from huggingface_hub import RepoFile, RepoFolder
from huggingface_hub.errors import SandboxError
from openenv.validation.runner import source_digest


TOY_NAMES = ("toy-alpha", "toy-beta")
TOY_SOURCE = {"kind": "hf", "repo_id": "example/toy-tasks", "revision": "b" * 40}


def toy_config(name=TOY_NAMES[0]):
    config = cloud._loads(
        (Path(__file__).parents[1] / "coding_audit/config.json").read_bytes()
    )
    config["execution"].update(
        agent_image="example/toy@sha256:" + "c" * 64,
        verifier_image="example/toy@sha256:" + "c" * 64,
        submission_paths=[f"/app/{name}.txt"],
        timeout_seconds=60,
        max_tool_output_bytes=1048576,
    )
    config["negative_cases"] = [
        {"id": "toy-empty", "script": f": > /app/{name}.txt"},
        {"id": "toy-wrong", "script": f"printf pear > /app/{name}.txt"},
    ]
    config["validation"].update(
        required_negative_cases=[case["id"] for case in config["negative_cases"]],
        target_attempts=2,
        reference_repeats=2,
    )
    config["seeds"] = [17, 29]
    config["generation"]["candidate_count"] = 1
    config["target"].update(max_steps=3, max_tokens=64, max_total_tokens=256)
    config["generator"] = dict(config["target"], max_steps=1)
    config["cloud"].update(results_repo="example/results", timeout_seconds=60)
    return cloud._config(config)[0]


def write_panel(directory, names=TOY_NAMES):
    tasks = []
    for name in names:
        config = toy_config(name)
        tasks.append(
            {
                "name": name,
                "task_path": f"fixtures/{name}",
                "image": config["execution"]["agent_image"],
                "submission_path": config["execution"]["submission_paths"][0],
                "config": config,
                "negative_counterexamples": [{"id": "toy-wrong", "expected": "apple"}],
                "annotations": {"caveat": "Synthetic fixture only"},
            }
        )
    path = directory / "panel.json"
    path.write_bytes(
        cloud._json_bytes(
            {
                "schema": "openenv-native-evaluation-panel-v1",
                "evaluation_only": True,
                "source": TOY_SOURCE,
                "tasks": tasks,
            }
        )
    )
    return path


def source_files(name):
    return {
        "task.toml": (
            '[task]\nname = "toy/' + name + '"\n'
            "[agent]\ntimeout_sec = 900.0\n[verifier]\ntimeout_sec = 900.0\n"
            '[environment]\ndocker_image = "example/' + name + ':fixture"\n'
            "cpus = 1\nmemory_mb = 2048\nstorage_mb = 10240\ngpus = 0\n"
        ).encode(),
        "instruction.md": b"Public task instruction\n",
        "solution/solve.sh": b"#!/bin/bash\necho native-reference\n",
        "tests/test.sh": b"#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n",
        "tests/test_outputs.py": b"# native verifier asset\n",
        "environment/fixture/task.toml": b"nested asset, never a separate task\n",
    }


def hf_fakes(files=None):
    files = (
        files
        if files is not None
        else {
            f"fixtures/{name}/{path}": data
            for name in TOY_NAMES
            for path, data in source_files(name).items()
        }
    )
    api = Mock()
    api.dataset_info.return_value = SimpleNamespace(
        sha=TOY_SOURCE["revision"], gated=False, disabled=False
    )

    def listing(repo_id, **kwargs):
        prefix = kwargs["path_in_repo"] + "/"
        yield RepoFolder(path=prefix + "tests", oid="0" * 40)
        for path, data in files.items():
            if path.startswith(prefix):
                yield RepoFile(
                    path=path,
                    size=len(data),
                    oid=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
                )

    api.list_repo_tree.side_effect = listing

    def download(repo_id, filename, **kwargs):
        destination = kwargs["local_dir"] / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(files[filename])
        return str(destination)

    return api, Mock(side_effect=download)


@pytest.fixture(autouse=True)
def no_live_network(monkeypatch):
    import socket

    monkeypatch.setattr(
        socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden"))
    )
    monkeypatch.setattr(
        cloud, "submit", Mock(side_effect=AssertionError("submission forbidden"))
    )
    for name in cloud.SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_prepare_pinned_selected_native_trees_and_consumable_manifest(tmp_path):
    api, download = hf_fakes()
    panel_path = write_panel(tmp_path)
    panel = cloud._loads(panel_path.read_bytes())
    output = tmp_path / "recipes"
    manifest = real_tasks.prepare(
        output, panel_path=panel_path, api=api, download=download
    )
    assert manifest == cloud._loads((output / "manifest.json").read_bytes())
    assert manifest["evaluation_only"] is True and manifest["status"] == "prepared"
    assert [row["name"] for row in manifest["tasks"]] == list(TOY_NAMES)
    panel_sha = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    assert manifest["evaluation_panel"] == {
        "path": str(panel_path),
        "sha256": panel_sha,
        "bytes": panel_path.stat().st_size,
    }
    for row, task in zip(manifest["tasks"], panel["tasks"]):
        root = Path(row["task_root"])
        assert row["source"] == {
            **TOY_SOURCE,
            "task_path": task["task_path"],
            "evaluation_panel_sha256": panel_sha,
            "task_file_hashes": {
                path: record["sha256"] for path, record in row["files"].items()
            },
        }
        assert row["source_digest"] == source_digest(root)
        assert row["source_image"].endswith(":fixture")
        assert row["evaluation_only"] is True
        for path, data in source_files(row["name"]).items():
            assert (root / path).read_bytes() == data
            assert not (root / path).is_symlink()
            assert row["files"][path]["sha256"] == hashlib.sha256(data).hexdigest()
        config = cloud._loads(Path(row["config_path"]).read_bytes())
        assert config == task["config"]
        assert row["budget"] == cloud._config(config)[1]
        assert row["negative_counterexamples"] == task["negative_counterexamples"]
        assert row["annotations"] == task["annotations"]
        assert "negative_counterexamples" not in config
    for call in api.list_repo_tree.call_args_list:
        assert call.args[0] == TOY_SOURCE["repo_id"]
        assert call.kwargs["revision"] == TOY_SOURCE["revision"]
        assert call.kwargs["path_in_repo"] in {"fixtures/" + name for name in TOY_NAMES}
        assert call.kwargs["recursive"] is True and call.kwargs["token"] is False
    for call in download.call_args_list:
        assert call.kwargs["revision"] == TOY_SOURCE["revision"]
        assert call.kwargs["repo_type"] == "dataset" and call.kwargs["token"] is False
    with pytest.raises(FileExistsError):
        real_tasks.prepare(output, panel_path=panel_path, api=api, download=download)


def test_bundle_and_cli_pass_exact_preparation_contract(tmp_path, monkeypatch, capsys):
    api, download = hf_fakes()
    panel_path = write_panel(tmp_path)
    prepare = Mock()
    monkeypatch.setattr(cloud, "prepare", prepare)
    monkeypatch.setattr(real_tasks, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(real_tasks, "hf_hub_download", download)
    output = tmp_path / "cli"
    real_tasks.main(
        [
            "--panel",
            str(panel_path),
            "--output",
            str(output),
            "--task",
            "toy-alpha",
            "--bundle",
        ]
    )
    row = json.loads((output / "manifest.json").read_text())["tasks"][0]
    call = prepare.call_args
    assert call.args[0] == Path(row["task_root"])
    assert call.args[1]["cloud"]["results_repo"] == "example/results"
    assert call.kwargs["source"] == row["source"]
    assert row["bundle_plan"] == str(call.args[2] / "plan.json")
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


def test_real_cloud_bundle_roundtrip_preserves_source_and_native_files(tmp_path):
    api, download = hf_fakes()
    panel_path = write_panel(tmp_path)
    manifest = real_tasks.prepare(
        tmp_path / "bundled",
        panel_path=panel_path,
        task_names=["toy-alpha"],
        bundle=True,
        api=api,
        download=download,
    )
    row = manifest["tasks"][0]
    plan, config, data, bundle_manifest = cloud._read_plan(row["bundle_plan"])
    _, files = cloud._checked_archive(data, plan["bundle_sha256"])
    assert plan["task_source"] == bundle_manifest["task_source"] == row["source"]
    assert config["execution"]["agent_image"] == row["execution_image"]
    for path, data in source_files("toy-alpha").items():
        assert files["task/" + path] == data


def test_bundle_rejects_native_file_change_and_restore_after_acquisition(
    tmp_path, monkeypatch
):
    api, download = hf_fakes()
    panel_path = write_panel(tmp_path)
    original_tree = cloud._tree

    def changed_verifier(root, **kwargs):
        root = Path(root)
        if root.name != "toy-alpha":
            return original_tree(root, **kwargs)
        verifier = root / "tests/test.sh"
        original = verifier.read_bytes()
        verifier.write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
        try:
            return original_tree(root, **kwargs)
        finally:
            verifier.write_bytes(original)

    monkeypatch.setattr(cloud, "_tree", changed_verifier)
    output = tmp_path / "recipes"
    with pytest.raises(ValueError, match="acquisition"):
        real_tasks.prepare(
            output,
            panel_path=panel_path,
            task_names=["toy-alpha"],
            bundle=True,
            api=api,
            download=download,
        )
    assert not (output / "bundles/toy-alpha/plan.json").exists()


@pytest.mark.parametrize(
    "case",
    [
        "revision",
        "gated",
        "escape",
        "environment",
        "file-cap",
        "total-cap",
        "hash",
        "symlink",
        "directory",
        "missing",
    ],
)
def test_acquisition_rejects_bad_source_before_materializing(
    tmp_path, monkeypatch, case
):
    name = "toy-alpha"
    api, download = hf_fakes()
    panel_path = write_panel(tmp_path)
    if case == "revision":
        api.dataset_info.return_value.sha = "main"
    elif case == "gated":
        api.dataset_info.return_value.gated = "manual"
    elif case in ("escape", "environment"):
        path = (
            "fixtures/elsewhere/task.toml"
            if case == "escape"
            else f"fixtures/{name}/.env"
        )
        api.list_repo_tree.side_effect = None
        api.list_repo_tree.return_value = [RepoFile(path=path, size=0, oid="0" * 40)]
    elif case in ("file-cap", "total-cap"):
        monkeypatch.setattr(
            real_tasks, "MAX_FILE_BYTES" if case == "file-cap" else "MAX_TOTAL_BYTES", 1
        )
    elif case == "missing":
        api.list_repo_tree.side_effect = None
        api.list_repo_tree.return_value = []
    else:
        original = download.side_effect

        def changed(*args, **kwargs):
            path = Path(original(*args, **kwargs))
            if case == "hash":
                path.write_bytes(b"x" * path.stat().st_size)
            elif case == "symlink":
                target = tmp_path / "outside"
                target.write_text("outside")
                path.unlink()
                path.symlink_to(target)
            else:
                path.unlink()
                path.mkdir()
            return path

        download.side_effect = changed
    destination = tmp_path / "task"
    with pytest.raises(ValueError):
        panel, _ = real_tasks._load_panel(panel_path)
        real_tasks.acquire_task(
            panel["tasks"][0],
            destination,
            source=panel["source"],
            api=api,
            download=download,
        )
    assert not destination.exists()
    if case in (
        "revision",
        "gated",
        "escape",
        "environment",
        "file-cap",
        "total-cap",
        "missing",
    ):
        download.assert_not_called()


@pytest.mark.parametrize("name", TOY_NAMES)
def test_same_image_fresh_solver_grader_transfer_with_sdk_fakes(tmp_path, name):
    api, download = hf_fakes()
    panel_path = write_panel(tmp_path)
    root = tmp_path / "native"
    panel, _ = real_tasks._load_panel(panel_path)
    task = next(task for task in panel["tasks"] if task["name"] == name)
    real_tasks.acquire_task(
        task, root, source=panel["source"], api=api, download=download
    )
    original_digest = source_digest(root)
    config = toy_config(name)
    execution = config["execution"]
    exported = task["submission_path"]
    submission_bytes = b"candidate file bytes\n"
    verifier_output = "APT: fetching native verifier dependencies 雪\n" * 2048
    assert 16384 < len(verifier_output.encode()) < 1048576
    jobs, created = {}, []
    jobs_api = Mock()

    def inspect_job(*, job_id, **kwargs):
        return SimpleNamespace(
            id=job_id,
            docker_image=execution["agent_image"],
            status=SimpleNamespace(stage=jobs[job_id]),
            owner=SimpleNamespace(name="owner"),
        )

    jobs_api.inspect_job.side_effect = inspect_job
    jobs_api.cancel_job.side_effect = lambda *, job_id, **kwargs: jobs.update(
        {job_id: "CANCELED"}
    )

    def factory(**kwargs):
        assert kwargs["image"] == execution["agent_image"]
        assert kwargs["forward_hf_token"] is False
        assert (
            "env" not in kwargs and "secrets" not in kwargs and "volumes" not in kwargs
        )
        identifier = f"fake-{len(created)}"
        jobs[identifier] = "RUNNING"
        data = {}
        directories = {"/", "/tmp"}

        def stat_file(path):
            if path in directories:
                return SimpleNamespace(type="dir", size=0)
            if path not in data:
                raise SandboxError("missing", status_code=404)
            return SimpleNamespace(type="file", size=len(data[path]))

        files = SimpleNamespace(
            MAX_READ_BYTES=1048576,
            data=data,
            stat=stat_file,
            exists=lambda path: path in data or path in directories,
            read=lambda path: data[path],
            write=lambda path, raw, **kwargs: data.update(
                {path: raw.encode() if isinstance(raw, str) else raw}
            ),
            mkdir=directories.add,
            delete=lambda path: data.pop(path),
        )

        def run(argv, **kwargs):
            assert kwargs["cwd"] == "/app"
            assert 0 < kwargs["timeout"] <= 60
            if argv == ["/bin/bash", "/tests/test.sh"]:
                assert data[exported] == submission_bytes
                assert data["/tests/test.sh"] == source_files(name)["tests/test.sh"]
                assert not any(path.startswith("/solution") for path in data)
                kwargs["on_stdout"](verifier_output)
                kwargs["on_stderr"]("Native verifier completed.\n")
                data["/logs/verifier/reward.txt"] = b"0\n"
            else:
                assert not any(
                    path.startswith(("/tests", "/solution")) for path in data
                )
                data[exported] = submission_bytes
                data["/app/not-exported.txt"] = b"not transferred"
            return SimpleNamespace(exit_code=0, timed_out=False)

        sandbox = SimpleNamespace(id=identifier, files=files, run=run)
        created.append(sandbox)
        return sandbox

    backend = HFTaskBackend(
        root, execution, token="execution-only", api=jobs_api, sandbox_factory=factory
    )
    submission = backend.execute_script(
        config["negative_cases"][0]["script"], "negative"
    )
    grade = backend.grade(submission)
    assert submission["error"] is None and grade["error"] is None
    assert submission["files"] == {exported: submission_bytes}
    assert grade["reward"] == 0 and grade["rawreward"] == b"0\n"
    assert grade["transcript"] == [
        {
            "exit_code": 0,
            "output": verifier_output + "Native verifier completed.\n",
            "output_truncated": False,
        }
    ]
    assert submission["cleanup_verified"] and grade["cleanup_verified"]
    assert len(created) == 2 and created[0].id != created[1].id
    assert "/app/not-exported.txt" not in created[1].files.data
    assert source_digest(root) == original_digest


def test_recipe_requires_explicit_panel_before_acquisition(tmp_path, monkeypatch):
    acquisition = Mock(side_effect=AssertionError("No acquisition"))
    monkeypatch.setattr(real_tasks, "HfApi", acquisition)
    with pytest.raises(TypeError, match="panel_path"):
        real_tasks.prepare(tmp_path / "recipes")
    with pytest.raises(SystemExit):
        real_tasks.main(["--output", str(tmp_path / "recipes")])
    acquisition.assert_not_called()
    assert not (tmp_path / "recipes").exists()


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "evaluation",
        "source",
        "repo",
        "revision",
        "empty",
        "duplicate",
        "name",
        "path",
        "image",
        "submission",
        "counterexamples",
        "annotations",
        "credentials",
        "duplicate-json",
    ],
)
def test_invalid_operator_panel_fails_before_acquisition(tmp_path, monkeypatch, case):
    panel_path = write_panel(tmp_path)
    panel = cloud._loads(panel_path.read_bytes())
    task = panel["tasks"][0]
    if case == "schema":
        panel["schema"] = "unknown"
    elif case == "evaluation":
        panel["evaluation_only"] = False
    elif case == "source":
        panel["source"]["kind"] = "local"
    elif case == "repo":
        panel["source"]["repo_id"] = "../escape/name"
    elif case == "revision":
        panel["source"]["revision"] = "main"
    elif case == "empty":
        panel["tasks"] = []
    elif case == "duplicate":
        panel["tasks"].append(task)
    elif case == "name":
        task["name"] = "../escape"
    elif case == "path":
        task["task_path"] = "fixtures/.env-private"
    elif case == "image":
        task["image"] = "example/other@sha256:" + "d" * 64
    elif case == "submission":
        task["submission_path"] = "/app/other.txt"
    elif case == "counterexamples":
        task["negative_counterexamples"] = "invalid"
    elif case == "annotations":
        task["annotations"] = []
    elif case == "credentials":
        task["annotations"] = {"credential": "hf_" + "Z" * 32}
    panel_path.write_bytes(cloud._json_bytes(panel))
    if case == "duplicate-json":
        panel_path.write_bytes(
            panel_path.read_bytes().replace(
                b'"evaluation_only":true',
                b'"evaluation_only":true,"evaluation_only":true',
            )
        )
    acquisition = Mock(side_effect=AssertionError("No acquisition"))
    monkeypatch.setattr(real_tasks, "HfApi", acquisition)
    with pytest.raises(ValueError):
        real_tasks.prepare(tmp_path / "recipes", panel_path=panel_path)
    acquisition.assert_not_called()
    assert not (tmp_path / "recipes").exists()


def test_recipe_hashes_exact_panel_bytes_and_loads_only_once(tmp_path, monkeypatch):
    panel_path = write_panel(tmp_path)
    # A noncanonical encoding must be hashed as supplied, rather than reserialized.
    original_bytes = b"  " + panel_path.read_bytes() + b"\n\n"
    panel_path.write_bytes(original_bytes)
    read_file = cloud._text_file
    reads = []

    def replace_after_read(path):
        result = read_file(path)
        if path == panel_path:
            reads.append(path)
            panel_path.write_text('{"replacement": true}')
        return result

    monkeypatch.setattr(cloud, "_text_file", replace_after_read)
    api, download = hf_fakes()
    manifest = real_tasks.prepare(
        tmp_path / "recipes", panel_path=panel_path, api=api, download=download
    )
    expected = hashlib.sha256(original_bytes).hexdigest()
    assert reads == [panel_path]
    assert manifest["evaluation_panel"]["sha256"] == expected
    assert manifest["evaluation_panel"]["bytes"] == len(original_bytes)
    assert all(
        row["source"]["evaluation_panel_sha256"] == expected
        for row in manifest["tasks"]
    )
    assert Path(manifest["tasks"][0]["config_path"]).read_bytes() == cloud._json_bytes(
        toy_config()
    )
