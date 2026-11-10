"""Offline cloud planning, input integrity, one-shot launch and failure evidence."""

import hashlib
import io
import json
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from examples.hub_validation_pipeline import cloud, full_run, spend

SHA = "a" * 40
UPLOAD_SHA = "b" * 40
SECRETS = {
    "HF_TOKEN": "results-write-credential-123",
    "HF_INFERENCE_TOKEN": "inference-only-credential-456",
    "HF_SANDBOX_TOKEN": "execution-only-credential-789",
}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src/openenv").mkdir(parents=True)
    (repo / "src/openenv/__init__.py").write_text("# source\n")
    (repo / "examples/hub_validation_pipeline").mkdir(parents=True)
    (repo / "examples/hub_validation_pipeline/cloud.py").write_bytes(
        Path(cloud.__file__).read_bytes()
    )
    shutil.copytree(
        Path(cloud.__file__).parents[1] / "coding_audit",
        repo / "examples/coding_audit",
        ignore=shutil.ignore_patterns("test_*", "__pycache__", ".pytest_cache"),
    )
    (repo / "examples/coding_audit/test_excluded.py").write_text(
        "raise AssertionError\n"
    )
    for name in ("pyproject.toml", "README.md", "LICENSE", "uv.lock"):
        (repo / name).write_text(name + "\n")
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "task.toml").write_text("version = '1.0'\n")
    (task / "instruction.md").write_text("Write an answer\n")
    (task / "tests/test.sh").write_text("printf 1\n")
    (task / "solution/solve.sh").write_text("printf answer > /app/answer\n")
    (task / "solution/solve.sh").chmod(0o755)
    config = {
        "schema_version": 1,
        "target": {
            "model_id": full_run.MODEL_SOURCE[0],
            "revision": full_run.MODEL_SOURCE[1],
            "provider": full_run.MODEL_SOURCE[2],
            "max_steps": 3,
        },
        "execution": {
            "flavor": "cpu-basic",
            "agent_image": "registry/agent@sha256:" + "c" * 64,
            "verifier_image": "registry/verifier@sha256:" + "d" * 64,
            "submission_paths": ["/app/answer"],
        },
        "validation": {
            "reference_repeats": 2,
            "target_attempts": 2,
            "required_negative_cases": ["wrong", "empty"],
        },
        "generation": {"candidate_count": 2},
        "negative_cases": [
            {"id": "wrong", "script": "printf wrong > /app/answer"},
            {"id": "empty", "script": "true"},
        ],
        "seeds": [7, 13],
        "cloud": {
            "results_repo": "org/private-results",
            "namespace": "org",
            "flavor": "cpu-basic",
            "timeout_seconds": 1800,
            "max_sandboxes": 40,
        },
    }
    monkeypatch.setattr(
        cloud, "_git_provenance", lambda root: {"commit": SHA, "dirty": True}
    )
    return repo, task, config


def prepared(inputs, tmp_path, name="prepared"):
    repo, task, config = inputs
    output = tmp_path / name
    plan = cloud.prepare(task, config, output, repo_root=repo)
    return output, plan


def launch_options(output, config):
    ledger = output.parent / "spend.json"
    spend.init(ledger, "1000")
    digest = hashlib.sha256((output / "plan.json").read_bytes()).hexdigest()
    cost = full_run.estimate([config])
    spend.reserve(
        ledger,
        output.name,
        cost["reservation_usd_rounded_up"],
        "Reviewed test plan",
        plan_costs={digest: cost["reservation_usd_exact"]},
    )
    return dict(
        spend_ledger=ledger, reservation_id=output.name, expected_plan_sha256=digest
    )


def fake_api():
    api = Mock()
    api.dataset_info.return_value = SimpleNamespace(private=True, sha=SHA)
    api.list_repo_files.return_value = ["corpus/task.toml"]
    api.create_commit.return_value = SimpleNamespace(oid=UPLOAD_SHA)
    api.run_job.return_value = SimpleNamespace(
        id="job-123", url="https://huggingface.co/jobs/org/job-123"
    )
    return api


def worker_root(inputs, tmp_path, monkeypatch):
    output, plan = prepared(inputs, tmp_path)
    root = tmp_path / "extracted"
    cloud.extract_bundle(
        (output / "bundle.zip").read_bytes(), root, plan["bundle_sha256"]
    )
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENENV_RESULTS_REPO", plan["results_repo"])
    return root, plan


def rewrite_archive(data, *, changes=None, additions=None, remove=()):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as old, zipfile.ZipFile(output, "w") as new:
        for member in old.infolist():
            if member.filename not in remove:
                new.writestr(
                    member, (changes or {}).get(member.filename, old.read(member))
                )
        for name, content, mode in additions or []:
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = mode << 16
            new.writestr(member, content)
    return output.getvalue()


def test_prepare_is_deterministic_offline_and_complete(inputs, tmp_path, monkeypatch):
    network = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr(urllib.request, "urlopen", network)
    repo, task, config = inputs
    (repo / "src/openenv/__pycache__").mkdir()
    (repo / "src/openenv/__pycache__/cached.pyc").write_bytes(b"\x00binary")
    (repo / ".env").write_text("never bundle me")
    (repo / ".venv").mkdir()
    output, plan = prepared(inputs, tmp_path)
    second, again = prepared(inputs, tmp_path, "second")
    assert plan == again
    data = (output / "bundle.zip").read_bytes()
    assert data == (second / "bundle.zip").read_bytes()
    assert plan["bundle_sha256"] == hashlib.sha256(data).hexdigest()
    manifest, files = cloud._checked_archive(data, plan["bundle_sha256"])
    assert set(files) == set(manifest["files"]) | {cloud.MANIFEST}
    assert files["uv.lock"] == b"uv.lock\n"
    assert {
        "examples/coding_audit/" + name
        for name in ("__init__.py", "__main__.py", "scoring.py", "hf.py", "runner.py")
    } <= files.keys()
    assert "examples/coding_audit/test_excluded.py" not in files
    assert manifest["git"] == {"commit": SHA, "dirty": True}
    assert manifest["target_source"]["revision"] == full_run.MODEL_SOURCE[1]
    assert manifest["task_source"]["identity"].startswith("sha256:")
    assert manifest["files"]["task/solution/solve.sh"]["mode"] == 0o755
    assert not any(
        ".env" in name or ".venv" in name or "__pycache__" in name for name in files
    )
    assert plan["budget"]["sandbox_jobs_max"] == 28
    assert plan["budget"]["sandbox_remote_job_lifetime_seconds"] == 86400
    assert plan["budget"]["concurrent_sandbox_jobs_max"] == 1
    assert plan["budget"]["inference_calls_max"] == 8
    assert plan["mcp_request"]["args"]["image"] == cloud.WORKER_IMAGE
    assert plan["mcp_request"]["required_env_after_upload"] == [
        "OPENENV_BUNDLE_REVISION"
    ]
    assert "secrets" not in plan["mcp_request"]["args"]
    network.assert_not_called()
    (task / "instruction.md").write_text("different task\n")
    _, changed = prepared(inputs, tmp_path, "changed")
    assert changed["bundle_sha256"] != plan["bundle_sha256"]


def test_prepare_fresh_directory_and_locked_inputs(inputs, tmp_path):
    repo, task, config = inputs
    output, _ = prepared(inputs, tmp_path)
    with pytest.raises(FileExistsError, match="fresh"):
        cloud.prepare(task, config, output, repo_root=repo)
    with pytest.raises(ValueError, match="outside"):
        cloud.prepare(task, config, task / "output", repo_root=repo)
    (repo / "uv.lock").unlink()
    with pytest.raises(FileNotFoundError):
        prepared(inputs, tmp_path, "missing-lock")


@pytest.mark.parametrize("bad", ["link", "dirlink", "binary", "secret", "env", "fifo"])
def test_prepare_rejects_unsafe_local_inputs(inputs, tmp_path, bad):
    _, task, _ = inputs
    if bad == "link":
        (task / "link").symlink_to(task / "instruction.md")
    elif bad == "dirlink":
        (task / "link").symlink_to(task / "tests", target_is_directory=True)
    elif bad == "binary":
        (task / "binary").write_bytes(b"\x00binary")
    elif bad == "secret":
        (task / "secret.txt").write_text("hf_" + "z" * 30)
    elif bad == "env":
        (task / ".env.local").write_text("password=123")
    else:
        import os

        os.mkfifo(task / "fifo")
    with pytest.raises((ValueError, UnicodeDecodeError)):
        prepared(inputs, tmp_path)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize(
    "kind",
    ["timeout", "sandboxes", "calls", "tag", "provider", "revision", "credentials"],
)
def test_preflight_rejects_unbounded_or_mutable_config(inputs, tmp_path, kind):
    config = inputs[2]
    if kind == "timeout":
        config["cloud"]["timeout_seconds"] = 0
    elif kind == "sandboxes":
        config["cloud"]["max_sandboxes"] = 1
    elif kind == "calls":
        config["cloud"]["max_inference_calls"] = 1
    elif kind == "tag":
        config["execution"]["agent_image"] = "python:latest"
    elif kind == "provider":
        config["target"]["provider"] = "auto"
    elif kind == "revision":
        config["target"]["revision"] = "main"
    else:
        config["execution"]["token"] = "do-not-persist"
    with pytest.raises(ValueError):
        prepared(inputs, tmp_path)


def test_archive_digest_inventory_and_size_limits(inputs, tmp_path, monkeypatch):
    output, plan = prepared(inputs, tmp_path)
    data = (output / "bundle.zip").read_bytes()
    with pytest.raises(ValueError, match="SHA256"):
        cloud.extract_bundle(data, tmp_path / "bad", "0" * 64)
    for changed in (
        rewrite_archive(data, changes={"task/instruction.md": b"tampered"}),
        rewrite_archive(data, remove=["task/instruction.md"]),
        rewrite_archive(data, additions=[("extra", b"extra", stat.S_IFREG | 0o644)]),
    ):
        with pytest.raises(ValueError, match="inventory"):
            cloud.extract_bundle(
                changed, tmp_path / "bad", hashlib.sha256(changed).hexdigest()
            )
        assert not (tmp_path / "bad").exists()
    monkeypatch.setattr(cloud, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        cloud.extract_bundle(data, tmp_path / "bad", plan["bundle_sha256"])


@pytest.mark.parametrize(
    "name,mode",
    [
        ("../escape", stat.S_IFREG | 0o644),
        ("/absolute", stat.S_IFREG | 0o644),
        ("task\\escape", stat.S_IFREG | 0o644),
        ("task//escape", stat.S_IFREG | 0o644),
        ("C:escape", stat.S_IFREG | 0o644),
        ("task/instruction.md/child", stat.S_IFREG | 0o644),
        ("TASK/INSTRUCTION.MD", stat.S_IFREG | 0o644),
        ("link", stat.S_IFLNK | 0o777),
        ("directory", stat.S_IFDIR | 0o755),
    ],
)
def test_archive_rejects_path_collisions_and_nonregular_files(
    inputs, tmp_path, name, mode
):
    output, _ = prepared(inputs, tmp_path)
    data = rewrite_archive(
        (output / "bundle.zip").read_bytes(), additions=[(name, b"payload", mode)]
    )
    with pytest.raises(ValueError):
        cloud.extract_bundle(data, tmp_path / "bad", hashlib.sha256(data).hexdigest())
    assert not (tmp_path / "bad").exists()


def test_bootstrap_uses_pinned_download_and_frozen_sync(inputs, tmp_path, monkeypatch):
    output, plan = prepared(inputs, tmp_path)
    data = (output / "bundle.zip").read_bytes()
    request = Mock(return_value=io.BytesIO(data))
    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *args: SimpleNamespace(open=request)
    )
    destination = tmp_path / "bootstrap"
    destination.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(destination))
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    spec = cloud._launch_spec(
        cloud._config(inputs[2])[0], plan["bundle_sha256"], UPLOAD_SHA
    )
    for name, value in {**spec["env"], **SECRETS}.items():
        monkeypatch.setenv(name, value)
    exec(spec["command"][2], {})
    assert "/resolve/" + UPLOAD_SHA + "/inputs/" in request.call_args.args[0].full_url
    assert (
        request.call_args.args[0].get_header("Authorization")
        == "Bearer " + SECRETS["HF_TOKEN"]
    )
    calls = [call.args[0] for call in run.call_args_list]
    assert calls[0][-2:] == ["install", "uv==0.9.3"]
    assert calls[1] == ["uv", "sync", "--frozen", "--no-dev"]
    assert calls[2][1:4] == ["-m", "examples.hub_validation_pipeline.cloud", "worker"]
    assert (
        destination / "bundle/task/instruction.md"
    ).read_bytes() == b"Write an answer\n"


def test_submit_real_runtime_secrets_pins_and_one_shot_receipt(inputs, tmp_path):
    output, plan = prepared(inputs, tmp_path)
    api = fake_api()
    receipt = cloud.submit(
        output / "plan.json",
        api=api,
        secrets=SECRETS,
        **launch_options(output, inputs[2]),
    )
    assert receipt["job_id"] == "job-123"
    launch = api.run_job.call_args.kwargs
    assert launch["secrets"] == SECRETS
    assert launch["token"] == SECRETS["HF_SANDBOX_TOKEN"]
    assert launch["env"]["OPENENV_BUNDLE_REVISION"] == UPLOAD_SHA
    assert launch["timeout"] == "1800s"
    assert launch["image"] == cloud.WORKER_IMAGE
    commit = api.create_commit.call_args.kwargs
    assert commit["parent_commit"] == SHA
    assert commit["create_pr"] is False
    assert [op.path_in_repo for op in commit["operations"]] == [
        "inputs/" + plan["bundle_sha256"] + "/bundle.zip",
        "inputs/" + plan["bundle_sha256"] + "/manifest.json",
        "runs/" + plan["run_id"] + "/claim.json",
    ]
    for path in output.iterdir():
        for value in SECRETS.values():
            assert value.encode() not in path.read_bytes()
    with pytest.raises(FileExistsError, match="receipt"):
        cloud.submit(
            output / "plan.json",
            api=api,
            secrets=SECRETS,
            **launch_options(output, inputs[2]),
        )
    assert api.run_job.call_count == 1


@pytest.mark.parametrize("case", ["public", "missing", "duplicate", "conflict"])
def test_no_launch_until_private_repo_and_exclusive_claim(inputs, tmp_path, case):
    output, plan = prepared(inputs, tmp_path)
    api = fake_api()
    if case == "public":
        api.dataset_info.return_value.private = False
    elif case == "missing":
        api.dataset_info.side_effect = RuntimeError("missing")
    elif case == "duplicate":
        api.list_repo_files.return_value = ["runs/" + plan["run_id"] + "/claim.json"]
    else:
        api.create_commit.side_effect = RuntimeError("parent conflict")
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        cloud.submit(
            output / "plan.json",
            api=api,
            secrets=SECRETS,
            **launch_options(output, inputs[2]),
        )
    api.run_job.assert_not_called()
    if case in ("public", "missing", "duplicate"):
        api.create_commit.assert_not_called()


def test_uncertain_launch_never_retries_or_leaks_exception(inputs, tmp_path):
    output, _ = prepared(inputs, tmp_path)
    api = fake_api()
    api.run_job.side_effect = RuntimeError(SECRETS["HF_TOKEN"])
    with pytest.raises(RuntimeError, match="unknown") as caught:
        cloud.submit(
            output / "plan.json",
            api=api,
            secrets=SECRETS,
            **launch_options(output, inputs[2]),
        )
    assert SECRETS["HF_TOKEN"] not in str(caught.value)
    receipt = json.loads((output / "submission.json").read_text())
    assert receipt["status"] == "launch_unknown"
    with pytest.raises(FileExistsError):
        cloud.submit(
            output / "plan.json",
            api=api,
            secrets=SECRETS,
            **launch_options(output, inputs[2]),
        )
    assert api.run_job.call_count == 1


@pytest.mark.parametrize(
    "values",
    [
        {},
        {**SECRETS, "HF_INFERENCE_TOKEN": "$HF_INFERENCE_TOKEN"},
        {**SECRETS, "HF_SANDBOX_TOKEN": SECRETS["HF_TOKEN"]},
    ],
)
def test_secret_values_required_and_distinct(inputs, tmp_path, values):
    output, _ = prepared(inputs, tmp_path)
    api = fake_api()
    with pytest.raises(ValueError):
        cloud.submit(
            output / "plan.json",
            api=api,
            secrets=values,
            **launch_options(output, inputs[2]),
        )
    api.dataset_info.assert_not_called()
    api.run_job.assert_not_called()


@pytest.mark.parametrize("section", ["launch", "git", "task_source", "target_source"])
def test_plan_tampering_stops_before_hub(inputs, tmp_path, section):
    output, _ = prepared(inputs, tmp_path)
    path = output / "plan.json"
    plan = json.loads(path.read_text())
    if section == "launch":
        plan["mcp_request"]["args"]["image"] = "python:latest"
    else:
        plan[section] = {
            "tampered": "reviewed provenance no longer matches bundled input"
        }
    path.write_text(json.dumps(plan))
    api = fake_api()
    with pytest.raises(ValueError, match="immutable"):
        cloud.submit(
            path, api=api, secrets=SECRETS, **launch_options(output, inputs[2])
        )
    api.dataset_info.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_worker_persists_report_and_partial_evidence_on_failure(
    inputs, tmp_path, monkeypatch, fail
):
    root, plan = worker_root(inputs, tmp_path, monkeypatch)
    api = fake_api()

    def runner(task, config, output):
        assert task == root / "task"
        assert not output.exists()
        output.mkdir()
        (output / "evidence.json").write_text('{"attempted":1}\n')
        if fail:
            raise RuntimeError(SECRETS["HF_TOKEN"])
        return {"accepted": False, "verdict": "skip"}

    if fail:
        with pytest.raises(RuntimeError, match="persisted"):
            cloud.worker(root, bundle_sha=plan["bundle_sha256"], api=api, runner=runner)
    else:
        report = cloud.worker(
            root, bundle_sha=plan["bundle_sha256"], api=api, runner=runner
        )
        assert report["status"] == "completed"
        assert report["accepted"] is False
    commit = api.create_commit.call_args.kwargs
    operations = {op.path_in_repo: op.path_or_fileobj for op in commit["operations"]}
    prefix = "runs/" + plan["run_id"] + "/"
    assert prefix + "audit/evidence.json" in operations
    report = json.loads(operations[prefix + "cloud-report.json"])
    assert report["status"] == ("error" if fail else "completed")
    assert report["accepted"] is False
    if fail:
        assert report["error_type"] == "RuntimeError"
    assert all(name.startswith(prefix) for name in operations)
    assert all(SECRETS["HF_TOKEN"].encode() not in data for data in operations.values())


def test_worker_inventory_failure_still_persists_error(inputs, tmp_path, monkeypatch):
    root, plan = worker_root(inputs, tmp_path, monkeypatch)
    (root / "task/instruction.md").write_text("modified")
    api, runner = fake_api(), Mock()
    with pytest.raises(RuntimeError, match="persisted"):
        cloud.worker(root, bundle_sha=plan["bundle_sha256"], api=api, runner=runner)
    runner.assert_not_called()
    assert (
        json.loads((root / "worker-output/cloud-report.json").read_text())["status"]
        == "error"
    )
    api.create_commit.assert_called_once()


def test_worker_upload_failure_retains_local_report(inputs, tmp_path, monkeypatch):
    root, plan = worker_root(inputs, tmp_path, monkeypatch)
    api = fake_api()
    api.create_commit.side_effect = RuntimeError("Hub unavailable")
    with pytest.raises(RuntimeError, match="upload failed"):
        cloud.worker(
            root,
            bundle_sha=plan["bundle_sha256"],
            api=api,
            runner=lambda *args: {"accepted": True},
        )
    report = json.loads((root / "worker-output/cloud-report.json").read_text())
    assert report["persistence_status"] == "error"


def test_admission_thresholds_survive_bundle_roundtrip_exactly(inputs, tmp_path):
    from decimal import Decimal

    repo, task, config = inputs
    config["validation"]["reward_max"] = Decimal("1.00000000000000000001")
    cloud.prepare(task, config, tmp_path / "precise", repo_root=repo)
    _, restored, _, _ = cloud._read_plan(tmp_path / "precise/plan.json")
    assert restored["validation"]["reward_max"] == Decimal("1.00000000000000000001")


@pytest.mark.parametrize("kind", ["hf", "harbor", "local"])
def test_pinned_discovery_sources_bind_into_cloud_bundle(inputs, tmp_path, kind):
    from openenv.validation.runner import source_digest

    repo, task, config = inputs
    revision = {
        "hf": "a" * 40,
        "harbor": "sha256:" + "b" * 64,
        "local": source_digest(task),
    }[kind]
    source = {
        "source": kind,
        "source_id": "org/repo",
        "revision": revision,
        "task_path": ".",
    }
    source["task_file_hashes"] = {
        name: hashlib.sha256(data).hexdigest()
        for name, (data, _) in cloud._tree(task).items()
    }
    plan = cloud.prepare(task, config, tmp_path / kind, repo_root=repo, source=source)
    assert plan["task_source"]["kind"] == kind
    assert plan["task_source"]["revision"] == revision


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        cloud._loads('{"cloud":{},"cloud":{}}')


def test_bootstrap_persists_dependency_failure_without_sdk(
    inputs, tmp_path, monkeypatch
):
    import base64

    output, plan = prepared(inputs, tmp_path)
    data = (output / "bundle.zip").read_bytes()
    requests = []

    def open_request(request, **kwargs):
        requests.append(request)
        if request.get_method() == "POST":
            return io.BytesIO(b'{"commitOid":"done"}')
        if "/resolve/" in request.full_url:
            return io.BytesIO(data)
        return io.BytesIO(b'{"private":true}')

    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_request)
    )
    destination = tmp_path / "bootstrap"
    destination.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(destination))
    monkeypatch.setattr(
        subprocess, "run", Mock(side_effect=RuntimeError(SECRETS["HF_TOKEN"]))
    )
    spec = cloud._launch_spec(
        cloud._config(inputs[2])[0], plan["bundle_sha256"], UPLOAD_SHA
    )
    for name, value in {**spec["env"], **SECRETS}.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="install_uv") as caught:
        exec(spec["command"][2], {})
    assert SECRETS["HF_TOKEN"] not in str(caught.value)
    commit = requests[-1]
    assert (
        commit.full_url
        == "https://huggingface.co/api/datasets/org/private-results/commit/main"
    )
    rows = [json.loads(line) for line in commit.data.splitlines()]
    assert (
        rows[1]["value"]["path"] == "runs/" + plan["run_id"] + "/bootstrap-error.json"
    )
    report = json.loads(base64.b64decode(rows[1]["value"]["content"]))
    assert report["stage"] == "install_uv"
    assert report["accepted"] is False
    assert report["error_type"] == "RuntimeError"


def test_cross_host_redirect_drops_results_token():
    redirect = cloud._HubRedirect()
    request = urllib.request.Request(
        "https://huggingface.co/resolve/blob",
        headers={"Authorization": "Bearer private", "Range": "bytes=0-"},
    )
    target = redirect.redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example/blob"
    )
    assert target.get_header("Authorization") is None
    assert target.get_header("Range") == "bytes=0-"
    same = redirect.redirect_request(
        request, None, 302, "Found", {}, "https://huggingface.co/blob"
    )
    assert same.get_header("Authorization") == "Bearer private"
    with pytest.raises(ValueError, match="HTTPS"):
        redirect.redirect_request(
            request, None, 302, "Found", {}, "http://cdn.example/blob"
        )


@pytest.mark.parametrize("kind", ["symlink", "binary", "credential"])
def test_worker_unsafe_evidence_still_uploads_error_only(
    inputs, tmp_path, monkeypatch, kind
):
    root, plan = worker_root(inputs, tmp_path, monkeypatch)
    api = fake_api()

    def runner(task, config, output):
        output.mkdir()
        if kind == "symlink":
            (output / "link").symlink_to(root / "coding-config.json")
        elif kind == "binary":
            (output / "binary").write_bytes(b"\x00binary")
        else:
            (output / "unsafe.json").write_text(json.dumps(SECRETS))
        return {"accepted": True}

    with pytest.raises(RuntimeError, match="persisted"):
        cloud.worker(root, bundle_sha=plan["bundle_sha256"], api=api, runner=runner)
    operations = api.create_commit.call_args.kwargs["operations"]
    assert len(operations) == 1
    assert operations[0].path_in_repo.endswith("/cloud-report.json")
    report = json.loads(operations[0].path_or_fileobj)
    assert report["status"] == "error"
    assert report["accepted"] is False
    assert report["evidence_omitted"] is True


def test_task_exclusions_match_runner_source_identity(inputs, tmp_path):
    from openenv.validation.runner import source_digest

    task = inputs[1]
    for name in (".git", ".worktrees", ".venv", "__pycache__", "outputs"):
        (task / name).mkdir()
        (task / name / "binary-secret").write_bytes(b"\x00ignored")
    before = source_digest(task)
    output, plan = prepared(inputs, tmp_path)
    destination = tmp_path / "snapshot"
    cloud.extract_bundle(
        (output / "bundle.zip").read_bytes(), destination, plan["bundle_sha256"]
    )
    assert source_digest(destination / "task") == before
    assert not (destination / "task/.worktrees").exists()
    assert not (destination / "task/outputs").exists()


@pytest.mark.parametrize(
    "fault",
    ["compose", "steps", "private-network", "phase-environment", "missing-verifier"],
)
def test_prepare_rejects_unsupported_task_before_controller_spending(
    inputs, tmp_path, fault
):
    repo, task, config = inputs
    if fault == "compose":
        (task / "environment").mkdir()
        (task / "environment/compose.yaml").write_text("services: {}\n")
    elif fault == "steps":
        (task / "steps").mkdir()
        (task / "steps/step.toml").write_text("name='step'\n")
    elif fault == "private-network":
        (task / "task.toml").write_text("[environment]\nallow_internet=false\n")
    elif fault == "phase-environment":
        (task / "task.toml").write_text("[verifier.environment]\nnetwork_mode='none'\n")
    else:
        (task / "tests/test.sh").unlink()
    output = tmp_path / "plan"
    with pytest.raises(ValueError):
        cloud.prepare(task, config, output, repo_root=repo)
    assert not output.exists()


def test_prepare_checks_the_frozen_task_contract_not_a_restored_source(
    inputs, tmp_path, monkeypatch
):
    repo, task, config = inputs
    original_tree = cloud._tree
    original = (task / "task.toml").read_bytes()

    def changed_during_snapshot(root, **kwargs):
        if Path(root) != task:
            return original_tree(root, **kwargs)
        (task / "task.toml").write_text("[environment]\nallow_internet=false\n")
        try:
            return original_tree(root, **kwargs)
        finally:
            (task / "task.toml").write_bytes(original)

    monkeypatch.setattr(cloud, "_tree", changed_during_snapshot)
    output = tmp_path / "plan"
    with pytest.raises(ValueError, match="network"):
        cloud.prepare(task, config, output, repo_root=repo)
    assert (task / "task.toml").read_bytes() == original
    assert not output.exists()


def test_bundle_retains_available_full_run_reproduction_helpers(inputs, tmp_path):
    repo, _, _ = inputs
    names = (
        "full_run.py",
        "real_tasks.py",
        "sources.py",
        "spend.py",
    )
    for name in names:
        path = repo / "examples/hub_validation_pipeline" / name
        path.write_text("{}\n" if name.endswith(".json") else "# helper\n")
    output, plan = prepared(inputs, tmp_path)
    _, files = cloud._checked_archive(
        (output / "bundle.zip").read_bytes(), plan["bundle_sha256"]
    )
    for name in names:
        relative = "examples/hub_validation_pipeline/" + name
        assert files[relative] == (repo / relative).read_bytes()


@pytest.mark.parametrize("source_kind", ["local", "hf", "harbor"])
def test_hub_driver_prepares_coding_audits_without_admitting_or_running_docker(
    inputs, tmp_path, monkeypatch, source_kind
):
    from argparse import Namespace

    from examples.hub_validation_pipeline import cloud, pipeline

    repo, task, raw_config = inputs
    hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, (data, _) in cloud._tree(task).items()
    }
    if source_kind != "local":
        record = dict(
            task_id="remote-test-task",
            path=str(task),
            source=source_kind,
            source_id="org/tasks",
            task_path=".",
            task_file_hashes=hashes,
            revision=SHA if source_kind == "hf" else "sha256:" + "b" * 64,
        )
        monkeypatch.setattr(pipeline.sources, "acquire", lambda *args: [record])
    config = tmp_path / "coding.json"
    config.write_text(json.dumps(raw_config))
    calls = []
    real_prepare = cloud.prepare

    def prepare(subject, config, output, *, source):
        calls.append((subject, source))
        assert (subject / "task.toml").is_file()
        if source_kind != "local":
            assert source["task_file_hashes"] == hashes
        return real_prepare(subject, config, output, repo_root=repo, source=source)

    monkeypatch.setattr(cloud, "prepare", prepare)
    monkeypatch.setattr(
        pipeline,
        "validate_task",
        lambda *args: pytest.fail("cloud mode must not invoke Docker validation"),
    )
    args = Namespace(
        output=tmp_path / "run",
        contract=None,
        coding_config=config,
        submit_hf_jobs=False,
        hf_dataset=["org/tasks@" + SHA] if source_kind == "hf" else [],
        harbor_dataset=["org/tasks@sha256:" + "b" * 64]
        if source_kind == "harbor"
        else [],
        local_dataset=[task] if source_kind == "local" else [],
        source="both",
        dataset_limit=0,
        tasks_per_dataset=1,
    )
    assert pipeline.run(args) == 0
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["phase"] == "prepared"
    assert summary["plans"] == 1 and summary["accepted"] == 0
    assert not summary["rfc008_complete"]
    planned = json.loads((args.output / "planned.jsonl").read_text())
    assert planned["accepted"] is False and planned["state"] == "prepared"
    assert (args.output / "accepted.jsonl").read_text() == ""
    assert calls[0][1]["revision"]


@pytest.mark.parametrize("ledger_state", ["missing", "exhausted", "blocked"])
@pytest.mark.parametrize("entry", ["cloud_cli", "pipeline", "submit"])
def test_public_launch_paths_require_spending_gates(
    inputs, tmp_path, monkeypatch, ledger_state, entry
):
    from argparse import Namespace

    import huggingface_hub
    from examples.hub_validation_pipeline import pipeline, spend

    repo, task, config = inputs
    ledger = tmp_path / "spend.json"
    if ledger_state != "missing":
        spend.init(ledger, "1")
        spend.reserve(ledger, "prior", "1", "Prior work")
        if ledger_state == "blocked":
            spend.settle(ledger, "prior", "2", "Overrun")
    api = fake_api()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: api)
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    output, _ = prepared(inputs, tmp_path)
    if entry == "cloud_cli":
        with pytest.raises(SystemExit):
            cloud.main(["submit", "--plan", str(output / "plan.json")])
    elif entry == "pipeline":
        config_path = tmp_path / "coding.json"
        config_path.write_bytes(cloud._json_bytes(config))
        real_prepare = cloud.prepare
        monkeypatch.setattr(
            cloud,
            "prepare",
            lambda *args, **kwargs: real_prepare(*args, repo_root=repo, **kwargs),
        )
        args = Namespace(
            output=tmp_path / "run",
            contract=None,
            coding_config=config_path,
            submit_hf_jobs=True,
            hf_dataset=[],
            harbor_dataset=[],
            local_dataset=[task],
            source="both",
            dataset_limit=0,
            tasks_per_dataset=1,
        )
        with pytest.raises(ValueError, match="prepared_batch"):
            pipeline.run(args)
    else:
        with pytest.raises(ValueError, match="reservation"):
            cloud.submit(output / "plan.json", api=api, secrets=SECRETS)
    api.run_job.assert_not_called()
    api.create_commit.assert_not_called()


@pytest.mark.parametrize(
    "case",
    [
        "missing_ledger",
        "unknown",
        "blocked",
        "settled",
        "unbound",
        "underpriced",
        "missing_hash",
        "wrong_hash",
        "overallocated",
        "exhausted",
    ],
)
def test_submit_rechecks_reserved_plan_cost_and_status(inputs, tmp_path, case):
    output, _ = prepared(inputs, tmp_path)
    options = launch_options(output, inputs[2])
    ledger, identifier = options["spend_ledger"], options["reservation_id"]
    if case == "missing_ledger":
        ledger.unlink()
    elif case == "unknown":
        options["reservation_id"] = "unreserved"
    elif case == "blocked":
        spend.reserve(ledger, "other", "1", "Other work")
        spend.settle(ledger, "other", "2", "Overrun")
    elif case == "settled":
        spend.settle(ledger, identifier, "0", "Completed work")
    elif case == "missing_hash":
        del options["expected_plan_sha256"]
    elif case == "wrong_hash":
        options["expected_plan_sha256"] = "0" * 64
    else:
        data = json.loads(ledger.read_bytes())
        row = data["reservations"][identifier]
        if case == "exhausted":
            data["reservations"] = {}
        elif case == "unbound":
            del row["plan_costs"]
        else:
            row["plan_costs"][options["expected_plan_sha256"]] = (
                "0" if case == "underpriced" else "1001"
            )
        ledger.write_text(json.dumps(data))
        if case == "exhausted":
            spend.reserve(ledger, "other", "1000", "Other work")
    api = fake_api()
    with pytest.raises(ValueError):
        cloud.submit(output / "plan.json", api=api, secrets=SECRETS, **options)
    api.run_job.assert_not_called()
    api.create_commit.assert_not_called()
    assert not (output / "submission.json").exists()


def test_local_revision_checks_captured_verifier_not_restored_live_tree(
    inputs, tmp_path, monkeypatch
):
    from openenv.validation.runner import source_digest

    repo, task, config = inputs
    revision = source_digest(task)
    original_tree = cloud._tree
    verifier = task / "tests/test.sh"
    original = verifier.read_bytes()
    captured = []

    def mutate_during_capture(root, **kwargs):
        if root != task:
            return original_tree(root, **kwargs)
        verifier.write_bytes(b"echo 1 > /logs/verifier/reward.txt\n")
        try:
            files = original_tree(root, **kwargs)
            captured.append(files["tests/test.sh"][0])
            return files
        finally:
            verifier.write_bytes(original)

    monkeypatch.setattr(cloud, "_tree", mutate_during_capture)
    source = {"kind": "local", "revision": revision, "task_path": "."}
    with pytest.raises(ValueError, match="Local source revision"):
        cloud.prepare(task, config, tmp_path / "race", repo_root=repo, source=source)
    assert source_digest(task) == revision
    assert captured == [b"echo 1 > /logs/verifier/reward.txt\n"]
    assert not (tmp_path / "race").exists()


@pytest.mark.parametrize("kind", ["hf", "harbor"])
def test_remote_identity_requires_verified_acquisition_hashes(inputs, tmp_path, kind):
    repo, task, config = inputs
    source = dict(
        kind=kind,
        revision="a" * 40 if kind == "hf" else "sha256:" + "b" * 64,
        task_path=".",
    )
    with pytest.raises(ValueError, match="verified acquisition"):
        cloud.prepare(task, config, tmp_path / kind, repo_root=repo, source=source)
    assert not (tmp_path / kind).exists()
