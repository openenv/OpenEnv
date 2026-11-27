"""Tests for `openenv.harbor.images` and `openenv harbor build-images`."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openenv.cli.commands.harbor import app
from openenv.harbor.images import (
    audit_environment,
    build_copy_layer_tar,
    build_harbor_images,
    classify_dockerfile,
    compute_environment_hash,
    DockerBuildxBuilder,
    HarborImageError,
    KanikoHFJobsBuilder,
    OCIRegistryClient,
    parse_image_ref,
    parse_kaniko_digest,
    sanitize_tag,
    update_task_toml,
    verify_harbor_images,
)
from typer.testing import CliRunner

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _sha(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class FakeOCIRegistry:
    """In-memory OCI Distribution v2 registry supporting Bearer auth, multi-arch, and uploads."""

    def __init__(self, *, require_push_auth: bool = True) -> None:
        self.require_push_auth = require_push_auth
        self.blobs: dict[str, dict[str, bytes]] = {}  # repo -> {digest: bytes}
        self.manifests: dict[
            str, dict[str, tuple[bytes, str]]
        ] = {}  # repo -> {ref: (bytes, media_type)}
        self.uploads: dict[str, str] = {}  # upload_id -> repo
        self.blob_uploads_count = 0
        self.expected_creds = ("testuser", "testpass")
        self.valid_push_token = "tok-push-ok"
        self.valid_pull_token = "tok-pull-ok"

    def seed_image(
        self,
        repo: str,
        tag: str,
        *,
        os_name: str = "linux",
        arch: str = "amd64",
        multi_arch: bool = False,
        workdir: str = "/",
    ) -> str:
        layer_raw = b"base-layer-tar-bytes"
        layer_gz = gzip.compress(layer_raw, mtime=0)
        layer_digest = _sha(layer_gz)
        config_doc = {
            "os": os_name,
            "architecture": arch,
            "config": {"Env": ["PATH=/usr/bin:/bin"], "WorkingDir": workdir},
            "rootfs": {"type": "layers", "diff_ids": [_sha(layer_raw)]},
            "history": [{"created_by": "base"}],
        }
        config_bytes = json.dumps(config_doc, sort_keys=True).encode()
        config_digest = _sha(config_bytes)
        manifest_doc = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "size": len(config_bytes),
                "digest": config_digest,
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "size": len(layer_gz),
                    "digest": layer_digest,
                }
            ],
        }
        manifest_bytes = json.dumps(manifest_doc, sort_keys=True).encode()
        manifest_digest = _sha(manifest_bytes)

        self.blobs.setdefault(repo, {})[layer_digest] = layer_gz
        self.blobs[repo][config_digest] = config_bytes
        self.manifests.setdefault(repo, {})[manifest_digest] = (
            manifest_bytes,
            manifest_doc["mediaType"],
        )

        if multi_arch:
            arm_doc = {**config_doc, "architecture": "arm64"}
            arm_cfg = json.dumps(arm_doc, sort_keys=True).encode()
            arm_cfg_d = _sha(arm_cfg)
            self.blobs[repo][arm_cfg_d] = arm_cfg
            arm_m = json.dumps(
                {
                    **manifest_doc,
                    "config": {
                        "mediaType": "application/vnd.oci.image.config.v1+json",
                        "size": len(arm_cfg),
                        "digest": arm_cfg_d,
                    },
                },
                sort_keys=True,
            ).encode()
            arm_m_d = _sha(arm_m)
            self.manifests[repo][arm_m_d] = (arm_m, manifest_doc["mediaType"])
            index_doc = {
                "schemaVersion": 2,
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [
                    {
                        "mediaType": manifest_doc["mediaType"],
                        "size": len(arm_m),
                        "digest": arm_m_d,
                        "platform": {"os": "linux", "architecture": "arm64"},
                    },
                    {
                        "mediaType": manifest_doc["mediaType"],
                        "size": len(manifest_bytes),
                        "digest": manifest_digest,
                        "platform": {"os": os_name, "architecture": arch},
                    },
                ],
            }
            index_bytes = json.dumps(index_doc, sort_keys=True).encode()
            index_digest = _sha(index_bytes)
            self.manifests[repo][index_digest] = (index_bytes, index_doc["mediaType"])
            self.manifests[repo][tag] = (index_bytes, index_doc["mediaType"])
            return manifest_digest

        self.manifests[repo][tag] = (manifest_bytes, manifest_doc["mediaType"])
        return manifest_digest

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/token":
            scope = request.url.params.get("scope", "")
            auth = request.headers.get("Authorization", "")
            if "push" in scope and self.require_push_auth:
                expected = (
                    "Basic "
                    + base64.b64encode(
                        f"{self.expected_creds[0]}:{self.expected_creds[1]}".encode()
                    ).decode()
                )
                if auth != expected:
                    return httpx.Response(401, json={"error": "unauthorized"})
                return httpx.Response(200, json={"token": self.valid_push_token})
            return httpx.Response(200, json={"token": self.valid_pull_token})

        auth = request.headers.get("Authorization", "")
        is_write = request.method in {"POST", "PUT"}

        if path.startswith("/v2/") and "/manifests/" in path:
            rest = path[len("/v2/") :]
            repo, ref = rest.split("/manifests/", 1)
            if is_write:
                if self.require_push_auth and auth != f"Bearer {self.valid_push_token}":
                    return httpx.Response(
                        401,
                        headers={
                            "WWW-Authenticate": (
                                f'Bearer realm="{request.url.scheme}://{request.url.netloc.decode()}/token",'
                                f'service="fake",scope="repository:{repo}:pull,push"'
                            )
                        },
                    )
                raw = request.content
                digest = _sha(raw)
                mt = request.headers.get("Content-Type", "")
                self.manifests.setdefault(repo, {})[ref] = (raw, mt)
                self.manifests[repo][digest] = (raw, mt)
                return httpx.Response(201, headers={"Docker-Content-Digest": digest})

            if auth not in {
                f"Bearer {self.valid_pull_token}",
                f"Bearer {self.valid_push_token}",
            }:
                return httpx.Response(
                    401,
                    headers={
                        "WWW-Authenticate": (
                            f'Bearer realm="{request.url.scheme}://{request.url.netloc.decode()}/token",'
                            f'service="fake",scope="repository:{repo}:pull"'
                        )
                    },
                )
            entry = self.manifests.get(repo, {}).get(ref)
            if not entry:
                return httpx.Response(404)
            raw, mt = entry
            return httpx.Response(
                200,
                content=raw,
                headers={
                    "Content-Type": mt,
                    "Docker-Content-Digest": _sha(raw),
                },
            )

        if path.startswith("/v2/") and "/blobs/uploads/" in path:
            rest = path[len("/v2/") :]
            repo, tail = rest.split("/blobs/uploads/", 1)
            if self.require_push_auth and auth != f"Bearer {self.valid_push_token}":
                return httpx.Response(
                    401,
                    headers={
                        "WWW-Authenticate": (
                            f'Bearer realm="{request.url.scheme}://{request.url.netloc.decode()}/token",'
                            f'service="fake",scope="repository:{repo}:pull,push"'
                        )
                    },
                )
            if request.method == "POST":
                mount = request.url.params.get("mount")
                src_repo = request.url.params.get("from")
                if mount and src_repo and mount in self.blobs.get(src_repo, {}):
                    self.blobs.setdefault(repo, {})[mount] = self.blobs[src_repo][mount]
                    return httpx.Response(201)
                uid = f"u{len(self.uploads) + 1}"
                self.uploads[uid] = repo
                return httpx.Response(
                    202, headers={"Location": f"/v2/{repo}/blobs/uploads/{uid}"}
                )
            if request.method == "PUT":
                digest = request.url.params.get("digest", "")
                raw = request.content
                assert _sha(raw) == digest
                self.blobs.setdefault(repo, {})[digest] = raw
                self.blob_uploads_count += 1
                return httpx.Response(201, headers={"Docker-Content-Digest": digest})

        if path.startswith("/v2/") and "/blobs/" in path:
            rest = path[len("/v2/") :]
            repo, digest = rest.split("/blobs/", 1)
            if auth not in {
                f"Bearer {self.valid_pull_token}",
                f"Bearer {self.valid_push_token}",
            }:
                return httpx.Response(
                    401,
                    headers={
                        "WWW-Authenticate": (
                            f'Bearer realm="{request.url.scheme}://{request.url.netloc.decode()}/token",'
                            f'service="fake",scope="repository:{repo}:pull"'
                        )
                    },
                )
            blob = self.blobs.get(repo, {}).get(digest)
            if blob is None:
                return httpx.Response(404)
            if request.method == "HEAD":
                return httpx.Response(200, headers={"Content-Length": str(len(blob))})
            return httpx.Response(200, content=blob)

        return httpx.Response(404)


def _write_task(
    root: Path,
    name: str,
    *,
    dockerfile: str | None = None,
    env_files: dict[str, str] | None = None,
    docker_image: str | None = None,
    tests_files: dict[str, str] | None = None,
    solution_files: dict[str, str] | None = None,
) -> Path:
    task_dir = root / name
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    env_block = "[environment]\ncpus = 1\nmemory_mb = 512\n"
    if docker_image is not None:
        env_block = f'[environment]\ndocker_image = "{docker_image}"\ncpus = 1\nmemory_mb = 512\n'
    (task_dir / "task.toml").write_text(
        f'version = "1.0"\n[agent]\ntimeout_sec = 300\n{env_block}\n'
        "[verifier]\ntimeout_sec = 300\n[verifier.environment]\ncpus = 1\n",
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Fix the bug.\n", encoding="utf-8")
    if dockerfile is not None:
        (task_dir / "environment" / "Dockerfile").write_text(
            dockerfile, encoding="utf-8"
        )
    for rel, content in (env_files or {}).items():
        p = task_dir / "environment" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    for rel, content in (tests_files or {}).items():
        p = task_dir / "tests" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    for rel, content in (solution_files or {}).items():
        p = task_dir / "solution" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return task_dir


def test_parse_image_ref_and_sanitize_tag():
    py = parse_image_ref("python:3.12-slim")
    assert py.registry == "registry-1.docker.io"
    assert py.display_registry == "docker.io"
    assert py.repository == "library/python"
    assert py.tag == "3.12-slim"
    assert py.canonical_repo == "docker.io/library/python"

    ghcr = parse_image_ref("ghcr.io/harness-hub/scale800:my-tag")
    assert ghcr.registry == "ghcr.io"
    assert ghcr.repository == "harness-hub/scale800"
    assert ghcr.tag == "my-tag"

    local = parse_image_ref("localhost:5000/org/app@sha256:" + "a" * 64)
    assert local.registry == "localhost:5000"
    assert local.digest == "sha256:" + "a" * 64

    long_a = sanitize_tag("a" * 150 + "1")
    long_b = sanitize_tag("a" * 150 + "2")
    assert len(long_a) <= 128 and len(long_b) <= 128 and long_a != long_b


def test_classify_dockerfile_copy_only_vs_run():
    copy_only = classify_dockerfile(
        """
        ARG BASE=ghcr.io/harness-hub/go-base:v1
        FROM --platform=linux/amd64 ${BASE}
        WORKDIR /testbed
        ENV CGO_ENABLED=0 PYTHONDONTWRITEBYTECODE=1
        COPY source/ /testbed/
        LABEL openenv.test="yes"
        """
    )
    assert copy_only.is_copy_only is True
    assert copy_only.base_image == "ghcr.io/harness-hub/go-base:v1"
    assert copy_only.workdir == "/testbed"
    assert copy_only.env == {"CGO_ENABLED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    assert copy_only.labels == {"openenv.test": "yes"}
    assert len(copy_only.copy_ops) == 1
    assert copy_only.copy_ops[0].sources == ("source/",)

    with_run = classify_dockerfile(
        """
        FROM python:3.12-slim
        WORKDIR /app
        COPY source/ /app/
        RUN pip install pytest
        """
    )
    assert with_run.is_copy_only is False
    assert "RUN" in with_run.reason

    multi_stage = classify_dockerfile(
        """
        FROM golang:1.22 AS build
        FROM debian:bookworm-slim
        COPY --from=build /bin/app /bin/app
        """
    )
    assert multi_stage.is_copy_only is False

    non_numeric_chown = classify_dockerfile(
        "FROM python:3.12-slim\nCOPY --chown=learner:learner source/ /app/\n"
    )
    assert non_numeric_chown.is_copy_only is False

    recursive_glob = classify_dockerfile("FROM python:3.12-slim\nCOPY **/*.py /app/\n")
    assert recursive_glob.is_copy_only is False
    assert "recursive '**' COPY glob" in recursive_glob.reason


def test_audit_environment_blocks_leaks_symlinks_and_secrets(tmp_path: Path):
    clean = _write_task(
        tmp_path,
        "clean",
        dockerfile="FROM python:3.12-slim\nCOPY source/ /app/\n",
        env_files={"source/app.py": "def f(): return 1\n"},
        tests_files={
            "grade.py": "assert f() == 2  # hidden grader content > 32 bytes\n"
        },
        solution_files={"reference/app.py": "def f(): return 2  # reference fix\n"},
    )
    hashes = audit_environment(clean)
    assert set(hashes) == {"Dockerfile", "source/app.py"}

    # Symlink rejected
    sym = _write_task(tmp_path, "sym", dockerfile="FROM python:3.12-slim\n")
    (sym / "environment" / "escape").symlink_to(tmp_path)
    with pytest.raises(HarborImageError, match="symlink"):
        audit_environment(sym)

    # Credential rejected
    cred = _write_task(
        tmp_path,
        "cred",
        dockerfile="FROM python:3.12-slim\n",
        env_files={"source/cfg.py": "TOKEN = 'hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456'\n"},
    )
    with pytest.raises(HarborImageError, match="huggingface token"):
        audit_environment(cred)

    # Hidden grader leak rejected
    leak = _write_task(
        tmp_path,
        "leak",
        dockerfile="FROM python:3.12-slim\nCOPY source/ /app/\n",
        env_files={
            "source/leaked.py": "assert f() == 2  # hidden grader content > 32 bytes\n"
        },
        tests_files={
            "grade.py": "assert f() == 2  # hidden grader content > 32 bytes\n"
        },
    )
    with pytest.raises(HarborImageError, match="matches hidden tests/grade.py"):
        audit_environment(leak)

    # Parent traversal in COPY rejected
    trav = _write_task(
        tmp_path,
        "trav",
        dockerfile="FROM python:3.12-slim\nCOPY ../tests /tests\n",
    )
    with pytest.raises(HarborImageError, match="parent traversal"):
        audit_environment(trav)


def test_build_copy_layer_tar_matches_docker_copy_semantics(tmp_path: Path):
    env_dir = tmp_path / "environment"
    (env_dir / "source" / "sub").mkdir(parents=True)
    (env_dir / "source" / "a.go").write_text("package a\n", encoding="utf-8")
    script = env_dir / "source" / "sub" / "run.sh"
    script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    script.chmod(0o755)

    parsed = classify_dockerfile(
        "FROM ghcr.io/org/base:v1\nWORKDIR /testbed\nCOPY source/ /testbed/\n"
    )
    gz1, diff1 = build_copy_layer_tar(env_dir, parsed.copy_ops)
    gz2, diff2 = build_copy_layer_tar(env_dir, parsed.copy_ops)
    assert gz1 == gz2 and diff1 == diff2
    assert gz1 is not None

    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(gz1)), mode="r:") as tar:
        names = {m.name: m for m in tar.getmembers()}
    assert "testbed/a.go" in names
    assert "testbed/sub/run.sh" in names
    assert "testbed/source/a.go" not in names
    assert names["testbed/sub/run.sh"].mode == 0o755


def test_build_copy_layer_tar_copies_file_into_explicit_workdir(tmp_path: Path):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "app.py").write_text("print('ok')\n", encoding="utf-8")
    parsed = classify_dockerfile(
        "FROM python:3.12-slim\nWORKDIR /app\nCOPY app.py /app\n"
    )

    gz, _ = build_copy_layer_tar(env_dir, parsed.copy_ops)
    assert gz is not None
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(gz)), mode="r:") as tar:
        names = {member.name: member for member in tar.getmembers()}

    assert names["app"].isdir()
    assert names["app/app.py"].isfile()


def test_oci_registry_resolve_multiarch_and_append_layer(tmp_path: Path):
    fake = FakeOCIRegistry()
    base_digest = fake.seed_image(
        "harness-hub/base", "v1", multi_arch=True, workdir="/orig"
    )
    transport = httpx.MockTransport(fake.handler)

    with (
        OCIRegistryClient(
            http=httpx.Client(transport=transport), anonymous=True
        ) as pub,
        OCIRegistryClient(
            http=httpx.Client(transport=transport),
            username="testuser",
            password="testpass",
        ) as reg,
    ):
        resolved_base, _, _ = pub.resolve_linux_amd64("ghcr.io/harness-hub/base:v1")
        assert resolved_base.registry_digest == base_digest
        assert resolved_base.public_pull_verified is True

        env_dir = tmp_path / "environment"
        (env_dir / "source").mkdir(parents=True)
        (env_dir / "source" / "main.go").write_text("package main\n", encoding="utf-8")
        parsed = classify_dockerfile(
            "FROM ghcr.io/harness-hub/base:v1\nWORKDIR /testbed\nENV GOFLAGS=-mod=readonly\nCOPY source/ /testbed/\n"
        )
        gz, diff_id = build_copy_layer_tar(env_dir, parsed.copy_ops)

        pushed = reg.append_layer(
            base_image=parsed.base_image,
            target_repo="ghcr.io/harness-hub/scale800",
            target_tag="task-1",
            layer_tar_gz=gz,
            diff_id=diff_id,
            workdir=parsed.workdir,
            env=parsed.env,
            base_client=pub,
        )
        assert pushed.image.startswith("ghcr.io/harness-hub/scale800@sha256:")

        # Confirm anonymous public pull resolves the pushed amd64 image & updated config
        verified, manifest, config = pub.resolve_linux_amd64(pushed.image)
        assert verified.registry_digest == pushed.registry_digest
        assert verified.public_pull_verified is True
        assert len(manifest["layers"]) == 2
        assert config["config"]["WorkingDir"] == "/testbed"
        assert "GOFLAGS=-mod=readonly" in config["config"]["Env"]
        assert config["rootfs"]["diff_ids"][-1] == diff_id

        # Second identical append deduplicates blobs (0 additional uploads)
        uploads_before = fake.blob_uploads_count
        reg.append_layer(
            base_image=parsed.base_image,
            target_repo="ghcr.io/harness-hub/scale800",
            target_tag="task-2",
            layer_tar_gz=gz,
            diff_id=diff_id,
            workdir=parsed.workdir,
            env=parsed.env,
            base_client=pub,
        )
        assert fake.blob_uploads_count == uploads_before


def test_oci_registry_rejects_arm64_only_image():
    fake = FakeOCIRegistry()
    fake.seed_image("org/arm", "v1", os_name="linux", arch="arm64")
    with OCIRegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(fake.handler)),
        anonymous=True,
    ) as client:
        with pytest.raises(HarborImageError, match="expected linux/amd64"):
            client.resolve_linux_amd64("ghcr.io/org/arm:v1")


def test_kaniko_hf_jobs_builder_uploads_runs_and_cleans_up(tmp_path: Path):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN pip install packaging\n", encoding="utf-8"
    )
    expected_digest = "sha256:" + "9" * 64

    calls: dict[str, Any] = {"added": [], "deleted": [], "run_job": None}

    class FakeHfApi:
        def whoami(self) -> dict[str, str]:
            return {"name": "harness-hub"}

        def create_bucket(self, bucket_id: str, **kwargs: Any) -> None:
            calls["bucket"] = (bucket_id, kwargs)

        def batch_bucket_files(
            self, bucket_id: str, *, add: Any = None, delete: Any = None
        ) -> None:
            if add:
                calls["added"].extend(add)
            if delete:
                calls["deleted"].extend(delete)

        def run_job(self, **kwargs: Any) -> SimpleNamespace:
            calls["run_job"] = kwargs
            return SimpleNamespace(id="job-kaniko-123")

        def inspect_job(self, *, job_id: str, namespace: str) -> SimpleNamespace:
            return SimpleNamespace(
                id=job_id,
                namespace=namespace,
                status=SimpleNamespace(stage="COMPLETED"),
            )

        def fetch_job_logs(
            self, *, job_id: str, namespace: str, follow: bool = False
        ) -> list[str]:
            return [
                "INFO[0001] Building stage",
                f"OPENENV_KANIKO_DIGEST={expected_digest}",
            ]

    builder = KanikoHFJobsBuilder(
        namespace="harness-hub",
        flavor="cpu-upgrade",
        cache_repo="ghcr.io/harness-hub/cache",
        poll_interval_s=0.0,
        api=FakeHfApi(),
    )
    digest, job_id = builder.build(
        env_dir=env_dir,
        env_hash="abcdef0123456789" * 4,
        target_repo="ghcr.io/harness-hub/tasks",
        target_tag="task-1",
        registry_username="bot",
        registry_password="secret-token-value",
    )

    assert digest == expected_digest
    assert job_id == "job-kaniko-123"
    assert calls["bucket"] == (
        "harness-hub/openenv-kaniko-builds",
        {"private": True, "exist_ok": True},
    )
    assert len(calls["added"]) == 1
    assert calls["deleted"] == [calls["added"][0][1]]

    spec = calls["run_job"]
    assert spec["flavor"] == "cpu-upgrade"
    assert "secret-token-value" not in " ".join(spec["command"])
    assert "KANIKO_DOCKER_CONFIG_B64" in spec["secrets"]
    decoded_cfg = json.loads(
        base64.b64decode(spec["secrets"]["KANIKO_DOCKER_CONFIG_B64"]).decode()
    )
    assert "ghcr.io" in decoded_cfg["auths"]
    assert "--cache=true" in spec["command"][-1]
    assert "--cache-repo=ghcr.io/harness-hub/cache" in spec["command"][-1]
    assert (
        parse_kaniko_digest(f"x\nOPENENV_KANIKO_DIGEST={expected_digest}\n")
        == expected_digest
    )


def test_update_task_toml_preserves_verifier_environment(tmp_path: Path):
    toml_path = tmp_path / "task.toml"
    toml_path.write_text(
        'version = "1.0"\n[environment]\ncpus = 1\n[verifier.environment]\ncpus = 2\n',
        encoding="utf-8",
    )
    pinned = "ghcr.io/harness-hub/tasks@sha256:" + "c" * 64
    update_task_toml(toml_path, docker_image=pinned, workdir="/workspace")
    parsed = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    assert parsed["environment"]["docker_image"] == pinned
    assert parsed["environment"]["workdir"] == "/workspace"
    assert "docker_image" not in parsed["verifier"]["environment"]

    # Replacing an existing tag updates in-place cleanly
    pinned2 = "ghcr.io/harness-hub/tasks@sha256:" + "d" * 64
    update_task_toml(toml_path, docker_image=pinned2)
    parsed2 = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    assert parsed2["environment"]["docker_image"] == pinned2


def test_build_harbor_images_deduplicates_routes_and_resumes(tmp_path: Path):
    fake = FakeOCIRegistry()
    fake.seed_image("harness-hub/go-base", "v1")
    shared_digest = fake.seed_image("harness-hub/office", "v1")
    run_digest = fake.seed_image("harness-hub/corpus", "run-built")
    transport = httpx.MockTransport(fake.handler)

    src = tmp_path / "src" / "tasks"
    # Two identical COPY-only tasks -> should deduplicate into ONE oci_append
    for name in ("go-a", "go-b"):
        _write_task(
            src,
            name,
            dockerfile="FROM ghcr.io/harness-hub/go-base:v1\nWORKDIR /testbed\nCOPY source/ /testbed/\n",
            env_files={"source/main.go": "package main\n"},
        )
    # One RUN task -> routes to run_builder
    _write_task(
        src,
        "py-run",
        dockerfile="FROM ghcr.io/harness-hub/go-base:v1\nRUN echo hi\n",
    )
    # One shared-image task (no Dockerfile) -> verifies & pins digest
    _write_task(
        src,
        "office-1",
        docker_image="ghcr.io/harness-hub/office:v1",
        env_files={"inputs/data.csv": "a,b\n1,2\n"},
    )

    run_calls: list[str] = []

    class StubRunBuilder:
        def build(self, **kwargs: Any) -> tuple[str, str]:
            run_calls.append(kwargs["target_tag"])
            return run_digest, "job-run-1"

    out_dir = tmp_path / "out"
    receipts_file = tmp_path / "receipts.json"
    per_task_dir = tmp_path / "per-task-receipts"

    with (
        OCIRegistryClient(
            http=httpx.Client(transport=transport),
            username="testuser",
            password="testpass",
        ) as reg,
        OCIRegistryClient(
            http=httpx.Client(transport=transport), anonymous=True
        ) as pub,
    ):
        summary = build_harbor_images(
            tmp_path / "src",
            registry="ghcr.io/harness-hub/corpus",
            builder="kaniko",
            output_dir=out_dir,
            receipts_path=receipts_file,
            per_task_receipts_dir=per_task_dir,
            run_builder=StubRunBuilder(),
            registry_client=reg,
            public_client=pub,
        )

        assert summary["total_tasks"] == 4
        assert summary["unique_environments"] == 3  # go-a & go-b deduped!
        assert summary["resolved"] == 4
        assert summary["failed"] == 0
        assert run_calls == ["py-run"]
        assert summary["tasks"]["go-a"]["image"] == summary["tasks"]["go-b"]["image"]
        assert (
            summary["tasks"]["office-1"]["image"]
            == f"ghcr.io/harness-hub/office@{shared_digest}"
        )
        assert sorted(p.name for p in per_task_dir.glob("*.json")) == [
            "go-a.json",
            "go-b.json",
            "office-1.json",
            "py-run.json",
        ]

        # Every emitted task.toml has a pullable pinned linux/amd64 image
        verified = verify_harbor_images(
            out_dir, anonymous=True, require_digest_pin=True, registry_client=pub
        )
        assert (
            verified["total"] == 4 and verified["ok"] == 4 and verified["failed"] == 0
        )

        # Resuming from per_task_receipts_dir alone skips all builds (even for a subset)
        run_calls.clear()
        resumed = build_harbor_images(
            tmp_path / "src",
            registry="ghcr.io/harness-hub/corpus",
            builder="kaniko",
            per_task_receipts_dir=per_task_dir,
            task_ids=["py-run", "go-a"],
            run_builder=StubRunBuilder(),
            registry_client=reg,
            public_client=pub,
        )
        assert resumed["total_tasks"] == 2
        assert resumed["resolved"] == 2
        assert run_calls == []


def test_environment_hash_ignores_files_the_dockerfile_never_copies(tmp_path: Path):
    run_df = "FROM python:3.12-slim\nWORKDIR /testbed\nRUN pip install pytest\n"
    a = _write_task(
        tmp_path, "a", dockerfile=run_df, env_files={"source/app.py": "x = 1\n"}
    )
    b = _write_task(
        tmp_path, "b", dockerfile=run_df, env_files={"source/app.py": "x = 2\n"}
    )
    # Uploaded by Harbor at runtime, not baked into the image: one build serves both tasks
    assert compute_environment_hash(
        a, assume_static_context=True
    ) == compute_environment_hash(b, assume_static_context=True)

    copy_df = "FROM python:3.12-slim\nCOPY source/ /app/\n"
    c = _write_task(
        tmp_path,
        "c",
        dockerfile=copy_df,
        env_files={"source/app.py": "x = 1\n", "notes.txt": "a"},
    )
    d = _write_task(
        tmp_path,
        "d",
        dockerfile=copy_df,
        env_files={"source/app.py": "x = 2\n", "notes.txt": "a"},
    )
    e = _write_task(
        tmp_path,
        "e",
        dockerfile=copy_df,
        env_files={"source/app.py": "x = 1\n", "notes.txt": "b"},
    )
    assert compute_environment_hash(
        c, assume_static_context=True
    ) != compute_environment_hash(d, assume_static_context=True)  # copied file differs
    assert compute_environment_hash(
        c, assume_static_context=True
    ) == compute_environment_hash(
        e, assume_static_context=True
    )  # only an uncopied file differs

    # A bind mount reads the context without COPY: every file counts
    mount_df = (
        "FROM python:3.12-slim\nRUN --mount=type=bind,target=/src pip install /src\n"
    )
    f = _write_task(tmp_path, "f", dockerfile=mount_df, env_files={"setup.py": "a"})
    g = _write_task(tmp_path, "g", dockerfile=mount_df, env_files={"setup.py": "b"})
    assert compute_environment_hash(
        f, assume_static_context=True
    ) != compute_environment_hash(g, assume_static_context=True)


@pytest.mark.parametrize("ignore_file", [".dockerignore", "Dockerfile.dockerignore"])
def test_environment_hash_tabs_and_ignore_rules(tmp_path: Path, ignore_file: str):
    tab_df = "FROM python:3.12-slim\nCOPY\tsource/ /app/\n"
    a = _write_task(
        tmp_path, "a", dockerfile=tab_df, env_files={"source/app.py": "x = 1\n"}
    )
    b = _write_task(
        tmp_path, "b", dockerfile=tab_df, env_files={"source/app.py": "x = 2\n"}
    )
    assert compute_environment_hash(
        a, assume_static_context=True
    ) != compute_environment_hash(b, assume_static_context=True)

    copy_df = "FROM python:3.12-slim\nCOPY source/ /app/\n"
    c = _write_task(
        tmp_path,
        "c",
        dockerfile=copy_df,
        env_files={"source/app.py": "x\n", ignore_file: "*.pyc\n"},
    )
    d = _write_task(
        tmp_path,
        "d",
        dockerfile=copy_df,
        env_files={"source/app.py": "x\n", ignore_file: "*.md\n"},
    )
    assert compute_environment_hash(
        c, assume_static_context=True
    ) != compute_environment_hash(d, assume_static_context=True)


@pytest.mark.parametrize(
    "instructions",
    [
        'ENV TEMPLATE="<<VERSION>>"\nCOPY source/ /app/\n',
        "RUN <<END-SCRIPT\necho hi\nEND-SCRIPT\nCOPY source/ /app/\n",
        "COPY sou" + "\\\n" + "rce/ /app/\n",
        "COPY [^a]* /app/\n",
        "COPY missing /app/\n",
    ],
)
def test_environment_hash_falls_back_for_ambiguous_sources(
    tmp_path: Path, instructions: str
):
    dockerfile = "FROM python:3.12-slim\n" + instructions
    a = _write_task(tmp_path, "a", dockerfile=dockerfile, env_files={"source/a": "1"})
    b = _write_task(tmp_path, "b", dockerfile=dockerfile, env_files={"source/a": "2"})
    assert compute_environment_hash(
        a, assume_static_context=True
    ) == compute_environment_hash(a)
    assert compute_environment_hash(a) != compute_environment_hash(b)


def test_cli_build_images_static_context_is_opt_in(tmp_path: Path):
    dockerfile = "FROM example.com/base:latest\nRUN echo ready\n"
    for name in ("a", "b"):
        _write_task(tmp_path, name, dockerfile=dockerfile, env_files={"app.py": name})
    args = ["build-images", str(tmp_path), "--plan", "--json"]
    for options, expected in [([], 2), (["--assume-static-context"], 1)]:
        result = CliRunner().invoke(app, args + options)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["unique_environments"] == expected


def test_cli_build_images_plan_preserves_distinct_pins(tmp_path: Path):
    for name in ("a", "b"):
        _write_task(
            tmp_path,
            name,
            dockerfile="FROM python:3.12-slim\n",
            docker_image=f"ghcr.io/example/base@sha256:{name * 64}",
        )
    result = CliRunner().invoke(
        app, ["build-images", str(tmp_path), "--plan", "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["actions"]["keep_pinned"] == 2
    assert data["unique_environments"] == 2


def test_cli_build_images_plan_json(tmp_path: Path):
    src = tmp_path / "suite" / "tasks"
    _write_task(
        src,
        "t-copy",
        dockerfile="FROM python:3.12-slim\nWORKDIR /app\nCOPY source/ /app/\n",
        env_files={"source/app.py": "x = 1\n"},
    )
    _write_task(
        src,
        "t-run",
        dockerfile="FROM python:3.12-slim\nRUN pip install pytest\n",
    )
    result = CliRunner().invoke(
        app, ["build-images", str(tmp_path / "suite"), "--plan", "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["total_tasks"] == 2
    assert data["actions"]["oci_append"] == 1
    assert data["actions"]["build"] == 1


def test_cli_verify_only_forwards_task_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, Any] = {}

    def fake_verify(root: Path, **kwargs: Any) -> dict[str, Any]:
        captured["root"] = root
        captured.update(kwargs)
        return {"total": 0, "ok": 0, "failed": 0, "tasks": {}}

    monkeypatch.setattr("openenv.harbor.images.verify_harbor_images", fake_verify)

    result = CliRunner().invoke(
        app,
        [
            "build-images",
            str(tmp_path),
            "--verify-only",
            "--task",
            "selected",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["task_ids"] == ["selected"]


def test_docker_buildx_builder_invokes_amd64_and_reads_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN echo ok\n", encoding="utf-8"
    )
    expected = "sha256:" + "7" * 64
    seen_cmd: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        seen_cmd.extend(cmd)
        meta_idx = cmd.index("--metadata-file") + 1
        Path(cmd[meta_idx]).write_text(
            json.dumps({"containerimage.digest": expected}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("openenv.harbor.images.subprocess.run", fake_run)
    builder = DockerBuildxBuilder(cache_repo="ghcr.io/org/cache")
    digest, job_id = builder.build(
        env_dir=env_dir,
        env_hash="0" * 64,
        target_repo="ghcr.io/org/repo",
        target_tag="t1",
        extra_dockerfile="RUN echo extra",
    )
    assert digest == expected
    assert job_id is None
    assert "--platform" in seen_cmd and "linux/amd64" in seen_cmd
    assert "--push" in seen_cmd
    assert "type=registry,ref=ghcr.io/org/cache" in seen_cmd


def test_copy_dot_destination_and_single_segment_globs(tmp_path: Path):
    env_dir = tmp_path / "environment"
    (env_dir / "nested").mkdir(parents=True)
    (env_dir / "app.py").write_text("root = 1\n", encoding="utf-8")
    (env_dir / "nested" / "app.py").write_text("nested = 2\n", encoding="utf-8")

    # `COPY app.py .` with `WORKDIR /app` must land at `app/app.py`, not overwrite `app` as a file.
    parsed_dot = classify_dockerfile(
        "FROM python:3.12-slim\nWORKDIR /app\nCOPY app.py .\n"
    )
    gz_dot, _ = build_copy_layer_tar(env_dir, parsed_dot.copy_ops)
    assert gz_dot is not None
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(gz_dot)), mode="r:") as tar:
        names = {m.name: m for m in tar.getmembers()}
    assert "app/app.py" in names
    assert names["app/app.py"].isfile()
    assert names["app"].isdir()

    # `COPY *.py /app/` must only match top-level `app.py`, not `nested/app.py`.
    parsed_glob = classify_dockerfile(
        "FROM python:3.12-slim\nWORKDIR /app\nCOPY *.py /app/\n"
    )
    gz_glob, _ = build_copy_layer_tar(env_dir, parsed_glob.copy_ops)
    assert gz_glob is not None
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(gz_glob)), mode="r:") as tar:
        extracted = tar.extractfile("app/app.py")
        assert extracted is not None
        assert extracted.read() == b"root = 1\n"


def test_output_dir_overlap_rejected_and_verify_only_honors_task_filter(
    tmp_path: Path,
):
    fake = FakeOCIRegistry()
    good_digest = fake.seed_image("harness-hub/good", "v1")
    transport = httpx.MockTransport(fake.handler)

    suite = tmp_path / "suite"
    single = _write_task(
        suite / "tasks",
        "good-task",
        docker_image=f"ghcr.io/harness-hub/good@{good_digest}",
    )
    _write_task(suite / "tasks", "missing-image-task")

    with OCIRegistryClient(
        http=httpx.Client(transport=transport), anonymous=True
    ) as pub:
        # Overlapping output_dir must fail before deleting source task.
        with pytest.raises(HarborImageError, match="output_dir must not overlap"):
            build_harbor_images(
                single,
                output_dir=single,
                public_client=pub,
                registry_client=pub,
            )
        assert (single / "task.toml").is_file()

        # Filtering verify_harbor_images to only good-task ignores missing-image-task.
        res = verify_harbor_images(
            suite,
            anonymous=True,
            require_digest_pin=True,
            task_ids=["good-task"],
            registry_client=pub,
        )
        assert res["total"] == 1 and res["ok"] == 1 and res["failed"] == 0
