"""Prepare immutable coding audits locally, then explicitly submit one HF Job.

Only ``submit`` writes to the Hub or launches compute. Jobs use three distinct
runtime secrets. Native Sandbox RPC exposes its execution bearer to root task
code; use reviewed tasks. A provider's served weights are not revision-attested.
"""

import argparse
import hashlib
import inspect
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath

WORKER_IMAGE = "docker.io/library/python@sha256:48d884813cf44e927cf64c5529dd630dd3b18338c6dc727991eef9b71e934a20"
UV_VERSION = "0.9.3"
SECRET_NAMES = ("HF_TOKEN", "HF_INFERENCE_TOKEN", "HF_SANDBOX_TOKEN")
MAX_FILES = 5000
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_BYTES = 160 * 1024 * 1024
MANIFEST = "bundle-manifest.json"
LIMITATIONS = [
    "Reviewed tasks only: Native Sandbox root code can observe its execution RPC token.",
    "Target revision records Hub source provenance; provider-served weights are not attested.",
    "Worker image digest is a launch reference, not runtime attestation.",
    "Hard termination or unavailable Hub can prevent final evidence upload.",
    "This supplemental coding audit does not establish full RFC008 compliance or trainability.",
]
_EXCLUDED = {".git", "__pycache__", ".venv", ".worktrees", "outputs"}
_FORBIDDEN = {
    ".env",
    "env",
    "secrets",
    "credentials",
    "logs",
    "outputs",
    "output",
    "runs",
}


def _json_bytes(value):
    from openenv.validation.runtime.json_values import dumps

    return (dumps(value) + "\n").encode()


def _loads(data):
    from decimal import Decimal

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON number")

    return json.loads(
        data, object_pairs_hook=pairs, parse_constant=invalid, parse_float=Decimal
    )


def _safe_name(name):
    if not isinstance(name, str):
        raise ValueError("Unsafe bundle path")
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in (".", "..") for part in path.parts)
        or any(ord(char) < 32 for char in name)
        or "\\" in name
        or ":" in name
    ):
        raise ValueError("Unsafe bundle path")
    return path


def _checked_archive(data, expected_sha=None):
    """Check every archive member before writing or executing any payload."""
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive size limit exceeded")
    if expected_sha is not None and hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Bundle SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if not 1 < len(members) <= MAX_FILES + 1:
            raise ValueError("Bundle file count limit exceeded")
        names = set()
        folded = set()
        total = 0
        for member in members:
            name = str(_safe_name(member.filename))
            mode = member.external_attr >> 16
            if stat.S_IFMT(mode) != stat.S_IFREG or member.flag_bits & 1:
                raise ValueError("Bundle members must be unencrypted regular files")
            if name in names or name.casefold() in folded:
                raise ValueError("Bundle path collision")
            names.add(name)
            folded.add(name.casefold())
            total += member.file_size
            if member.file_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("Bundle size limit exceeded")
        for name in names:
            if any(
                str(parent).casefold() in folded
                for parent in PurePosixPath(name).parents
                if str(parent) != "."
            ):
                raise ValueError("Bundle file/directory collision")
        if MANIFEST not in names:
            raise ValueError("Bundle inventory missing")
        manifest = _loads(archive.read(MANIFEST))
        inventory = manifest["files"]
        if not isinstance(inventory, dict) or set(inventory) != names - {MANIFEST}:
            raise ValueError("Bundle inventory is incomplete")
        files = {}
        for member in members:
            if member.filename == MANIFEST:
                continue
            content = archive.read(member)
            record = inventory[member.filename]
            if (
                not isinstance(record, dict)
                or record.get("size") != len(content)
                or record.get("sha256") != hashlib.sha256(content).hexdigest()
                or record.get("mode") not in (0o644, 0o755)
                or stat.S_IMODE(member.external_attr >> 16) != record["mode"]
            ):
                raise ValueError("Bundle inventory hash, size or mode mismatch")
            files[member.filename] = content
        files[MANIFEST] = archive.read(MANIFEST)
    return manifest, files


def extract_bundle(data, destination, expected_sha):
    """Extract a fully verified bundle into a fresh directory, without extractall."""
    manifest, files = _checked_archive(data, expected_sha)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = destination.joinpath(*_safe_name(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
        path.chmod(manifest["files"].get(name, {}).get("mode", 0o644))
    return manifest


def _text_file(path):
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"Only regular source files are allowed: {path.name}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Source file size limit exceeded")
    with path.open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES or b"\0" in content:
        raise ValueError("Binary or oversized source file")
    content.decode("utf-8")
    _no_secrets(content)
    return content, 0o755 if mode & 0o111 else 0o644


def _no_secrets(content, secrets=()):
    if re.search(rb"hf_[A-Za-z0-9]{16,}", content) or re.search(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", content
    ):
        raise ValueError("Credential material is not allowed in persisted artifacts")
    if any(secret and secret.encode() in content for secret in secrets):
        raise ValueError("Runtime credential is not allowed in persisted artifacts")
    for name in SECRET_NAMES:
        value = os.environ.get(name)
        if value and value.encode() in content:
            raise ValueError("Runtime credential is not allowed in persisted artifacts")


def _tree(root, *, source=False):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Source root must be a real directory")
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        excluded = _EXCLUDED | ({".pytest_cache", ".ruff_cache"} if source else set())
        dirs[:] = sorted(name for name in dirs if name not in excluded)
        files = sorted(name for name in files if name not in excluded)
        for name in dirs + files:
            if (parent / name).is_symlink():
                raise ValueError("Symlinks are not allowed in bundles")
        if any(name.lower() in _FORBIDDEN for name in dirs + files):
            raise ValueError(
                "Environment, secret, log or output files are not bundle inputs"
            )
        for name in sorted(files):
            path = parent / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("Nonregular bundle input")
            if source and name.endswith((".pyc", ".pyo")):
                continue
            relative = path.relative_to(root).as_posix()
            if name.startswith(".env") or path.suffix.lower() in (
                ".pem",
                ".key",
                ".log",
                ".zip",
                ".tar",
                ".gz",
            ):
                raise ValueError(
                    "Environment, secret, binary or log file is not a bundle input"
                )
            if source and not (
                path.suffix == ".py"
                or (
                    path.suffix == ".json"
                    and any(part in ("schemas", "policies") for part in path.parts)
                )
            ):
                continue
            result[relative] = _text_file(path)
            if (
                len(result) > MAX_FILES
                or sum(len(item[0]) for item in result.values()) > MAX_TOTAL_BYTES
            ):
                raise ValueError("Source tree size limit exceeded")
    return result


def _positive(value, name, *, maximum=10000, minimum=1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _check_acquisition_files(source, task_files):
    """Bind a captured task to optional hashes verified during acquisition."""
    if "task_file_hashes" not in source:
        return
    expected = source["task_file_hashes"]
    if (
        not isinstance(expected, dict)
        or set(expected) != set(task_files)
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in expected.values()
        )
        or any(
            hashlib.sha256(data).hexdigest() != expected[name]
            for name, data in task_files.items()
        )
    ):
        raise ValueError("Captured task files disagree with verified acquisition")


def _config(config):
    from examples.coding_audit.runner import coding_budget, normalize_coding_config

    _no_secrets(_json_bytes(config))
    config = _loads(_json_bytes(config))
    cloud = config.setdefault("cloud", {})
    for key, value in {
        "flavor": "cpu-basic",
        "timeout_seconds": 1800,
        "max_sandboxes": 40,
    }.items():
        cloud.setdefault(key, value)
    config = normalize_coding_config(config)
    cloud = config["cloud"]
    if "results_repo" not in cloud:
        raise ValueError("Supply cloud.results_repo explicitly")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", cloud["results_repo"]):
        raise ValueError("cloud.results_repo must be org/repo")
    if cloud["flavor"] != "cpu-basic":
        raise ValueError("This bounded audit worker requires flavor cpu-basic")
    if cloud.get("namespace") is not None and not re.fullmatch(
        r"[\w.-]+", cloud["namespace"]
    ):
        raise ValueError("Invalid Job namespace")
    _positive(cloud["timeout_seconds"], "timeout_seconds", maximum=86400)
    budget = coding_budget(config, has_reference=True)
    if (
        budget["sandbox_jobs_max"] > cloud["max_sandboxes"]
        or budget["inference_calls_max"] > cloud["max_inference_calls"]
    ):
        raise ValueError(
            "Audit upper bound exceeds max_sandboxes or max_inference_calls"
        )
    budget.update(
        job_timeout_seconds=cloud["timeout_seconds"],
        max_sandboxes=cloud["max_sandboxes"],
        max_inference_calls=cloud["max_inference_calls"],
        output_token_upper_bound=config["generation"]["candidate_count"]
        * config["generator"]["max_tokens"]
        + len(config["seeds"])
        * config["target"]["max_steps"]
        * config["target"]["max_tokens"],
        cost_usd=None,
        sandbox_remote_job_lifetime_seconds=86400,
        concurrent_sandbox_jobs_max=1,
        orphan_job_lifetime_note="SDK 1.33.0 hardcodes a 24-hour child Job lifetime; command/start/idle deadlines are shorter but cannot replace this server-side orphan cap. Collection stops after unverified cleanup.",
        cost_note="Call/token counts bound requests, not current provider prices or input-token billing.",
    )
    return config, budget


def _git_provenance(root):
    def git(*args):
        return (
            subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
    }


class _HubRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("Hub download redirect requires HTTPS")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            urllib.parse.urlsplit(req.full_url).netloc
            != urllib.parse.urlsplit(newurl).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def _bootstrap_main(uv_version):
    import base64
    import tempfile

    revision = os.environ["OPENENV_BUNDLE_REVISION"]
    sha = os.environ["OPENENV_BUNDLE_SHA256"]
    repo = os.environ["OPENENV_RESULTS_REPO"]
    token = os.environ["HF_TOKEN"]
    if (
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not re.fullmatch(r"[0-9a-f]{64}", sha)
        or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo)
    ):
        raise ValueError("Invalid immutable input identity")
    opener = urllib.request.build_opener(_HubRedirect())
    headers = {"Authorization": "Bearer " + token}
    stage = "download"
    try:
        url = (
            "https://huggingface.co/datasets/"
            + repo
            + "/resolve/"
            + revision
            + "/inputs/"
            + sha
            + "/bundle.zip"
        )
        with opener.open(
            urllib.request.Request(url, headers=headers), timeout=120
        ) as response:
            data = response.read(MAX_ARCHIVE_BYTES + 1)
        root = Path(tempfile.mkdtemp(prefix="openenv-coding-")) / "bundle"
        stage = "integrity"
        extract_bundle(data, root, sha)
        stage = "install_uv"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "uv==" + uv_version], check=True
        )
        stage = "sync"
        subprocess.run(["uv", "sync", "--frozen", "--no-dev"], cwd=root, check=True)
        stage = "worker"
        subprocess.run(
            [
                str(root / ".venv/bin/python"),
                "-m",
                "examples.hub_validation_pipeline.cloud",
                "worker",
                "--root",
                str(root),
                "--bundle-sha",
                sha,
            ],
            cwd=root,
            check=True,
        )
    except BaseException as error:
        # Stdlib fallback also covers dependency failures before HfApi is installed.
        report = {
            "status": "error",
            "accepted": False,
            "stage": stage,
            "error_type": type(error).__name__,
            "bundle_sha256": sha,
        }
        try:
            info_url = "https://huggingface.co/api/datasets/" + repo
            with opener.open(
                urllib.request.Request(info_url, headers=headers), timeout=30
            ) as response:
                info = _loads(response.read(1024 * 1024))
            if info.get("private") is not True:
                raise ValueError("Results repository is not private")
            records = [
                {
                    "key": "header",
                    "value": {
                        "summary": "Persist coding bootstrap error",
                        "description": "",
                    },
                },
                {
                    "key": "file",
                    "value": {
                        "path": "runs/coding-" + sha + "/bootstrap-error.json",
                        "encoding": "base64",
                        "content": base64.b64encode(
                            json.dumps(report).encode()
                        ).decode(),
                    },
                },
            ]
            payload = "".join(json.dumps(row) + "\n" for row in records).encode()
            request = urllib.request.Request(
                info_url + "/commit/main",
                data=payload,
                headers={**headers, "Content-Type": "application/x-ndjson"},
                method="POST",
            )
            with opener.open(request, timeout=60) as response:
                response.read(1024 * 1024)
        except Exception:
            print("Bootstrap error evidence upload failed", file=sys.stderr)
        raise RuntimeError(
            "Coding bootstrap failed during " + stage + "; inspect runs/coding-" + sha
        ) from None


def _bootstrap_code():
    # Reuse the exact stdlib-only archive validator in the remote bootstrap.
    header = "import hashlib, io, json, os, re, stat, subprocess, sys, urllib.parse, urllib.request, zipfile\nfrom pathlib import Path, PurePosixPath\n"
    for name in (
        "MAX_FILES",
        "MAX_FILE_BYTES",
        "MAX_TOTAL_BYTES",
        "MAX_ARCHIVE_BYTES",
        "MANIFEST",
    ):
        header += f"{name} = {globals()[name]!r}\n"
    header += "\n".join(
        inspect.getsource(function)
        for function in (
            _loads,
            _safe_name,
            _checked_archive,
            extract_bundle,
            _HubRedirect,
            _bootstrap_main,
        )
    )
    return header + f"\n_bootstrap_main({UV_VERSION!r})\n"


def _launch_spec(config, digest, revision=None):
    cloud = config["cloud"]
    env = {
        "OPENENV_RESULTS_REPO": cloud["results_repo"],
        "OPENENV_BUNDLE_SHA256": digest,
        "PYTHONUNBUFFERED": "1",
    }
    if revision is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Input upload did not return a full commit SHA")
        env["OPENENV_BUNDLE_REVISION"] = revision
    spec = {
        "image": WORKER_IMAGE,
        "command": ["python", "-c", _bootstrap_code()],
        "env": env,
        "flavor": cloud["flavor"],
        "timeout": f"{cloud['timeout_seconds']}s",
        "labels": {"openenv_bundle": digest, "openenv_profile": "coding-audit-v1"},
    }
    if cloud.get("namespace"):
        spec["namespace"] = cloud["namespace"]
    return spec


def prepare(task_root, config, output_dir, *, repo_root=None, source=None):
    """Write a deterministic bundle and reviewable launch plan without network IO."""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    task_root, output_dir = Path(task_root), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Prepare requires a fresh output directory")
    if output_dir.resolve().is_relative_to(task_root.resolve()):
        raise ValueError("Output directory must be outside the task")
    config, budget = _config(config)
    from examples.coding_audit.runner import validate_coding_task

    validate_coding_task(task_root, config)
    files = {}
    for name in ("pyproject.toml", "README.md", "LICENSE", "uv.lock"):
        files[name] = _text_file(repo_root / name)
    files.update(
        {
            "src/openenv/" + name: value
            for name, value in _tree(repo_root / "src/openenv", source=True).items()
        }
    )
    audit_files = {
        name: value
        for name, value in _tree(
            repo_root / "examples/coding_audit", source=True
        ).items()
        if not PurePosixPath(name).name.startswith("test_")
    }
    if (
        not {"__init__.py", "__main__.py", "scoring.py", "hf.py", "runner.py"}
        <= audit_files.keys()
    ):
        raise ValueError("Operator coding-audit production package is incomplete")
    files.update(
        {"examples/coding_audit/" + name: value for name, value in audit_files.items()}
    )
    worker_path = "examples/hub_validation_pipeline/cloud.py"
    files[worker_path] = _text_file(repo_root / worker_path)
    for name in (
        "full_run.py",
        "real_tasks.py",
        "sources.py",
        "spend.py",
    ):
        relative = "examples/hub_validation_pipeline/" + name
        if (repo_root / relative).exists():
            files[relative] = _text_file(repo_root / relative)
    task_files = _tree(task_root)
    if not task_files or "task.toml" not in task_files:
        raise ValueError("Local task must contain task.toml")
    # Check the exact captured bytes: a live source can change and be restored
    # between its initial preflight and inventory collection.
    with tempfile.TemporaryDirectory(prefix="openenv-cloud-preflight-") as temporary:
        frozen = Path(temporary)
        for name, (content, mode) in task_files.items():
            path = frozen / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(mode)
        validate_coding_task(frozen, config)
    files.update({"task/" + name: value for name, value in task_files.items()})
    files["coding-config.json"] = (_json_bytes(config), 0o644)
    if source is None:
        source = {
            "kind": "local",
            "identity": "sha256:"
            + hashlib.sha256(
                _json_bytes(
                    {
                        name: hashlib.sha256(value[0]).hexdigest()
                        for name, value in task_files.items()
                    }
                )
            ).hexdigest(),
        }
    else:
        source = dict(source)
        kind = source.get("kind", source.get("source"))
        if (
            source.get("kind")
            and source.get("source")
            and source["kind"] != source["source"]
        ):
            raise ValueError("Source kind declarations disagree")
        revision = source.get("revision", "")
        patterns = {
            "hf": r"[0-9a-f]{40}",
            "harbor": r"sha256:[0-9a-f]{64}",
            "local": r"[0-9a-f]{64}",
        }
        if (
            kind not in patterns
            or not isinstance(revision, str)
            or not re.fullmatch(patterns[kind], revision)
        ):
            raise ValueError("Dataset source requires its full immutable revision")
        if source.get("task_path") != ".":
            _safe_name(source["task_path"])
        if kind == "local":
            from openenv.validation.runner import source_digest

            if source_digest(task_root) != revision:
                raise ValueError("Local source revision differs from the prepared task")
        source["kind"] = kind
    _check_acquisition_files(
        source, {name: value[0] for name, value in task_files.items()}
    )
    manifest = {
        "schema": "openenv-coding-bundle-v1",
        "git": _git_provenance(repo_root),
        "task_source": source,
        "target_source": {
            name: config["target"][name]
            for name in ("model_id", "revision", "provider")
        },
        "files": {
            name: {
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "mode": mode,
            }
            for name, (content, mode) in sorted(files.items())
        },
        "inventory_note": "Every payload file is inventoried; the outer ZIP digest also binds this manifest.",
    }
    files[MANIFEST] = (_json_bytes(manifest), 0o644)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, (content, mode) in sorted(files.items()):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(member, content)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    _checked_archive(data, digest)
    plan = {
        "schema": "openenv-coding-cloud-v1",
        "bundle": "bundle.zip",
        "bundle_sha256": digest,
        "run_id": "coding-" + digest,
        "results_repo": config["cloud"]["results_repo"],
        "manifest_sha256": hashlib.sha256(files[MANIFEST][0]).hexdigest(),
        "git": manifest["git"],
        "task_source": source,
        "target_source": manifest["target_source"],
        "budget": budget,
        "secret_names": list(SECRET_NAMES),
        "mcp_request": {
            "operation": "run",
            "args": _launch_spec(config, digest),
            "required_env_after_upload": ["OPENENV_BUNDLE_REVISION"],
            "secret_names": list(SECRET_NAMES),
        },
        "limitations": LIMITATIONS,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "bundle.zip").write_bytes(data)
    (output_dir / "plan.json").write_bytes(_json_bytes(plan))
    return plan


def _read_plan(plan_path, *, expected_sha256=None):
    plan_path = Path(plan_path)
    plan_bytes = plan_path.read_bytes()
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or hashlib.sha256(plan_bytes).hexdigest() != expected_sha256
    ):
        raise ValueError("Prepared plan changed before submission")
    plan = _loads(plan_bytes)
    if (
        plan.get("schema") != "openenv-coding-cloud-v1"
        or plan.get("bundle") != "bundle.zip"
    ):
        raise ValueError("Unsupported cloud plan")
    data = _read_bundle(plan_path.parent / "bundle.zip")
    manifest, files = _checked_archive(data, plan["bundle_sha256"])
    _check_acquisition_files(
        manifest["task_source"],
        {
            name.removeprefix("task/"): value
            for name, value in files.items()
            if name.startswith("task/")
        },
    )
    config, budget = _config(_loads(files["coding-config.json"]))
    if (
        plan["run_id"] != "coding-" + plan["bundle_sha256"]
        or plan["results_repo"] != config["cloud"]["results_repo"]
        or plan["budget"] != budget
        or plan["manifest_sha256"] != hashlib.sha256(files[MANIFEST]).hexdigest()
        or any(
            plan.get(name) != manifest.get(name)
            for name in ("git", "task_source", "target_source")
        )
        or manifest["target_source"]
        != {
            name: config["target"][name]
            for name in ("model_id", "revision", "provider")
        }
        or plan["mcp_request"]["args"] != _launch_spec(config, plan["bundle_sha256"])
    ):
        raise ValueError("Plan does not match its immutable bundle")
    return plan, config, data, manifest


def _read_bundle(path):
    if (
        not stat.S_ISREG(path.lstat().st_mode)
        or path.stat().st_size > MAX_ARCHIVE_BYTES
    ):
        raise ValueError("Bundle must be a bounded regular file")
    with path.open("rb") as stream:
        return stream.read(MAX_ARCHIVE_BYTES + 1)


def _runtime_secrets(supplied=None):
    from huggingface_hub import get_token

    values = (
        dict(supplied)
        if supplied is not None
        else {
            "HF_TOKEN": get_token(),
            **{name: os.environ.get(name) for name in SECRET_NAMES[1:]},
        }
    )
    if set(values) != set(SECRET_NAMES) or any(
        not isinstance(values.get(name), str)
        or not values[name]
        or values[name].startswith("$")
        for name in SECRET_NAMES
    ):
        raise ValueError(
            "Provide HF_TOKEN, HF_INFERENCE_TOKEN and HF_SANDBOX_TOKEN as runtime secrets"
        )
    if len(set(values.values())) != len(SECRET_NAMES):
        raise ValueError("All three runtime secret values must be distinct")
    return values


def _private_repo(api, repo, token):
    try:
        info = api.dataset_info(repo, token=token)
    except Exception:
        raise RuntimeError(
            "Results dataset must already exist and be accessible with HF_TOKEN; create a private dataset before submission"
        ) from None
    if (
        info.private is not True
        or not isinstance(info.sha, str)
        or not re.fullmatch(r"[0-9a-f]{40}", info.sha)
    ):
        raise ValueError(
            "Results dataset must already be private with a known commit SHA"
        )
    return info


def _write_receipt(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def stage_inputs(plan_path, *, api=None, secrets=None, expected_plan_sha256=None):
    """Claim a run and upload inputs; return exact launch kwargs without secrets.

    This writes to the private Hub, but does not launch compute. A stale/uncertain
    receipt requires manual reconciliation. Never automatically retry a launch.
    An expected plan SHA binds the parsed bytes to the caller's reviewed budget.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    plan_path = Path(plan_path)
    plan, config, data, manifest = _read_plan(
        plan_path, expected_sha256=expected_plan_sha256
    )
    secrets = _runtime_secrets(secrets)
    _no_secrets(data, secrets.values())
    api = api or HfApi(token=secrets["HF_TOKEN"])
    info = _private_repo(api, plan["results_repo"], secrets["HF_TOKEN"])
    receipt_path = plan_path.parent / "submission.json"
    receipt = {
        "run_id": plan["run_id"],
        "status": "reserved",
        "claim_nonce": uuid.uuid4().hex,
    }
    try:
        with receipt_path.open("xb") as stream:
            stream.write(_json_bytes(receipt))
    except FileExistsError:
        raise FileExistsError(
            "Submission receipt exists; reuse/review existing evidence explicitly instead of rerunning"
        ) from None
    prefix = "runs/" + plan["run_id"] + "/"
    try:
        names = api.list_repo_files(
            plan["results_repo"],
            repo_type="dataset",
            revision=info.sha,
            token=secrets["HF_TOKEN"],
        )
        if any(name.startswith(prefix) for name in names):
            raise FileExistsError(
                "Run already claimed on Hub; reuse existing evidence explicitly"
            )
        commit = api.create_commit(
            repo_id=plan["results_repo"],
            repo_type="dataset",
            revision="main",
            create_pr=False,
            parent_commit=info.sha,
            token=secrets["HF_TOKEN"],
            commit_message="Claim immutable coding audit " + plan["run_id"],
            operations=[
                CommitOperationAdd(
                    path_in_repo="inputs/" + plan["bundle_sha256"] + "/bundle.zip",
                    path_or_fileobj=data,
                ),
                CommitOperationAdd(
                    path_in_repo="inputs/" + plan["bundle_sha256"] + "/manifest.json",
                    path_or_fileobj=_json_bytes(manifest),
                ),
                CommitOperationAdd(
                    path_in_repo=prefix + "claim.json",
                    path_or_fileobj=_json_bytes(receipt),
                ),
            ],
        )
        spec = _launch_spec(config, plan["bundle_sha256"], commit.oid)
        receipt.update(
            status="inputs_uploaded",
            input_revision=commit.oid,
            launch_spec=spec,
            secret_names=list(SECRET_NAMES),
        )
        _write_receipt(receipt_path, receipt)
        (plan_path.parent / "mcp-request.json").write_bytes(
            _json_bytes(
                {"operation": "run", "args": spec, "secret_names": list(SECRET_NAMES)}
            )
        )
    except Exception as error:
        receipt.update(status="staging_failed", error_type=type(error).__name__)
        _write_receipt(receipt_path, receipt)
        raise
    return receipt


def submit(plan_path, *, api=None, secrets=None, expected_plan_sha256=None):
    """Upload/claim immutable inputs and launch once, resolving actual secrets."""
    from huggingface_hub import HfApi

    secrets = _runtime_secrets(secrets)
    api = api or HfApi(token=secrets["HF_TOKEN"])
    receipt = stage_inputs(
        plan_path,
        api=api,
        secrets=secrets,
        expected_plan_sha256=expected_plan_sha256,
    )
    receipt_path = Path(plan_path).parent / "submission.json"
    receipt["status"] = "launch_pending"
    _write_receipt(receipt_path, receipt)
    try:
        job = api.run_job(
            **receipt["launch_spec"], secrets=secrets, token=secrets["HF_SANDBOX_TOKEN"]
        )
    except Exception as error:
        receipt.update(status="launch_unknown", error_type=type(error).__name__)
        _write_receipt(receipt_path, receipt)
        raise RuntimeError(
            "Job launch outcome unknown; inspect the submission receipt and HF Jobs before any manual retry"
        ) from None
    receipt.update(status="submitted", job_id=job.id, job_url=job.url)
    _write_receipt(receipt_path, receipt)
    try:
        api.upload_file(
            repo_id=receipt["launch_spec"]["env"]["OPENENV_RESULTS_REPO"],
            repo_type="dataset",
            path_in_repo="runs/" + receipt["run_id"] + "/submission.json",
            path_or_fileobj=_json_bytes(receipt),
            token=secrets["HF_TOKEN"],
            commit_message="Record coding Job " + job.id,
        )
    except Exception as error:
        receipt.update(
            receipt_sync_status="error", receipt_sync_error_type=type(error).__name__
        )
        _write_receipt(receipt_path, receipt)
    return receipt


def worker(root, *, bundle_sha=None, api=None, runner=None):
    """Run the bundled audit and persist report/evidence even after runner errors."""
    from huggingface_hub import HfApi

    root = Path(root)
    secrets = _runtime_secrets()
    api = api or HfApi(token=secrets["HF_TOKEN"])
    digest = bundle_sha or os.environ.get("OPENENV_BUNDLE_SHA256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Worker requires bundle SHA256")
    output = root / "worker-output"
    output.mkdir(exist_ok=False)
    run_id = "coding-" + digest
    # Repo identity comes from bootstrap env so invalid config still has a sink.
    repo = os.environ.get("OPENENV_RESULTS_REPO")
    report = {
        "status": "error",
        "accepted": False,
        "run_id": run_id,
        "bundle_sha256": digest,
        "limitations": LIMITATIONS,
    }
    failure = None
    try:
        manifest = _loads((root / MANIFEST).read_bytes())
        for name, record in manifest["files"].items():
            path = root.joinpath(*_safe_name(name).parts)
            if any(
                parent.is_symlink()
                for parent in [
                    path,
                    *(root / parent for parent in path.relative_to(root).parents),
                ]
            ):
                raise ValueError("Worker payload contains a symlink")
            data, mode = _text_file(path)
            if (
                hashlib.sha256(data).hexdigest() != record["sha256"]
                or len(data) != record["size"]
                or mode != record["mode"]
            ):
                raise ValueError("Worker inventory mismatch")
        config, budget = _config(_loads((root / "coding-config.json").read_bytes()))
        if repo != config["cloud"]["results_repo"]:
            raise ValueError("Worker results repository disagrees with launch")
        _private_repo(api, repo, secrets["HF_TOKEN"])
        if runner is None:
            from examples.coding_audit.runner import run_coding_validation

            runner = run_coding_validation
        result = runner(root / "task", config, output / "audit")
        if not isinstance(result, dict):
            raise TypeError("Coding runner must return a report object")
        report.update(
            status="completed",
            report=result,
            budget=budget,
            git=manifest["git"],
            task_source=manifest["task_source"],
            target_source=manifest["target_source"],
        )
        report["accepted"] = result.get("accepted") is True
    except BaseException as error:
        failure = error
        report.update(status="error", accepted=False, error_type=type(error).__name__)
        # SDK exceptions can contain credentials; persist type, never raw repr.
    finally:
        (output / "cloud-report.json").write_bytes(_json_bytes(report))
        try:
            _private_repo(api, repo, secrets["HF_TOKEN"])
            # A malformed artifact must not prevent persistence of the error report.
            try:
                evidence = _tree(output)
                for data, _ in evidence.values():
                    _no_secrets(data, secrets.values())
            except Exception as error:
                failure = error
                report.pop("report", None)
                report.update(
                    status="error",
                    accepted=False,
                    evidence_collection_error_type=type(error).__name__,
                    evidence_omitted=True,
                )
                report_data = _json_bytes(report)
                (output / "cloud-report.json").write_bytes(report_data)
                evidence = {"cloud-report.json": (report_data, 0o644)}
            from huggingface_hub import CommitOperationAdd

            api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                commit_message="Persist coding audit " + run_id,
                operations=[
                    CommitOperationAdd(
                        path_in_repo="runs/" + run_id + "/" + name, path_or_fileobj=data
                    )
                    for name, (data, _) in sorted(evidence.items())
                ],
                token=secrets["HF_TOKEN"],
            )
        except Exception as error:
            report.update(
                status="error",
                accepted=False,
                persistence_status="error",
                persistence_error_type=type(error).__name__,
            )
            (output / "cloud-report.json").write_bytes(_json_bytes(report))
            raise RuntimeError(
                "Evidence upload failed; local worker-output/cloud-report.json retained"
            ) from None
    if failure is not None:
        raise RuntimeError(
            "Coding worker failed; error report and available evidence persisted"
        ) from None
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser(
        "prepare", help="Build local immutable inputs; no network or paid compute"
    )
    local.add_argument("--task", type=Path, required=True)
    local.add_argument("--config", type=Path, required=True)
    local.add_argument("--output", type=Path, required=True)
    launch = commands.add_parser(
        "submit", help="Upload private inputs and launch one paid HF Job"
    )
    launch.add_argument("--plan", type=Path, required=True)
    internal = commands.add_parser("worker", help=argparse.SUPPRESS)
    internal.add_argument("--root", type=Path, required=True)
    internal.add_argument("--bundle-sha", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.task, _loads(args.config.read_bytes()), args.output)
    elif args.command == "submit":
        result = submit(args.plan)
    else:
        result = worker(args.root, bundle_sha=args.bundle_sha)
    print(_json_bytes(result).decode(), end="")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        from huggingface_hub import get_token

        message = str(error)
        for value in [get_token(), *(os.environ.get(name) for name in SECRET_NAMES)]:
            if value:
                message = message.replace(value, "[REDACTED]")
        message = re.sub(r"hf_[A-Za-z0-9]{16,}", "[REDACTED]", message)
        print(
            "Cloud operation failed (" + type(error).__name__ + "): " + message[:1200],
            file=sys.stderr,
        )
        sys.exit(1)
