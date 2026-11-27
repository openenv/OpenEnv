"""Build, append, push, and verify OCI container images for Harbor tasks.

HF Sandbox (and other prebuilt-image Harbor runtimes) require each task's
``task.toml`` to set ``[environment].docker_image`` to an already-published,
publicly pullable ``linux/amd64`` image. This module provides a Spaces-free
workflow targeting standard OCI registries (GHCR, Docker Hub, private/local
OCI v2 registries):

1. **Audit** each task's ``environment/`` directory for symlinks, unsafe
   ``COPY``/``ADD`` paths, verifier/solution leakage, and embedded credentials.
2. **Classify & deduplicate** environments by content hash:
   - Tasks with no ``environment/Dockerfile`` and an existing ``docker_image``
     only verify and digest-pin the shared image (Harbor uploads
     ``environment/`` at sandbox start).
   - Tasks whose ``Dockerfile`` is declarative ``COPY``/``WORKDIR``/``ENV``
     only (no ``RUN``) take the pure-Python **OCI layer append** fast path —
     no Docker daemon and no remote build VM, <1 s per task with full base
     layer deduplication.
   - Tasks whose ``Dockerfile`` runs ``RUN`` instructions build via either
     **Kaniko on Hugging Face Jobs** (``--builder kaniko``, no persistent VM
     or cluster) or **Docker Buildx** (``--builder docker``).
3. **Pin & verify** every resulting manifest as ``<repo>@sha256:<digest>``
   (confirming ``linux/amd64`` config and anonymous public pullability) and
   write ``[environment].docker_image`` into ``task.toml``.
"""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import quote, urljoin, urlsplit

import httpx

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from .tasks import _task_dirs_from_directory

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PINNED_IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
DEFAULT_KANIKO_IMAGE = "ghcr.io/osscontainertools/kaniko:debug"
KANIKO_DIGEST_MARKER = "OPENENV_KANIKO_DIGEST="

_OCI_INDEX = "application/vnd.oci.image.index.v1+json"
_OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
_OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
_OCI_LAYER_GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
_DOCKER_MANIFEST_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
_DOCKER_MANIFEST_V2 = "application/vnd.docker.distribution.manifest.v2+json"
_DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"
_DOCKER_LAYER_GZIP = "application/vnd.docker.image.rootfs.diff.tar.gzip"

_ACCEPT_MANIFEST = ", ".join(
    (
        _OCI_INDEX,
        _OCI_MANIFEST,
        _DOCKER_MANIFEST_LIST,
        _DOCKER_MANIFEST_V2,
    )
)

_COPY_ONLY_INSTRUCTIONS = frozenset(
    {"FROM", "WORKDIR", "ENV", "COPY", "LABEL", "ARG", "EXPOSE"}
)

_CREDENTIAL_PATTERNS = (
    ("huggingface token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("github classic token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("github fine-grained token", re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    (
        "private key",
        re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
)

_HIDDEN_VERIFIER_FILES = frozenset(
    {
        "test.sh",
        "grade.py",
        "contract.json",
        "test_driver.py",
        "test_results.py",
        "expected.json",
        "cases.json",
    }
)


class HarborImageError(RuntimeError):
    """Raised when auditing, building, pushing, or verifying a Harbor image fails."""


@dataclass(frozen=True)
class ImageRef:
    """Parsed OCI image reference."""

    registry: str
    display_registry: str
    repository: str
    tag: str | None = None
    digest: str | None = None

    @property
    def canonical_repo(self) -> str:
        return f"{self.display_registry}/{self.repository}"

    @property
    def reference(self) -> str:
        if self.digest:
            return self.digest
        return self.tag or "latest"

    def with_tag(self, tag: str) -> "ImageRef":
        return ImageRef(
            registry=self.registry,
            display_registry=self.display_registry,
            repository=self.repository,
            tag=tag,
            digest=None,
        )

    def pinned(self, digest: str) -> str:
        if not DIGEST_RE.fullmatch(digest):
            raise ValueError(f"Invalid OCI digest: {digest!r}")
        return f"{self.canonical_repo}@{digest}"

    def tagged(self, tag: str | None = None) -> str:
        chosen = tag or self.tag or "latest"
        return f"{self.canonical_repo}:{chosen}"


@dataclass(frozen=True)
class CopyOp:
    """One declarative Dockerfile COPY operation."""

    sources: tuple[str, ...]
    destination: str
    workdir: str
    uid: int = 0
    gid: int = 0


@dataclass(frozen=True)
class ParsedDockerfile:
    """Structured view of a Dockerfile for classification and OCI layer append."""

    base_image: str
    is_copy_only: bool
    reason: str
    copy_ops: tuple[CopyOp, ...] = ()
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedImage:
    """An immutable, verified linux/amd64 OCI image manifest."""

    image: str
    registry_digest: str
    index_digest: str | None
    config: dict[str, Any]
    public_pull_verified: bool = False
    tag: str | None = None


@dataclass(frozen=True)
class TaskImagePlan:
    """Planned image operation for a single Harbor task."""

    task_id: str
    task_dir: Path
    action: str  # "keep_pinned" | "verify_shared" | "oci_append" | "build"
    env_hash: str
    reason: str
    base_image: str | None = None
    existing_docker_image: str | None = None
    target_tag: str | None = None
    parsed_dockerfile: ParsedDockerfile | None = None
    context_files: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskImageReceipt:
    """Recorded build/verification result for one Harbor task."""

    task_id: str
    status: str  # "resolved" | "built_no_push" | "failed"
    action: str
    builder: str
    env_hash: str
    image: str | None = None
    registry_digest: str | None = None
    tag: str | None = None
    config: dict[str, Any] | None = None
    public_pull_verified: bool = False
    context_files: dict[str, str] = field(default_factory=dict)
    job_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_localhost(registry: str) -> bool:
    host = registry.split(":", 1)[0].lower()
    return host in {"localhost", "127.0.0.1", "[::1]", "::1"}


def parse_image_ref(raw: str) -> ImageRef:
    """Parse an OCI image reference into normalized registry, repo, tag, and digest."""
    value = raw.strip()
    if not value or any(ch.isspace() for ch in value):
        raise ValueError(f"Invalid OCI image reference: {raw!r}")

    digest: str | None = None
    if "@" in value:
        value, digest = value.split("@", 1)
        if not DIGEST_RE.fullmatch(digest):
            raise ValueError(f"Invalid digest in image reference: {raw!r}")

    slash_idx = value.find("/")
    colon_idx = value.rfind(":")
    tag: str | None = None
    if colon_idx != -1 and colon_idx > slash_idx:
        tag = value[colon_idx + 1 :]
        value = value[:colon_idx]
        if not tag:
            raise ValueError(f"Empty tag in image reference: {raw!r}")

    parts = value.split("/")
    first = parts[0]
    if len(parts) == 1 or (
        "." not in first and ":" not in first and first != "localhost"
    ):
        display_registry = "docker.io"
        registry = "registry-1.docker.io"
        repository = f"library/{value}" if len(parts) == 1 else value
    else:
        display_registry = first.lower()
        registry = (
            "registry-1.docker.io"
            if display_registry in {"docker.io", "index.docker.io"}
            else display_registry
        )
        repository = "/".join(parts[1:])

    if not repository or repository.startswith("/") or "//" in repository:
        raise ValueError(f"Invalid repository path in image reference: {raw!r}")

    return ImageRef(
        registry=registry,
        display_registry=display_registry,
        repository=repository,
        tag=tag,
        digest=digest,
    )


def sanitize_tag(name: str, *, prefix: str = "") -> str:
    """Convert an arbitrary task ID into a valid OCI tag (<= 128 chars)."""
    raw = f"{prefix}{name}"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).lstrip(".-")
    if not cleaned:
        cleaned = "task"
    if len(cleaned) <= 128 and cleaned == raw:
        return cleaned
    if len(cleaned) > 110:
        digest = _sha256_bytes(raw.encode("utf-8"))[:12]
        cleaned = f"{cleaned[:110].rstrip('.-')}-{digest}"
    elif cleaned != raw:
        digest = _sha256_bytes(raw.encode("utf-8"))[:8]
        cleaned = f"{cleaned[:118].rstrip('.-')}-{digest}"
    return cleaned[:128]


def discover_task_dirs(root: Path) -> list[tuple[str, Path]]:
    """Return sorted ``(task_id, task_dir)`` pairs for a task or dataset directory."""
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Task or dataset directory does not exist: {root}")
    if (resolved / "task.toml").is_file():
        return [(resolved.name, resolved)]

    task_dirs = _task_dirs_from_directory(resolved, validate=False)
    valid = [p for p in task_dirs if (p / "task.toml").is_file()]
    if not valid:
        raise HarborImageError(f"No Harbor tasks (task.toml) found under {root}")

    base = resolved / "tasks" if (resolved / "tasks").is_dir() else resolved
    seen: set[str] = set()
    out: list[tuple[str, Path]] = []
    for task_dir in valid:
        rel = task_dir.relative_to(base).as_posix()
        task_id = rel.replace("/", "__") if "/" in rel else task_dir.name
        if task_id in seen:
            raise HarborImageError(f"Duplicate task ID discovered: {task_id}")
        seen.add(task_id)
        out.append((task_id, task_dir))
    return out


def _join_dockerfile_lines(text: str) -> list[str]:
    """Join backslash-continued Dockerfile lines and strip comments/blanks."""
    logical: list[str] = []
    buf = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not buf and (not stripped or stripped.startswith("#")):
            continue
        if buf and stripped.startswith("#"):
            # Dockerfile parsers allow comment lines between continuations.
            continue
        if raw_line.rstrip().endswith("\\"):
            buf += raw_line.rstrip()[:-1] + " "
        else:
            buf += raw_line
            if buf.strip():
                logical.append(buf.strip())
            buf = ""
    if buf.strip():
        logical.append(buf.strip())
    return logical


def _expand_args(value: str, args: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return args.get(name, match.group(0))

    return re.sub(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace, value
    )


def _parse_chown(spec: str) -> tuple[int, int] | None:
    parts = spec.split(":", 1)
    uid_str = parts[0]
    gid_str = parts[1] if len(parts) == 2 else uid_str
    if uid_str.isdigit() and gid_str.isdigit():
        return int(uid_str), int(gid_str)
    return None


def _parse_kv_pairs(rest: str, args: dict[str, str]) -> dict[str, str]:
    tokens = shlex.split(rest)
    if not tokens:
        return {}
    if "=" in tokens[0]:
        out: dict[str, str] = {}
        for token in tokens:
            if "=" not in token:
                raise ValueError(f"Malformed KEY=VALUE pair: {rest!r}")
            key, val = token.split("=", 1)
            out[key] = _expand_args(val, args)
        return out
    # Legacy single-pair syntax: ENV KEY value with spaces
    key, _, val = rest.strip().partition(" ")
    return {key: _expand_args(val.strip(), args)}


def _normalize_container_path(path: str, workdir: str) -> str:
    if path.startswith("/"):
        joined = PurePosixPath(path)
    else:
        joined = PurePosixPath(workdir) / path
    parts: list[str] = []
    for part in joined.parts:
        if part in {"", "/", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    normalized = "/" + "/".join(parts)
    if path.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def classify_dockerfile(text: str) -> ParsedDockerfile:
    """Parse a Dockerfile and decide if pure-Python OCI layer append suffices."""
    instructions = _join_dockerfile_lines(text)
    if not instructions:
        raise HarborImageError("Dockerfile is empty")

    args: dict[str, str] = {}
    from_count = 0
    base_image: str | None = None
    is_copy_only = True
    reasons: list[str] = []
    workdir = "/"
    explicit_workdir: str | None = None
    env: dict[str, str] = {}
    labels: dict[str, str] = {}
    copy_ops: list[CopyOp] = []

    for line in instructions:
        parts = line.split(None, 1)
        op = parts[0].upper()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if op == "ARG":
            if "=" in rest:
                key, val = rest.split("=", 1)
                args[key.strip()] = _expand_args(
                    val.strip().strip('"').strip("'"), args
                )
            continue

        if op == "FROM":
            from_count += 1
            tokens = shlex.split(rest)
            filtered: list[str] = []
            idx = 0
            while idx < len(tokens):
                tok = tokens[idx]
                if tok.startswith("--platform="):
                    idx += 1
                    continue
                if tok.upper() == "AS":
                    # Stage alias is fine for single-stage, multi-stage is caught by from_count > 1.
                    idx += 2
                    continue
                filtered.append(tok)
                idx += 1
            if not filtered:
                raise HarborImageError(f"Malformed FROM instruction: {line!r}")
            expanded = _expand_args(filtered[0], args)
            if from_count == 1:
                base_image = expanded
            else:
                is_copy_only = False
                reasons.append("multi-stage Dockerfile")
            continue

        if from_count == 0:
            raise HarborImageError(f"Instruction before FROM: {line!r}")

        if op not in _COPY_ONLY_INSTRUCTIONS:
            is_copy_only = False
            if op not in reasons:
                reasons.append(f"contains {op}")
            continue

        if op == "WORKDIR":
            raw_wd = _expand_args(shlex.split(rest)[0], args) if rest else "/"
            workdir = _normalize_container_path(raw_wd, workdir).rstrip("/") or "/"
            explicit_workdir = workdir
        elif op == "ENV":
            try:
                env.update(_parse_kv_pairs(rest, args))
            except ValueError:
                is_copy_only = False
                reasons.append("complex ENV")
        elif op == "LABEL":
            try:
                labels.update(_parse_kv_pairs(rest, args))
            except ValueError:
                is_copy_only = False
                reasons.append("complex LABEL")
        elif op == "EXPOSE":
            continue
        elif op == "COPY":
            uid, gid = 0, 0
            copy_rest = rest
            # Strip leading COPY flags like --chown=... or --from=...
            flag_tokens: list[str] = []
            while copy_rest.startswith("--"):
                tok, _, tail = copy_rest.partition(" ")
                flag_tokens.append(tok)
                copy_rest = tail.lstrip()

            unsupported_flag = False
            for flag in flag_tokens:
                if flag.startswith("--chown="):
                    parsed_ids = _parse_chown(flag.split("=", 1)[1])
                    if parsed_ids is None:
                        is_copy_only = False
                        unsupported_flag = True
                        reasons.append(f"non-numeric {flag}")
                    else:
                        uid, gid = parsed_ids
                elif flag.startswith("--chmod="):
                    is_copy_only = False
                    unsupported_flag = True
                    reasons.append("COPY --chmod")
                else:
                    is_copy_only = False
                    unsupported_flag = True
                    reasons.append(f"COPY {flag.split('=', 1)[0]}")

            if copy_rest.startswith("["):
                try:
                    items = json.loads(copy_rest)
                except json.JSONDecodeError as exc:
                    raise HarborImageError(
                        f"Invalid JSON COPY syntax: {line!r}"
                    ) from exc
                if not isinstance(items, list) or len(items) < 2:
                    raise HarborImageError(f"Malformed COPY instruction: {line!r}")
                items = [_expand_args(str(item), args) for item in items]
            else:
                items = [_expand_args(tok, args) for tok in shlex.split(copy_rest)]
                if len(items) < 2:
                    raise HarborImageError(f"Malformed COPY instruction: {line!r}")

            if any(
                "**" in PurePosixPath(source.lstrip("/")).parts for source in items[:-1]
            ):
                is_copy_only = False
                unsupported_flag = True
                reasons.append("recursive '**' COPY glob")

            if not unsupported_flag:
                copy_ops.append(
                    CopyOp(
                        sources=tuple(items[:-1]),
                        destination=items[-1],
                        workdir=workdir,
                        uid=uid,
                        gid=gid,
                    )
                )

    if base_image is None:
        raise HarborImageError("Dockerfile is missing a FROM instruction")

    reason = (
        "declarative COPY/WORKDIR/ENV only (OCI layer append)"
        if is_copy_only
        else "; ".join(reasons)
    )
    return ParsedDockerfile(
        base_image=base_image,
        is_copy_only=is_copy_only,
        reason=reason,
        copy_ops=tuple(copy_ops),
        workdir=explicit_workdir,
        env=env,
        labels=labels,
    )


def audit_environment(
    task_dir: Path,
    *,
    extra_dockerfile: str | None = None,
) -> dict[str, str]:
    """Audit a task's ``environment/`` directory before building/pushing a public image.

    Returns a mapping of relative POSIX paths in ``environment/`` to their SHA-256
    hex digests. Raises ``HarborImageError`` on any safety or leakage violation.
    """
    env_dir = task_dir / "environment"
    if not env_dir.is_dir():
        return {}

    problems: list[str] = []
    file_hashes: dict[str, str] = {}
    env_blobs: dict[str, bytes] = {}

    for path in sorted(env_dir.rglob("*")):
        rel = path.relative_to(env_dir).as_posix()
        if path.is_symlink():
            problems.append(f"symlink in environment/: {rel}")
            continue
        if path.is_dir():
            continue
        data = path.read_bytes()
        file_hashes[rel] = _sha256_bytes(data)
        env_blobs[rel] = data
        for label, pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(data):
                problems.append(f"possible {label} in environment/{rel}")

    dockerfile_path = env_dir / "Dockerfile"
    if dockerfile_path.is_file():
        df_text = dockerfile_path.read_text(encoding="utf-8")
        if extra_dockerfile:
            df_text = df_text.rstrip() + "\n" + extra_dockerfile.strip() + "\n"
        for line in _join_dockerfile_lines(df_text):
            parts = line.split(None, 1)
            op = parts[0].upper()
            if op not in {"COPY", "ADD"} or len(parts) < 2:
                continue
            rest = parts[1].strip()
            while rest.startswith("--"):
                _, _, rest = rest.partition(" ")
                rest = rest.lstrip()
            try:
                items = json.loads(rest) if rest.startswith("[") else shlex.split(rest)
            except (json.JSONDecodeError, ValueError):
                items = rest.split()
            for src in items[:-1]:
                src_str = str(src)
                if op == "ADD" and src_str.startswith(("http://", "https://")):
                    problems.append(f"remote URL in ADD instruction: {src_str}")
                posix = PurePosixPath(src_str)
                if ".." in posix.parts:
                    problems.append(f"parent traversal in {op}: {src_str}")
                if posix.parts and posix.parts[0] in {"tests", "solution"}:
                    problems.append(f"Dockerfile {op} references {posix.parts[0]}/")

    # Check against hidden verifier/solution files if present alongside environment/.
    tests_dir = task_dir / "tests"
    if tests_dir.is_dir():
        hidden_digests: dict[str, str] = {}
        for name in _HIDDEN_VERIFIER_FILES:
            candidate = tests_dir / name
            if candidate.is_file() and not candidate.is_symlink():
                raw = candidate.read_bytes()
                if len(raw) >= 32:
                    hidden_digests[_sha256_bytes(raw)] = f"tests/{name}"
        for rel, digest in file_hashes.items():
            if digest in hidden_digests:
                problems.append(
                    f"environment/{rel} matches hidden {hidden_digests[digest]}"
                )

    solution_dir = task_dir / "solution"
    if solution_dir.is_dir():
        sol_digests: dict[str, str] = {}
        for p in sorted(solution_dir.rglob("*")):
            if p.is_file() and not p.is_symlink():
                raw = p.read_bytes()
                if len(raw) >= 16:
                    sol_digests[_sha256_bytes(raw)] = (
                        "solution/" + p.relative_to(solution_dir).as_posix()
                    )
        for rel, digest in file_hashes.items():
            if digest in sol_digests:
                problems.append(f"environment/{rel} matches {sol_digests[digest]}")

    if problems:
        raise HarborImageError(
            f"{task_dir.name}: environment audit failed: {'; '.join(problems)}"
        )
    return file_hashes


def _context_sources(dockerfile_text: str) -> list[str] | None:
    """Build-context paths a Dockerfile reads through COPY/ADD, or ``None`` when that cannot be told statically.

    Files under ``environment/`` that no instruction reads (e.g. a source tree Harbor uploads at runtime) cannot
    change the image, so they must not split otherwise identical builds.
    """
    sources: list[str] = []
    for line in _join_dockerfile_lines(dockerfile_text):
        op, _, rest = line.partition(" ")
        if "--mount" in rest:  # RUN --mount=type=bind reads the context without COPY
            return None
        if op.upper() not in {"COPY", "ADD"}:
            continue
        flags = []
        rest = rest.strip()
        while rest.startswith("--"):
            flag, _, rest = rest.partition(" ")
            flags.append(flag)
            rest = rest.lstrip()
        if any(
            flag.startswith("--from=") for flag in flags
        ):  # another stage or image, not the context
            continue
        try:
            items = json.loads(rest) if rest.startswith("[") else shlex.split(rest)
        except (json.JSONDecodeError, ValueError):
            return None
        for source in items[:-1]:
            source = str(source)
            if op.upper() == "ADD" and "://" in source:  # a URL, not the context
                continue
            if "$" in source or "**" in PurePosixPath(source.lstrip("/")).parts:
                return None
            sources.append(source)
    return sources


def compute_environment_hash(
    task_dir: Path,
    *,
    extra_dockerfile: str | None = None,
    existing_docker_image: str | None = None,
) -> str:
    """Deterministic SHA-256 of a task's effective build context: the Dockerfile and the files it copies."""
    env_dir = task_dir / "environment"
    digest = hashlib.sha256()
    if extra_dockerfile:
        digest.update(b"EXTRA_DOCKERFILE\0")
        digest.update(extra_dockerfile.strip().encode("utf-8") + b"\0")
    if not env_dir.is_dir() or not (env_dir / "Dockerfile").is_file():
        digest.update(b"SHARED_IMAGE\0")
        digest.update((existing_docker_image or "").encode("utf-8") + b"\0")
        return digest.hexdigest()

    dockerfile = env_dir / "Dockerfile"
    sources = _context_sources(
        dockerfile.read_text(encoding="utf-8") + "\n" + (extra_dockerfile or "")
    )
    if sources is None:
        paths = set(env_dir.rglob("*"))
    else:
        paths = {dockerfile}
        for source in sources:
            for match in _resolve_copy_sources(env_dir, source):
                paths |= {match, *match.rglob("*")} if match.is_dir() else {match}
    for path in sorted(paths):
        if path.is_dir() or path.is_symlink():
            continue
        rel = path.relative_to(env_dir).as_posix()
        mode = 0o755 if (path.stat().st_mode & 0o111) else 0o644
        raw = path.read_bytes()
        digest.update(f"{rel}\0{mode:o}\0".encode("utf-8"))
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def _read_docker_config_auth(registry: str) -> tuple[str, str] | None:
    """Read inline Basic auth from ~/.docker/config.json if present."""
    config_path = (
        Path(os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker")))
        / "config.json"
    )
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    auths = data.get("auths") or {}
    candidates = [
        registry,
        f"https://{registry}",
        f"https://{registry}/v1/",
        f"https://{registry}/v2/",
    ]
    if registry in {"registry-1.docker.io", "docker.io"}:
        candidates.extend(
            [
                "https://index.docker.io/v1/",
                "index.docker.io",
                "docker.io",
                "registry-1.docker.io",
            ]
        )
    for host in candidates:
        entry = auths.get(host)
        if isinstance(entry, dict) and entry.get("auth"):
            try:
                decoded = base64.b64decode(entry["auth"]).decode("utf-8")
                if ":" in decoded:
                    user, pwd = decoded.split(":", 1)
                    return user, pwd
            except Exception:  # noqa: BLE001
                continue
    return None


def resolve_registry_credentials(
    registry: str,
    *,
    username: str | None = None,
    password: str | None = None,
) -> tuple[str, str] | None:
    """Resolve registry credentials from explicit args, env vars, or Docker config."""
    if username and password:
        return username, password

    env_user = os.environ.get("OPENENV_REGISTRY_USERNAME") or os.environ.get(
        "REGISTRY_USERNAME"
    )
    env_pass = os.environ.get("OPENENV_REGISTRY_PASSWORD") or os.environ.get(
        "REGISTRY_PASSWORD"
    )
    if env_user and env_pass:
        return env_user, env_pass

    if registry == "ghcr.io":
        token = (
            password
            or os.environ.get("GHCR_TOKEN")
            or os.environ.get("CR_PAT")
            or os.environ.get("GITHUB_TOKEN")
        )
        user = (
            username
            or os.environ.get("GHCR_USERNAME")
            or os.environ.get("GITHUB_ACTOR")
            or os.environ.get("GITHUB_USER")
            or "x-access-token"
        )
        if token:
            return user, token
    elif registry in {"registry-1.docker.io", "docker.io"}:
        user = username or os.environ.get("DOCKER_USERNAME")
        pwd = (
            password
            or os.environ.get("DOCKER_PASSWORD")
            or os.environ.get("DOCKER_TOKEN")
        )
        if user and pwd:
            return user, pwd

    return _read_docker_config_auth(registry)


def build_docker_config_json(registry: str, username: str, password: str) -> bytes:
    """Build a minimal Docker ``config.json`` with Basic auth for one registry."""
    host = (
        "https://index.docker.io/v1/"
        if registry == "registry-1.docker.io"
        else registry
    )
    encoded = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    payload = {"auths": {host: {"auth": encoded}}}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _parse_www_authenticate(header: str) -> tuple[str, dict[str, str]]:
    header = header.strip()
    if " " not in header:
        return header.lower(), {}
    scheme, rest = header.split(" ", 1)
    fields = dict(re.findall(r'(\w+)="([^"]*)"', rest))
    return scheme.lower(), fields


class OCIRegistryClient:
    """Minimal OCI Distribution v2 client for manifest verification and layer append."""

    def __init__(
        self,
        *,
        http: httpx.Client | None = None,
        username: str | None = None,
        password: str | None = None,
        anonymous: bool = False,
        timeout: float = 60.0,
    ) -> None:
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=timeout, follow_redirects=False)
        self._username = None if anonymous else username
        self._password = None if anonymous else password
        self._anonymous = anonymous
        self._tokens: dict[tuple[str, str], str] = {}

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "OCIRegistryClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _base_url(self, registry: str) -> str:
        scheme = "http" if _is_localhost(registry) else "https"
        return f"{scheme}://{registry}"

    def _credentials_for(self, registry: str) -> tuple[str, str] | None:
        if self._anonymous:
            return None
        return resolve_registry_credentials(
            registry, username=self._username, password=self._password
        )

    def _send_with_redirects(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
        params: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.Response:
        cur_url = url
        cur_headers = dict(headers or {})
        cur_params = params
        cur_auth = auth
        for _ in range(8):
            parsed = urlsplit(cur_url)
            if parsed.scheme not in {"https", "http"} or (
                parsed.scheme == "http" and not _is_localhost(parsed.netloc)
            ):
                raise HarborImageError(f"Refusing insecure registry URL: {cur_url}")
            req = self._http.build_request(
                method,
                cur_url,
                headers=cur_headers,
                content=content,
                params=cur_params,
            )
            resp = self._http.send(req, auth=cur_auth, follow_redirects=False)
            if resp.status_code not in {301, 302, 303, 307, 308}:
                return resp
            location = resp.headers.get("Location")
            if not location:
                return resp
            dest = urljoin(cur_url, location)
            if urlsplit(dest).netloc != parsed.netloc:
                cur_headers.pop("Authorization", None)
                cur_auth = None
            cur_url = dest
            cur_params = None
        raise HarborImageError(f"Too many redirects requesting {url}")

    def _request(
        self,
        method: str,
        registry: str,
        path_or_url: str,
        *,
        scope_key: str,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        url = (
            path_or_url
            if path_or_url.startswith(("http://", "https://"))
            else f"{self._base_url(registry)}{path_or_url}"
        )
        req_headers = dict(headers or {})
        token_cache_key = (registry, scope_key)
        cached = self._tokens.get(token_cache_key)
        if cached:
            req_headers["Authorization"] = cached

        resp = self._send_with_redirects(
            method, url, headers=req_headers, content=content, params=params
        )
        if resp.status_code != 401:
            return resp

        challenge = resp.headers.get("WWW-Authenticate", "")
        scheme, fields = _parse_www_authenticate(challenge)
        creds = self._credentials_for(registry)

        if scheme == "basic":
            if not creds:
                return resp
            encoded = base64.b64encode(f"{creds[0]}:{creds[1]}".encode("utf-8")).decode(
                "ascii"
            )
            auth_header = f"Basic {encoded}"
            self._tokens[token_cache_key] = auth_header
            req_headers["Authorization"] = auth_header
            return self._send_with_redirects(
                method, url, headers=req_headers, content=content, params=params
            )

        if scheme != "bearer" or "realm" not in fields:
            return resp

        token_params: dict[str, str] = {}
        if "service" in fields:
            token_params["service"] = fields["service"]
        if "scope" in fields:
            token_params["scope"] = fields["scope"]

        auth_resp = self._send_with_redirects(
            "GET",
            fields["realm"],
            params=token_params,
            auth=httpx.BasicAuth(creds[0], creds[1]) if creds else None,
        )
        if auth_resp.status_code != 200:
            return resp
        payload = auth_resp.json()
        bearer = payload.get("token") or payload.get("access_token")
        if not bearer:
            return resp
        auth_header = f"Bearer {bearer}"
        self._tokens[token_cache_key] = auth_header
        req_headers["Authorization"] = auth_header
        return self._send_with_redirects(
            method, url, headers=req_headers, content=content, params=params
        )

    def _get_manifest_raw(
        self, ref: ImageRef, reference: str | None = None
    ) -> tuple[str, bytes, str]:
        target_ref = reference or ref.reference
        resp = self._request(
            "GET",
            ref.registry,
            f"/v2/{ref.repository}/manifests/{target_ref}",
            scope_key=f"pull:{ref.repository}",
            headers={"Accept": _ACCEPT_MANIFEST},
        )
        if resp.status_code != 200:
            raise HarborImageError(
                f"Failed to read manifest {ref.canonical_repo}:{target_ref} "
                f"(HTTP {resp.status_code})"
            )
        raw = resp.content
        computed = f"sha256:{_sha256_bytes(raw)}"
        header_digest = resp.headers.get("Docker-Content-Digest")
        if header_digest and header_digest != computed:
            raise HarborImageError(
                f"Manifest digest mismatch for {ref.canonical_repo}: "
                f"header={header_digest} computed={computed}"
            )
        if DIGEST_RE.fullmatch(target_ref) and target_ref != computed:
            raise HarborImageError(
                f"Pinned manifest digest mismatch for {ref.canonical_repo}@{target_ref}"
            )
        media_type = resp.headers.get("Content-Type", "").split(";", 1)[0].strip()
        return computed, raw, media_type

    def _get_blob(self, ref: ImageRef, digest: str) -> bytes:
        resp = self._request(
            "GET",
            ref.registry,
            f"/v2/{ref.repository}/blobs/{digest}",
            scope_key=f"pull:{ref.repository}",
        )
        if resp.status_code != 200:
            raise HarborImageError(
                f"Failed to read blob {ref.canonical_repo}@{digest} "
                f"(HTTP {resp.status_code})"
            )
        raw = resp.content
        if f"sha256:{_sha256_bytes(raw)}" != digest:
            raise HarborImageError(
                f"Blob digest mismatch for {ref.canonical_repo}@{digest}"
            )
        return raw

    def resolve_linux_amd64(
        self, image: str | ImageRef
    ) -> tuple[ResolvedImage, dict[str, Any], dict[str, Any]]:
        """Resolve an image reference to a verified linux/amd64 manifest & config."""
        ref = parse_image_ref(image) if isinstance(image, str) else image
        digest, raw, media_type = self._get_manifest_raw(ref)
        document = json.loads(raw)
        index_digest: str | None = None

        if "manifests" in document:
            index_digest = digest
            matches = [
                child
                for child in document["manifests"]
                if child.get("platform", {}).get("os") == "linux"
                and child.get("platform", {}).get("architecture") == "amd64"
                and not child.get("annotations", {}).get("vnd.docker.reference.type")
            ]
            if not matches:
                raise HarborImageError(
                    f"Image {ref.canonical_repo}:{ref.reference} has no linux/amd64 manifest"
                )
            child = matches[0]
            digest, raw, media_type = self._get_manifest_raw(ref, child["digest"])
            if "size" in child and len(raw) != child["size"]:
                raise HarborImageError("Child manifest size mismatch")
            document = json.loads(raw)

        if document.get("schemaVersion") != 2 or "config" not in document:
            raise HarborImageError(
                f"Schema-2 OCI/Docker manifest required for {ref.canonical_repo}"
            )
        if not document.get("mediaType") and media_type:
            document["mediaType"] = media_type

        config_desc = document["config"]
        config_raw = self._get_blob(ref, config_desc["digest"])
        if len(config_raw) != config_desc["size"]:
            raise HarborImageError("Image config blob size mismatch")
        config_doc = json.loads(config_raw)
        if config_doc.get("os") != "linux" or config_doc.get("architecture") != "amd64":
            raise HarborImageError(
                f"Image {ref.canonical_repo}@{digest} is "
                f"{config_doc.get('os')}/{config_doc.get('architecture')}, expected linux/amd64"
            )

        resolved = ResolvedImage(
            image=ref.pinned(digest),
            registry_digest=digest,
            index_digest=index_digest,
            config={
                "digest": config_desc["digest"],
                "sha256": _sha256_bytes(config_raw),
                "os": "linux",
                "architecture": "amd64",
            },
            public_pull_verified=self._anonymous,
            tag=ref.tag,
        )
        return resolved, document, config_doc

    def _blob_exists(self, ref: ImageRef, digest: str) -> bool:
        resp = self._request(
            "HEAD",
            ref.registry,
            f"/v2/{ref.repository}/blobs/{digest}",
            scope_key=f"push:{ref.repository}",
        )
        return resp.status_code == 200

    def _mount_blob(self, target: ImageRef, source: ImageRef, digest: str) -> bool:
        if target.registry != source.registry:
            return False
        resp = self._request(
            "POST",
            target.registry,
            f"/v2/{target.repository}/blobs/uploads/",
            scope_key=f"push:{target.repository}:pull:{source.repository}",
            params={"mount": digest, "from": source.repository},
        )
        return resp.status_code == 201

    def _upload_blob(self, target: ImageRef, digest: str, data: bytes) -> None:
        if self._blob_exists(target, digest):
            return
        init = self._request(
            "POST",
            target.registry,
            f"/v2/{target.repository}/blobs/uploads/",
            scope_key=f"push:{target.repository}",
        )
        if init.status_code not in {202, 201}:
            raise HarborImageError(
                f"Failed to initiate blob upload to {target.canonical_repo} "
                f"(HTTP {init.status_code}): {init.text[:300]}"
            )
        if init.status_code == 201:
            return
        location = init.headers.get("Location")
        if not location:
            raise HarborImageError("Registry did not return upload Location header")
        upload_url = urljoin(f"{self._base_url(target.registry)}/", location)
        sep = "&" if "?" in upload_url else "?"
        final_url = f"{upload_url}{sep}digest={quote(digest, safe=':')}"
        put = self._request(
            "PUT",
            target.registry,
            final_url,
            scope_key=f"push:{target.repository}",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(data)),
            },
            content=data,
        )
        if put.status_code not in {201, 202}:
            raise HarborImageError(
                f"Failed to upload blob {digest} to {target.canonical_repo} "
                f"(HTTP {put.status_code}): {put.text[:300]}"
            )

    def _put_manifest(
        self, target: ImageRef, tag: str, manifest_bytes: bytes, media_type: str
    ) -> str:
        resp = self._request(
            "PUT",
            target.registry,
            f"/v2/{target.repository}/manifests/{tag}",
            scope_key=f"push:{target.repository}",
            headers={"Content-Type": media_type},
            content=manifest_bytes,
        )
        if resp.status_code not in {200, 201}:
            raise HarborImageError(
                f"Failed to push manifest {target.tagged(tag)} "
                f"(HTTP {resp.status_code}): {resp.text[:300]}"
            )
        computed = f"sha256:{_sha256_bytes(manifest_bytes)}"
        returned = resp.headers.get("Docker-Content-Digest")
        if returned and returned != computed:
            raise HarborImageError(
                f"Pushed manifest digest mismatch: returned={returned} computed={computed}"
            )
        return computed

    def append_layer(
        self,
        *,
        base_image: str,
        target_repo: str,
        target_tag: str,
        layer_tar_gz: bytes | None,
        diff_id: str | None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
        base_client: "OCIRegistryClient | None" = None,
    ) -> ResolvedImage:
        """Append a single deterministic layer + config updates to a base image."""
        base_ref = parse_image_ref(base_image)
        target_ref = parse_image_ref(target_repo).with_tag(target_tag)
        reader = base_client or self

        try:
            _, base_manifest, base_config = reader.resolve_linux_amd64(base_ref)
        except HarborImageError:
            if reader is self:
                raise
            reader = self
            _, base_manifest, base_config = reader.resolve_linux_amd64(base_ref)
        manifest_mt = base_manifest.get("mediaType") or _OCI_MANIFEST
        is_docker = "docker" in manifest_mt
        config_mt = _DOCKER_CONFIG if is_docker else _OCI_CONFIG
        layer_mt = _DOCKER_LAYER_GZIP if is_docker else _OCI_LAYER_GZIP

        # Ensure every base layer exists in target_repo (skipping already-present blobs).
        layers = list(base_manifest.get("layers", []))
        for layer in layers:
            digest = layer["digest"]
            if self._blob_exists(target_ref, digest):
                continue
            if self._mount_blob(target_ref, base_ref, digest):
                continue
            blob = reader._get_blob(base_ref, digest)
            self._upload_blob(target_ref, digest, blob)

        new_config = copy.deepcopy(base_config)
        new_config["created"] = "1970-01-01T00:00:00Z"
        cfg = new_config.setdefault("config", {})
        if workdir:
            cfg["WorkingDir"] = workdir
        if env:
            existing_env: dict[str, str] = {}
            for entry in cfg.get("Env") or []:
                k, _, v = entry.partition("=")
                existing_env[k] = v
            existing_env.update(env)
            cfg["Env"] = [f"{k}={v}" for k, v in sorted(existing_env.items())]
        if labels:
            existing_labels = dict(cfg.get("Labels") or {})
            existing_labels.update(labels)
            cfg["Labels"] = existing_labels

        history = list(new_config.get("history") or [])
        if layer_tar_gz is not None and diff_id is not None:
            layer_digest = f"sha256:{_sha256_bytes(layer_tar_gz)}"
            self._upload_blob(target_ref, layer_digest, layer_tar_gz)
            layers.append(
                {
                    "mediaType": layer_mt,
                    "size": len(layer_tar_gz),
                    "digest": layer_digest,
                }
            )
            new_config.setdefault("rootfs", {"type": "layers", "diff_ids": []})
            new_config["rootfs"]["diff_ids"].append(diff_id)
            history.append(
                {
                    "created": "1970-01-01T00:00:00Z",
                    "created_by": "openenv harbor build-images (oci-append)",
                }
            )
        else:
            history.append(
                {
                    "created": "1970-01-01T00:00:00Z",
                    "created_by": "openenv harbor build-images (config-only)",
                    "empty_layer": True,
                }
            )
        new_config["history"] = history

        config_bytes = json.dumps(
            new_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        config_digest = f"sha256:{_sha256_bytes(config_bytes)}"
        self._upload_blob(target_ref, config_digest, config_bytes)

        new_manifest = {
            "schemaVersion": 2,
            "mediaType": _DOCKER_MANIFEST_V2 if is_docker else _OCI_MANIFEST,
            "config": {
                "mediaType": config_mt,
                "size": len(config_bytes),
                "digest": config_digest,
            },
            "layers": layers,
        }
        manifest_bytes = json.dumps(
            new_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest_digest = self._put_manifest(
            target_ref,
            target_tag,
            manifest_bytes,
            new_manifest["mediaType"],
        )
        return ResolvedImage(
            image=target_ref.pinned(manifest_digest),
            registry_digest=manifest_digest,
            index_digest=None,
            config={
                "digest": config_digest,
                "sha256": _sha256_bytes(config_bytes),
                "os": "linux",
                "architecture": "amd64",
            },
            public_pull_verified=False,
            tag=target_tag,
        )


def _resolve_copy_sources(context_dir: Path, pattern: str) -> list[Path]:
    clean = pattern.lstrip("/")
    if clean in {".", ""}:
        return [context_dir]
    ctx_resolved = context_dir.resolve()
    if any(ch in clean for ch in ("*", "?", "[")):
        if "**" in PurePosixPath(clean).parts:
            raise HarborImageError(
                f"Recursive '**' COPY globs are not supported by OCI append: {pattern!r}"
            )
        matched = sorted(
            p
            for p in context_dir.glob(clean)
            if (p.resolve() == ctx_resolved or p.resolve().is_relative_to(ctx_resolved))
        )
        if not matched:
            raise HarborImageError(f"COPY pattern matched no files: {pattern!r}")
        return matched
    candidate = (context_dir / clean).resolve()
    if not (candidate == ctx_resolved or candidate.is_relative_to(ctx_resolved)):
        raise HarborImageError(f"COPY source escapes build context: {pattern!r}")
    if not candidate.exists():
        raise HarborImageError(f"COPY source not found in environment/: {pattern!r}")
    return [candidate]


def _copy_destination_is_dir(op: CopyOp, dest_norm: str) -> bool:
    raw = op.destination.strip()
    if (
        raw in {".", "..", "/"}
        or raw.endswith(("/", "/.", "/.."))
        or dest_norm == "/"
        or dest_norm.endswith("/")
        or len(op.sources) > 1
    ):
        return True
    # Destination equal to WORKDIR (e.g. WORKDIR /app + COPY f /app) is an existing directory.
    if op.workdir and dest_norm.rstrip("/") == op.workdir.rstrip("/"):
        return True
    return False


def build_copy_layer_tar(
    context_dir: Path, copy_ops: tuple[CopyOp, ...]
) -> tuple[bytes | None, str | None]:
    """Create a deterministic gzipped OCI layer tar for declarative COPY operations.

    Returns ``(gz_bytes, diff_id)`` where ``diff_id`` is ``sha256:<uncompressed>``,
    or ``(None, None)`` when there are no COPY operations.
    """
    if not copy_ops:
        return None, None

    entries: dict[str, tuple[str, bytes | None, int, int, int]] = {}
    # path -> ("dir"|"file", bytes|None, mode, uid, gid)

    def ensure_parents(rel_posix: str, uid: int, gid: int) -> None:
        parts = PurePosixPath(rel_posix).parts[:-1]
        cur = ""
        for part in parts:
            cur = f"{cur}/{part}" if cur else part
            entries.setdefault(cur, ("dir", None, 0o755, uid, gid))

    for op in copy_ops:
        dest_norm = _normalize_container_path(op.destination, op.workdir)
        dest_is_dir = _copy_destination_is_dir(op, dest_norm)
        dest_rel = dest_norm.strip("/")

        resolved_sources: list[Path] = []
        for src in op.sources:
            resolved_sources.extend(_resolve_copy_sources(context_dir, src))

        if len(resolved_sources) > 1:
            dest_is_dir = True

        for src_path in resolved_sources:
            if src_path.is_dir():
                if dest_rel:
                    ensure_parents(dest_rel + "/_", op.uid, op.gid)
                for child in sorted(src_path.rglob("*")):
                    if child.is_symlink():
                        raise HarborImageError(
                            f"Symlink not allowed in COPY source: {child}"
                        )
                    sub = child.relative_to(src_path).as_posix()
                    target = f"{dest_rel}/{sub}" if dest_rel else sub
                    if child.is_dir():
                        ensure_parents(target + "/_", op.uid, op.gid)
                        entries[target] = ("dir", None, 0o755, op.uid, op.gid)
                    else:
                        ensure_parents(target, op.uid, op.gid)
                        mode = 0o755 if (child.stat().st_mode & 0o111) else 0o644
                        entries[target] = (
                            "file",
                            child.read_bytes(),
                            mode,
                            op.uid,
                            op.gid,
                        )
            else:
                target = (
                    f"{dest_rel}/{src_path.name}"
                    if (dest_is_dir and dest_rel)
                    else (src_path.name if dest_is_dir else dest_rel)
                )
                ensure_parents(target, op.uid, op.gid)
                mode = 0o755 if (src_path.stat().st_mode & 0o111) else 0o644
                entries[target] = (
                    "file",
                    src_path.read_bytes(),
                    mode,
                    op.uid,
                    op.gid,
                )

    raw_buf = io.BytesIO()
    with tarfile.open(fileobj=raw_buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(entries):
            kind, data, mode, uid, gid = entries[path]
            info = tarfile.TarInfo(name=path)
            info.uid = uid
            info.gid = gid
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = mode
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                assert data is not None
                info.type = tarfile.REGTYPE
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    raw_bytes = raw_buf.getvalue()
    diff_id = f"sha256:{_sha256_bytes(raw_bytes)}"
    gz_buf = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=gz_buf, compresslevel=6, mtime=0
    ) as gz:
        gz.write(raw_bytes)
    return gz_buf.getvalue(), diff_id


def pack_build_context_tar_gz(
    env_dir: Path, *, extra_dockerfile: str | None = None
) -> bytes:
    """Pack an audited ``environment/`` directory into a deterministic ``.tar.gz``."""
    buf = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=buf, compresslevel=6, mtime=0
    ) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(env_dir.rglob("*")):
                if path.is_symlink() or path.is_dir():
                    continue
                rel = path.relative_to(env_dir).as_posix()
                data = path.read_bytes()
                if rel == "Dockerfile" and extra_dockerfile:
                    data = (
                        data.decode("utf-8").rstrip()
                        + "\n"
                        + extra_dockerfile.strip()
                        + "\n"
                    ).encode("utf-8")
                info = tarfile.TarInfo(name=rel)
                info.size = len(data)
                info.mode = 0o755 if (path.stat().st_mode & 0o111) else 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def parse_kaniko_digest(logs: str) -> str:
    """Extract the pinned ``sha256:...`` digest emitted by our Kaniko job wrapper."""
    matches = re.findall(
        rf"{re.escape(KANIKO_DIGEST_MARKER)}(sha256:[0-9a-f]{{64}})", logs
    )
    if not matches:
        raise HarborImageError(
            "Kaniko job finished without reporting a valid image digest"
        )
    return matches[-1]


class KanikoHFJobsBuilder:
    """Build and push Dockerfiles via Kaniko running inside Hugging Face Jobs."""

    def __init__(
        self,
        *,
        namespace: str | None = None,
        flavor: str = "cpu-basic",
        timeout: str = "30m",
        kaniko_image: str = DEFAULT_KANIKO_IMAGE,
        bucket_name: str = "openenv-kaniko-builds",
        cache_repo: str | None = None,
        poll_interval_s: float = 5.0,
        no_push: bool = False,
        hf_token: str | None = None,
        api: Any | None = None,
    ) -> None:
        self.namespace = namespace
        self.flavor = flavor
        self.timeout = timeout
        self.kaniko_image = kaniko_image
        self.bucket_name = bucket_name
        self.cache_repo = cache_repo
        self.poll_interval_s = poll_interval_s
        self.no_push = no_push
        self.hf_token = hf_token
        self._api = api
        self._bucket_ready = False
        self._bucket_lock = threading.Lock()

    def _get_api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.hf_token)
        return self._api

    def _effective_namespace(self) -> str:
        with self._bucket_lock:
            if self.namespace:
                return self.namespace
            who = self._get_api().whoami()
            name = (
                who.get("name") if isinstance(who, dict) else getattr(who, "name", None)
            )
            if not name:
                raise HarborImageError(
                    "Could not determine Hugging Face namespace; pass --namespace"
                )
            self.namespace = name
            return name

    def _ensure_bucket(self) -> str:
        ns = self._effective_namespace()
        bucket_id = (
            self.bucket_name if "/" in self.bucket_name else f"{ns}/{self.bucket_name}"
        )
        with self._bucket_lock:
            if not self._bucket_ready:
                self._get_api().create_bucket(bucket_id, private=True, exist_ok=True)
                self._bucket_ready = True
        return bucket_id

    def build_command(
        self,
        *,
        context_filename: str,
        destination: str,
        has_auth_secret: bool,
    ) -> list[str]:
        """Build the ``/busybox/sh -c`` command executed inside the Kaniko HF Job."""
        kaniko_args = [
            "/kaniko/executor",
            f"--context=tar:///context/{context_filename}",
            "--dockerfile=Dockerfile",
            f"--destination={destination}",
            "--custom-platform=linux/amd64",
            "--reproducible",
            "--digest-file=/tmp/digest",
        ]
        if self.no_push:
            kaniko_args.append("--no-push")
        if self.cache_repo:
            kaniko_args.extend(
                [
                    "--cache=true",
                    f"--cache-repo={self.cache_repo}",
                    "--cache-copy-layers",
                ]
            )

        steps = ["set -eu"]
        if has_auth_secret:
            steps.extend(
                [
                    "umask 077",
                    "/busybox/mkdir -p /kaniko/.docker",
                    'printf "%s" "$KANIKO_DOCKER_CONFIG_B64" | /busybox/base64 -d > /kaniko/.docker/config.json',
                    "unset KANIKO_DOCKER_CONFIG_B64",
                ]
            )
        steps.append(" ".join(shlex.quote(arg) for arg in kaniko_args))
        steps.append(f'printf "\\n{KANIKO_DIGEST_MARKER}%s\\n" "$(cat /tmp/digest)"')
        return ["/busybox/sh", "-c", " && ".join(steps)]

    def build(
        self,
        *,
        env_dir: Path,
        env_hash: str,
        target_repo: str,
        target_tag: str,
        extra_dockerfile: str | None = None,
        registry_username: str | None = None,
        registry_password: str | None = None,
    ) -> tuple[str, str]:
        """Run one Kaniko build on HF Jobs and return ``(digest, job_id)``."""
        from huggingface_hub import Volume

        api = self._get_api()
        namespace = self._effective_namespace()
        bucket_id = self._ensure_bucket()
        target_ref = parse_image_ref(target_repo)
        destination = target_ref.tagged(target_tag)

        secrets: dict[str, str] = {}
        if not self.no_push:
            creds = resolve_registry_credentials(
                target_ref.registry,
                username=registry_username,
                password=registry_password,
            )
            if not creds:
                raise HarborImageError(
                    f"No push credentials found for {target_ref.registry}. "
                    "Set GHCR_TOKEN / REGISTRY_PASSWORD or pass --registry-user / --registry-password."
                )
            cfg_bytes = build_docker_config_json(
                target_ref.registry, creds[0], creds[1]
            )
            secrets["KANIKO_DOCKER_CONFIG_B64"] = base64.b64encode(cfg_bytes).decode(
                "ascii"
            )

        run_id = uuid.uuid4().hex[:12]
        context_filename = f"{env_hash[:16]}.tar.gz"
        bucket_folder = f"contexts/{run_id}"
        bucket_path = f"{bucket_folder}/{context_filename}"
        tar_bytes = pack_build_context_tar_gz(
            env_dir, extra_dockerfile=extra_dockerfile
        )
        api.batch_bucket_files(bucket_id, add=[(tar_bytes, bucket_path)])

        job_id = ""
        try:
            command = self.build_command(
                context_filename=context_filename,
                destination=destination,
                has_auth_secret=bool(secrets),
            )
            job = api.run_job(
                image=self.kaniko_image,
                command=command,
                secrets=secrets or None,
                flavor=self.flavor,
                timeout=self.timeout,
                namespace=namespace,
                name=f"openenv-kaniko-{env_hash[:10]}",
                labels={
                    "openenv-harbor-build": run_id,
                    "openenv-env-hash": env_hash[:16],
                },
                volumes=[
                    Volume(
                        type="bucket",
                        source=bucket_id,
                        mount_path="/context",
                        read_only=True,
                        path=bucket_folder,
                    )
                ],
            )
            job_id = job.id
            transient_errors = 0
            while True:
                try:
                    info = api.inspect_job(job_id=job_id, namespace=namespace)
                    transient_errors = 0
                except Exception:  # noqa: BLE001 - retry transient Hub polling glitches
                    transient_errors += 1
                    if transient_errors > 5:
                        raise
                    time.sleep(min(30.0, self.poll_interval_s * transient_errors))
                    continue
                stage = getattr(getattr(info, "status", None), "stage", None)
                if stage == "COMPLETED":
                    break
                if stage in {"ERROR", "CANCELED", "CANCELLED", "TIMEOUT", "FAILED"}:
                    tail = "\n".join(
                        list(
                            api.fetch_job_logs(
                                job_id=job_id, namespace=namespace, follow=False
                            )
                        )[-40:]
                    )
                    raise HarborImageError(
                        f"Kaniko HF Job {job_id} ended with stage={stage}:\n{tail}"
                    )
                time.sleep(self.poll_interval_s)

            logs = "\n".join(
                api.fetch_job_logs(job_id=job_id, namespace=namespace, follow=False)
            )
            digest = parse_kaniko_digest(logs)
            return digest, job_id
        finally:
            try:
                api.batch_bucket_files(bucket_id, delete=[bucket_path])
            except Exception:  # noqa: BLE001 - best-effort cleanup of temp context
                pass


class DockerBuildxBuilder:
    """Build and push Dockerfiles via local or VM ``docker buildx build``."""

    def __init__(
        self,
        *,
        cache_repo: str | None = None,
        no_push: bool = False,
        timeout_s: int = 2400,
    ) -> None:
        self.cache_repo = cache_repo
        self.no_push = no_push
        self.timeout_s = timeout_s

    def build(
        self,
        *,
        env_dir: Path,
        env_hash: str,  # noqa: ARG002 - parity with KanikoHFJobsBuilder
        target_repo: str,
        target_tag: str,
        extra_dockerfile: str | None = None,
        registry_username: str | None = None,  # noqa: ARG002
        registry_password: str | None = None,  # noqa: ARG002
    ) -> tuple[str, str | None]:
        target_ref = parse_image_ref(target_repo)
        destination = target_ref.tagged(target_tag)

        with tempfile.TemporaryDirectory(prefix="openenv-buildx-") as tmp_str:
            tmp = Path(tmp_str)
            stage_dir = tmp / "context"
            shutil.copytree(env_dir, stage_dir)
            if extra_dockerfile:
                df = stage_dir / "Dockerfile"
                df.write_text(
                    df.read_text(encoding="utf-8").rstrip()
                    + "\n"
                    + extra_dockerfile.strip()
                    + "\n",
                    encoding="utf-8",
                )
            meta_path = tmp / "metadata.json"
            cmd = [
                "docker",
                "buildx",
                "build",
                "--platform",
                "linux/amd64",
                "--provenance=false",
                "--metadata-file",
                str(meta_path),
                "-t",
                destination,
            ]
            if self.cache_repo:
                cmd.extend(
                    [
                        "--cache-from",
                        f"type=registry,ref={self.cache_repo}",
                        "--cache-to",
                        f"type=registry,ref={self.cache_repo},mode=max",
                    ]
                )
            if self.no_push:
                cmd.append("--load")
            else:
                cmd.append("--push")
            cmd.append(str(stage_dir))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            if result.returncode != 0:
                tail = (result.stderr or result.stdout)[-1200:]
                raise HarborImageError(
                    f"docker buildx build failed for {destination}:\n{tail}"
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            digest = meta.get("containerimage.digest") or meta.get(
                "containerimage.config.digest"
            )
            if not digest or not DIGEST_RE.fullmatch(digest):
                raise HarborImageError(
                    f"docker buildx metadata missing valid containerimage.digest: {meta}"
                )
            return digest, None


def update_task_toml(
    task_toml_path: Path,
    *,
    docker_image: str,
    workdir: str | None = None,
) -> str:
    """Set ``[environment].docker_image`` (and optional ``workdir``) in ``task.toml``.

    Preserves surrounding formatting and validates the resulting TOML document.
    """
    original = task_toml_path.read_text(encoding="utf-8")
    # Confirm input parses cleanly before editing.
    tomllib.loads(original)

    lines = original.splitlines(keepends=True)
    env_start: int | None = None
    env_end = len(lines)
    section_re = re.compile(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$")
    key_re = re.compile(r"^(\s*)([A-Za-z0-9_-]+)\s*=")

    for idx, line in enumerate(lines):
        m = section_re.match(line)
        if m:
            section = m.group(1).strip()
            if section == "environment" and env_start is None:
                env_start = idx
            elif env_start is not None:
                env_end = idx
                break

    image_line = f"docker_image = {json.dumps(docker_image)}\n"
    workdir_line = f"workdir = {json.dumps(workdir)}\n" if workdir is not None else None

    if env_start is None:
        suffix = "" if (not original or original.endswith("\n")) else "\n"
        addition = f"{suffix}\n[environment]\n{image_line}"
        if workdir_line:
            addition += workdir_line
        updated = original + addition
    else:
        body = list(lines[env_start + 1 : env_end])
        replaced_image = False
        replaced_workdir = False
        for i, line in enumerate(body):
            km = key_re.match(line)
            if not km:
                continue
            key = km.group(2)
            if key == "docker_image":
                body[i] = f"{km.group(1)}{image_line}"
                replaced_image = True
            elif key == "workdir" and workdir_line is not None:
                body[i] = f"{km.group(1)}{workdir_line}"
                replaced_workdir = True
        if not replaced_image:
            body.insert(0, image_line)
        if workdir_line is not None and not replaced_workdir:
            body.insert(1, workdir_line)
        updated = "".join(lines[: env_start + 1] + body + lines[env_end:])

    parsed = tomllib.loads(updated)
    if parsed.get("environment", {}).get("docker_image") != docker_image:
        raise HarborImageError(f"Failed to update docker_image in {task_toml_path}")
    if workdir is not None and parsed.get("environment", {}).get("workdir") != workdir:
        raise HarborImageError(f"Failed to update workdir in {task_toml_path}")

    task_toml_path.write_text(updated, encoding="utf-8")
    return updated


def plan_harbor_images(
    root: Path,
    *,
    registry: str | None = None,
    tag_prefix: str = "",
    extra_dockerfile: str | None = None,
    skip_pinned: bool = True,
    allow_oci_append: bool = True,
    task_ids: list[str] | None = None,
) -> list[TaskImagePlan]:
    """Audit, classify, and plan image operations for every Harbor task under ``root``."""
    tasks = discover_task_dirs(root)
    if task_ids is not None:
        wanted = set(task_ids)
        tasks = [(tid, tdir) for tid, tdir in tasks if tid in wanted]
        missing = sorted(wanted - {tid for tid, _ in tasks})
        if missing:
            raise HarborImageError(
                f"Requested task ID(s) not found under {root}: {missing}"
            )
    plans: list[TaskImagePlan] = []

    for task_id, task_dir in tasks:
        cfg = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        env_cfg = cfg.get("environment") or {}
        existing = env_cfg.get("docker_image")
        dockerfile_path = task_dir / "environment" / "Dockerfile"
        has_dockerfile = dockerfile_path.is_file()

        context_files = audit_environment(task_dir, extra_dockerfile=extra_dockerfile)
        env_hash = compute_environment_hash(
            task_dir,
            extra_dockerfile=extra_dockerfile,
            existing_docker_image=existing,
        )

        if not has_dockerfile:
            if not existing:
                raise HarborImageError(
                    f"{task_id}: has neither environment/Dockerfile nor [environment].docker_image"
                )
            action = (
                "keep_pinned"
                if (skip_pinned and PINNED_IMAGE_RE.fullmatch(existing))
                else "verify_shared"
            )
            plans.append(
                TaskImagePlan(
                    task_id=task_id,
                    task_dir=task_dir,
                    action=action,
                    env_hash=env_hash,
                    reason=(
                        "already digest-pinned shared image (no Dockerfile)"
                        if action == "keep_pinned"
                        else "shared image without Dockerfile; resolve and pin linux/amd64 digest"
                    ),
                    base_image=existing,
                    existing_docker_image=existing,
                    context_files=context_files,
                )
            )
            continue

        if (
            skip_pinned
            and existing
            and PINNED_IMAGE_RE.fullmatch(existing)
            and not extra_dockerfile
        ):
            plans.append(
                TaskImagePlan(
                    task_id=task_id,
                    task_dir=task_dir,
                    action="keep_pinned",
                    env_hash=env_hash,
                    reason="task.toml already has a digest-pinned docker_image",
                    base_image=existing,
                    existing_docker_image=existing,
                    context_files=context_files,
                )
            )
            continue

        df_text = dockerfile_path.read_text(encoding="utf-8")
        if extra_dockerfile:
            df_text = df_text.rstrip() + "\n" + extra_dockerfile.strip() + "\n"
        parsed = classify_dockerfile(df_text)
        action = "oci_append" if (allow_oci_append and parsed.is_copy_only) else "build"
        target_tag = sanitize_tag(task_id, prefix=tag_prefix)
        if registry is not None:
            parse_image_ref(registry)  # validate target registry early

        plans.append(
            TaskImagePlan(
                task_id=task_id,
                task_dir=task_dir,
                action=action,
                env_hash=env_hash,
                reason=parsed.reason,
                base_image=parsed.base_image,
                existing_docker_image=existing,
                target_tag=target_tag,
                parsed_dockerfile=parsed,
                context_files=context_files,
            )
        )

    return plans


def verify_harbor_images(
    root: Path,
    *,
    anonymous: bool = True,
    require_digest_pin: bool = False,
    concurrency: int = 8,
    task_ids: list[str] | None = None,
    registry_client: OCIRegistryClient | None = None,
) -> dict[str, Any]:
    """Verify every Harbor task's ``docker_image`` is pullable as ``linux/amd64``."""
    tasks = discover_task_dirs(root)
    if task_ids is not None:
        wanted = set(task_ids)
        tasks = [(tid, tdir) for tid, tdir in tasks if tid in wanted]
        missing = sorted(wanted - {tid for tid, _ in tasks})
        if missing:
            raise HarborImageError(
                f"Requested task ID(s) not found under {root}: {missing}"
            )
    results: dict[str, dict[str, Any]] = {}

    def check_one(item: tuple[str, Path]) -> tuple[str, dict[str, Any]]:
        task_id, task_dir = item
        cfg = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        image = (cfg.get("environment") or {}).get("docker_image")
        if not image:
            return task_id, {
                "ok": False,
                "image": None,
                "error": "missing [environment].docker_image in task.toml",
            }
        if require_digest_pin and not PINNED_IMAGE_RE.fullmatch(image):
            return task_id, {
                "ok": False,
                "image": image,
                "error": "docker_image is not pinned to a @sha256 digest",
            }
        client = registry_client or OCIRegistryClient(anonymous=anonymous)
        try:
            resolved, _, _ = client.resolve_linux_amd64(image)
            return task_id, {
                "ok": True,
                "image": image,
                "pinned_image": resolved.image,
                "registry_digest": resolved.registry_digest,
                "public_pull_verified": resolved.public_pull_verified,
            }
        except Exception as exc:  # noqa: BLE001
            return task_id, {"ok": False, "image": image, "error": str(exc)}
        finally:
            if registry_client is None:
                client.close()

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for task_id, rec in pool.map(check_one, tasks):
            results[task_id] = rec

    ok_count = sum(1 for r in results.values() if r["ok"])
    return {
        "total": len(results),
        "ok": ok_count,
        "failed": len(results) - ok_count,
        "tasks": results,
    }


def build_harbor_images(
    root: Path,
    *,
    registry: str | None = None,
    builder: str = "auto",
    output_dir: Path | None = None,
    in_place: bool = False,
    receipts_path: Path | None = None,
    per_task_receipts_dir: Path | None = None,
    task_ids: list[str] | None = None,
    tag_prefix: str = "",
    extra_dockerfile: str | None = None,
    skip_pinned: bool = True,
    allow_oci_append: bool = True,
    strip_dockerfile_for_shared: bool = False,
    verify_public: bool = True,
    no_push: bool = False,
    concurrency: int = 4,
    namespace: str | None = None,
    flavor: str = "cpu-basic",
    timeout: str = "30m",
    kaniko_image: str = DEFAULT_KANIKO_IMAGE,
    bucket_name: str = "openenv-kaniko-builds",
    cache_repo: str | None = None,
    registry_username: str | None = None,
    registry_password: str | None = None,
    run_builder: Any | None = None,
    registry_client: OCIRegistryClient | None = None,
    public_client: OCIRegistryClient | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build/append/verify OCI images for Harbor tasks and stamp ``task.toml`` pins."""
    if builder not in {"auto", "kaniko", "docker", "oci-append"}:
        raise ValueError(f"Unsupported builder: {builder!r}")
    if output_dir is not None and in_place:
        raise ValueError("Pass either output_dir or in_place, not both")

    plans = plan_harbor_images(
        root,
        registry=registry,
        tag_prefix=tag_prefix,
        extra_dockerfile=extra_dockerfile,
        skip_pinned=skip_pinned,
        allow_oci_append=allow_oci_append,
        task_ids=task_ids,
    )

    if output_dir is not None:
        out_resolved = output_dir.resolve()
        src_resolved = root.resolve()
        if (
            out_resolved == src_resolved
            or out_resolved.is_relative_to(src_resolved)
            or src_resolved.is_relative_to(out_resolved)
            or any(
                out_resolved == p.task_dir.resolve()
                or out_resolved.is_relative_to(p.task_dir.resolve())
                or p.task_dir.resolve().is_relative_to(out_resolved)
                for p in plans
            )
        ):
            raise HarborImageError(
                "output_dir must not overlap the input task/dataset directory; "
                "use --in-place to update task.toml in place"
            )

    needs_registry = any(p.action in {"oci_append", "build"} for p in plans)
    if needs_registry and not registry and not no_push:
        raise HarborImageError(
            "A target --registry (e.g. ghcr.io/your-org/your-repo) is required to push built images"
        )
    effective_registry = registry or "ghcr.io/local/no-push"

    # Load resumable receipts if present (single manifest and/or per-task directory).
    prior_receipts: dict[str, dict[str, Any]] = {}
    if receipts_path and receipts_path.is_file():
        loaded = json.loads(receipts_path.read_text(encoding="utf-8"))
        prior_receipts.update(
            loaded.get("tasks", loaded) if isinstance(loaded, dict) else {}
        )
    if per_task_receipts_dir and per_task_receipts_dir.is_dir():
        for p in sorted(per_task_receipts_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(rec, dict) and rec.get("task_id"):
                prior_receipts[rec["task_id"]] = rec

    # Choose RUN builder lazily only if needed.
    needs_run_builder = any(p.action == "build" for p in plans)
    active_run_builder = run_builder
    active_builder_name = builder
    if needs_run_builder and active_run_builder is None:
        if builder == "oci-append":
            run_tasks = [p.task_id for p in plans if p.action == "build"]
            raise HarborImageError(
                f"Tasks require RUN instructions (cannot use --builder oci-append): {run_tasks[:5]}"
            )
        chosen = builder
        if chosen == "auto":
            chosen = (
                "kaniko" if (namespace or shutil.which("docker") is None) else "docker"
            )
        active_builder_name = chosen
        if chosen == "kaniko":
            active_run_builder = KanikoHFJobsBuilder(
                namespace=namespace,
                flavor=flavor,
                timeout=timeout,
                kaniko_image=kaniko_image,
                bucket_name=bucket_name,
                cache_repo=cache_repo,
                no_push=no_push,
            )
        else:
            active_run_builder = DockerBuildxBuilder(
                cache_repo=cache_repo,
                no_push=no_push,
            )

    # Deduplicate work by (action, env_hash).
    groups: dict[tuple[str, str], list[TaskImagePlan]] = {}
    for plan in plans:
        groups.setdefault((plan.action, plan.env_hash), []).append(plan)

    own_reg = registry_client is None
    own_pub = public_client is None
    reg_client = registry_client or OCIRegistryClient(
        username=registry_username, password=registry_password
    )
    pub_client = public_client or OCIRegistryClient(anonymous=True)

    receipts_by_task: dict[str, TaskImageReceipt] = {}

    def verify_pinned(image_ref: str) -> ResolvedImage:
        verifier = pub_client if verify_public else reg_client
        resolved, _, _ = verifier.resolve_linux_amd64(image_ref)
        return resolved

    def execute_group(
        key: tuple[str, str], members: list[TaskImagePlan]
    ) -> list[TaskImageReceipt]:
        action, env_hash = key
        rep = members[0]

        # Check resumable cache on representative task.
        cached = prior_receipts.get(rep.task_id)
        if (
            not no_push
            and isinstance(cached, dict)
            and cached.get("status") == "resolved"
            and cached.get("env_hash") == env_hash
            and cached.get("image")
            and (
                action in {"keep_pinned", "verify_shared"}
                or cached["image"].startswith(
                    parse_image_ref(effective_registry).canonical_repo + "@"
                )
            )
        ):
            if progress:
                progress(
                    f"reuse   {rep.task_id} ({len(members)} task(s)) -> {cached['image']}"
                )
            return [
                TaskImageReceipt(
                    task_id=m.task_id,
                    status="resolved",
                    action=m.action,
                    builder=cached.get("builder", "cached"),
                    env_hash=m.env_hash,
                    image=cached["image"],
                    registry_digest=cached.get("registry_digest"),
                    tag=cached.get("tag"),
                    config=cached.get("config"),
                    public_pull_verified=bool(cached.get("public_pull_verified")),
                    context_files=m.context_files,
                    job_id=cached.get("job_id"),
                )
                for m in members
            ]

        try:
            if action in {"keep_pinned", "verify_shared"}:
                assert rep.base_image is not None
                resolved = verify_pinned(rep.base_image)
                if progress:
                    progress(
                        f"verify  {rep.task_id} ({len(members)} task(s)) -> {resolved.image}"
                    )
                return [
                    TaskImageReceipt(
                        task_id=m.task_id,
                        status="resolved",
                        action=m.action,
                        builder="verify",
                        env_hash=m.env_hash,
                        image=resolved.image,
                        registry_digest=resolved.registry_digest,
                        tag=resolved.tag,
                        config=resolved.config,
                        public_pull_verified=resolved.public_pull_verified,
                        context_files=m.context_files,
                    )
                    for m in members
                ]

            tag = rep.target_tag or sanitize_tag(rep.task_id, prefix=tag_prefix)
            if len(members) > 1:
                tag = sanitize_tag(f"env-{env_hash[:12]}", prefix=tag_prefix)

            if action == "oci_append":
                assert rep.parsed_dockerfile is not None
                layer_gz, diff_id = build_copy_layer_tar(
                    rep.task_dir / "environment", rep.parsed_dockerfile.copy_ops
                )
                if no_push:
                    if progress:
                        progress(
                            f"dry-oci {rep.task_id} ({len(members)} task(s)) diff={diff_id}"
                        )
                    return [
                        TaskImageReceipt(
                            task_id=m.task_id,
                            status="built_no_push",
                            action=m.action,
                            builder="oci-append",
                            env_hash=m.env_hash,
                            registry_digest=diff_id,
                            tag=tag,
                            context_files=m.context_files,
                        )
                        for m in members
                    ]
                pushed = reg_client.append_layer(
                    base_image=rep.parsed_dockerfile.base_image,
                    target_repo=effective_registry,
                    target_tag=tag,
                    layer_tar_gz=layer_gz,
                    diff_id=diff_id,
                    workdir=rep.parsed_dockerfile.workdir,
                    env=rep.parsed_dockerfile.env or None,
                    labels=rep.parsed_dockerfile.labels or None,
                    base_client=pub_client,
                )
                resolved = verify_pinned(pushed.image)
                if progress:
                    progress(
                        f"append  {rep.task_id} ({len(members)} task(s)) -> {resolved.image}"
                    )
                return [
                    TaskImageReceipt(
                        task_id=m.task_id,
                        status="resolved",
                        action=m.action,
                        builder="oci-append",
                        env_hash=m.env_hash,
                        image=resolved.image,
                        registry_digest=resolved.registry_digest,
                        tag=tag,
                        config=resolved.config,
                        public_pull_verified=resolved.public_pull_verified,
                        context_files=m.context_files,
                    )
                    for m in members
                ]

            # Full RUN build (Kaniko on HF Jobs or Docker Buildx)
            assert active_run_builder is not None
            digest, job_id = active_run_builder.build(
                env_dir=rep.task_dir / "environment",
                env_hash=env_hash,
                target_repo=effective_registry,
                target_tag=tag,
                extra_dockerfile=extra_dockerfile,
                registry_username=registry_username,
                registry_password=registry_password,
            )
            if no_push:
                if progress:
                    progress(
                        f"built   {rep.task_id} ({len(members)} task(s)) digest={digest}"
                    )
                return [
                    TaskImageReceipt(
                        task_id=m.task_id,
                        status="built_no_push",
                        action=m.action,
                        builder=active_builder_name,
                        env_hash=m.env_hash,
                        registry_digest=digest,
                        tag=tag,
                        context_files=m.context_files,
                        job_id=job_id,
                    )
                    for m in members
                ]
            pinned = parse_image_ref(effective_registry).pinned(digest)
            resolved = verify_pinned(pinned)
            if progress:
                progress(
                    f"built   {rep.task_id} ({len(members)} task(s)) -> {resolved.image}"
                )
            return [
                TaskImageReceipt(
                    task_id=m.task_id,
                    status="resolved",
                    action=m.action,
                    builder=active_builder_name,
                    env_hash=m.env_hash,
                    image=resolved.image,
                    registry_digest=resolved.registry_digest,
                    tag=tag,
                    config=resolved.config,
                    public_pull_verified=resolved.public_pull_verified,
                    context_files=m.context_files,
                    job_id=job_id,
                )
                for m in members
            ]
        except Exception as exc:  # noqa: BLE001
            if progress:
                progress(f"FAIL    {rep.task_id}: {exc}")
            return [
                TaskImageReceipt(
                    task_id=m.task_id,
                    status="failed",
                    action=m.action,
                    builder="oci-append"
                    if action == "oci_append"
                    else active_builder_name,
                    env_hash=m.env_hash,
                    context_files=m.context_files,
                    error=str(exc),
                )
                for m in members
            ]

    def _persist_per_task_receipt(receipt: TaskImageReceipt) -> None:
        if per_task_receipts_dir is None:
            return
        per_task_receipts_dir.mkdir(parents=True, exist_ok=True)
        dest = per_task_receipts_dir / f"{receipt.task_id}.json"
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(dest)

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [
                pool.submit(execute_group, key, members)
                for key, members in groups.items()
            ]
            for fut in as_completed(futures):
                for receipt in fut.result():
                    receipts_by_task[receipt.task_id] = receipt
                    _persist_per_task_receipt(receipt)
    finally:
        if own_reg:
            reg_client.close()
        if own_pub:
            pub_client.close()

    # Materialize updated task.toml files when output_dir or in_place is requested.
    if not no_push and (output_dir is not None or in_place):
        for plan in plans:
            rec = receipts_by_task[plan.task_id]
            if rec.status != "resolved" or not rec.image:
                continue
            if output_dir is not None:
                dest_dir = (
                    output_dir
                    if (root.resolve() / "task.toml").is_file()
                    else output_dir / "tasks" / plan.task_id
                )
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(plan.task_dir, dest_dir)
                target_task_dir = dest_dir
            else:
                target_task_dir = plan.task_dir

            if strip_dockerfile_for_shared and plan.action in {
                "keep_pinned",
                "verify_shared",
            }:
                df = target_task_dir / "environment" / "Dockerfile"
                if df.is_file():
                    df.unlink()

            update_task_toml(target_task_dir / "task.toml", docker_image=rec.image)

    summary = {
        "total_tasks": len(plans),
        "unique_environments": len(groups),
        "resolved": sum(r.status == "resolved" for r in receipts_by_task.values()),
        "built_no_push": sum(
            r.status == "built_no_push" for r in receipts_by_task.values()
        ),
        "failed": sum(r.status == "failed" for r in receipts_by_task.values()),
        "actions": {
            action: sum(p.action == action for p in plans)
            for action in ("keep_pinned", "verify_shared", "oci_append", "build")
        },
        "registry": effective_registry if needs_registry else None,
        "tasks": {
            tid: receipts_by_task[tid].to_dict() for tid in sorted(receipts_by_task)
        },
    }

    if receipts_path is not None:
        receipts_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_receipts = receipts_path.with_suffix(receipts_path.suffix + ".tmp")
        tmp_receipts.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp_receipts.replace(receipts_path)

    return summary
