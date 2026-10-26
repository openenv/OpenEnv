# SPDX-License-Identifier: BSD-3-Clause

"""Read metadata from an immutable Git tree, never from candidate imports."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from urllib.parse import quote

from .errors import CatalogError
from .models import github_repository, relative_path

MAX_METADATA_BYTES = 1024 * 1024
MAX_TREE_BYTES = 16 * 1024 * 1024
MAX_ENVIRONMENTS = 10_000
GIT_TIMEOUT_SECONDS = 30


class MetadataError(CatalogError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class GitMetadataSource:
    """A bounded metadata reader pinned to one Git commit."""

    def __init__(
        self,
        repository: Path,
        repository_uri: str,
        revision: str,
        root: str,
    ):
        try:
            self.repository_id = github_repository(repository_uri)
            self.root = relative_path(root)
        except ValueError as error:
            raise CatalogError(str(error)) from error
        self.repository = repository
        self.uri = repository_uri
        self.revision = (
            self._git(
                ["rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
                256,
            )
            .decode("ascii")
            .strip()
        )
        rows = self._git(["ls-tree", "-r", "-z", self.revision], MAX_TREE_BYTES).split(
            b"\x00"
        )
        self.files: dict[str, tuple[str, str]] = {}
        for row in rows:
            if not row:
                continue
            metadata, name = row.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split()
            if kind == "blob":
                try:
                    filename = name.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise CatalogError(
                        "Repository metadata paths must be UTF-8"
                    ) from error
                self.files[filename] = (mode, oid)
        prefix = "" if self.root == "." else self.root + "/"
        if not any(path.startswith(prefix) for path in self.files):
            raise CatalogError("The configured inventory root does not exist")
        self.environments = sorted(
            path.rsplit("/", 1)[0]
            for path in self.files
            if path.startswith(prefix)
            and path.endswith("/openenv.yaml")
            and len(path[len(prefix) :].split("/")) == 2
        )
        if len(self.environments) > MAX_ENVIRONMENTS:
            raise CatalogError("Repository inventory exceeds the environment limit")

    def _git(self, arguments: list[str], limit: int) -> bytes:
        environment = dict(os.environ)
        environment["GIT_NO_REPLACE_OBJECTS"] = "1"
        environment["GIT_NO_LAZY_FETCH"] = "1"
        environment["GIT_TERMINAL_PROMPT"] = "0"
        try:
            process = subprocess.Popen(
                ["git", "--no-pager", "-C", str(self.repository), *arguments],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
            )
        except OSError as error:
            raise CatalogError("Cannot read the configured Git revision") from error
        timer = threading.Timer(GIT_TIMEOUT_SECONDS, process.kill)
        timer.start()
        try:
            if process.stdout is None:
                raise CatalogError("Cannot open the Git metadata stream")
            output = process.stdout.read(limit + 1)
            if len(output) > limit:
                process.kill()
                raise CatalogError("Git metadata exceeds the configured size limit")
            if process.wait() != 0:
                raise CatalogError("Cannot read the configured Git revision")
            return output
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout is not None:
                process.stdout.close()

    def read(self, path: str, *, required: bool = False) -> str | None:
        relative_path(path)
        item = self.files.get(path)
        if item is None:
            if required:
                raise MetadataError(
                    "missing_metadata", f"Required metadata is absent: {path}"
                )
            return None
        mode, oid = item
        if mode == "120000":
            raise MetadataError(
                "symlink_metadata", f"Metadata must be a regular tracked file: {path}"
            )
        size = int(self._git(["cat-file", "-s", oid], 32).decode("ascii"))
        if size > MAX_METADATA_BYTES:
            raise MetadataError("metadata_limit", f"Metadata file is too large: {path}")
        try:
            return self._git(["cat-file", "blob", oid], MAX_METADATA_BYTES).decode(
                "utf-8"
            )
        except UnicodeDecodeError as error:
            raise MetadataError(
                "invalid_metadata", f"Metadata is not UTF-8: {path}"
            ) from error

    def url(self, path: str) -> str:
        relative_path(path)
        return (
            f"https://github.com/{self.repository_id}/blob/{self.revision}/"
            f"{quote(path, safe='/')}"
        )
