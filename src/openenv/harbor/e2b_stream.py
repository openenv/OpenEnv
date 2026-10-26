"""E2B uploads using the SDK's streaming gzip transport and a total deadline."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

from harbor.environments.e2b import E2BEnvironment
from tenacity import retry, stop_after_attempt, wait_exponential


class E2BStreamingEnvironment(E2BEnvironment):
    """Preserve file contents while bounding large artifact upload memory and time.

    Selected explicitly through Harbor's native environment import path. Retrying
    a file upload overwrites the same target with the same source; commands and
    model requests retain their existing policies. The SDK decompresses in envd.
    """

    @retry(
        stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10), reraise=True
    )
    async def upload_file(self, source_path: Path | str, target_path: str):
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        with Path(source_path).open("rb") as stream:
            await asyncio.wait_for(
                self._sandbox.files.write(
                    target_path,
                    stream,
                    gzip=True,
                    use_octet_stream=True,
                    request_timeout=30,
                ),
                timeout=120,
            )

    @retry(
        stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10), reraise=True
    )
    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        source = Path(source_dir)
        paths = sorted(p for p in source.rglob("*") if p.is_file())
        for start in range(0, len(paths), self._UPLOAD_BATCH_SIZE):
            with ExitStack() as stack:
                entries = [
                    {
                        "path": str(
                            PurePosixPath(target_dir)
                            / path.relative_to(source).as_posix()
                        ),
                        "data": stack.enter_context(path.open("rb")),
                    }
                    for path in paths[start : start + self._UPLOAD_BATCH_SIZE]
                ]
                await asyncio.wait_for(
                    self._sandbox.files.write_files(
                        entries, gzip=True, use_octet_stream=True, request_timeout=30
                    ),
                    timeout=120,
                )
