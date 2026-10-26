"""Dataset discovery: resolve Harbor task sets and serve them over OpenEnv's Task API.

`HarborTaskProvider` satisfies `openenv.core.env_server.interfaces.TaskProvider`, so the HTTP routes
(`/{env}/splits`, `/{env}/tasks`, `/{env}/task`, ...) come for free once an environment exposes it.

Two constraints from that Protocol shape the design, and both are easy to violate:

  * **it must be side-effect free.** Discovery must not boot a sandbox or start a job.
  * **it must work on a freshly constructed instance.** The route handlers build a throwaway
    environment per request purely to answer, so anything expensive has to be cached at module level
    rather than on `self`, or every `/task` call re-downloads a dataset.

Three source kinds, resolved by shape:

    AdithyaSK/data_agent_rl_environment_train   HF dataset repo  -> snapshot_download
    /path/to/tasks                              local directory
    terminal-bench@1.0                          Harbor registry name@version

Harbor has no HuggingFace path of its own — its datasets are git repos or Harbor Hub packages. But an
HF dataset laid out as `tasks/<name>/` is already a directory of Harbor task dirs, so downloading it
and pointing Harbor's local mode at the result needs no new concepts.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .models import HarborTaskRef

# Resolution is expensive (a download on first use) and the Task API constructs a throwaway
# environment per HTTP request, so the cache has to outlive the instance.
_CACHE: dict[str, list[Path]] = {}
_LOCK = threading.Lock()


def _is_hf_repo(spec: str) -> bool:
    """`org/name`, not a path and not `name@version`."""
    return (
        "/" in spec
        and "@" not in spec
        and not spec.startswith((".", "/", "~"))
        and len(spec.split("/")) == 2
    )


# Validating a task means reading and parsing several files inside it. That is 0.8 ms per task on
# local SSD, so 2s for a 2238-task suite, and a mounted bucket is an order of magnitude slower per
# read: listing a dataset then costs minutes and the Task API times out before answering.
#
# So discovery lists directories and does not validate. A task that is malformed surfaces as a failed
# rollout, with Harbor's own error, instead of being silently absent from the listing. That is also
# the more honest behaviour: filtering during discovery makes a broken task look like it was never
# in the dataset, and shifts the index of every task after it.
_VALIDATE_TASKS = os.environ.get("OPENENV_VALIDATE_TASKS", "").lower() in (
    "1",
    "true",
    "yes",
)


_MAX_TASK_DEPTH = 6


def _nested_task_dirs(base: Path) -> list[Path]:
    """Every folder under `base` holding a `task.toml`, by relative path; a task's own subfolders
    are not searched.

    A symlinked folder is not followed: it can point outside the dataset, or back into it, which
    would recurse until the thread dies. Neither is anything deeper than `_MAX_TASK_DEPTH` levels.
    """
    found: list[Path] = []

    def walk(folder: Path, depth: int) -> None:
        for p in sorted(folder.iterdir()):
            if p.name.startswith(".") or p.is_symlink() or not p.is_dir():
                continue
            if (p / "task.toml").is_file():
                found.append(p)
            elif depth < _MAX_TASK_DEPTH:
                walk(p, depth + 1)

    walk(base, 1)
    return found


def _task_dirs_from_directory(
    root: Path, *, validate: bool | None = None
) -> list[Path]:
    """Task dirs under `root`, preferring a `tasks/` subdir when present.

    Args:
        root (`Path`):
            Dataset root, either containing `tasks/` or being the task directory itself.
        validate (`bool`, *optional*):
            Check each directory with Harbor's `Task.is_valid_dir`. Defaults to
            `$OPENENV_VALIDATE_TASKS`, off, because it costs a file read per task and discovery is
            on the latency path for every `/splits` and `/task` call.
    """
    base = root / "tasks" if (root / "tasks").is_dir() else root
    candidates = sorted(
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if candidates and not any((p / "task.toml").is_file() for p in candidates):
        # Grouped: `tasks/<field>/<subfield>/<task>/` (terminal-bench-science). Only when no top
        # level folder is a task, so a flat dataset keeps exactly the order, and so the indexes, it
        # always had.
        candidates = _nested_task_dirs(base)
    if not (_VALIDATE_TASKS if validate is None else validate):
        return candidates
    try:
        from harbor.models.task.task import Task
    except ImportError:
        return candidates
    return [p for p in candidates if Task.is_valid_dir(p, disable_verification=True)]


def resolve_task_dirs(
    spec: str, *, refresh: bool = False, tqdm_class: Any = None
) -> list[Path]:
    """Resolve a dataset spec to an ordered list of Harbor task directories.

    `tqdm_class` is handed to the Hub download, for a caller that shows its progress.

    Order is stable (sorted by directory name) because a task's *index* is its identity everywhere
    downstream — a trainer's dataset row, a `run_rollout` argument, a result. An unstable order would
    silently change which task an index refers to between runs.
    """
    with _LOCK:
        if not refresh and spec in _CACHE:
            return _CACHE[spec]

    path = Path(spec).expanduser()
    if path.is_dir():
        dirs = _task_dirs_from_directory(path)
    elif _is_hf_repo(spec):
        dirs = _task_dirs_from_directory(
            _materialise_hf_dataset(spec, tqdm_class=tqdm_class)
        )
    else:
        dirs = _registry_task_dirs(spec)

    if not dirs:
        raise ValueError(
            f"no Harbor tasks found for {spec!r}. Expected an HF dataset repo laid out as "
            "`tasks/<name>/`, a local directory of task dirs, or a Harbor registry `name@version`."
        )

    with _LOCK:
        _CACHE[spec] = dirs
    return dirs


# Task dirs must contain REAL FILES, not symlinks.
#
# The default HF cache is a symlink farm: every file under `snapshots/<rev>/` points at
# `../../blobs/<sha>`. Harbor uploads a task's `tests/` directory to the sandbox by tarring it, and
# tar faithfully preserves symlinks — so the sandbox receives `test.sh -> ../../../blobs/09b32…`,
# pointing at a path that does not exist there. Bash reports a dangling symlink as
# "No such file or directory", which makes it look like the upload failed when the entry is right
# there in `ls`.
#
# That cost a long debugging session: `upload_dir` appeared to work, `chmod +x` as root succeeded,
# and only `ls -la` revealed the arrow. Backends differ in whether they hit it — E2B's upload path
# does not preserve symlinks, Modal's tar-based one does — so it presents as "Modal is broken".
#
# `local_dir=` makes huggingface_hub write real files instead of populating the symlink cache, which
# fixes it for every backend at once and needs no Harbor change.
_DATASET_ROOT = Path(
    os.environ.get("OPENENV_DATASET_CACHE")
    or (Path.home() / ".cache" / "openenv" / "harbor-datasets")
)


# A Harbor task suite is thousands of tiny files (a `task.toml`, a Dockerfile, a test script per
# task), so wall clock is dominated by per-file round trips rather than bytes. Raising concurrency is
# the lever that matters; `hf_transfer` optimises large-file throughput and does comparatively little
# here, but costs nothing when it is installed.
_DOWNLOAD_WORKERS = int(os.environ.get("OPENENV_DATASET_WORKERS", "32"))


def _materialise_hf_dataset(spec: str, *, tqdm_class: Any = None) -> Path:
    """Download an HF dataset as real files and return its local root.

    Mounting beats downloading where it is available: a deployed Space can attach the dataset repo
    as a read-only volume and pass the mount path as the dataset spec, which skips this entirely.
    `openenv harbor push` does that automatically. This path is for local runs.
    """
    from huggingface_hub import snapshot_download

    target = _DATASET_ROOT / spec.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        spec,
        repo_type="dataset",
        allow_patterns=["tasks/**"],
        local_dir=str(target),
        max_workers=_DOWNLOAD_WORKERS,
        tqdm_class=tqdm_class,
    )
    return target


def has_symlinks(task_dir: Path) -> list[Path]:
    """Any symlinks under `task_dir`. Non-empty means uploads to a tar-based backend will break."""
    return [p for p in task_dir.rglob("*") if p.is_symlink()]


def _registry_task_dirs(spec: str) -> list[Path]:
    """A Harbor registry dataset, e.g. `terminal-bench@1.0`. Downloads on first use."""
    from harbor.models.job.config import DatasetConfig
    from openenv.core.utils import run_async_safely

    name, _, version = spec.partition("@")
    config = DatasetConfig(name=name, version=version or None)
    # Discovery is reached from async callers too (`run_batch` is a coroutine), and `asyncio.run`
    # cannot be called from a running loop.
    task_configs = run_async_safely(config.get_task_configs(disable_verification=True))
    return [Path(str(t.get_local_path())) for t in task_configs]


def _dataset_folder(spec: str, task_dir: Path) -> Path | None:
    """The folder a dataset's files must stay in, with a link to that folder itself followed (an
    operator may serve `~/datasets/current`): the local folder, the Hub download, or Harbor's cache
    for a registry task. `None` for a registry task Harbor keeps elsewhere (a local path its
    registry names), which Harbor has already resolved."""
    path = Path(spec).expanduser()
    if path.is_dir():
        return path.resolve()
    if _is_hf_repo(spec):
        return (_DATASET_ROOT / spec.replace("/", "__")).resolve()
    try:
        from harbor.constants import CACHE_DIR
    except ImportError:
        return None
    if task_dir.absolute().is_relative_to(CACHE_DIR.absolute()):
        return CACHE_DIR.resolve()
    return None


def task_root(spec: str | None, task_dir: Path) -> Path | None:
    """The one folder a task's files are read from to be shown (the Task API's instruction, and
    everything the UI shows): `task_dir` resolved, or `None` when a link (the task folder, `tasks/`,
    anything between) takes it outside its dataset, and then nothing in it is read. Without a
    dataset to anchor on, the task folder may not be a link. Rollouts are not affected: Harbor
    reads a task for itself."""
    real = task_dir.resolve()
    folder = _dataset_folder(spec, task_dir) if spec else None
    if folder is None:
        return None if task_dir.is_symlink() else real
    return real if real.is_relative_to(folder) else None


def own_file(root: Path | None, name: str) -> Path:
    """`root / name` when it resolves inside the task's root (see `task_root`), else `OSError`,
    which readers already treat as a missing file."""
    if root is None:
        raise OSError("this task's folder is outside its dataset")
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise OSError(f"{name} points outside its task")
    return path


def read_instruction(
    task_dir: Path, *, limit: int = 4000, spec: str | None = None
) -> str:
    """The task's prompt, for previewing in discovery. Truncated: this is not the authoritative copy.

    The sandbox gets the real instruction from Harbor at run time. Serving a huge prompt over the
    Task API for every listed task would make `list_tasks` enormous for no benefit. Read only
    inside the task's dataset (`task_root`), since the Task API is public on a Space.
    """
    try:
        path = own_file(task_root(spec, task_dir), "instruction.md")
    except OSError:
        return ""
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace").strip()
    return text if len(text) <= limit else text[:limit] + "\n…"


def prefetch(datasets: list[str]) -> dict[str, Any]:
    """Resolve every dataset up front, downloading if needed.

    Called before the server accepts traffic. Two reasons it is worth doing eagerly rather than on
    first use: a 2000-task HF repo takes real time to fetch, and a caller who mistypes a dataset
    name should learn at startup rather than when the first rollout 404s. Failures are collected
    rather than raised, so one bad dataset does not stop the server serving the good ones.

    Args:
        datasets (`list[str]`):
            Dataset specs — HF repo id, local path, or Harbor `name@version`.

    Returns:
        `dict` mapping each spec to `{"num_tasks": int}` or `{"error": str}`.
    """
    report: dict[str, Any] = {}
    for spec in datasets:
        try:
            report[spec] = {"num_tasks": len(resolve_task_dirs(spec))}
        except Exception as exc:  # noqa: BLE001 - one broken dataset must not hide the others
            report[spec] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return report


class HarborTaskProvider:
    """Serves one or more Harbor datasets as OpenEnv splits.

    A split IS a dataset spec: start the server with two datasets and you get two splits. That keeps
    the mapping obvious in both directions — a split name is something you can paste back into
    `--dataset` — rather than inventing a train/test split Harbor does not have.
    """

    def __init__(self, datasets: list[str] | None = None) -> None:
        self._datasets = list(datasets or [])

    # --- TaskProvider protocol ------------------------------------------
    def list_splits(self) -> list[dict[str, Any]]:
        splits = []
        for spec in self._datasets:
            try:
                n = len(resolve_task_dirs(spec))
                splits.append({"name": spec, "num_tasks": n})
            except Exception as exc:  # noqa: BLE001 - a broken dataset must not hide the good ones
                splits.append({"name": spec, "num_tasks": 0, "error": str(exc)[:200]})
        return splits

    def num_tasks(self, split: str) -> int:
        return len(resolve_task_dirs(self._check(split)))

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        spec = self._check(split)
        return [
            self._ref(spec, i, d).model_dump()
            for i, d in enumerate(resolve_task_dirs(spec))
        ]

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        spec = self._check(split)
        dirs = resolve_task_dirs(spec)
        if not 0 <= index < len(dirs):
            raise IndexError(
                f"task index {index} out of range for {spec!r} ({len(dirs)} tasks)"
            )
        return self._ref(spec, index, dirs[index]).model_dump()

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        spec = self._check(split)
        dirs = resolve_task_dirs(spec)
        return [
            self._ref(spec, i, d).model_dump()
            for i, d in list(enumerate(dirs))[start:stop]
        ]

    # --- internals -------------------------------------------------------
    def task_dir(self, split: str, index: int) -> Path:
        """The on-disk task dir for an index. Used by the rollout path, not by discovery."""
        dirs = resolve_task_dirs(self._check(split))
        if not 0 <= index < len(dirs):
            raise IndexError(f"task index {index} out of range ({len(dirs)} tasks)")
        return dirs[index]

    def _check(self, split: str) -> str:
        if not self._datasets:
            raise ValueError("this server was started with no datasets; pass --dataset")
        if not split:
            return self._datasets[0]
        if split not in self._datasets:
            raise ValueError(
                f"unknown split {split!r}; served splits are {self._datasets}"
            )
        return split

    @staticmethod
    def _ref(spec: str, index: int, task_dir: Path) -> HarborTaskRef:
        return HarborTaskRef(
            index=index,
            task_id=str(task_dir),
            task_name=task_dir.name,
            dataset=spec,
            instruction=read_instruction(task_dir, spec=spec),
        )
