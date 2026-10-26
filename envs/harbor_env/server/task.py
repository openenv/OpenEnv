# SPDX-License-Identifier: BSD-3-Clause

"""Parsing and discovery for Harbor task directories.

A Harbor task is a directory on disk. This module turns that directory into a
[`HarborTask`] value object and provides a [`TaskCatalog`] for discovering many
of them. Nothing here executes anything — see `sandbox.py` for that.

The layout (https://www.harborframework.com/docs/tasks) is:

```text
<task>/
  task.toml              # config + metadata (required)
  instruction.md         # the prompt shown to the agent
  environment/           # how to build the sandbox
    Dockerfile           #   ...or docker-compose.yaml, or seed files (see below)
  tests/                 # copied to /tests at verify time
    test.sh              #   the verifier; writes /logs/verifier/reward.{json,txt}
  solution/              # copied to /solution by the oracle agent
    solve.sh
```

Two spellings of the schema field are accepted: Harbor writes `schema_version`,
while [Repo2RLEnv](https://github.com/huggingface/Repo2RLEnv) writes `version`.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator


if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


logger = logging.getLogger(__name__)

TASK_CONFIG_FILE = "task.toml"
INSTRUCTION_FILE = "instruction.md"
TEST_SCRIPT = "test.sh"
SOLVE_SCRIPT = "solve.sh"

#: Files in `environment/` that describe *how to build* the sandbox rather than
#: what to seed it with. See "Environment" in the Harbor task docs.
_COMPOSE_FILES = ("docker-compose.yaml", "docker-compose.yml")
_DOCKERFILE = "Dockerfile"

#: How deep [`TaskCatalog`] descends when looking for task directories. Enough
#: for a flat directory, the `tasks/<id>/` layout that `repo2rlenv push` writes
#: to the Hub, and one level of grouping on top of that.
_MAX_DISCOVERY_DEPTH = 4

DEFAULT_AGENT_TIMEOUT_S = 1800.0
DEFAULT_VERIFIER_TIMEOUT_S = 300.0
DEFAULT_BUILD_TIMEOUT_S = 1800.0

#: Hugging Face dataset reference, optionally scheme-qualified and pinned:
#: `hf://datasets/<org>/<name>@<revision>` or just `<org>/<name>`.
_HUB_REFERENCE = re.compile(
    r"^(?:hf://datasets/)?(?P<repo_id>[\w.-]+/[\w.-]+)(?:@(?P<revision>[\w./-]+))?$"
)

#: Tasks shipped with this environment, used when no source is configured.
BUNDLED_TASKS_DIR = Path(__file__).resolve().parent.parent / "examples" / "tasks"


class TaskFormatError(ValueError):
    """Raised when a directory is not a well-formed Harbor task."""


#: Harbor's network policies, from `[environment].network_mode` and the
#: per-phase `[agent]` / `[verifier]` overrides.
NETWORK_NONE = "no-network"
NETWORK_PUBLIC = "public"
NETWORK_ALLOWLIST = "allowlist"

_NETWORK_MODES = frozenset({NETWORK_NONE, NETWORK_PUBLIC, NETWORK_ALLOWLIST})


@dataclass(frozen=True)
class NetworkPolicy:
    """What network a task's phases are allowed to reach.

    Harbor declares a baseline in `[environment]` and lets `[agent]` and
    `[verifier]` narrow it. A task that asks for `no-network` is usually
    testing offline behaviour, so granting it the daemon's default bridge is
    both a wrong result and an escape hatch.

    Args:
        baseline (`str`, *optional*, defaults to `"public"`):
            `[environment].network_mode`.
        agent (`str`, *optional*):
            `[agent].network_mode`, when the task overrides the baseline.
        verifier (`str`, *optional*):
            `[verifier].network_mode`, when the task overrides the baseline.
        allowed_hosts (`tuple[str, ...]`):
            Hosts named by an `allowlist` policy.
    """

    baseline: str = NETWORK_PUBLIC
    agent: str | None = None
    verifier: str | None = None
    allowed_hosts: tuple[str, ...] = ()

    def for_phase(self, phase: str) -> str:
        """The effective mode for `"agent"` or `"verifier"`."""
        override = self.agent if phase == "agent" else self.verifier
        return override or self.baseline

    @property
    def modes(self) -> frozenset[str]:
        """The effective mode of each phase.

        The baseline is deliberately *not* unioned in: when both phases override
        it, the baseline never takes effect, and including it would (for
        example) disable networking for a task whose agent and verifier both
        asked for `public`.
        """
        return frozenset({self.for_phase("agent"), self.for_phase("verifier")})

    @property
    def restricted(self) -> bool:
        """Whether any phase asks for something other than open internet."""
        return bool(self.modes - {NETWORK_PUBLIC})


@dataclass(frozen=True)
class ResourceLimits:
    """`[environment]` resource caps, as Harbor spells them.

    Args:
        cpus (`int`, *optional*):
            `[environment].cpus`.
        memory_mb (`int`, *optional*):
            `[environment].memory_mb`.
        storage_mb (`int`, *optional*):
            `[environment].storage_mb`.
        gpus (`int`, *optional*):
            `[environment].gpus`.
    """

    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None

    @property
    def declared(self) -> bool:
        """Whether the task asked for any limit at all."""
        return any(
            value is not None
            for value in (self.cpus, self.memory_mb, self.storage_mb, self.gpus)
        )


@dataclass(frozen=True)
class HarborTask:
    """An immutable view of one Harbor task directory.

    Args:
        task_id (`str`):
            Stable identifier used to select the task at `reset()`. It is the
            task directory's path relative to the catalog root, so nested
            layouts stay unambiguous.
        path (`Path`):
            Absolute path to the task directory.
        name (`str`):
            The `[task].name` field, conventionally `<org>/<slug>`.
        description (`str`):
            The `[task].description` field.
        instruction (`str`):
            Contents of `instruction.md` — the prompt handed to the agent.
        schema_version (`str`):
            Declared task-format version (`schema_version` or `version`).
        metadata (`dict[str, Any]`):
            The free-form `[metadata]` table, verbatim.
        docker_image (`str` or `None`):
            Pre-built image from `[environment].docker_image`, if any.
        environment_env (`dict[str, str]`):
            `[environment].env` — variables exported for the whole episode.
        verifier_env (`dict[str, str]`):
            `[verifier].env` — variables exported only while verifying.
        agent_timeout_s (`float`):
            `[agent].timeout_sec`.
        verifier_timeout_s (`float`):
            `[verifier].timeout_sec`.
        build_timeout_s (`float`):
            `[environment].build_timeout_sec`.
        os_name (`str`):
            `[environment].os`. Only `"linux"` is supported here.
        network ([`NetworkPolicy`]):
            Baseline `[environment]` network policy, and the per-phase
            `[agent]` / `[verifier]` overrides Harbor allows.
        resources ([`ResourceLimits`]):
            `[environment].cpus` / `memory_mb` / `storage_mb` / `gpus`.
        agent_user (`str` or `None`):
            `[agent].user` — the account agent commands run as.
        verifier_user (`str` or `None`):
            `[verifier].user` — the account the verifier runs as.
        declared_workdir (`str` or `None`):
            `[environment].workdir`, which overrides the image's `WORKDIR`.
    """

    task_id: str
    path: Path
    name: str
    description: str
    instruction: str
    schema_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    docker_image: str | None = None
    environment_env: dict[str, str] = field(default_factory=dict)
    verifier_env: dict[str, str] = field(default_factory=dict)
    agent_timeout_s: float = DEFAULT_AGENT_TIMEOUT_S
    verifier_timeout_s: float = DEFAULT_VERIFIER_TIMEOUT_S
    build_timeout_s: float = DEFAULT_BUILD_TIMEOUT_S
    os_name: str = "linux"
    network: NetworkPolicy = field(default_factory=lambda: NetworkPolicy())
    resources: ResourceLimits = field(default_factory=lambda: ResourceLimits())
    agent_user: str | None = None
    verifier_user: str | None = None
    declared_workdir: str | None = None

    # --- directory layout ---------------------------------------------------

    @property
    def tests_dir(self) -> Path:
        """The `tests/` directory, whether or not it exists."""
        return self.path / "tests"

    @property
    def solution_dir(self) -> Path:
        """The `solution/` directory, whether or not it exists."""
        return self.path / "solution"

    @property
    def environment_dir(self) -> Path:
        """The `environment/` directory, whether or not it exists."""
        return self.path / "environment"

    @property
    def test_script(self) -> Path | None:
        """`tests/test.sh`, or `None` when the task ships no verifier."""
        script = self.tests_dir / TEST_SCRIPT
        return script if script.is_file() else None

    @property
    def solve_script(self) -> Path | None:
        """`solution/solve.sh`, or `None` when the task ships no oracle."""
        script = self.solution_dir / SOLVE_SCRIPT
        return script if script.is_file() else None

    @property
    def dockerfile(self) -> Path | None:
        """`environment/Dockerfile`, or `None`."""
        candidate = self.environment_dir / _DOCKERFILE
        return candidate if candidate.is_file() else None

    @property
    def compose_file(self) -> Path | None:
        """`environment/docker-compose.y{a,}ml`, or `None`."""
        for filename in _COMPOSE_FILES:
            candidate = self.environment_dir / filename
            if candidate.is_file():
                return candidate
        return None

    @property
    def seed_files(self) -> tuple[Path, ...]:
        """Files uploaded into the working directory when the episode starts.

        Harbor only uploads `environment/` when it holds no build recipe: "If
        you omit both `environment/Dockerfile` and `environment/docker-compose.yaml`,
        any other files in `environment/` are uploaded into the container workdir".
        """
        if not self.environment_dir.is_dir():
            return ()
        if self.dockerfile is not None or self.compose_file is not None:
            return ()
        return tuple(sorted(p for p in self.environment_dir.rglob("*") if p.is_file()))

    @property
    def needs_image(self) -> bool:
        """Whether the task's starting state lives inside a container image.

        A task that ships seed files carries its own starting state and can be
        reproduced anywhere. Anything else (a `Dockerfile`, a compose file, or a
        pre-built `docker_image`) puts the starting state in an image, so only
        the Docker execution mode can run it faithfully.
        """
        if self.seed_files:
            return False
        return (
            self.dockerfile is not None
            or self.compose_file is not None
            or bool(self.docker_image)
        )

    # --- loading ------------------------------------------------------------

    @classmethod
    def load(
        cls, path: str | os.PathLike[str], task_id: str | None = None
    ) -> HarborTask:
        """Load a task from its directory.

        Args:
            path (`str` or `os.PathLike`):
                The task directory (the one containing `task.toml`).
            task_id (`str`, *optional*):
                Identifier to record. Defaults to the directory name.

        Returns:
            [`HarborTask`]

        Raises:
            [`TaskFormatError`]: If `task.toml` is missing or unparsable, or the
                task targets an unsupported container OS.
        """
        task_dir = Path(path).expanduser().resolve()
        config_path = task_dir / TASK_CONFIG_FILE
        if not config_path.is_file():
            raise TaskFormatError(
                f"{task_dir} is not a Harbor task: no {TASK_CONFIG_FILE}"
            )

        try:
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TaskFormatError(f"could not parse {config_path}: {exc}") from exc

        task_table = _table(config, "task")
        environment = _table(config, "environment")
        os_name = str(environment.get("os", "linux")).lower()
        if os_name != "linux":
            raise TaskFormatError(
                f"{task_dir} targets [environment].os = {os_name!r}; "
                "harbor_env only supports Linux tasks"
            )

        instruction_path = task_dir / INSTRUCTION_FILE
        instruction = (
            instruction_path.read_text(encoding="utf-8")
            if instruction_path.is_file()
            else str(task_table.get("description", ""))
        )

        return cls(
            task_id=task_id or task_dir.name,
            path=task_dir,
            name=str(task_table.get("name", task_dir.name)),
            description=str(task_table.get("description", "")),
            instruction=instruction,
            # Harbor spells it `schema_version`; Repo2RLEnv writes `version`.
            schema_version=str(
                config.get("schema_version") or config.get("version") or "unknown"
            ),
            metadata=_table(config, "metadata"),
            docker_image=environment.get("docker_image") or None,
            environment_env=_str_map(environment.get("env")),
            verifier_env=_str_map(_table(config, "verifier").get("env")),
            agent_timeout_s=_timeout(config, "agent", DEFAULT_AGENT_TIMEOUT_S),
            verifier_timeout_s=_timeout(config, "verifier", DEFAULT_VERIFIER_TIMEOUT_S),
            build_timeout_s=_positive_float(
                environment.get("build_timeout_sec"), DEFAULT_BUILD_TIMEOUT_S
            ),
            os_name=os_name,
            network=_network_policy(config, environment),
            resources=ResourceLimits(
                cpus=_positive_int(environment.get("cpus"), "cpus"),
                memory_mb=_positive_int(environment.get("memory_mb"), "memory_mb"),
                storage_mb=_positive_int(environment.get("storage_mb"), "storage_mb"),
                gpus=_positive_int(environment.get("gpus"), "gpus"),
            ),
            agent_user=_user(_table(config, "agent").get("user")),
            verifier_user=_user(_table(config, "verifier").get("user")),
            declared_workdir=str(environment["workdir"])
            if environment.get("workdir")
            else None,
        )

    def summary(self) -> dict[str, Any]:
        """A JSON-serializable digest, surfaced in observations and state.

        Deliberately omits [`path`]: the observation reaches the agent, and
        naming the task directory on the server would point straight at
        `solution/` and `tests/`. [`HarborState.task_path`] still carries it for
        training orchestration, which is on the infrastructure side.
        """
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "schema_version": self.schema_version,
            "needs_image": self.needs_image,
            "docker_image": self.docker_image,
            "has_verifier": self.test_script is not None,
            "has_solution": self.solve_script is not None,
        }


class TaskCatalog:
    """Discovers Harbor tasks under a root directory.

    Any directory containing a `task.toml` is a task, and the search does not
    descend into one once found. That covers a flat directory of tasks, the
    `tasks/<id>/` layout `repo2rlenv push` writes to the Hub, and a single task
    directory passed directly.

    Args:
        root (`str` or `os.PathLike`):
            Directory to scan.

    Examples:

    ```python
    catalog = TaskCatalog("./tasks")
    catalog.task_ids()          # ['fix-sum-bug', 'pallets__click-1234']
    task = catalog.get("fix-sum-bug")
    ```
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self._discovered: dict[str, Path] | None = None

    def task_ids(self) -> list[str]:
        """Sorted identifiers of every task under the root."""
        return sorted(self._task_dirs())

    def tasks(self) -> Iterator[HarborTask]:
        """Yield every task under the root, in identifier order."""
        for task_id in self.task_ids():
            yield self.get(task_id)

    def refresh(self) -> None:
        """Forget the cached scan so the next lookup re-reads the filesystem."""
        self._discovered = None

    def get(self, task_id: str) -> HarborTask:
        """Load one task by identifier.

        Accepts either the full identifier (path relative to the root) or a bare
        directory name when that name is unique.

        Raises:
            `KeyError`: If no task matches, or a bare name is ambiguous.
        """
        available = self._task_dirs()
        if task_id in available:
            return HarborTask.load(available[task_id], task_id=task_id)

        matches = [tid for tid in available if Path(tid).name == task_id]
        if len(matches) == 1:
            return HarborTask.load(available[matches[0]], task_id=matches[0])
        if len(matches) > 1:
            raise KeyError(
                f"task id {task_id!r} is ambiguous under {self.root}; "
                f"use one of {sorted(matches)}"
            )
        raise KeyError(
            f"unknown task {task_id!r} under {self.root}; "
            f"available: {sorted(available) or '<none>'}"
        )

    def _task_dirs(self) -> dict[str, Path]:
        """Map identifier -> task directory, scanning the tree at most once.

        A served task set is fixed for the life of the server, so the scan is
        cached; call [`refresh`] after changing it on disk.
        """
        if self._discovered is None:
            self._discovered = self._scan()
        return self._discovered

    def _scan(self) -> dict[str, Path]:
        if not self.root.is_dir():
            return {}
        if (self.root / TASK_CONFIG_FILE).is_file():
            return {self.root.name: self.root}

        found: dict[str, Path] = {}
        stack = [(self.root, 0)]
        while stack:
            directory, depth = stack.pop()
            if depth > _MAX_DISCOVERY_DEPTH:
                continue
            for child in sorted(directory.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if (child / TASK_CONFIG_FILE).is_file():
                    found[child.relative_to(self.root).as_posix()] = child
                else:
                    stack.append((child, depth + 1))
        return found


@lru_cache(maxsize=None)
def resolve_task_source(source: str | os.PathLike[str] | None) -> Path:
    """Resolve a configured task source to a local directory.

    Results are cached, so a server that starts one environment per session
    downloads a Hub dataset once and then trains against a fixed snapshot.

    Args:
        source (`str` or `os.PathLike`, *optional*):
            A local directory, or a Hugging Face dataset holding Harbor tasks —
            `hf://datasets/<org>/<name>`, optionally pinned with `@<revision>`.
            A bare `<org>/<name>` is treated as a dataset when no such local path
            exists. Defaults to the tasks bundled with this environment.

    Returns:
        `Path`: A local directory to hand to [`TaskCatalog`].

    Examples:

    ```python
    resolve_task_source("./tasks")
    resolve_task_source("hf://datasets/my-org/click-pr-tasks@v1")
    ```
    """
    if source is None or not str(source).strip():
        return BUNDLED_TASKS_DIR

    text = str(source).strip()
    local = Path(text).expanduser()
    if local.exists():
        return local.resolve()

    match = _HUB_REFERENCE.match(text)
    if match is None:
        raise FileNotFoundError(
            f"task source {text!r} is neither an existing directory nor a Hugging Face "
            "dataset reference like 'hf://datasets/<org>/<name>'"
        )

    from huggingface_hub import snapshot_download

    repo_id = match.group("repo_id")
    logger.info("downloading Harbor tasks from dataset %s", repo_id)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=match.group("revision"),
            repo_type="dataset",
        )
    )


def _table(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return `config[key]` when it is a table, else an empty one."""
    value = config.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _str_map(value: Any) -> dict[str, str]:
    """Coerce a TOML table into a `str -> str` environment mapping."""
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def _timeout(config: dict[str, Any], section: str, fallback: float) -> float:
    return _positive_float(_table(config, section).get("timeout_sec"), fallback)


def _network_policy(
    config: dict[str, Any], environment: dict[str, Any]
) -> NetworkPolicy:
    """Read the baseline policy plus the per-phase overrides Harbor allows.

    `allow_internet` is Harbor's deprecated spelling; it is still honoured so
    older tasks are not silently granted more network than they asked for.
    """
    baseline = _network_mode(environment.get("network_mode"))
    if baseline is None:
        allow_internet = environment.get("allow_internet")
        if allow_internet is False:
            baseline = NETWORK_NONE
        else:
            baseline = NETWORK_PUBLIC

    hosts = environment.get("allowed_hosts")
    allowed = tuple(str(h) for h in hosts) if isinstance(hosts, list) else ()
    return NetworkPolicy(
        baseline=baseline,
        agent=_network_mode(_table(config, "agent").get("network_mode")),
        verifier=_network_mode(_table(config, "verifier").get("network_mode")),
        allowed_hosts=allowed,
    )


def _network_mode(value: Any) -> str | None:
    """Normalize a declared mode, rejecting anything we do not understand.

    An unrecognized mode must not silently degrade to `public` — that is the
    exact failure this parsing exists to prevent.
    """
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode not in _NETWORK_MODES:
        raise TaskFormatError(
            f"unknown network_mode {value!r}; expected one of {sorted(_NETWORK_MODES)}"
        )
    return mode


def _user(value: Any) -> str | None:
    """`[agent].user` / `[verifier].user`, which Harbor allows as a name or UID."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return str(value).strip()


def _positive_int(value: Any, field: str) -> int | None:
    """Parse a resource limit, distinguishing "absent" from "malformed".

    Returning `None` for a bad value would read as "no limit declared", so a
    typo would silently buy the task unlimited resources *and* stop the local
    backend refusing it. An invalid declaration is a format error.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TaskFormatError(f"[environment].{field} must be a positive integer")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise TaskFormatError(
            f"[environment].{field} must be a positive integer, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise TaskFormatError(
            f"[environment].{field} must be a positive integer, got {parsed}"
        )
    return parsed


def _positive_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback
