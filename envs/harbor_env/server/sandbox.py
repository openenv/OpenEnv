# SPDX-License-Identifier: BSD-3-Clause

"""Execution backends that implement Harbor's sandbox contract.

Harbor tasks are written against a fixed filesystem layout inside the sandbox:

| Path              | Contents                                                |
| ----------------- | ------------------------------------------------------- |
| working directory | what the agent edits (the image's `WORKDIR`)            |
| `/tests`          | the task's `tests/`, staged by the harness before verify |
| `/solution`       | the task's `solution/`, staged only for the oracle       |
| `/logs/verifier`  | where the verifier writes `reward.json` / `reward.txt`   |
| `/logs/agent`     | scratch space for agent logs                             |

[`Sandbox`] encodes that layout and the episode phases once; subclasses only
provide five primitives (boot, exec, read, write, upload). Two backends ship:

- [`DockerSandbox`] — runs inside the task's own image. This is the faithful
  backend and the only one that can run tasks whose starting state lives in an
  image (everything `repo2rlenv` emits with an `environment/Dockerfile`).
- [`LocalSandbox`] — runs as subprocesses under a per-episode root directory.
  No Docker, so it works on Hugging Face Spaces, and it is exact for
  *self-contained* tasks that ship their starting state as `environment/` seed
  files.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


try:  # In-repo import (PYTHONPATH=src:envs)
    from harbor_env.server.reward import read_reward, RewardReport
    from harbor_env.server.task import (
        HarborTask,
        NETWORK_ALLOWLIST,
        NETWORK_NONE,
    )
except ImportError:  # Standalone image layout
    from reward import read_reward, RewardReport
    from task import HarborTask, NETWORK_ALLOWLIST, NETWORK_NONE


logger = logging.getLogger(__name__)

#: Exit code convention for a command killed by its timeout (GNU `timeout`).
TIMEOUT_EXIT_CODE = 124

#: Harbor's absolute paths, matched to detect task scripts that hardcode them
#: while the local sandbox is not rooted at `/`. `/workspace` is included
#: because it is the working directory every `repo2rlenv` image declares.
_ABSOLUTE_HARBOR_PATHS = re.compile(
    r"(?<![\w/])/(?:logs|tests|solution|workspace)(?:/|\b)"
)

#: Substitutes `${VAR}` / `$VAR` in `[environment].env` values from the server's
#: own environment, matching how `harbor run` resolves them.
_ENV_REFERENCE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


#: Variables a subprocess genuinely needs to run. Everything else in the
#: server's environment — API keys, Hub tokens, cloud credentials — is withheld
#: from task commands, which are attacker-controlled input in every threat
#: model that matters. A task that needs a secret declares it in
#: `[environment].env`, which is resolved explicitly by `expand_env_refs`.
_ENV_PASSTHROUGH = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "SHELL",
    "USER",
    "SYSTEMROOT",
)


def _baseline_environ() -> dict[str, str]:
    """The minimal environment a local subprocess starts from."""
    return {name: os.environ[name] for name in _ENV_PASSTHROUGH if name in os.environ}


class SandboxError(RuntimeError):
    """Raised when a sandbox cannot be created or cannot run a task."""


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one command executed inside a sandbox."""

    exit_code: int
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Whether the command exited successfully."""
        return self.exit_code == 0 and not self.timed_out


@dataclass(frozen=True)
class SandboxPaths:
    """Harbor's special paths, resolved for a concrete sandbox."""

    workdir: str
    tests: str = "/tests"
    solution: str = "/solution"
    logs: str = "/logs"

    @property
    def logs_verifier(self) -> str:
        return posixpath.join(self.logs, "verifier")

    @property
    def logs_agent(self) -> str:
        return posixpath.join(self.logs, "agent")

    def as_env(self, *, agent_visible: bool = False) -> dict[str, str]:
        """Path layout as environment variables.

        Portable task scripts should prefer these over hardcoded absolute paths
        so the same `tests/test.sh` runs under `harbor run` and under
        [`LocalSandbox`]; `TEST_DIR` is included for Terminal-Bench lineage
        scripts that expect it.

        Args:
            agent_visible (`bool`, *optional*, defaults to `False`):
                Return only the paths an agent legitimately needs. The verifier
                and oracle directories are withheld: they are where the grading
                logic and the answer live, and only the verifier and oracle
                themselves need to be told where they are.
        """
        agent_paths = {
            "HARBOR_WORKDIR": self.workdir,
            "HARBOR_AGENT_LOGS_DIR": self.logs_agent,
        }
        if agent_visible:
            return agent_paths
        return {
            **agent_paths,
            "HARBOR_TESTS_DIR": self.tests,
            "HARBOR_SOLUTION_DIR": self.solution,
            "HARBOR_LOGS_DIR": self.logs_verifier,
            "TEST_DIR": self.tests,
        }


class Sandbox(ABC):
    """Harbor's sandbox contract, independent of how commands are executed.

    Subclasses implement [`_boot`], [`exec`], [`read_text`], [`write_text`] and
    [`upload_dir`]; everything else — the path layout and the episode phases —
    is shared.
    """

    paths: SandboxPaths

    #: Human-readable backend name, surfaced in observations.
    mode: str = "unknown"

    #: Episode-wide variables from `[environment].env`, resolved by [`start`].
    task_env: dict[str, str]

    #: Account agent commands run as, from `[agent].user`. `None` means the
    #: image's own default.
    agent_user: str | None = None

    # --- backend primitives -------------------------------------------------

    @abstractmethod
    def _boot(self, task: HarborTask) -> SandboxPaths:
        """Bring the backend up for `task` and return its resolved paths."""

    @abstractmethod
    def exec(
        self,
        command: str,
        *,
        timeout_s: float,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        user: str | None = None,
        agent_visible: bool = False,
    ) -> ExecResult:
        """Run `command` through `bash` and return its merged output.

        Args:
            agent_visible (`bool`, *optional*, defaults to `False`):
                Whether this command is the agent's. Agent commands are not told
                where the verifier and oracle directories live.
            user (`str`, *optional*):
                Account to run as, from `[agent].user` / `[verifier].user`.
                Backends that cannot switch user must reject such tasks in
                [`_boot`] rather than ignore this.
        """

    @abstractmethod
    def read_text(self, path: str, user: str | None = None) -> str | None:
        """Read a text file, returning `None` when it does not exist.

        Args:
            user (`str`, *optional*):
                Read as this account. Agent reads must pass the task's
                `[agent].user`, or a task that declared an unprivileged agent
                would still read root-only files through this path.
        """

    @abstractmethod
    def write_text(self, path: str, content: str, user: str | None = None) -> None:
        """Write a text file, creating parent directories as needed.

        Args:
            user (`str`, *optional*):
                Write as this account, so an unprivileged agent cannot create
                files it could not have created with `exec`.
        """

    @abstractmethod
    def upload_dir(self, source: Path, destination: str) -> None:
        """Copy a host directory into the sandbox at `destination`."""

    @abstractmethod
    def close(self) -> None:
        """Release every resource held by the sandbox."""

    # --- episode phases -----------------------------------------------------

    def start(self, task: HarborTask) -> None:
        """Boot the backend and lay out Harbor's directories.

        Seeds the working directory from the task's `environment/` files when
        the task ships them. `tests/` and `solution/` are deliberately *not*
        staged here — see [`stage_tests`] and [`stage_solution`].
        """
        self.task_env = expand_env_refs(task.environment_env)
        self.agent_user = task.agent_user
        self.paths = self._boot(task)
        self.mkdirs(self.paths.workdir, self.paths.logs_verifier, self.paths.logs_agent)
        # The directories above are created by the image's default user (root).
        # A task that asks for `[agent].user` would otherwise get a working
        # directory it cannot write to, which breaks every `write` action and
        # any edit the agent makes.
        self.chown(task.agent_user, self.paths.workdir, self.paths.logs_agent)
        if task.seed_files:
            self.upload_dir(task.environment_dir, self.paths.workdir)
            self.chown(task.agent_user, self.paths.workdir)

    def stage_tests(self, task: HarborTask) -> None:
        """Copy `tests/` to `/tests`, as Harbor's harness does before verifying.

        Staging happens at verification time rather than at reset so the agent
        never sees the verifier while it is working. The destination is cleared
        first: the agent shares the filesystem and could otherwise pre-create
        files there that the real verifier would then read.
        """
        if task.tests_dir.is_dir():
            self._replace_dir(task.tests_dir, self.paths.tests)

    def stage_solution(self, task: HarborTask) -> None:
        """Copy `solution/` to `/solution`, as Harbor's oracle agent does."""
        if task.solution_dir.is_dir():
            self._replace_dir(task.solution_dir, self.paths.solution)

    def _replace_dir(self, source: Path, destination: str) -> None:
        """Upload `source` over a freshly emptied `destination`."""
        self.exec(f"rm -rf {_quote(destination)}", timeout_s=60.0, workdir="/")
        self.upload_dir(source, destination)

    def run_verifier(self, task: HarborTask) -> ExecResult:
        """Stage `tests/` and run `tests/test.sh`.

        Raises:
            [`SandboxError`]: If the task ships no `tests/test.sh`.
        """
        if task.test_script is None:
            raise SandboxError(
                f"task {task.task_id!r} has no tests/test.sh, so it cannot be graded"
            )
        # Start from an empty verifier log directory: the reward must describe
        # *this* run. The agent shares the sandbox and could otherwise plant a
        # reward file that survives a verifier which fails before writing one.
        self.exec(
            f"rm -rf {_quote(self.paths.logs_verifier)}", timeout_s=60.0, workdir="/"
        )
        self.mkdirs(self.paths.logs_verifier)
        self.stage_tests(task)
        # Same reason as the working directory: a `[verifier].user` that cannot
        # write its reward file would make every task unscorable.
        self.chown(task.verifier_user, self.paths.logs_verifier, self.paths.tests)
        script = posixpath.join(self.paths.tests, task.test_script.name)
        return self.exec(
            f"bash {_quote(script)}",
            timeout_s=task.verifier_timeout_s,
            env={**self.task_env, **expand_env_refs(task.verifier_env)},
            user=task.verifier_user,
        )

    def run_solution(self, task: HarborTask) -> ExecResult:
        """Stage `solution/` and run `solution/solve.sh` (Harbor's oracle agent).

        Raises:
            [`SandboxError`]: If the task ships no `solution/solve.sh`.
        """
        if task.solve_script is None:
            raise SandboxError(
                f"task {task.task_id!r} has no solution/solve.sh, so it has no oracle"
            )
        self.stage_solution(task)
        self.chown(task.agent_user, self.paths.solution)
        script = posixpath.join(self.paths.solution, task.solve_script.name)
        return self.exec(
            f"bash {_quote(script)}",
            timeout_s=task.agent_timeout_s,
            env=self.task_env,
            user=task.agent_user,
        )

    def reward_report(self) -> RewardReport:
        """Read the verifier's verdict out of `/logs/verifier`."""
        return read_reward(
            lambda name: self.read_text(posixpath.join(self.paths.logs_verifier, name))
        )

    def explain_missing_reward(self, task: HarborTask) -> str | None:
        """Backend-specific hint for why a verifier produced no reward file."""
        del task
        return None

    def chown(self, user: str | None, *paths: str) -> None:
        """Hand `paths` to `user`, so a non-root phase can write to them.

        A no-op when the task declares no user. Runs as the image's default
        account, which is the only one able to change ownership.
        """
        if not user or not paths:
            return
        quoted = " ".join(_quote(path) for path in paths)
        result = self.exec(
            f"chown -R {_quote(user)} {quoted}", timeout_s=120.0, workdir="/"
        )
        if not result.ok:
            raise SandboxError(
                f"could not give {user!r} ownership of {list(paths)}: "
                f"{result.output.strip()}"
            )

    def resolve_agent_path(self, relative: str) -> str:
        """Resolve an agent-supplied path, refusing anything outside the workdir.

        Two checks, because either alone is insufficient:

        1. [`resolve_within`] rejects absolute paths and lexical `..` escapes.
        2. The result is then canonicalized *on the sandbox's own filesystem*,
           which is the only way to catch a symlink the agent planted with
           `exec`. Without this, `ln -s /tests t` followed by
           `read path="t/test.sh"` walks straight out of the working directory
           and reads the grader.

        Raises:
            `ValueError`: If the path escapes the working directory, before or
                after symlink resolution.
        """
        target = resolve_within(self.paths.workdir, relative)
        real = self.real_path(target)
        # Canonicalize the base as well: the working directory itself often
        # sits behind a link (macOS `/var` -> `/private/var`), and comparing a
        # resolved path against an unresolved base would reject every legitimate
        # access.
        base = self.real_path(self.paths.workdir)
        if real != base and not real.startswith(base.rstrip("/") + "/"):
            raise ValueError(
                f"path escapes the working directory through a link: {relative!r}"
            )
        return real

    @abstractmethod
    def real_path(self, path: str) -> str:
        """Canonicalize `path` on the sandbox's filesystem, resolving symlinks.

        The final component need not exist — a `write` legitimately creates it —
        but every parent must, so a link in the middle cannot hide.
        """

    def mkdirs(self, *paths: str) -> None:
        """Create directories inside the sandbox.

        Runs from `/` so it works before the working directory exists — a fresh
        image often declares a `WORKDIR` that is not in the filesystem yet.
        """
        quoted = " ".join(_quote(path) for path in paths)
        result = self.exec(f"mkdir -p {quoted}", timeout_s=60.0, workdir="/")
        if not result.ok:
            raise SandboxError(
                f"could not create {list(paths)}: {result.output.strip()}"
            )


class LocalSandbox(Sandbox):
    """Runs a task as subprocesses under a per-episode root directory.

    Every Harbor path is materialized under `root` (`<root>/workspace`,
    `<root>/tests`, `<root>/logs/verifier`, ...) and exported through the
    `HARBOR_*` variables from [`SandboxPaths.as_env`]. That keeps concurrent
    episodes isolated and needs no privileges, at the cost of not being able to
    serve scripts that hardcode Harbor's absolute paths. Such a script writes
    its reward outside the episode, which surfaces as an unscorable episode
    rather than a wrong score — [`explain_missing_reward`] then names the
    offending path.

    Note that this backend is a *filesystem* boundary, not a *security* one:
    `exec` runs shell commands as the server's own user, with the server's own
    environment, so a task or a policy can read whatever that user can and can
    reach the network. Serve task sets you trust here, and use [`DockerSandbox`]
    for anything else.

    Args:
        root (`str` or `os.PathLike`, *optional*):
            Directory to root the episode in. Defaults to a fresh temporary
            directory that is removed on [`close`].
    """

    mode = "local"

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self._owns_root = root is None
        self._root = (
            Path(root).expanduser().resolve()
            if root
            else Path(tempfile.mkdtemp(prefix="harbor-episode-"))
        )
        self.paths = SandboxPaths(workdir=str(self._root / "workspace"))
        self.task_env = {}

    @property
    def root(self) -> Path:
        """The episode's root directory on the host."""
        return self._root

    def _boot(self, task: HarborTask) -> SandboxPaths:
        if task.needs_image:
            raise SandboxError(
                f"task {task.task_id!r} keeps its starting state in a container image "
                f"({_image_source(task)}), which the local backend cannot reproduce. "
                "Run the server with HARBOR_MODE=docker, or use a task that ships its "
                "files in environment/."
            )
        # Fail closed on anything this backend cannot actually enforce. A task
        # that asked for no-network, a CPU cap or a dedicated user has made a
        # statement about how it must be graded; running it anyway under the
        # server's own account would quietly produce a result the task never
        # sanctioned.
        unenforceable: list[str] = []
        if task.network.restricted:
            unenforceable.append(
                f"network_mode={sorted(task.network.modes - {'public'})}"
            )
        if task.resources.declared:
            unenforceable.append("resource limits")
        if task.agent_user or task.verifier_user:
            unenforceable.append("a declared user")
        if unenforceable:
            raise SandboxError(
                f"task {task.task_id!r} declares {', '.join(unenforceable)}, which the "
                "local backend cannot enforce — it runs subprocesses as the server's "
                "own user with the host's network. Run the server with "
                "HARBOR_MODE=docker."
            )
        if task.docker_image:
            # The task ships its starting state as seed files, so the state is
            # reproducible here — but the image it named supplies the toolchain,
            # and the host's may differ. Say so rather than let a version skew
            # surface as a mis-graded episode.
            logger.warning(
                "task %s declares [environment].docker_image = %s; the local backend "
                "runs on the host toolchain instead, so results may differ from "
                "`harbor run`. Use HARBOR_MODE=docker for a faithful result.",
                task.task_id,
                task.docker_image,
            )

        self._root.mkdir(parents=True, exist_ok=True)
        return SandboxPaths(
            workdir=str(self._root / "workspace"),
            tests=str(self._root / "tests"),
            solution=str(self._root / "solution"),
            logs=str(self._root / "logs"),
        )

    def exec(
        self,
        command: str,
        *,
        timeout_s: float,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        user: str | None = None,
        agent_visible: bool = False,
    ) -> ExecResult:
        if user:  # pragma: no cover - _boot rejects these tasks up front
            raise SandboxError(
                "the local backend cannot run commands as another user; "
                "use HARBOR_MODE=docker"
            )
        cwd = Path(workdir or self.paths.workdir)
        cwd.mkdir(parents=True, exist_ok=True)
        process_env = _baseline_environ()
        process_env.update(self.paths.as_env(agent_visible=agent_visible))
        process_env.update(env or {})
        # Keep the task's tree free of interpreter droppings.
        process_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        try:
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=cwd,
                env=process_env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=TIMEOUT_EXIT_CODE,
                output=_decode(exc.stdout) + _decode(exc.stderr),
                timed_out=True,
            )
        return ExecResult(completed.returncode, completed.stdout + completed.stderr)

    def real_path(self, path: str) -> str:
        target = Path(path)
        if target.exists():
            return str(target.resolve())
        # A `write` may be creating the final component; resolving the parent
        # still catches a symlinked directory on the way there.
        return str(target.parent.resolve() / target.name)

    def read_text(self, path: str, user: str | None = None) -> str | None:
        del user  # _boot refuses tasks that declare one
        target = Path(path)
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8", errors="replace")

    def write_text(self, path: str, content: str, user: str | None = None) -> None:
        del user  # _boot refuses tasks that declare one
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def upload_dir(self, source: Path, destination: str) -> None:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)

    def close(self) -> None:
        if self._owns_root:
            shutil.rmtree(self._root, ignore_errors=True)

    def explain_missing_reward(self, task: HarborTask) -> str | None:
        """Point at hardcoded absolute paths, the usual cause in local mode.

        A verifier that writes to `/logs/verifier` instead of `$HARBOR_LOGS_DIR`
        lands outside this episode's root — or fails outright without root
        privileges — and its reward never reaches us.
        """
        if self.paths.logs.startswith("/logs"):
            return None  # Rooted at /, so absolute paths resolve correctly.
        if task.test_script is None:
            return None
        text = task.test_script.read_text(encoding="utf-8", errors="replace")
        match = _ABSOLUTE_HARBOR_PATHS.search(text)
        if match is None:
            return None
        return (
            f"{task.test_script.name} references {match.group(0)!r}, an absolute path "
            f"that only exists inside a container. Run the server with HARBOR_MODE=docker, "
            f"or have the script write to $HARBOR_LOGS_DIR so it stays portable across "
            f"both backends."
        )


class DockerSandbox(Sandbox):
    """Runs a task inside its own container image — Harbor's native backend.

    The image comes from `[environment].docker_image` when set, otherwise it is
    built from `environment/Dockerfile`; tasks that ship only `environment/` seed
    files run on `default_image`. Files are streamed in over the Docker API
    rather than bind-mounted, so the environment server can itself be
    containerized.

    Args:
        default_image (`str`, *optional*, defaults to `"python:3.12-slim"`):
            Image used for tasks that declare no image of their own.
        keep_container (`bool`, *optional*, defaults to `False`):
            Leave the container in place after [`close`], for debugging.
    """

    mode = "docker"

    #: Used when the image declares no `WORKDIR`; matches Harbor's default.
    DEFAULT_WORKDIR = "/app"

    def __init__(
        self,
        default_image: str = "python:3.12-slim",
        keep_container: bool = False,
    ) -> None:
        self.paths = SandboxPaths(workdir=self.DEFAULT_WORKDIR)
        self.task_env = {}
        self.default_image = default_image
        self.keep_container = keep_container
        self._client = None
        self._container = None
        self._image = ""
        self._has_timeout: bool | None = None
        self._agent_user: str | None = None

    @property
    def image(self) -> str:
        """The image backing this sandbox."""
        return self._image

    def _boot(self, task: HarborTask) -> SandboxPaths:
        client = self._connect()
        self._image = self._resolve_image(task, client)
        self._container = client.containers.run(
            image=self._image,
            command=["sleep", "infinity"],
            detach=True,
            environment=dict(self.task_env),
            labels={"openenv.env": "harbor_env", "openenv.task": task.task_id},
            auto_remove=False,
            **_container_limits(task),
        )
        self._agent_user = task.agent_user
        return SandboxPaths(
            workdir=task.declared_workdir or self._image_workdir(client)
        )

    def exec(
        self,
        command: str,
        *,
        timeout_s: float,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        user: str | None = None,
        agent_visible: bool = False,
    ) -> ExecResult:
        container = self._require_container()
        process_env = dict(self.paths.as_env(agent_visible=agent_visible))
        process_env.update(env or {})
        wrapped = self._with_timeout(command, timeout_s)
        exit_code, output = container.exec_run(
            cmd=["bash", "-c", wrapped],
            workdir=workdir or self.paths.workdir,
            environment=process_env,
            user=user or "",
            demux=False,
        )
        text = output.decode("utf-8", errors="replace") if output else ""
        # The Docker API reports a null exit code when the exec never started.
        status = int(exit_code) if exit_code is not None else 1
        return ExecResult(
            exit_code=status,
            output=text,
            timed_out=status == TIMEOUT_EXIT_CODE,
        )

    def real_path(self, path: str) -> str:
        # `readlink -m` canonicalizes every component and, unlike `-f`, does not
        # require any of them to exist — a `write` legitimately creates nested
        # directories. Symlinks that *do* exist are still resolved, which is the
        # only part that matters for confinement.
        result = self.exec(f"readlink -m {_quote(path)}", timeout_s=60.0)
        canonical = result.output.strip()
        if not result.ok or not canonical:
            raise SandboxError(
                f"could not canonicalize {path!r} inside the sandbox; refusing the "
                "action rather than trusting an unresolved path"
            )
        return canonical

    def read_text(self, path: str, user: str | None = None) -> str | None:
        result = self.exec(f"cat {_quote(path)}", timeout_s=60.0, user=user)
        return result.output if result.ok else None

    def write_text(self, path: str, content: str, user: str | None = None) -> None:
        if user:
            # put_archive extracts as root, which would let an unprivileged
            # agent create root-owned files — and, through a symlink out of the
            # working directory, create them anywhere. Write through the shell
            # as the declared user instead, so the kernel applies its
            # permissions. Base64 keeps arbitrary bytes off the command line.
            payload = base64.b64encode(content.encode("utf-8")).decode("ascii")
            result = self.exec(
                f"mkdir -p {_quote(posixpath.dirname(path))} && "
                f"printf %s {_quote(payload)} | base64 -d > {_quote(path)}",
                timeout_s=120.0,
                user=user,
            )
            if not result.ok:
                raise SandboxError(
                    f"could not write {path} as {user!r}: {result.output.strip()}"
                )
            return
        container = self._require_container()
        directory, name = posixpath.split(path)
        self.mkdirs(directory)
        container.put_archive(directory, _tar_bytes({name: content.encode("utf-8")}))

    def upload_dir(self, source: Path, destination: str) -> None:
        container = self._require_container()
        self.mkdirs(destination)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for entry in sorted(source.rglob("*")):
                # rglob yields directories as well as files, and `add` recurses
                # by default — leaving it on would archive every nested file
                # once via its parent and again on its own iteration.
                archive.add(
                    entry,
                    arcname=entry.relative_to(source).as_posix(),
                    recursive=False,
                )
        container.put_archive(destination, stream.getvalue())

    def close(self) -> None:
        if self._container is not None and not self.keep_container:
            try:
                self._container.remove(force=True)
            except Exception:  # pragma: no cover - best-effort teardown
                logger.warning("could not remove container", exc_info=True)
        self._container = None
        self._client = None

    # --- docker helpers -----------------------------------------------------

    def _connect(self):
        if self._client is None:
            try:
                import docker
            except ImportError as exc:
                raise SandboxError(
                    "HARBOR_MODE=docker needs the `docker` package: pip install docker"
                ) from exc
            try:
                self._client = docker.from_env()
                self._client.ping()
            except Exception as exc:
                raise SandboxError(f"could not reach the Docker daemon: {exc}") from exc
        return self._client

    def _resolve_image(self, task: HarborTask, client) -> str:
        if task.docker_image:
            return _ensure_pulled(client, task.docker_image)

        if task.dockerfile is not None:
            tag = f"openenv-harbor/{_slug(task.task_id)}:latest"
            logger.info("building %s from %s", tag, task.dockerfile)
            client.images.build(
                path=str(task.environment_dir),
                dockerfile=task.dockerfile.name,
                tag=tag,
                rm=True,
                timeout=task.build_timeout_s,
            )
            return tag

        if task.compose_file is not None:
            raise SandboxError(
                f"task {task.task_id!r} uses environment/{task.compose_file.name}; "
                "multi-container tasks are not supported yet — run it with `harbor run`"
            )
        # Self-contained task: no image of its own, so run the seed files on the
        # backend's default image.
        return _ensure_pulled(client, self.default_image)

    def _image_workdir(self, client) -> str:
        try:
            config = client.images.get(self._image).attrs.get("Config", {})
            return str(config.get("WorkingDir") or self.DEFAULT_WORKDIR)
        except Exception:  # pragma: no cover - image metadata is best-effort
            return self.DEFAULT_WORKDIR

    def _with_timeout(self, command: str, timeout_s: float) -> str:
        """Bound `command` with GNU `timeout` when the image provides it."""
        if self._has_timeout is None:
            container = self._require_container()
            probe, _ = container.exec_run(cmd=["sh", "-c", "command -v timeout"])
            self._has_timeout = int(probe) == 0
        if not self._has_timeout:
            return command
        return f"timeout -k 5 {int(max(timeout_s, 1))} bash -c {_quote(command)}"

    def _require_container(self):
        if self._container is None:
            raise SandboxError("sandbox is not running; call start() first")
        return self._container


def _container_limits(task: HarborTask) -> dict[str, object]:
    """Translate the task's declared policies into `containers.run` kwargs.

    Raises:
        [`SandboxError`]: If the task asks for a policy this backend cannot
            enforce faithfully. Failing closed matters more than breadth here:
            silently granting a `no-network` task the daemon's default bridge
            would both corrupt the result and hand a sandboxed process the
            internet.
    """
    kwargs: dict[str, object] = {}

    modes = task.network.modes
    if NETWORK_ALLOWLIST in modes:
        raise SandboxError(
            f"task {task.task_id!r} declares network_mode='allowlist' "
            f"(hosts: {list(task.network.allowed_hosts) or 'unspecified'}), which "
            "needs a filtering proxy this backend does not run. Grade it with "
            "`harbor run`, which implements the allowlist."
        )
    if modes == {NETWORK_NONE}:
        kwargs["network_mode"] = "none"
    elif NETWORK_NONE in modes:
        # Per-phase split: one phase wants isolation, another wants the network.
        # A container has a single network, so honouring both is impossible —
        # take the restrictive one rather than the permissive one.
        logger.warning(
            "task %s asks for different network modes per phase (%s); applying the "
            "most restrictive (no-network) for the whole episode",
            task.task_id,
            sorted(modes),
        )
        kwargs["network_mode"] = "none"

    limits = task.resources
    if limits.cpus is not None:
        # Docker expresses CPU count as a quota against a 100ms period.
        kwargs["cpu_period"] = 100_000
        kwargs["cpu_quota"] = int(limits.cpus * 100_000)
    if limits.memory_mb is not None:
        kwargs["mem_limit"] = f"{limits.memory_mb}m"
    if limits.storage_mb is not None:
        # Docker accepts `storage_opt` on drivers that cannot honour it — the
        # option lands in HostConfig and is then simply not enforced, which is
        # the worst outcome: a task believes it is capped and is not. Refuse it
        # rather than pretend.
        raise SandboxError(
            f"task {task.task_id!r} requests storage_mb={limits.storage_mb}; this "
            "backend cannot guarantee a filesystem quota (Docker silently ignores "
            "storage_opt on drivers without quota support). Grade it with `harbor run`."
        )
    if limits.gpus is not None:
        raise SandboxError(
            f"task {task.task_id!r} requests {limits.gpus} GPU(s); this backend does "
            "not allocate accelerators. Grade it with `harbor run`."
        )
    return kwargs


def create_sandbox(mode: str, **kwargs) -> Sandbox:
    """Instantiate the sandbox backend named by `mode`.

    Args:
        mode (`str`):
            `"local"` or `"docker"`.

    Returns:
        [`Sandbox`]

    Raises:
        [`SandboxError`]: If `mode` is not a known backend.
    """
    backends = {"local": LocalSandbox, "docker": DockerSandbox}
    try:
        backend = backends[mode.lower()]
    except KeyError:
        raise SandboxError(
            f"unknown sandbox mode {mode!r}; expected one of {sorted(backends)}"
        ) from None
    return backend(**kwargs)


def resolve_within(base: str, relative: str) -> str:
    """Resolve `relative` under `base`, rejecting anything that escapes it.

    Used for every agent-supplied path so the agent cannot read or write outside
    its working directory — in particular, it cannot reach `/tests` or
    `/solution`.

    Raises:
        `ValueError`: If `relative` is empty, is absolute, names the working
            directory itself, or escapes `base`.
    """
    if not relative.strip():
        raise ValueError("path must name a file relative to the working directory")
    if posixpath.isabs(relative):
        raise ValueError(
            f"path must be relative to the working directory: {relative!r}"
        )
    base_norm = posixpath.normpath(base)
    target = posixpath.normpath(posixpath.join(base_norm, relative))
    if target == base_norm:
        # `""`, `"."` and `"a/.."` all land here. Allowing it would mean reading
        # a directory, or — far worse — writing a regular file over the
        # working directory itself.
        raise ValueError(
            f"path must name a file inside the working directory, not the "
            f"directory itself: {relative!r}"
        )
    if not target.startswith(base_norm.rstrip("/") + "/"):
        raise ValueError(f"path escapes the working directory: {relative!r}")
    return target


def expand_env_refs(mapping: dict[str, str]) -> dict[str, str]:
    """Resolve `${VAR}` / `$VAR` references in task-declared environment values.

    Harbor's `[environment].env` may reference variables supplied by the person
    running the task (`env = { HF_TOKEN = "${HF_TOKEN}" }`). Unset references
    expand to the empty string, so a missing secret never leaks the literal
    `${HF_TOKEN}` into the sandbox.
    """
    return {
        key: _ENV_REFERENCE.sub(
            lambda m: os.environ.get(m.group(1) or m.group(2), ""), value
        )
        for key, value in mapping.items()
    }


def _ensure_pulled(client, image: str) -> str:
    try:
        client.images.get(image)
    except Exception:
        logger.info("pulling %s", image)
        client.images.pull(image)
    return image


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-") or "task"


def _image_source(task: HarborTask) -> str:
    if task.docker_image:
        return f"[environment].docker_image = {task.docker_image!r}"
    if task.dockerfile is not None:
        return "environment/Dockerfile"
    return "environment/docker-compose.yaml"


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")
