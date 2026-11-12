"""Native Harbor execution: fresh trials graded through RFC 008 check ids.

No served-protocol evidence is synthesized. On local Docker, a do-nothing trial
builds the image, or pulls the task's digest-pinned one, and the oracle, its replay
and any canaries run pinned to that image. On `hf-sandbox`, every trial starts at
once from the task's digest-pinned image, each in its own Hugging Face Job.
"""

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from functools import partial
from importlib.metadata import distribution
from pathlib import Path

from .graders import GraderRegistry
from .graders.semantic import CanaryFloorGrader, FloorGapGrader, OracleMaxGrader
from .parsers.harbor_toml import (
    _unique_object,
    canary_scripts,
    HARBOR_PROFILE,
    read_harbor_contract,
    read_harbor_task,
)
from .policy import apply_policy
from .report import CheckResult, ValidationReportV2
from .runtime.artifacts import write_runtime_bundle
from .runtime.contracts import OracleEvidence
from .runtime.json_values import dumps
from .runtime.scheduler import execute_graders
from .types import CheckStatus, Lane, Level

_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
# A tag can move between validation and training; a digest cannot.
_PINNED_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")
_RESOURCE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\Z")
_PROJECT = re.compile(
    r"openenv-validation-[a-f0-9]+-(nop|oracle|oracle-replay|canary-[1-8])__env"
)
_MAX_REWARD_BYTES = 1024 * 1024
# Failures the trial's teardown records over any earlier one.
_HOST_FAILURES = {
    "HarborTeardownError",
    "CleanupNotVerified",
    "RewardArtifactDrift",
    "UnsafeRewardArtifact",
    "RetainedImageDrift",
    "RetainedImageUnverified",
}
_TERMINAL_STAGES = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
_JOB_STOP_TIMEOUT_S = 60
# Inside the 600-second validator start budget, so Sandbox.create finishes first.
_SANDBOX_START_TIMEOUT_S = 540
# Checks this binding measures; every other local check is a named SKIP.
_IMPLEMENTED = {
    "runtime.startup",
    "runtime.reward_well_formed",
    "runtime.episode_determinism",
    "semantic.oracle_max",
    "semantic.floor_gap",
    "semantic.canary_floor",
}


class NativeEvidenceError(ValueError):
    """A credential-safe diagnostic written by the trusted native collector."""


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class DockerInspector:
    """Trusted host inspection and removal limited to run-owned resources."""

    def __init__(self):
        self._owned_volumes = {}
        self._retained_image = None

    def _docker(self, *args):
        result = subprocess.run(
            ["docker", *args], capture_output=True, timeout=30, check=True
        )
        if len(result.stdout) > 4 * 1024 * 1024:
            raise NativeEvidenceError("Docker inspection exceeded its output limit")
        return result.stdout.decode()

    def preflight(self):
        info = json.loads(self._docker("info", "--format", "{{json .}}"))
        if info["OSType"] != "linux":
            raise NativeEvidenceError("Harbor requires a Linux Docker daemon")
        self._docker("compose", "version")
        return {
            "os": info["OSType"],
            "daemon_id": info["ID"],
            "server_version": info["ServerVersion"],
        }

    def inventory(self, project):
        if not _PROJECT.fullmatch(project):
            raise NativeEvidenceError("not a run-owned Harbor project")
        label = f"label=com.docker.compose.project={project}"
        inventory = {}
        for kind, args in (
            ("containers", ("ps", "-aq", "--no-trunc")),
            ("networks", ("network", "ls", "-q", "--no-trunc")),
            ("volumes", ("volume", "ls", "-q")),
        ):
            ids = self._docker(*args, "--filter", label).split()
            if any(not _RESOURCE_ID.fullmatch(item) for item in ids):
                raise NativeEvidenceError("invalid Docker resource identity")
            inventory[kind] = ids
        images = self._docker(
            "image", "ls", "-q", "--no-trunc", project + "-main:latest"
        ).split()
        if len(images) > 1 or any(not _IMAGE_ID.fullmatch(image) for image in images):
            raise NativeEvidenceError("invalid Compose image identity")
        inventory["images"] = images
        return inventory

    def capture(self, project, retained_tag, *, retain=True):
        inventory = self.inventory(project)
        if retained_tag != "openenv-validation:" + project.split("-", 3)[2]:
            raise NativeEvidenceError(
                "retained image tag differs from its run-owned project"
            )
        if len(inventory["containers"]) != 1:
            raise NativeEvidenceError("Harbor requires exactly one main container")
        item = json.loads(self._docker("inspect", inventory["containers"][0]))[0]
        labels = item["Config"]["Labels"]
        if (
            labels.get("com.docker.compose.project") != project
            or labels.get("com.docker.compose.service") != "main"
        ):
            raise NativeEvidenceError(
                "container does not match the native main service"
            )
        if not item["State"]["Running"] or not _IMAGE_ID.fullmatch(item["Image"]):
            raise NativeEvidenceError(
                "native container is not running with an immutable image"
            )
        if retain:
            self._retained_image = (retained_tag, item["Image"])
            self._docker("image", "tag", item["Image"], retained_tag)
        retained = json.loads(self._docker("image", "inspect", retained_tag))[0]
        if (retain and retained["Id"] != item["Image"]) or retained["Os"] != "linux":
            raise NativeEvidenceError("retained image binding changed")
        host = item["HostConfig"]
        # Dockerfile VOLUME creates anonymous volumes without Compose labels.
        volumes = [
            mount["Name"] for mount in item["Mounts"] if mount["Type"] == "volume"
        ]
        if any(not _RESOURCE_ID.fullmatch(name) for name in volumes):
            raise NativeEvidenceError("invalid mounted volume identity")
        self._owned_volumes[project] = volumes
        configuration = {
            "image_ref": item["Image"],
            "user": item["Config"]["User"],
            "workdir": item["Config"]["WorkingDir"],
            "environment_digest": _digest(sorted(item["Config"].get("Env") or [])),
            "entrypoint": item["Config"].get("Entrypoint"),
            "command": item["Config"].get("Cmd"),
            "host": {
                key: host.get(key)
                for key in (
                    "NanoCpus",
                    "Memory",
                    "MemorySwap",
                    "PidsLimit",
                    "ReadonlyRootfs",
                    "Privileged",
                    "CapAdd",
                    "CapDrop",
                    "SecurityOpt",
                    "Devices",
                )
            },
            "network_mode": "compose-default"
            if host["NetworkMode"] == project + "_default"
            else host["NetworkMode"],
            "mounts": sorted(
                [
                    {
                        "type": mount["Type"],
                        "destination": mount["Destination"],
                        "rw": mount["RW"],
                    }
                    for mount in item["Mounts"]
                ],
                key=lambda mount: mount["destination"],
            ),
        }
        return {
            "container_id": item["Id"],
            "project": project,
            "image_ref": item["Image"],
            "retained_image_tag": retained_tag,
            "configuration": configuration,
            "configuration_digest": _digest(configuration),
            "owned_volumes": volumes,
        }

    def cleanup(self, project):
        before = self.inventory(project)
        for container in before["containers"]:
            item = json.loads(self._docker("inspect", container))[0]
            if item["Config"]["Labels"].get("com.docker.compose.project") != project:
                raise NativeEvidenceError("cleanup container ownership changed")
            volumes = [
                mount["Name"] for mount in item["Mounts"] if mount["Type"] == "volume"
            ]
            if any(not _RESOURCE_ID.fullmatch(name) for name in volumes):
                raise NativeEvidenceError("invalid cleanup volume identity")
            self._owned_volumes.setdefault(project, []).extend(volumes)
        # Even a successful docker rm -v can leave an anonymous volume behind.
        registered = self._owned_volumes.get(project, [])
        existing = (
            set(self._docker("volume", "ls", "-q").split()) if registered else set()
        )
        before["volumes"] = sorted(
            set(before["volumes"]) | (set(registered) & existing)
        )
        for key, args in (
            ("containers", ("rm", "-f", "-v")),
            ("networks", ("network", "rm")),
            ("volumes", ("volume", "rm")),
        ):
            remaining = before[key]
            if key == "volumes" and remaining:
                existing = set(self._docker("volume", "ls", "-q").split())
                remaining = [name for name in remaining if name in existing]
            if remaining:
                self._docker(*args, *remaining)
        for image_ref in before["images"]:
            tag = project + "-main:latest"
            image = json.loads(self._docker("image", "inspect", tag))[0]
            labels = image["Config"]["Labels"]
            if (
                image["Id"] != image_ref
                or labels.get("com.docker.compose.project") != project
                or labels.get("com.docker.compose.service") != "main"
            ):
                raise NativeEvidenceError(
                    "Compose image ownership changed before removal"
                )
            self._docker("image", "rm", "--no-prune", tag)
        after = self.inventory(project)
        if registered:
            existing = set(self._docker("volume", "ls", "-q").split())
            after["volumes"] = sorted(
                set(after["volumes"]) | (set(registered) & existing)
            )
        return {"verified": not any(after.values()), "before": before, "after": after}

    def verify_image(self, tag, image_ref):
        image = json.loads(self._docker("image", "inspect", tag))[0]
        return image["Id"] == image_ref

    def remove_image(self):
        if self._retained_image is not None:
            tag, image_ref = self._retained_image
            if not self.verify_image(tag, image_ref):
                raise NativeEvidenceError(
                    "retained image ownership changed before removal"
                )
            self._docker("image", "rm", "--no-prune", tag)
            self._retained_image = None


def _job_labels(trial_name):
    # openenv-validation-<run id>-<role>
    run_id, role = trial_name.split("-", 3)[2:]
    return {"openenv-validation-run": run_id, "openenv-validation-role": role}


class HFJobsInspector:
    """
    Trusted HF Jobs checks limited to sandbox jobs labelled with this run.

    The image identity is the digest requested from HF Jobs and read back from the
    job; no runtime digest is measured inside the sandbox.
    """

    def __init__(self, image, flavor, namespace, resources):
        self.image = image
        self.flavor = flavor
        self.namespace = namespace
        self._resources = resources
        self._api = None

    def preflight(self):
        from huggingface_hub import HfApi

        self._api = HfApi()
        if self.namespace is None:
            self.namespace = self._api.whoami()["name"]
        hardware = {item.name: item for item in self._api.list_jobs_hardware()}
        if self.flavor not in hardware:
            raise NativeEvidenceError(f"unknown HF Jobs flavor {self.flavor}")
        item = hardware[self.flavor]
        # The Hub lists specs as "2 vCPU", "16 GB" and "50 GB".
        cpu, memory_gb, disk_gb = (
            int(spec.split()[0])
            for spec in (item.cpu, item.ram, item.ephemeral_storage)
        )
        if (
            cpu < self._resources.cpu
            or memory_gb * 1000 < self._resources.memory_mb
            or disk_gb * 1000 < self._resources.disk_mb
        ):
            raise NativeEvidenceError(
                f"HF Jobs flavor {self.flavor} ({item.cpu}, {item.ram} RAM, "
                + f"{item.ephemeral_storage} disk) is smaller than the contract resources"
            )
        return {
            "flavor": self.flavor,
            "namespace": self.namespace,
            "hardware": {
                "cpu": item.cpu,
                "ram": item.ram,
                "disk": item.ephemeral_storage,
            },
        }

    def _jobs(self, project):
        if not _PROJECT.fullmatch(project):
            raise NativeEvidenceError("not a run-owned Harbor project")
        labels = _job_labels(project.removesuffix("__env"))
        jobs = list(self._api.list_jobs(labels=labels, namespace=self.namespace))
        # Cleanup cancels what this returns, so recheck the server-side filter.
        if any(
            not labels.items() <= (job.labels or {}).items()
            or job.owner.name != self.namespace
            for job in jobs
        ):
            raise NativeEvidenceError("HF Jobs returned a job outside this run")
        return jobs

    def inventory(self, project):
        return {"jobs": sorted(job.id for job in self._jobs(project))}

    def capture(self, project, retained_tag, *, retain=True):
        jobs = self._jobs(project)
        if len(jobs) != 1:
            raise NativeEvidenceError("HF Sandbox trial requires exactly one job")
        job = jobs[0]
        if (
            job.status.stage != "RUNNING"
            or job.docker_image != self.image
            or job.flavor != self.flavor
        ):
            raise NativeEvidenceError(
                "HF Sandbox job is not running the pinned image on the requested flavor"
            )
        configuration = {
            "image_ref": job.docker_image,
            "flavor": self.flavor,
            "namespace": self.namespace,
        }
        return {
            "container_id": job.id,
            "project": project,
            "image_ref": job.docker_image,
            "configuration": configuration,
            "configuration_digest": _digest(configuration),
        }

    def cleanup(self, project):
        # A start abandoned at its deadline can still create a job, so keep looking.
        deadline = time.monotonic() + _JOB_STOP_TIMEOUT_S
        cancelled = set()
        while True:
            stages = {job.id: job.status.stage for job in self._jobs(project)}
            running = {
                job_id
                for job_id, stage in stages.items()
                if stage not in _TERMINAL_STAGES
            }
            for job_id in running - cancelled:
                self._api.cancel_job(job_id=job_id, namespace=self.namespace)
                cancelled.add(job_id)
            if not running or time.monotonic() > deadline:
                return {
                    "verified": not running,
                    "jobs": stages,
                    "cancelled": sorted(cancelled),
                }
            time.sleep(2)

    def verify_image(self, tag, image_ref):
        return image_ref == self.image

    def remove_image(self):
        pass  # Nothing is pulled or tagged on this host.


def _unsupported(task, contract, sandbox):
    config = task.config
    env = config.environment
    if task.has_steps:
        return "multi-step Harbor tasks"
    if env.os.value != "linux":
        return "non-Linux Harbor tasks"
    if env.docker_image is not None and not _PINNED_IMAGE.fullmatch(env.docker_image):
        return "task-declared environment.docker_image without a sha256 digest pin"
    if config.verifier.environment is not None or (
        config.verifier.environment_mode is not None
        and config.verifier.environment_mode.value != "shared"
    ):
        return "separate Harbor verifier environments"
    if (
        env.docker_image is None
        and not (task.task_dir / "environment/Dockerfile").is_file()
    ) or any(
        path.name.lower()
        in {"docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"}
        for path in (task.task_dir / "environment").rglob("*")
    ):
        return "tasks without one Dockerfile or pinned image, or with task-authored Compose"
    if contract.resources.gpus or env.gpus or env.tpu or env.mcp_servers:
        return "GPU/TPU or external MCP requirements"
    if contract.requires_credentials:
        return "external credential requirements"
    if env.network_mode.value != "public" or any(
        phase.network_mode is not None and phase.network_mode.value != "public"
        for phase in (config.agent, config.verifier)
    ):
        return "non-public or differing phase network policies"
    if env.env or config.verifier.env or config.solution.env:
        return "task environment variables (including host credential substitutions)"
    if env.healthcheck or env.skills_dir or config.verifier.collect or config.artifacts:
        return "healthcheck, skills, collect or artifact extensions"
    if sandbox == "hf-sandbox":
        if env.docker_image is None:
            return (
                "hf-sandbox tasks without environment.docker_image; build and push "
                + "the image, then pin it as name@sha256:<digest>"
            )
        if config.agent.user is not None or config.verifier.user is not None:
            return "agent or verifier users on hf-sandbox, which cannot switch users"
    return None


async def _create_trial(config):
    from harbor.environments.hf_sandbox import HFSandboxEnvironment
    from harbor.trial.trial import Trial

    trial = await Trial.create(config)
    environment = trial.agent_environment
    trial._openenv_cancelled_compose = []
    if isinstance(environment, HFSandboxEnvironment):
        environment.start = partial(
            _start_sandbox,
            environment,
            labels=_job_labels(config.trial_name),
            namespace=config.environment.kwargs["namespace"],
        )
    else:
        environment._run_docker_compose_command = partial(
            _run_compose, environment, cancelled=trial._openenv_cancelled_compose
        )
    return trial


async def _start_sandbox(environment, *, force_build, labels, namespace):
    # Harbor's hf-sandbox start, plus labels that let cleanup find every run-owned job.
    from huggingface_hub import Sandbox

    environment._sandbox = await asyncio.to_thread(
        Sandbox.create,
        image=environment.task_env_config.docker_image,
        flavor=environment._flavor,
        idle_timeout=environment._idle_timeout,
        namespace=namespace,
        labels=labels,
        start_timeout=_SANDBOX_START_TIMEOUT_S,
    )
    if workdir := environment.task_env_config.workdir:
        await asyncio.to_thread(environment._sandbox.files.mkdir, workdir)
    await environment.ensure_dirs(environment._mount_targets(writable_only=True))
    await environment._upload_environment_dir_after_start()


async def _run_compose(
    environment,
    command,
    check=True,
    timeout_sec=None,
    stdin_data=None,
    on_output=None,
    *,
    cancelled,
):
    # Harbor's Compose collector does not reap the client on outer cancellation.
    runtime = type(environment).runtime()
    argv = [*runtime.compose, "--project-name", environment.session_id]
    if runtime.supports_compose_project_directory:
        argv.extend(["--project-directory", str(environment.environment_dir.resolve())])
    for path in environment._docker_compose_paths:
        argv.extend(["-f", str(path.resolve())])
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *argv,
            *command,
            env=environment._compose_env_vars(include_os_env=True),
            # A task with a prebuilt image may ship no environment directory.
            cwd=str(environment.environment_dir.resolve())
            if environment.environment_dir.is_dir()
            else None,
            stdin=asyncio.subprocess.PIPE
            if stdin_data is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    )
    try:
        process = await asyncio.shield(spawn)
        if on_output is not None:
            result = await environment._collect_streamed_output(
                process,
                timeout_sec=timeout_sec,
                stdin_data=stdin_data,
                on_output=on_output,
            )
        else:
            result = await environment._collect_buffered_output(
                process, timeout_sec=timeout_sec, stdin_data=stdin_data
            )
        if check and result.return_code != 0:
            raise RuntimeError("Harbor Compose command failed")
        return result
    except BaseException as exc:
        # Finish spawning before cleanup, even if cancellation raced with exec.
        process = await spawn
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        if isinstance(exc, asyncio.CancelledError) and command[0] in {
            "build",
            "pull",
            "up",
        }:
            cancelled.append(
                {
                    "command": command[0],
                    "project": environment.session_id,
                    "client_group_killed": True,
                    "client_reaped": True,
                    "daemon_completion_verified": False,
                }
            )
        raise


def _read_output(path, root):
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise NativeEvidenceError("native output may not contain symbolic links")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_REWARD_BYTES:
            raise NativeEvidenceError(
                "native reward output must be a bounded regular file"
            )
        raw = stream.read(_MAX_REWARD_BYTES + 1)
        if len(raw) > _MAX_REWARD_BYTES:
            raise NativeEvidenceError("native reward output exceeded its size bound")
        return raw


def _reward(raw_files, parsed, contract):
    if "reward.json" in raw_files:
        raw = json.loads(
            raw_files["reward.json"],
            object_pairs_hook=_unique_object,
            parse_float=Decimal,
        )
    elif "reward.txt" in raw_files:
        raw = {"reward": Decimal(raw_files["reward.txt"].decode().strip())}
    else:
        raise NativeEvidenceError("verifier produced no raw reward file")
    if not isinstance(raw, dict) or not raw or contract.reward_key not in raw:
        raise NativeEvidenceError(
            "verifier reward map is empty or lacks the declared key"
        )
    low, high = (Decimal(str(value)) for value in contract.reward.range)
    for value in raw.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, Decimal))
            or not Decimal(value).is_finite()
            or not low <= value <= high
        ):
            raise NativeEvidenceError(
                "every reward must be finite, nonboolean and within the declared range"
            )
    if (
        not isinstance(parsed, dict)
        or any(isinstance(value, bool) for value in parsed.values())
        or raw != {key: Decimal(str(value)) for key, value in parsed.items()}
    ):
        raise NativeEvidenceError("raw verifier rewards differ from the Harbor result")
    return raw


async def _trial(
    root,
    role,
    agent,
    work,
    contract,
    inspector,
    run_id,
    source_binding,
    expected=None,
    canary=None,
):
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )
    from harbor.trial.hooks import TrialEvent

    from .providers.docker import DockerValidationProvider
    from .runner import _DIGEST_EXCLUDED_DIRS, source_digest

    name = f"openenv-validation-{run_id}-{role}"
    project = name + "__env"
    retained_tag = f"openenv-validation:{run_id}"
    record = {
        "role": role,
        "agent": agent,
        "project": project,
        "raw_rewards": {},
        "cleanup": {"verified": False},
    }
    raw_files = {}
    trial = None
    phase = "trial creation"
    registered = False
    hf = isinstance(inspector, HFJobsInspector)
    try:
        # Verify ownership before Trial.create can allocate anything.
        record["pre_allocation_inventory"] = await asyncio.to_thread(
            inspector.inventory, project
        )
        if any(record["pre_allocation_inventory"].values()):
            raise NativeEvidenceError(
                "run-owned project was not empty before allocation"
            )
        registered = True
        task_copy = work / role / "task"
        shutil.copytree(
            root,
            task_copy,
            symlinks=True,
            ignore=shutil.ignore_patterns(*_DIGEST_EXCLUDED_DIRS),
        )
        copied_digest = source_digest(task_copy)
        record["unmodified_task_digest"] = copied_digest
        if copied_digest != source_binding:
            raise NativeEvidenceError(
                "private execution task differs from the validated source digest"
            )
        overlay = {}
        if expected:
            task = read_harbor_task(task_copy)
            task.config.environment.docker_image = retained_tag
            (task_copy / "task.toml").write_text(task.config.model_dump_toml())
            overlay["environment.docker_image"] = retained_tag
        if canary is not None:
            # The oracle agent runs the canary in place of the reference solution.
            script = (task_copy / canary).read_bytes()
            (task_copy / "solution/solve.sh").write_bytes(script)
            overlay["solution/solve.sh"] = {
                "canary": canary,
                "sha256": hashlib.sha256(script).hexdigest(),
            }
        digest_before = source_digest(task_copy)
        record["execution_task_digest"] = digest_before
        record["task_overlay"] = overlay
        budget = contract.resources.episode_timeout_s
        build_budget = (
            DockerValidationProvider().build_timeout_s if not expected else budget
        )
        config = TrialConfig(
            task=TaskConfig(path=task_copy.resolve()),
            agent=AgentConfig(
                name=agent,
                override_timeout_sec=budget,
                override_setup_timeout_sec=budget,
            ),
            environment=EnvironmentConfig(
                type=EnvironmentType.HF_SANDBOX,
                delete=True,
                kwargs={
                    "flavor": inspector.flavor,
                    "namespace": inspector.namespace,
                    # Idle shutdown must outlast the longest single command.
                    "job_timeout": math.ceil(budget) + 600,
                },
            )
            if hf
            else EnvironmentConfig(
                type=EnvironmentType.DOCKER,
                delete=False,
                force_build=False,
                override_cpus=contract.resources.cpu,
                override_memory_mb=contract.resources.memory_mb,
                override_storage_mb=contract.resources.disk_mb,
                kwargs={"auto_mounting": False, "keep_containers": True},
            ),
            verifier=VerifierConfig(override_timeout_sec=budget),
            trial_name=name,
            trials_dir=(work / role / "trials").resolve(),
        )
        effective_config = config.model_dump(mode="json")
        effective_config["task"]["path"] = "<private-task-copy>"
        effective_config["trials_dir"] = "<private-trials>"
        effective_config.pop("trial_name")
        record["trial_config_digest"] = _digest(effective_config)
        record["resources"] = contract.resources.model_dump()
        trial = await _create_trial(config)
        record["harbor_build_timeout_s"] = trial._environment_build_timeout_sec
        trial._environment_build_timeout_sec = build_budget
        if not expected:
            record["build_timeout_s"] = build_budget
            start_environment = trial.agent_environment.start

            async def build_and_start(*, force_build):
                nonlocal phase
                async with asyncio.timeout(build_budget):
                    await start_environment(force_build=force_build)
                phase = "agent setup"

            trial.agent_environment.start = build_and_start

        async def environment_start(_event):
            nonlocal phase
            phase = "environment build/start"

        async def capture(_event):
            nonlocal phase
            phase = "agent"
            if trial.agent_environment.session_id != project:
                raise NativeEvidenceError(
                    "Harbor sandbox identity differs from the registered project"
                )
            identity = await asyncio.to_thread(
                inspector.capture, project, retained_tag, retain=not expected
            )
            record.update(identity)
            if expected and (
                identity["image_ref"] != expected["image_ref"]
                or identity["configuration_digest"] != expected["configuration_digest"]
            ):
                raise NativeEvidenceError(
                    "fresh native trial image or effective container configuration drifted"
                )

        async def capture_exit(_event):
            # Harbor writes the marker only for a nonzero exit. Later sandbox
            # log synchronization can overwrite it, so this reading is trusted.
            if agent == "oracle":
                marker = _read_output(
                    trial.paths.agent_dir / "exit-code.txt", trial.paths.trial_dir
                )
                record["oracle_exit_observed_at_agent_end"] = True
                record["oracle_exit_code"] = int(marker) if marker is not None else 0
                if marker is not None:
                    raw_files["oracle-exit-code.txt"] = marker

        async def verification_start(_event):
            nonlocal phase
            phase = "verifier"

        trial.add_hook(TrialEvent.ENVIRONMENT_START, environment_start)
        trial.add_hook(TrialEvent.AGENT_START, capture)
        trial.add_hook(TrialEvent.AGENT_END, capture_exit)
        trial.add_hook(TrialEvent.VERIFICATION_START, verification_start)
        result = await trial.run()
        if source_digest(task_copy) != digest_before:
            raise NativeEvidenceError(
                "private execution task changed during validation"
            )
        if result.exception_info is not None:
            failure = result.exception_info.exception_type
            if failure in {
                "RewardFileNotFoundError",
                "RewardFileEmptyError",
                "VerifierOutputParseError",
            }:
                if "container_id" not in record:
                    raise NativeEvidenceError(
                        "Harbor reward failure has no container identity"
                    )
                record["reward_error"] = True
                raise NativeEvidenceError(f"Harbor verifier reward failed ({failure})")
            if failure == "EnvironmentStartTimeoutError":
                raise NativeEvidenceError(
                    f"validator build/start exceeded its {build_budget:g}-second budget"
                )
            subject_phase = {
                "SandboxBuildFailedError": "environment build/start",
                "AgentSetupTimeoutError": "agent setup",
                "AgentTimeoutError": "agent",
                "VerifierTimeoutError": "verifier",
            }.get(failure)
            if failure == "RuntimeError" and phase == "environment build/start":
                subject_phase = phase
            if subject_phase:
                record["subject_error"] = subject_phase
                raise NativeEvidenceError(
                    f"Harbor subject failed during {subject_phase} ({failure})"
                )
            raise NativeEvidenceError("Harbor Trial returned exception_info")
        if "container_id" not in record:
            raise NativeEvidenceError("Harbor Trial has no actual AGENT_START identity")
        for filename, path in (
            ("reward.json", trial.paths.reward_json_path),
            ("reward.txt", trial.paths.reward_text_path),
        ):
            raw = _read_output(path, trial.paths.trial_dir)
            if raw is not None:
                raw_files[filename] = raw
        if agent == "oracle":
            if not record.get("oracle_exit_observed_at_agent_end"):
                raise NativeEvidenceError(
                    "oracle has no trusted AGENT_END exit observation"
                )
            marker = _read_output(
                trial.paths.agent_dir / "exit-code.txt", trial.paths.trial_dir
            )
            if marker is not None and int(marker) != record["oracle_exit_code"]:
                raise NativeEvidenceError("oracle exit marker changed after AGENT_END")
        parsed = result.verifier_result.rewards if result.verifier_result else None
        record["reward_error"] = True
        record["rewards"] = _reward(raw_files, parsed, contract)
        record.pop("reward_error")
    except (Exception, KeyboardInterrupt) as exc:
        record["failure"] = type(exc).__name__
        if isinstance(exc, TimeoutError):
            record["reason"] = (
                f"validator build/start exceeded its {build_budget:g}-second budget"
                if phase == "environment build/start"
                else f"unexpected Harbor timeout during {phase}"
            )
        # Error categories are safe; task-authored exception messages/logs are not.
        if isinstance(exc, NativeEvidenceError):
            record["reason"] = str(exc)
    finally:
        if trial is not None:
            try:
                async with asyncio.timeout(30):
                    await trial.agent_environment.stop(delete=hf)
            except (Exception, KeyboardInterrupt):
                record["failure"] = "HarborTeardownError"
        if registered:
            try:
                record["cleanup"] = await asyncio.to_thread(inspector.cleanup, project)
                if trial is not None and trial._openenv_cancelled_compose:
                    record["cleanup"].update(
                        {
                            "verified": False,
                            "pending_operations": trial._openenv_cancelled_compose,
                        }
                    )
                    record["reason"] = (
                        "Docker daemon completion after cancelled Compose allocation is unverified"
                    )
                if not record["cleanup"]["verified"]:
                    record["failure"] = "CleanupNotVerified"
            except (Exception, KeyboardInterrupt) as exc:
                record["cleanup"] = {"verified": False, "failure": type(exc).__name__}
                record["failure"] = "CleanupNotVerified"
        else:
            record["cleanup"] = {
                "verified": False,
                "reason": "project ownership was not verified before allocation",
            }
        if trial is not None:
            # Preserve raw verifier files even when Harbor rejects their values.
            for filename, path in (
                ("reward.json", trial.paths.reward_json_path),
                ("reward.txt", trial.paths.reward_text_path),
                ("oracle-exit-code.txt", trial.paths.agent_dir / "exit-code.txt"),
            ):
                try:
                    raw = _read_output(path, trial.paths.trial_dir)
                    if raw is not None:
                        if filename in raw_files and raw_files[filename] != raw:
                            record["failure"] = "RewardArtifactDrift"
                        if filename == "oracle-exit-code.txt" and filename in raw_files:
                            continue
                        raw_files[filename] = raw
                except (ValueError, OSError):
                    record["failure"] = "UnsafeRewardArtifact"
    return record, raw_files


async def _collect(root, contract, work, inspector, source_binding, canaries=()):
    run_id = uuid.uuid4().hex
    # The replay makes episode determinism measurable before any semantic check.
    plan = [
        ("nop", "nop", None),
        ("oracle", "oracle", None),
        ("oracle-replay", "oracle", None),
        *(
            (f"canary-{index}", "oracle", canary)
            for index, canary in enumerate(canaries, start=1)
        ),
    ]
    if isinstance(inspector, HFJobsInspector):
        # Each running trial blocks a worker thread for every sandbox command, and
        # a trial waiting for a free thread would spend its own phase budget.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=len(plan) + 4)
        )
    host = await asyncio.to_thread(inspector.preflight)
    run_trial = partial(
        _trial,
        root,
        work=work,
        contract=contract,
        inspector=inspector,
        run_id=run_id,
        source_binding=source_binding,
    )
    if isinstance(inspector, HFJobsInspector):
        # A pinned image needs no build, so every trial starts at once.
        outcomes = await asyncio.gather(
            *(run_trial(role, agent, canary=canary) for role, agent, canary in plan)
        )
    else:
        outcomes = []
        for role, agent, canary in plan:
            expected = outcomes[0][0] if outcomes else None
            outcomes.append(
                await run_trial(role, agent, expected=expected, canary=canary)
            )
            if "failure" in outcomes[-1][0]:
                break
    trials, files = [], {}
    for trial, raw in outcomes:
        trials.append(trial)
        for name, value in raw.items():
            relative = f"harbor-trials/{trial['role']}/{name}"
            files[relative] = value
            trial["raw_rewards"][name] = {
                "path": relative,
                "sha256": hashlib.sha256(value).hexdigest(),
            }
    if trials and "image_ref" in trials[0]:
        try:
            if not inspector.verify_image(
                f"openenv-validation:{run_id}", trials[0]["image_ref"]
            ):
                trials[-1]["failure"] = "RetainedImageDrift"
        except Exception:
            trials[-1]["failure"] = "RetainedImageUnverified"
    return host, trials, files


def _result(check_id, status, reason, measured=None):
    return CheckResult(
        check_id=check_id,
        status=status,
        evidence=[reason],
        measured=measured or {},
        duration_s=0,
    )


def _reward_only(trial):
    # `reward_error` marks a failure raised while parsing the verifier's reward:
    # the package's defect, unless teardown or cleanup later found a host problem.
    return trial.get("reward_error") and trial.get("failure") not in _HOST_FAILURES


def _skip_reason(entry, contract):
    if entry.check_id not in _IMPLEMENTED:
        if entry.level is Level.STATIC:
            return "grader not implemented in this build"
        return f"not implemented for the harbor_task binding ({HARBOR_PROFILE})"
    if entry.check_id == "semantic.canary_floor" and contract.canaries is None:
        return "no canaries declared in validation/openenv.json"
    return "unmet dependency: native Harbor trials"


def validate_harbor(
    subject,
    *,
    max_level,
    policy,
    results,
    digest_before,
    artifacts_dir,
    report_target=None,
    skip_build=False,
    provider=None,
    sandbox="docker",
    flavor="cpu-basic",
    namespace=None,
):
    """Run fresh native Harbor trials and grade them up to the requested level."""
    from .runner import source_digest

    started = time.monotonic()
    contract = read_harbor_contract(subject.root)
    semantic = max_level >= Level.SEMANTIC
    canaries = (
        canary_scripts(subject.root, contract.canaries)
        if semantic and contract.canaries is not None
        else []
    )
    record = {
        "schema_version": "2",
        "profile": HARBOR_PROFILE,
        "sandbox": sandbox,
        "task_effective_digest": digest_before,
        "original_source_digest": contract.original_source_digest,
        "contract_digest": _digest(contract.model_dump(mode="json")),
        "reward_key": contract.reward_key,
        "cleanup": {"verified": False},
        "trials": [],
    }
    harbor_dist = distribution("harbor")
    record["harbor"] = {"version": harbor_dist.version}
    direct_url = harbor_dist.read_text("direct_url.json")
    if direct_url:
        installed = json.loads(direct_url)
        record["harbor"]["commit_id"] = installed.get("vcs_info", {}).get("commit_id")
    files = {}
    inspector = DockerInspector()
    try:
        attempted = False
        task = read_harbor_task(subject.root)
        problem = _unsupported(task, contract, sandbox)
        if sandbox == "hf-sandbox":
            inspector = HFJobsInspector(
                task.config.environment.docker_image,
                flavor,
                namespace,
                contract.resources,
            )
        if skip_build:
            problem = "image build skipped by --skip-build"
        if provider is not None:
            problem = (
                "served-package providers, including --space; "
                + "run Harbor trials on HF Jobs with --sandbox hf-sandbox"
            )
        if any(result.status is not CheckStatus.PASS for result in results):
            problem = "unmet dependency: static.manifest"
        startup = _result(
            "runtime.startup",
            CheckStatus.SKIP,
            f"unsupported by {HARBOR_PROFILE}: {problem}",
        )
        reward_result = _result(
            "runtime.reward_well_formed",
            CheckStatus.SKIP,
            "unmet dependency: native Harbor trials",
        )
        determinism = _result(
            "runtime.episode_determinism",
            CheckStatus.SKIP,
            "unmet dependency: native Harbor trials",
        )
        evidence = None
        checks = []
        semantic_graded = False
        if (
            semantic
            and subject.manifest.capabilities.oracle is None
            and all(result.status is CheckStatus.PASS for result in results)
        ):
            # RFC 008: a missing oracle is a graded FAIL, never a SKIP.
            semantic_graded = True
            checks.append(
                CheckResult(
                    check_id="semantic.oracle_max",
                    status=CheckStatus.FAIL,
                    evidence=["the task has no oracle (solution/solve.sh)"],
                    remediation=(
                        "Add solution/solve.sh that drives the task to a correct solution."
                    ),
                    duration_s=0,
                )
            )
        if not problem:
            if subject.manifest.capabilities.oracle is None:
                startup = _result(
                    "runtime.startup",
                    CheckStatus.SKIP,
                    "native Harbor trials need solution/solve.sh",
                )
            else:
                attempted = True
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="openenv-harbor-validation-"
                    ) as temp:
                        host, trials, files = asyncio.run(
                            _collect(
                                subject.root,
                                contract,
                                Path(temp),
                                inspector,
                                digest_before,
                                canaries,
                            )
                        )
                    record["trials"] = trials
                    if sandbox == "hf-sandbox":
                        record["hf_jobs"] = host
                        record["limitations"] = [
                            "The image is the digest requested from HF Jobs and read "
                            + "back from each job, not a digest measured at runtime."
                        ]
                    else:
                        record["daemon"] = host
                    record["cleanup"] = {
                        "verified": bool(trials)
                        and all(trial["cleanup"]["verified"] for trial in trials)
                    }
                    broken = [
                        trial
                        for trial in trials
                        if "failure" in trial and not _reward_only(trial)
                    ]
                    malformed = any(_reward_only(trial) for trial in trials)
                    containers = [trial.get("container_id") for trial in trials]
                    if (
                        broken
                        or (not malformed and len(trials) != 3 + len(canaries))
                        or len(set(containers)) != len(containers)
                    ):
                        subject_failures = broken and all(
                            trial.get("subject_error")
                            and trial["failure"] not in _HOST_FAILURES
                            for trial in broken
                        )
                        startup = _result(
                            "runtime.startup",
                            CheckStatus.FAIL if subject_failures else CheckStatus.ERROR,
                            "native Harbor subject failed during "
                            + ", ".join(
                                sorted({trial["subject_error"] for trial in broken})
                            )
                            if subject_failures
                            else "native Harbor trial, binding or cleanup failed",
                            {
                                "failures": [
                                    trial.get("reason", trial.get("failure"))
                                    for trial in broken
                                ]
                            },
                        )
                        startup.evidence.extend(
                            trial["reason"] for trial in broken if "reason" in trial
                        )
                        if malformed:
                            reward_result = _result(
                                "runtime.reward_well_formed",
                                CheckStatus.FAIL,
                                "native verifier reward violates the explicit contract",
                            )
                    else:
                        record["image_ref"] = trials[0]["image_ref"]
                        startup = _result(
                            "runtime.startup",
                            CheckStatus.PASS,
                            "fresh native Harbor trials ran from one immutable image "
                            "with verified cleanup",
                            {
                                "profile": HARBOR_PROFILE,
                                "image_ref": trials[0]["image_ref"],
                                "roles": [trial["role"] for trial in trials],
                                "task_effective_digest": digest_before,
                            },
                        )
                        source_image = task.config.environment.docker_image
                        if source_image is not None:
                            record["source_image"] = source_image
                            startup.measured["source_image"] = source_image
                        if sandbox == "hf-sandbox":
                            startup.measured.update(
                                sandbox=sandbox,
                                flavor=host["flavor"],
                                namespace=host["namespace"],
                                job_ids=containers,
                            )
                        if malformed:
                            reward_result = _result(
                                "runtime.reward_well_formed",
                                CheckStatus.FAIL,
                                "native verifier reward violates the explicit contract",
                            )
                            determinism = _result(
                                "runtime.episode_determinism",
                                CheckStatus.SKIP,
                                "unmet dependency: runtime.reward_well_formed",
                            )
                        else:
                            reward_result = _result(
                                "runtime.reward_well_formed",
                                CheckStatus.PASS,
                                "every raw verifier reward map matches Harbor and "
                                "satisfies the declared key/range",
                                {
                                    "profile": HARBOR_PROFILE,
                                    "reward_key": contract.reward_key,
                                },
                            )
                    complete_baseline = (
                        len(trials) >= 3
                        and all(
                            "failure" not in trial and trial["cleanup"]["verified"]
                            for trial in trials[:3]
                        )
                        and len(set(containers[:3])) == 3
                    )
                    if complete_baseline:
                        nop, oracle, replay, *canary_trials = trials
                        same = (
                            oracle["rewards"] == replay["rewards"]
                            and oracle["oracle_exit_code"] == replay["oracle_exit_code"]
                        )
                        determinism = _result(
                            "runtime.episode_determinism",
                            CheckStatus.PASS if same else CheckStatus.FAIL,
                            "the oracle replay in a fresh container "
                            + (
                                "reproduced every reward and the exit code"
                                if same
                                else "produced different rewards or a different exit code"
                            ),
                            {
                                "rewards": [
                                    json.loads(dumps(trial["rewards"]))
                                    for trial in (oracle, replay)
                                ],
                                "exit_codes": [
                                    trial["oracle_exit_code"]
                                    for trial in (oracle, replay)
                                ],
                            },
                        )
                        key = contract.reward_key
                        record.update(
                            {
                                "floor_reward": nop["rewards"][key],
                                "oracle_reward": oracle["rewards"][key],
                            }
                        )
                        evidence = OracleEvidence(
                            floor=Fraction(nop["rewards"][key]),
                            oracle=tuple(
                                Fraction(trial["rewards"][key])
                                for trial in (oracle, replay)
                            ),
                            oracle_exit_codes=tuple(
                                trial["oracle_exit_code"] for trial in (oracle, replay)
                            ),
                            canaries=tuple(
                                (
                                    trial["task_overlay"]["solution/solve.sh"][
                                        "canary"
                                    ],
                                    Fraction(trial["rewards"][key]),
                                )
                                for trial in canary_trials
                                if "failure" not in trial
                                and trial["cleanup"]["verified"]
                            ),
                        )
                except (Exception, KeyboardInterrupt) as exc:
                    record["failure"] = type(exc).__name__
                    reason = f"native Harbor collection failed ({type(exc).__name__})"
                    if isinstance(exc, NativeEvidenceError):
                        record["reason"] = str(exc)
                        reason += ": " + str(exc)
                    startup = _result(
                        "runtime.startup",
                        CheckStatus.ERROR,
                        reason,
                    )
                if semantic:
                    registry = GraderRegistry()
                    for grader in (
                        OracleMaxGrader(),
                        FloorGapGrader(),
                        CanaryFloorGrader(),
                    ):
                        registry.register(grader)
                    selected = registry.select(subject.manifest, max_level)
                    if evidence is not None and len(evidence.canaries) != len(canaries):
                        selected = [
                            grader
                            for grader in selected
                            if grader.check_id != "semantic.canary_floor"
                        ]
                        checks.append(
                            _result(
                                "semantic.canary_floor",
                                CheckStatus.SKIP,
                                "unmet dependency: complete canary trials with verified cleanup",
                            )
                        )
                    checks.extend(
                        execute_graders(
                            selected,
                            replace(subject, oracle_evidence=evidence),
                            prior=[startup, reward_result, determinism],
                        )
                    )
        try:
            unchanged = source_digest(subject.root) == digest_before
        except (ValueError, OSError):
            unchanged = False
        if not unchanged:
            record["failure"] = "SourceDrift"
            startup = _result(
                "runtime.startup",
                CheckStatus.ERROR,
                "task source changed during native validation",
            )
            reward_result, determinism, *checks = (
                _result(
                    check.check_id,
                    CheckStatus.SKIP,
                    "unmet dependency: unchanged native task source",
                )
                for check in (reward_result, determinism, *checks)
            )
        startup.duration_s = time.monotonic() - started
        record["cleanup"].update(
            {"required": attempted, "completed": record["cleanup"]["verified"]}
        )
        results = [*results, startup, reward_result, determinism, *checks]
        present = {result.check_id for result in results}
        for entry in policy.entries_for_lane(Lane.LOCAL).values():
            if entry.level <= max_level and entry.check_id not in present:
                results.append(
                    _result(
                        entry.check_id,
                        CheckStatus.SKIP,
                        _skip_reason(entry, contract),
                    )
                )
        if any(
            check.status in {CheckStatus.FAIL, CheckStatus.ERROR} for check in results
        ):
            try:
                inspector.remove_image()
            except (Exception, KeyboardInterrupt) as exc:
                startup.status = CheckStatus.ERROR
                startup.evidence.append(
                    f"retained image removal failed ({type(exc).__name__})"
                )
                record["cleanup"].update(verified=False, completed=False)
        elif "image_ref" in record and sandbox == "docker":
            tag = trials[0]["retained_image_tag"]
            startup.measured["retained_image_tag"] = tag
            record["retained_image_tag"] = tag
        levels = [Level.STATIC]
        if attempted:
            levels.append(Level.RUNTIME)
        if semantic and (attempted or semantic_graded):
            levels.append(Level.SEMANTIC)
        report = ValidationReportV2(
            report_schema_version="2",
            target=str(subject.root) if report_target is None else report_target,
            source_digest=digest_before,
            signature=subject.manifest.signature,
            manifest=subject.manifest,
            policy_version=policy.policy_version,
            lane=Lane.LOCAL,
            levels_run=levels,
            results=results,
            verdict=apply_policy(results, policy, Lane.LOCAL),
        )
        if artifacts_dir is not None:
            write_runtime_bundle(
                artifacts_dir,
                report,
                cleanup=record["cleanup"],
                provider={
                    "binding": "harbor_task",
                    "profile": HARBOR_PROFILE,
                    "sandbox": sandbox,
                },
            )
            files["harbor-evidence.json"] = (dumps(record) + "\n").encode()
            checksums = (artifacts_dir / "SHA256SUMS").read_text()
            for name, raw in sorted(files.items()):
                path = artifacts_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                path.chmod(0o600)
                checksums += f"{hashlib.sha256(raw).hexdigest()}  {name}\n"
            (artifacts_dir / "SHA256SUMS").write_text(checksums)
        return report
    except BaseException as original:
        try:
            inspector.remove_image()
        except (Exception, KeyboardInterrupt) as exc:
            record["cleanup"].update(
                verified=False, completed=False, failure=type(exc).__name__
            )
            original.add_note(
                f"retained image removal failed ({type(exc).__name__}); cleanup incomplete"
            )
        raise
