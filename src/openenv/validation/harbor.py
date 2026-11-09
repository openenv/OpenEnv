"""Native Harbor oracle/floor pilot; no served-protocol evidence is synthesized."""

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from decimal import Decimal
from fractions import Fraction
from importlib.metadata import distribution
from pathlib import Path

from .parsers.harbor_toml import HARBOR_PROFILE, read_harbor_contract, read_harbor_task
from .policy import apply_policy
from .report import CheckResult, ValidationReportV2
from .runtime.artifacts import write_runtime_bundle
from .runtime.json_values import dumps
from .types import CheckStatus, Lane, Level

_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RESOURCE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\Z")
_MAX_REWARD_BYTES = 1024 * 1024


class NativeEvidenceError(ValueError):
    """A credential-safe diagnostic written by the trusted native collector."""


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class DockerInspector:
    """Trusted host inspection and removal limited to exact run-owned projects."""

    def __init__(self):
        self._owned_volumes = {}

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
            raise NativeEvidenceError("the Harbor pilot requires a Linux Docker daemon")
        self._docker("compose", "version")
        return {
            "os": info["OSType"],
            "daemon_id": info["ID"],
            "server_version": info["ServerVersion"],
        }

    def inventory(self, project):
        if not re.fullmatch(r"openenv-validation-[a-f0-9]+-(nop|oracle)__env", project):
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
        return inventory

    def capture(self, project, retained_tag, *, retain=True):
        inventory = self.inventory(project)
        if len(inventory["containers"]) != 1:
            raise NativeEvidenceError(
                "the Harbor pilot requires exactly one main container"
            )
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


def _unsupported(task, contract):
    config = task.config
    env = config.environment
    if task.has_steps:
        return "multi-step Harbor tasks"
    if env.os.value != "linux":
        return "non-Linux Harbor tasks"
    if config.verifier.environment is not None or (
        config.verifier.environment_mode is not None
        and config.verifier.environment_mode.value != "shared"
    ):
        return "separate Harbor verifier environments"
    if not (task.task_dir / "environment/Dockerfile").is_file() or any(
        path.name.lower()
        in {"docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"}
        for path in (task.task_dir / "environment").rglob("*")
    ):
        return "tasks without one Dockerfile or with task-authored Compose"
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
    return None


async def _create_trial(config):
    from harbor.trial.trial import Trial

    return await Trial.create(config)


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
    from .parsers.harbor_toml import _unique_object

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
    root, role, work, contract, inspector, run_id, source_binding, expected=None
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

    from .runner import _DIGEST_EXCLUDED_DIRS, source_digest

    name = f"openenv-validation-{run_id}-{role}"
    project = name + "__env"
    retained_tag = f"openenv-validation:{run_id}"
    record = {
        "role": role,
        "project": project,
        "raw_rewards": {},
        "cleanup": {"verified": False},
    }
    raw_files = {}
    trial = None
    # Register and verify ownership before Trial.create can allocate anything.
    if any(inspector.inventory(project).values()):
        raise NativeEvidenceError("run-owned project was not empty before allocation")
    try:
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
        if expected:
            task = read_harbor_task(task_copy)
            task.config.environment.docker_image = retained_tag
            (task_copy / "task.toml").write_text(task.config.model_dump_toml())
        digest_before = source_digest(task_copy)
        record["execution_task_digest"] = digest_before
        record["task_overlay"] = (
            {"environment.docker_image": retained_tag} if expected else {}
        )
        budget = contract.resources.episode_timeout_s
        config = TrialConfig(
            task=TaskConfig(path=task_copy.resolve()),
            agent=AgentConfig(
                name=role,
                override_timeout_sec=budget,
                override_setup_timeout_sec=budget,
            ),
            environment=EnvironmentConfig(
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
        async with asyncio.timeout(budget):
            trial = await _create_trial(config)

            async def capture(_event):
                if trial.agent_environment.session_id != project:
                    raise NativeEvidenceError(
                        "Harbor sandbox identity differs from the registered project"
                    )
                identity = inspector.capture(project, retained_tag, retain=not expected)
                record.update(identity)
                if expected and (
                    identity["image_ref"] != expected["image_ref"]
                    or identity["configuration_digest"]
                    != expected["configuration_digest"]
                ):
                    raise NativeEvidenceError(
                        "fresh native trial image or effective container configuration drifted"
                    )

            async def capture_exit(_event):
                if role == "oracle":
                    marker = _read_output(
                        trial.paths.agent_dir / "exit-code.txt", trial.paths.trial_dir
                    )
                    record["oracle_exit_observed_at_agent_end"] = True
                    record["oracle_exit_code"] = (
                        int(marker) if marker is not None else 0
                    )
                    if marker is not None:
                        raw_files["oracle-exit-code.txt"] = marker
                    if record["oracle_exit_code"] != 0:
                        raise NativeEvidenceError(
                            "oracle exit-code.txt records a nonzero exit"
                        )

            trial.add_hook(TrialEvent.AGENT_START, capture)
            trial.add_hook(TrialEvent.AGENT_END, capture_exit)
            result = await trial.run()
        if source_digest(task_copy) != digest_before:
            raise NativeEvidenceError(
                "private execution task changed during validation"
            )
        if result.exception_info is not None:
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
        if role == "oracle":
            if not record.get("oracle_exit_observed_at_agent_end"):
                raise NativeEvidenceError(
                    "oracle has no trusted AGENT_END exit observation"
                )
            marker = _read_output(
                trial.paths.agent_dir / "exit-code.txt", trial.paths.trial_dir
            )
            if marker is not None and int(marker) != 0:
                raise NativeEvidenceError("oracle exit-code.txt records a nonzero exit")
        parsed = result.verifier_result.rewards if result.verifier_result else None
        record["reward_error"] = True
        record["rewards"] = _reward(raw_files, parsed, contract)
        record.pop("reward_error")
    except (Exception, KeyboardInterrupt) as exc:
        record["failure"] = type(exc).__name__
        # Error categories are safe; task-authored exception messages/logs are not.
        if isinstance(exc, NativeEvidenceError):
            record["reason"] = str(exc)
    finally:
        if trial is not None:
            try:
                async with asyncio.timeout(30):
                    await trial.agent_environment.stop(delete=False)
            except (Exception, KeyboardInterrupt):
                record["failure"] = "HarborTeardownError"
        try:
            record["cleanup"] = inspector.cleanup(project)
            if not record["cleanup"]["verified"]:
                record["failure"] = "CleanupNotVerified"
        except (Exception, KeyboardInterrupt) as exc:
            record["cleanup"] = {"verified": False, "failure": type(exc).__name__}
            record["failure"] = "CleanupNotVerified"
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


async def _collect(root, contract, work, inspector, source_binding):
    run_id = uuid.uuid4().hex
    daemon = inspector.preflight()
    trials, files = [], {}
    for role in ("nop", "oracle"):
        trial, raw = await _trial(
            root,
            role,
            work,
            contract,
            inspector,
            run_id,
            source_binding,
            trials[0] if trials else None,
        )
        trials.append(trial)
        for name, value in raw.items():
            relative = f"harbor-trials/{role}/{name}"
            files[relative] = value
            trial["raw_rewards"][name] = {
                "path": relative,
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        if "failure" in trial:
            break
    if trials and "image_ref" in trials[0]:
        try:
            if not inspector.verify_image(
                f"openenv-validation:{run_id}", trials[0]["image_ref"]
            ):
                trials[-1]["failure"] = "RetainedImageDrift"
        except Exception:
            trials[-1]["failure"] = "RetainedImageUnverified"
    return daemon, trials, files


def _result(check_id, status, reason, measured=None):
    return CheckResult(
        check_id=check_id,
        status=status,
        evidence=[reason],
        measured=measured or {},
        duration_s=0,
    )


def validate_harbor(
    subject,
    *,
    max_level,
    policy,
    results,
    digest_before,
    artifacts_dir,
    skip_build=False,
    provider=None,
):
    """Return the usual report plus explicitly bounded native pilot evidence."""
    from .runner import source_digest

    started = time.monotonic()
    contract = read_harbor_contract(subject.root)
    record = {
        "schema_version": "1",
        "profile": HARBOR_PROFILE,
        "task_effective_digest": digest_before,
        "original_source_digest": contract.original_source_digest,
        "contract_digest": _digest(contract.model_dump(mode="json")),
        "reward_key": contract.reward_key,
        "oracle_max": {"passed": False},
        "floor_gap": {"passed": False},
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
    attempted = False
    problem = _unsupported(read_harbor_task(subject.root), contract)
    if skip_build:
        problem = "image build skipped by --skip-build"
    if provider is not None:
        problem = "native Harbor binding requires its own Docker Trial provider"
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
    if not problem:
        if subject.manifest.capabilities.oracle is None:
            startup = _result(
                "runtime.startup",
                CheckStatus.FAIL,
                "native pilot requires solution/solve.sh",
            )
        else:
            attempted = True
            try:
                with tempfile.TemporaryDirectory(
                    prefix="openenv-harbor-validation-"
                ) as temp:
                    daemon, trials, files = asyncio.run(
                        _collect(
                            subject.root,
                            contract,
                            Path(temp),
                            DockerInspector(),
                            digest_before,
                        )
                    )
                record.update({"daemon": daemon, "trials": trials})
                record["cleanup"] = {
                    "verified": bool(trials)
                    and all(trial["cleanup"]["verified"] for trial in trials)
                }
                failed = [trial for trial in trials if "failure" in trial]
                if failed or len(trials) != 2:
                    startup = _result(
                        "runtime.startup",
                        CheckStatus.ERROR,
                        "native Harbor trial, binding or cleanup failed",
                        {
                            "failures": [
                                trial.get("reason", trial.get("failure"))
                                for trial in failed
                            ]
                        },
                    )
                    if any(trial.get("reward_error") for trial in failed):
                        reward_result = _result(
                            "runtime.reward_well_formed",
                            CheckStatus.FAIL,
                            "native verifier reward violates the explicit contract",
                        )
                else:
                    floor, oracle = (
                        trial["rewards"][contract.reward_key] for trial in trials
                    )
                    maximum = Decimal(str(contract.reward.range[1]))
                    oracle_pass = Fraction(oracle) >= Fraction(maximum) - Fraction(
                        str(contract.reward.oracle_tolerance)
                    )
                    gap_pass = Fraction(oracle) - Fraction(floor) >= Fraction(
                        str(contract.reward.floor_margin)
                    ) and Fraction(maximum) - Fraction(floor) >= Fraction(
                        str(contract.reward.floor_margin)
                    )
                    record.update(
                        {
                            "image_ref": trials[0]["image_ref"],
                            "retained_image_tag": trials[0]["retained_image_tag"],
                            "floor_reward": floor,
                            "oracle_reward": oracle,
                            "oracle_max": {
                                "passed": oracle_pass,
                                "declared_max": maximum,
                                "tolerance": contract.reward.oracle_tolerance,
                            },
                            "floor_gap": {
                                "passed": gap_pass,
                                "comparison": "exact-rational",
                                "measured_gap": oracle - floor,
                                "declared_margin": contract.reward.floor_margin,
                            },
                        }
                    )
                    distinct = trials[0]["container_id"] != trials[1]["container_id"]
                    startup = _result(
                        "runtime.startup",
                        CheckStatus.PASS
                        if oracle_pass and gap_pass and distinct
                        else CheckStatus.FAIL,
                        "fresh native Harbor trials with verified cleanup; pilot oracle/floor gates evaluated",
                        {
                            key: json.loads(dumps(record[key]))
                            for key in (
                                "profile",
                                "image_ref",
                                "floor_reward",
                                "oracle_reward",
                                "oracle_max",
                                "floor_gap",
                                "cleanup",
                                "task_effective_digest",
                            )
                        },
                    )
                    reward_result = _result(
                        "runtime.reward_well_formed",
                        CheckStatus.PASS,
                        "both raw verifier reward maps match Harbor and satisfy the declared key/range",
                        {"profile": HARBOR_PROFILE, "reward_key": contract.reward_key},
                    )
            except (Exception, KeyboardInterrupt) as exc:
                record["failure"] = type(exc).__name__
                startup = _result(
                    "runtime.startup",
                    CheckStatus.ERROR,
                    f"native Harbor collection failed ({type(exc).__name__})",
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
        reward_result = _result(
            "runtime.reward_well_formed",
            CheckStatus.SKIP,
            "unmet dependency: unchanged native task source",
        )
    startup.duration_s = time.monotonic() - started
    record["cleanup"].update(
        {"required": attempted, "completed": record["cleanup"]["verified"]}
    )
    results = [*results, startup, reward_result]
    present = {result.check_id for result in results}
    for entry in policy.entries_for_lane(Lane.LOCAL).values():
        if entry.level <= max_level and entry.check_id not in present:
            results.append(
                _result(
                    entry.check_id,
                    CheckStatus.SKIP,
                    f"out of profile: {HARBOR_PROFILE}; native process binding has no implemented evidence for this check",
                )
            )
    report = ValidationReportV2(
        report_schema_version="2",
        target=str(subject.root),
        source_digest=digest_before,
        signature=subject.manifest.signature,
        manifest=subject.manifest,
        policy_version=policy.policy_version,
        lane=Lane.LOCAL,
        levels_run=[Level.STATIC, Level.RUNTIME] if attempted else [Level.STATIC],
        results=results,
        verdict=apply_policy(results, policy, Lane.LOCAL),
    )
    if artifacts_dir is not None:
        write_runtime_bundle(
            artifacts_dir,
            report,
            cleanup=record["cleanup"],
            provider={"binding": "harbor_task", "profile": HARBOR_PROFILE},
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
